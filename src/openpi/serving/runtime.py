"""Runtime coordinator for the managed VLM/FM deployment.

This module is deliberately independent from JAX model construction.  The runtime
process owns the public protocol and coordinates two private role servers; the role
servers continue to use :class:`MultiProcessPolicy` for model execution.  The
legacy single-process and multi-process entry points do not import this module.
"""

from __future__ import annotations

import logging
import threading
from typing import Any
import uuid

from openpi_client.websocket_client_policy import WebsocketClientPolicy

logger = logging.getLogger(__name__)


class RuntimeCoordinator:
    """Coordinate VLM prefix construction and FM inference behind one endpoint.

    The public client never needs to know about cache IDs or the private worker
    endpoints.  A plain observation performs a complete prefix-plus-FM inference;
    streaming requests can set ``update_prefix`` when a new visual context is
    needed.  The coordinator is intentionally synchronous in this first runtime
    implementation, while worker processes remain isolated on their assigned GPUs.
    """

    # WebsocketPolicyServer uses this flag to route requests through
    # ``infer_with_session`` and to reset session state when a connection closes.
    tactile_history_enabled = True

    def __init__(
        self,
        *,
        vlm_host: str,
        vlm_port: int,
        fm_host: str,
        fm_port: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._vlm_host = vlm_host
        self._vlm_port = vlm_port
        self._fm_host = fm_host
        self._fm_port = fm_port
        self._metadata = metadata or {}
        self._vlm = WebsocketClientPolicy(vlm_host, vlm_port)
        self._fm = WebsocketClientPolicy(fm_host, fm_port)
        # A synchronous WebSocket connection is request/response ordered and is
        # not safe to use concurrently from multiple public handler threads.
        self._vlm_io_lock = threading.Lock()
        self._fm_io_lock = threading.Lock()
        self._cache_version: str | None = None
        self._cache_id: int | None = None
        self._session_locks: dict[str, threading.Lock] = {}
        self._session_locks_lock = threading.Lock()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._session_locks_lock:
            return self._session_locks.setdefault(session_id, threading.Lock())

    def _vlm_infer(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._vlm_io_lock:
            return self._vlm.infer(request)

    def _fm_infer(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._fm_io_lock:
            return self._fm.infer(request)

    def _refresh_prefix(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Build a prefix in VLM and activate it in FM through the runtime."""
        request_id = uuid.uuid4().hex
        marker = self._fm_infer({"op": "observation_started", "request_id": request_id})
        descriptor = self._vlm_infer({
            "op": "encode_prefix_to_shared_memory",
            "observation": observation,
            "observation_request_id": request_id,
            "observation_chunk_id": marker.get("observation_chunk_id"),
        })
        artifact_id = descriptor.get("artifact_id")
        try:
            activated = self._fm_infer({"op": "activate_shared_memory_prefix", "cache": descriptor})
            self._cache_id = int(activated["cache_id"])
            self._cache_version = activated["cache_version"]
        finally:
            if isinstance(artifact_id, str):
                try:
                    self._vlm_infer({"op": "release_shared_memory", "artifact_id": artifact_id})
                except Exception:
                    # Preserve an activation error (or the action result) while
                    # allowing the VLM process's atexit cleanup to reclaim a
                    # segment if the control connection is already unhealthy.
                    logger.exception("Failed to release shared-memory artifact %s", artifact_id)
        return {
            "cache_id": self._cache_id,
            "cache_version": self._cache_version,
            "status": activated.get("status", "active"),
        }

    def _reset(self, request: dict[str, Any]) -> dict[str, Any]:
        result = self._fm_infer({
            "op": "reset_stream",
            "session_id": request.get("session_id", "default"),
            "clear_prefix_cache": bool(request.get("clear_prefix_cache", False)),
        })
        if result.get("prefix_cache_cleared"):
            self._cache_id = None
            self._cache_version = None
        return result

    def infer_with_session(self, request: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        request = dict(request)
        request.setdefault("session_id", session_id)
        return self.infer(request)

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op", "infer")
        session_id = request.get("session_id", "default")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")

        with self._session_lock(session_id):
            if op in ("reset", "reset_stream"):
                return self._reset(request)

            if op in ("update_vlm", "encode_prefix"):
                if "observation" not in request:
                    raise ValueError(f"{op} requires an observation")
                result = self._refresh_prefix(request["observation"])
                result["protocol_version"] = 1
                return result

            if op == "stream_infer":
                update_prefix = bool(request.get("update_prefix", self._cache_version is None))
            elif op == "infer":
                # A plain policy client expects every infer call to condition on
                # its current image/prompt, just like the legacy single-process
                # server.  Callers can opt out for repeated FM-only requests.
                update_prefix = bool(request.get("update_prefix", True))
            else:
                raise ValueError(f"Unsupported Runtime operation: {op}")

            if "observation" not in request:
                raise ValueError(f"{op} requires an observation")
            if update_prefix or self._cache_version is None:
                self._refresh_prefix(request["observation"])

            fm_request = dict(request)
            fm_request["op"] = op
            fm_request["session_id"] = session_id
            if self._cache_version is not None:
                fm_request["expected_cache_version"] = self._cache_version
            fm_request.pop("update_prefix", None)
            return self._fm_infer(fm_request)

    def reset_tactile_history(self, session_id: str) -> None:
        """Release streaming state when a public websocket connection closes."""
        try:
            with self._session_lock(session_id):
                self._fm_infer({
                    "op": "reset_stream",
                    "session_id": session_id,
                    "clear_prefix_cache": False,
                })
        except Exception:
            logger.exception("Failed to reset Runtime session %s", session_id)
        self._cache_id = None
        self._cache_version = None

    def close(self) -> None:
        for client in (self._vlm, self._fm):
            websocket = getattr(client, "_ws", None)
            if websocket is not None:
                websocket.close()
