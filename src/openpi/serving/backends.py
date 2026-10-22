"""Worker backends used by the unified inference Runtime.

The Runtime only sees the small interface below.  ``LocalBackend`` keeps the
prefix cache as a JAX pytree in one process, while ``ProcessBackend`` hides the
worker WebSockets and shared-memory artifact lifecycle.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Literal
import uuid

from openpi_client.websocket_client_policy import WebsocketClientPolicy

from openpi.policies import multi_process_policy as _multi_process_policy
from openpi.policies import policy as _policy
from openpi.serving.runtime import CacheInfo

logger = logging.getLogger(__name__)


def _runtime_model_metadata(model: Any) -> dict[str, Any]:
    return {
        name: getattr(model, name)
        for name in (
            "streaming",
            "action_horizon",
            "streaming_chunk_size",
            "gate_history_length",
            "gate_sources",
        )
        if hasattr(model, name)
    }


class LocalBackend:
    """Run VLM and FM computations in one process using direct JAX objects."""

    def __init__(
        self,
        policy: _policy.Policy,
        *,
        streaming_inference_mode: Literal["chunk", "action_head", "chunk_heads"] = "chunk",
    ):
        self._policy = policy
        # MultiProcessPolicy already contains the streaming/cache implementation
        # used by the split workers.  Here it is only a local JAX coordinator; no
        # worker thread or network transport is created.
        self._worker = _multi_process_policy.MultiProcessPolicy(
            policy, "fm", streaming_inference_mode=streaming_inference_mode
        )
        self._lock = threading.RLock()

    @property
    def metadata(self) -> dict[str, Any]:
        model = self._policy._model  # noqa: SLF001
        return {
            **self._policy.metadata,
            "backend": "local",
            "runtime_model": _runtime_model_metadata(model),
        }

    def warmup(self) -> None:
        with self._lock:
            self._worker.warmup()

    def update_prefix(
        self,
        observation: dict[str, Any],
        *,
        session_id: str,
        executed_action_id: int | None = None,
    ) -> CacheInfo:
        with self._lock:
            result = self._worker.encode_prefix_local(
                observation,
                session_id=session_id,
                executed_action_id=executed_action_id,
            )
            return CacheInfo(int(result["cache_id"]), result["cache_version"])

    def infer(
        self,
        request: dict[str, Any],
        *,
        session_id: str,
        cache_version: str | None,
    ) -> dict[str, Any]:
        with self._lock:
            payload = dict(request)
            payload["session_id"] = session_id
            if cache_version is not None:
                payload["expected_cache_version"] = cache_version
            return self._worker.infer(payload)

    def reset(self, *, session_id: str, clear_prefix_cache: bool) -> dict[str, Any]:
        with self._lock:
            return self._worker.infer(
                {
                    "op": "reset_stream",
                    "session_id": session_id,
                    "clear_prefix_cache": clear_prefix_cache,
                }
            )

    def close(self) -> None:
        return None


class ProcessBackend:
    """Coordinate the private VLM/FM processes through control WebSockets."""

    # Prefix encoding runs in a separate VLM process and can overlap the active
    # FM inference. LocalBackend intentionally doesn't advertise this capability.
    supports_async_prefix_refresh = True

    def __init__(
        self,
        *,
        vlm_host: str,
        vlm_port: int,
        fm_host: str,
        fm_port: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._vlm = WebsocketClientPolicy(vlm_host, vlm_port)
        self._fm = WebsocketClientPolicy(fm_host, fm_port)
        self._metadata = metadata or {}
        self._vlm_io_lock = threading.Lock()
        self._fm_io_lock = threading.Lock()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self._metadata,
            "backend": "process",
            "vlm_worker_metadata": self._vlm.get_server_metadata(),
            "fm_worker_metadata": self._fm.get_server_metadata(),
        }

    def warmup(self) -> None:
        # Worker startup performs model warmup before its websocket is opened.
        return None

    def _vlm_infer(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._vlm_io_lock:
            return self._vlm.infer(request)

    def _fm_infer(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._fm_io_lock:
            return self._fm.infer(request)

    def update_prefix(
        self,
        observation: dict[str, Any],
        *,
        session_id: str,
        executed_action_id: int | None = None,
    ) -> CacheInfo:
        refresh = self.begin_prefix_refresh(
            observation,
            session_id=session_id,
            executed_action_id=executed_action_id,
        )
        return self.finish_prefix_refresh(refresh)

    def begin_prefix_refresh(
        self,
        observation: dict[str, Any],
        *,
        session_id: str,
        executed_action_id: int | None = None,
    ) -> dict[str, Any]:
        """Register the observation/action boundary before launching VLM work."""
        del session_id  # The VLM cache is selected by the Runtime session state.
        request_id = f"runtime-{uuid.uuid4().hex}"
        marker_request = {"op": "observation_started", "request_id": request_id}
        if executed_action_id is not None:
            marker_request["executed_action_id"] = int(executed_action_id)
        marker = self._fm_infer(marker_request)
        return {
            "observation": observation,
            "request_id": request_id,
            "observation_chunk_id": marker.get("observation_chunk_id"),
        }

    def finish_prefix_refresh(self, refresh: dict[str, Any]) -> CacheInfo:
        """Encode a registered VLM prefix and activate its cache on FM."""
        descriptor = self._vlm_infer(
            {
                "op": "encode_prefix_to_shared_memory",
                "observation": refresh["observation"],
                "observation_request_id": refresh["request_id"],
                "observation_chunk_id": refresh["observation_chunk_id"],
            }
        )
        artifact_id = descriptor.get("artifact_id")
        try:
            activated = self._fm_infer({"op": "activate_shared_memory_prefix", "cache": descriptor})
        finally:
            if isinstance(artifact_id, str):
                try:
                    self._vlm_infer({"op": "release_shared_memory", "artifact_id": artifact_id})
                except Exception:
                    logger.exception("Failed to release shared-memory artifact %s", artifact_id)
        return CacheInfo(int(activated["cache_id"]), activated["cache_version"])

    def infer(
        self,
        request: dict[str, Any],
        *,
        session_id: str,
        cache_version: str | None,
    ) -> dict[str, Any]:
        payload = dict(request)
        payload["session_id"] = session_id
        if cache_version is not None:
            payload["expected_cache_version"] = cache_version
        return self._fm_infer(payload)

    def reset(self, *, session_id: str, clear_prefix_cache: bool) -> dict[str, Any]:
        return self._fm_infer(
            {
                "op": "reset_stream",
                "session_id": session_id,
                "clear_prefix_cache": clear_prefix_cache,
            }
        )

    def close(self) -> None:
        for client in (self._vlm, self._fm):
            websocket = getattr(client, "_ws", None)
            if websocket is not None:
                websocket.close()
