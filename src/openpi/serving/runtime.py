"""Model-agnostic session and request coordinator for inference backends."""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class CacheInfo:
    cache_id: int
    cache_version: str


class WorkerBackend(Protocol):
    """Operations Runtime needs from a local or process-backed model worker."""

    @property
    def metadata(self) -> dict[str, Any]: ...

    def warmup(self) -> None: ...

    def update_prefix(
        self,
        observation: dict[str, Any],
        *,
        session_id: str,
        executed_action_id: int | None = None,
    ) -> CacheInfo: ...

    def infer(
        self,
        request: dict[str, Any],
        *,
        session_id: str,
        cache_version: str | None,
    ) -> dict[str, Any]: ...

    def reset(self, *, session_id: str, clear_prefix_cache: bool) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclasses.dataclass
class _SessionState:
    """Context retained for one continuous client task/episode."""

    cache_id: int | None = None
    cache_version: str | None = None
    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)


class RuntimeCoordinator:
    """Coordinate sessions and requests without knowing the backend transport."""

    # WebsocketPolicyServer uses this flag to route requests through
    # ``infer_with_session`` and to reset session state when a connection closes.
    tactile_history_enabled = True

    def __init__(
        self,
        backend: WorkerBackend | None = None,
        *,
        vlm_host: str | None = None,
        vlm_port: int | None = None,
        fm_host: str | None = None,
        fm_port: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if backend is None:
            if vlm_host is None or vlm_port is None or fm_host is None or fm_port is None:
                raise ValueError("backend or all VLM/FM worker endpoints are required")
            from openpi.serving.backends import ProcessBackend

            backend = ProcessBackend(
                vlm_host=vlm_host,
                vlm_port=vlm_port,
                fm_host=fm_host,
                fm_port=fm_port,
                metadata=metadata,
            )
        self._backend = backend
        self._metadata = metadata or {}
        self._sessions: dict[str, _SessionState] = {}
        self._sessions_lock = threading.Lock()

    @property
    def metadata(self) -> dict[str, Any]:
        return {**self._backend.metadata, **self._metadata}

    def warmup(self) -> None:
        self._backend.warmup()

    def _session(self, session_id: str) -> _SessionState:
        with self._sessions_lock:
            return self._sessions.setdefault(session_id, _SessionState())

    def infer_with_session(self, request: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        payload = dict(request)
        payload.setdefault("session_id", session_id)
        return self.infer(payload)

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op", "infer")
        session_id = request.get("session_id", "default")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")

        session = self._session(session_id)
        with session.lock:
            if op in ("reset", "reset_stream"):
                return self._reset(session_id, session, request)

            if op in ("update_vlm", "encode_prefix"):
                if "observation" not in request:
                    raise ValueError(f"{op} requires an observation")
                self._update_prefix(session, request)
                return {
                    "protocol_version": 1,
                    "cache_id": session.cache_id,
                    "cache_version": session.cache_version,
                    "status": "active",
                }

            if op == "stream_infer":
                update_prefix = bool(request.get("update_prefix", session.cache_version is None))
            elif op == "infer":
                # A plain policy client expects each observation to condition the
                # action on its current image and prompt, as in the legacy server.
                update_prefix = bool(request.get("update_prefix", True))
            else:
                raise ValueError(f"Unsupported Runtime operation: {op}")

            if "observation" not in request:
                raise ValueError(f"{op} requires an observation")
            if update_prefix or session.cache_version is None:
                self._update_prefix(session, request)

            inference_request = dict(request)
            inference_request.pop("update_prefix", None)
            return self._backend.infer(
                inference_request,
                session_id=session_id,
                cache_version=session.cache_version,
            )

    def _update_prefix(self, session: _SessionState, request: dict[str, Any]) -> CacheInfo:
        info = self._backend.update_prefix(
            request["observation"],
            session_id=request.get("session_id", "default"),
            executed_action_id=request.get("executed_action_id"),
        )
        session.cache_id = info.cache_id
        session.cache_version = info.cache_version
        return info

    def _reset(self, session_id: str, session: _SessionState, request: dict[str, Any]) -> dict[str, Any]:
        clear_prefix_cache = request.get("clear_prefix_cache", False)
        if not isinstance(clear_prefix_cache, bool):
            raise ValueError("clear_prefix_cache must be a boolean")
        result = self._backend.reset(session_id=session_id, clear_prefix_cache=clear_prefix_cache)
        if clear_prefix_cache:
            session.cache_id = None
            session.cache_version = None
        return result

    def reset_tactile_history(self, session_id: str) -> None:
        """Reset state when the public websocket connection closes."""
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            return
        try:
            with session.lock:
                self._backend.reset(session_id=session_id, clear_prefix_cache=False)
        except Exception:
            logger.exception("Failed to reset Runtime session %s", session_id)
        finally:
            with self._sessions_lock:
                self._sessions.pop(session_id, None)

    def close(self) -> None:
        self._backend.close()
