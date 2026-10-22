"""Model-agnostic session and request coordinator for inference backends."""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections import deque
from typing import Any, Protocol

import numpy as np

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
    # These fields are owned by Runtime.  The public client does not send any
    # cache or execution metadata for normal inference frames.
    execution_id: int = 0
    last_prefix_execution_id: int | None = None
    action_queue: deque = dataclasses.field(default_factory=deque)
    state_history: deque = dataclasses.field(default_factory=deque)
    left_marker_history: deque = dataclasses.field(default_factory=deque)
    right_marker_history: deque = dataclasses.field(default_factory=deque)
    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)


class RuntimeCoordinator:
    """Coordinate sessions and requests without knowing the backend transport."""

    # WebsocketPolicyServer uses this flag to route requests through
    # ``infer_with_session`` and to reset session state when a connection closes.
    tactile_history_enabled = True
    # WebsocketPolicyServer uses this marker to keep timing/cache metadata out
    # of the observation/action wire protocol.
    observation_action_protocol = True

    def __init__(
        self,
        backend: WorkerBackend | None = None,
        *,
        vlm_host: str | None = None,
        vlm_port: int | None = None,
        fm_host: str | None = None,
        fm_port: int | None = None,
        metadata: dict[str, Any] | None = None,
        streaming_horizon: int = 50,
        streaming_watermark: int = 40,
        num_steps: int = 10,
        history_length: int | None = None,
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
        if streaming_horizon <= 0:
            raise ValueError("streaming_horizon must be positive")
        if streaming_watermark < 0 or streaming_watermark >= streaming_horizon:
            raise ValueError("streaming_watermark must be in [0, streaming_horizon)")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if history_length is not None and history_length <= 0:
            raise ValueError("history_length must be positive")
        self._streaming_horizon = int(streaming_horizon)
        self._streaming_watermark = int(streaming_watermark)
        self._refresh_after = self._streaming_horizon - self._streaming_watermark
        self._num_steps = int(num_steps)
        backend_metadata = backend.metadata
        model_metadata = backend_metadata.get("runtime_model") or backend_metadata.get(
            "fm_worker_metadata", {}
        ).get("runtime_model", {})
        if history_length is None:
            gate_sources = tuple(model_metadata.get("gate_sources", ()))
            history_length = (
                int(model_metadata.get("gate_history_length", 1))
                if any(source in gate_sources for source in ("state", "tactile"))
                else 1
            )
        self._history_length = int(history_length)
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
        # The Runtime public protocol is deliberately small: a normal frame is
        # just an observation and the response is just one action.  Control
        # messages such as reset_stream remain available for lifecycle use.
        if "op" not in request and "observation" not in request:
            return self._infer_observation(request)

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

    def _infer_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Run one synchronous observation/action step for the public client.

        The Runtime owns execution ids, prefix refresh scheduling, and the
        action queue.  A chunk-mode worker may therefore produce several
        actions while the client still receives exactly one action per frame.
        """
        session_id = observation.get("session_id", "default")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        # session_id is an internal field added by infer_with_session.  It must
        # never be part of the observation sent by a normal client.
        model_observation = dict(observation)
        model_observation.pop("session_id", None)
        session = self._session(session_id)
        with session.lock:
            model_observation = self._append_observation_history(model_observation, session)
            if not session.action_queue:
                execution_id = session.execution_id
                needs_refresh = (
                    session.cache_version is None
                    or session.last_prefix_execution_id is None
                    or execution_id - session.last_prefix_execution_id >= self._refresh_after
                )
                if needs_refresh:
                    self._update_prefix(
                        session,
                        {
                            "observation": model_observation,
                            "session_id": session_id,
                            "executed_action_id": execution_id,
                        },
                    )
                    session.last_prefix_execution_id = execution_id

                result = self._backend.infer(
                    {
                        "op": "stream_infer",
                        "observation": model_observation,
                        "executed_action_id": execution_id,
                        "num_steps": self._num_steps,
                    },
                    session_id=session_id,
                    cache_version=session.cache_version,
                )
                actions = np.asarray(result.get("actions"))
                if actions.ndim == 1:
                    actions = actions[None, :]
                if actions.ndim != 2 or actions.shape[0] == 0:
                    raise ValueError("Runtime worker must return a non-empty action sequence")
                session.action_queue.extend(np.asarray(action) for action in actions[1:])
                action = np.asarray(actions[0])
            else:
                action = np.asarray(session.action_queue.popleft())

            session.execution_id += 1
            return {"actions": action}

    def _append_observation_history(
        self, observation: dict[str, Any], session: _SessionState
    ) -> dict[str, Any]:
        """Retain every uploaded state/tactile frame, including queued-action steps."""
        result = dict(observation)
        if "state" in result:
            state = np.asarray(result["state"], dtype=np.float32)
            if state.ndim != 1:
                raise ValueError("Observation-only Runtime expects one current state vector per frame")
            if self._history_length > 1:
                if not session.state_history:
                    session.state_history = deque(maxlen=self._history_length)
                session.state_history.append(state.copy())
                padded_state = [session.state_history[0]] * (
                    self._history_length - len(session.state_history)
                ) + list(session.state_history)
                result["state"] = np.stack(padded_state, axis=0)

        marker_keys = ("left_marker", "right_marker")
        present = tuple(key in result for key in marker_keys)
        if present[0] != present[1]:
            raise ValueError("left_marker and right_marker must be provided together")
        if present[0] and self._history_length > 1:
            histories = (session.left_marker_history, session.right_marker_history)
            for key, history in zip(marker_keys, histories, strict=True):
                marker = np.asarray(result[key], dtype=np.float32)
                if marker.ndim != 3:
                    raise ValueError(f"{key} must be a single marker frame, got shape {marker.shape}")
                if history.maxlen != self._history_length:
                    history = deque(history, maxlen=self._history_length)
                    if key == "left_marker":
                        session.left_marker_history = history
                    else:
                        session.right_marker_history = history
                history.append(marker.copy())
                padded = [history[0]] * (self._history_length - len(history)) + list(history)
                result[key] = np.stack(padded, axis=0)
        return result

    def _reset(self, session_id: str, session: _SessionState, request: dict[str, Any]) -> dict[str, Any]:
        clear_prefix_cache = request.get("clear_prefix_cache", False)
        if not isinstance(clear_prefix_cache, bool):
            raise ValueError("clear_prefix_cache must be a boolean")
        result = self._backend.reset(session_id=session_id, clear_prefix_cache=clear_prefix_cache)
        session.execution_id = 0
        session.last_prefix_execution_id = None
        session.action_queue.clear()
        session.state_history.clear()
        session.left_marker_history.clear()
        session.right_marker_history.clear()
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
