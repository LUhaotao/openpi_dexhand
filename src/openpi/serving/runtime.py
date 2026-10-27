"""Model-agnostic session and request coordinator for inference backends."""

from __future__ import annotations

from collections import deque
import concurrent.futures
import dataclasses
import logging
import threading
from typing import Any, Literal, Protocol

import numpy as np

logger = logging.getLogger(__name__)

InferenceMode = Literal["stream", "stream-head", "diffusion"]


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
    # Public Runtime action delivery is at-least-once: keep the exact response
    # until the client acknowledges it on the following control request.
    next_action_id: int = 0
    last_acked_action_id: int | None = None
    pending_action_id: int | None = None
    pending_action_response: dict[str, Any] | None = None
    state_history: deque = dataclasses.field(default_factory=deque)
    left_marker_history: deque = dataclasses.field(default_factory=deque)
    right_marker_history: deque = dataclasses.field(default_factory=deque)
    prefix_refresh_future: concurrent.futures.Future[CacheInfo] | None = None
    prefix_refresh_execution_id: int | None = None
    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)


class RuntimeCoordinator:
    """Coordinate Runtime sessions without knowing worker transport or algorithm."""

    # WebsocketPolicyServer uses this flag to route requests through
    # ``infer_with_session`` and to reset session state when a connection closes.
    tactile_history_enabled = True
    # The Runtime endpoint serves one active control stream. Keep its server-owned
    # session when the public WebSocket reconnects; episode reset remains explicit.
    retain_session_on_disconnect = True
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
        inference_mode: InferenceMode = "stream",
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
        if inference_mode not in ("stream", "stream-head", "diffusion"):
            raise ValueError("inference_mode must be 'stream', 'stream-head', or 'diffusion'")
        if history_length is not None and history_length <= 0:
            raise ValueError("history_length must be positive")
        self._streaming_horizon = int(streaming_horizon)
        self._streaming_watermark = int(streaming_watermark)
        self._refresh_after = self._streaming_horizon - self._streaming_watermark
        self._num_steps = int(num_steps)
        self._inference_mode = inference_mode
        backend_metadata = backend.metadata
        model_metadata = backend_metadata.get("runtime_model") or backend_metadata.get("fm_worker_metadata", {}).get(
            "runtime_model", {}
        )
        if history_length is None:
            gate_sources = tuple(model_metadata.get("gate_sources", ()))
            history_length = (
                int(model_metadata.get("gate_history_length", 1))
                if any(source in gate_sources for source in ("state", "tactile"))
                else 1
            )
        self._history_length = int(history_length)
        # In process mode, VLM encoding and FM cache activation may overlap an
        # action inference. LocalBackend shares one mutable JAX policy and
        # therefore keeps prefix updates synchronous.
        self._prefix_refresh_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="runtime-vlm-refresh"
        )
        self._sessions: dict[str, _SessionState] = {}
        self._sessions_lock = threading.Lock()

    @property
    def metadata(self) -> dict[str, Any]:
        return {**self._backend.metadata, "inference_mode": self._inference_mode, **self._metadata}

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

        if self._inference_mode == "diffusion" and op == "infer":
            if "observation" not in request:
                raise ValueError("infer requires an observation")
            observation = dict(request["observation"])
            observation["session_id"] = session_id
            return self._infer_observation(observation)
        if self._inference_mode == "diffusion" and op in ("stream_infer", "update_vlm", "encode_prefix"):
            raise ValueError(f"{op} is unavailable in diffusion inference mode")

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
        action queue. A worker may therefore produce several actions while the
        client still receives exactly one action per frame.
        """
        session_id = observation.get("session_id", "default")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        # session_id is an internal field added by infer_with_session.  It must
        # never be part of the observation sent by a normal client.
        model_observation = dict(observation)
        model_observation.pop("session_id", None)
        action_id = model_observation.pop("runtime_action_id", None)
        ack_action_id = model_observation.pop("runtime_ack_action_id", None)
        self._validate_action_id(action_id, "runtime_action_id", allow_none=True)
        self._validate_action_id(ack_action_id, "runtime_ack_action_id", allow_none=True)
        session = self._session(session_id)
        with session.lock:
            replay = self._resolve_pending_action(session, action_id, ack_action_id)
            if replay is not None:
                return replay
            self._collect_prefix_refresh(session)
            model_observation = self._append_observation_history(model_observation, session)
            if not session.action_queue:
                execution_id = session.execution_id
                if self._inference_mode == "diffusion":
                    result = self._backend.infer(
                        {
                            "op": "infer",
                            "observation": model_observation,
                            "num_steps": self._num_steps,
                        },
                        session_id=session_id,
                        cache_version=None,
                    )
                    actions = np.asarray(result.get("actions"))
                    if actions.ndim == 1:
                        actions = actions[None, :]
                    if actions.ndim != 2 or actions.shape[0] == 0:
                        raise ValueError("Runtime worker must return a non-empty action sequence")
                    session.action_queue.extend(np.asarray(action) for action in actions[1:])
                    action = np.asarray(actions[0])
                    session.execution_id += 1
                    return self._make_action_response(session, action, action_id)

                needs_refresh = (
                    session.cache_version is None
                    or session.last_prefix_execution_id is None
                    or execution_id - session.last_prefix_execution_id >= self._refresh_after
                )
                if needs_refresh and session.prefix_refresh_future is None:
                    if session.cache_version is not None and getattr(
                        self._backend, "supports_async_prefix_refresh", False
                    ):
                        self._schedule_prefix_refresh(
                            session,
                            model_observation,
                            session_id=session_id,
                            execution_id=execution_id,
                        )
                    elif session.cache_version is None or not getattr(
                        self._backend, "supports_async_prefix_refresh", False
                    ):
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
                streaming = result.get("streaming")
                active_cache_version = streaming.get("cache_version") if isinstance(streaming, dict) else None
                if not isinstance(active_cache_version, str):
                    active_cache_version = result.get("cache_version")
                if isinstance(active_cache_version, str):
                    session.cache_version = active_cache_version
                actions = np.asarray(result.get("actions"))
                if actions.ndim == 1:
                    actions = actions[None, :]
                if actions.ndim != 2 or actions.shape[0] == 0:
                    raise ValueError("Runtime worker must return a non-empty action sequence")
                session.action_queue.extend(np.asarray(action) for action in actions[1:])
                action = np.asarray(actions[0])
            else:
                action = np.asarray(session.action_queue.popleft())

            self._collect_prefix_refresh(session)
            session.execution_id += 1
            return self._make_action_response(session, action, action_id)

    @staticmethod
    def _validate_action_id(value: Any, name: str, *, allow_none: bool) -> None:
        if value is None and allow_none:
            return
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    @staticmethod
    def _copy_action_response(response: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in response.items()
        }

    def _resolve_pending_action(
        self,
        session: _SessionState,
        action_id: int | None,
        ack_action_id: int | None,
    ) -> dict[str, Any] | None:
        """Replay an unacknowledged action or accept its client acknowledgement."""
        if action_id is None:
            if ack_action_id is not None:
                raise ValueError("runtime_ack_action_id requires runtime_action_id")
            if session.pending_action_id is not None:
                raise ValueError("runtime_action_id is required while an action is pending")
            return None

        pending_id = session.pending_action_id
        if pending_id is not None:
            if action_id == pending_id:
                if ack_action_id != session.last_acked_action_id:
                    raise ValueError(
                        "Action acknowledgement mismatch: "
                        f"expected {session.last_acked_action_id}, got {ack_action_id}"
                    )
                if session.pending_action_response is None:
                    raise RuntimeError("Pending Runtime action is missing its response")
                return self._copy_action_response(session.pending_action_response)
            if ack_action_id != pending_id:
                raise ValueError(
                    f"Action acknowledgement mismatch: expected {pending_id}, got {ack_action_id}"
                )
            if action_id != pending_id + 1:
                raise ValueError(f"Expected runtime_action_id {pending_id + 1}, got {action_id}")
            session.last_acked_action_id = pending_id
            session.pending_action_id = None
            session.pending_action_response = None
        else:
            if ack_action_id != session.last_acked_action_id:
                raise ValueError(
                    "Action acknowledgement mismatch: "
                    f"expected {session.last_acked_action_id}, got {ack_action_id}"
                )
            if action_id != session.next_action_id:
                raise ValueError(f"Expected runtime_action_id {session.next_action_id}, got {action_id}")
        return None

    def _make_action_response(
        self,
        session: _SessionState,
        action: np.ndarray,
        action_id: int | None,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {"actions": np.array(action, copy=True)}
        if action_id is not None:
            response["action_id"] = action_id
            session.pending_action_id = action_id
            session.pending_action_response = self._copy_action_response(response)
            session.next_action_id = action_id + 1
        return response

    def _schedule_prefix_refresh(
        self,
        session: _SessionState,
        observation: dict[str, Any],
        *,
        session_id: str,
        execution_id: int,
    ) -> None:
        begin_refresh = getattr(self._backend, "begin_prefix_refresh", None)
        finish_refresh = getattr(self._backend, "finish_prefix_refresh", None)
        if callable(begin_refresh) and callable(finish_refresh):
            # Record the delay-conditioning marker before this action's FM
            # request. The expensive VLM encode/cache activation then overlaps
            # that inference in the background.
            refresh = begin_refresh(
                observation,
                session_id=session_id,
                executed_action_id=execution_id,
            )
            session.prefix_refresh_future = self._prefix_refresh_executor.submit(finish_refresh, refresh)
        else:
            session.prefix_refresh_future = self._prefix_refresh_executor.submit(
                self._backend.update_prefix,
                observation,
                session_id=session_id,
                executed_action_id=execution_id,
            )
        session.prefix_refresh_execution_id = execution_id
        # Record the requested boundary now. If refresh fails, collection
        # clears this marker and the next observation retries it.
        session.last_prefix_execution_id = execution_id

    @staticmethod
    def _apply_prefix_result(session: _SessionState, info: CacheInfo) -> None:
        session.cache_id = info.cache_id
        session.cache_version = info.cache_version

    def _collect_prefix_refresh(self, session: _SessionState, *, wait: bool = False) -> None:
        future = session.prefix_refresh_future
        if future is None or (not wait and not future.done()):
            return
        execution_id = session.prefix_refresh_execution_id
        session.prefix_refresh_future = None
        session.prefix_refresh_execution_id = None
        try:
            self._apply_prefix_result(session, future.result())
        except Exception:
            logger.exception("Runtime VLM prefix refresh failed")
            session.last_prefix_execution_id = None
        else:
            if execution_id is not None:
                session.last_prefix_execution_id = execution_id

    def _append_observation_history(self, observation: dict[str, Any], session: _SessionState) -> dict[str, Any]:
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
                padded_state = [session.state_history[0]] * (self._history_length - len(session.state_history)) + list(
                    session.state_history
                )
                result["state"] = np.stack(padded_state, axis=0)

        marker_keys = ("left_marker", "right_marker")
        present = tuple(key in result for key in marker_keys)
        if present[0] != present[1]:
            raise ValueError("left_marker and right_marker must be provided together")
        if present[0] and self._history_length > 1:
            histories = (session.left_marker_history, session.right_marker_history)
            for key, history_deque in zip(marker_keys, histories, strict=True):
                marker = np.asarray(result[key], dtype=np.float32)
                if marker.ndim != 3:
                    raise ValueError(f"{key} must be a single marker frame, got shape {marker.shape}")
                if history_deque.maxlen != self._history_length:
                    resized_history = deque(history_deque, maxlen=self._history_length)
                    if key == "left_marker":
                        session.left_marker_history = resized_history
                    else:
                        session.right_marker_history = resized_history
                else:
                    resized_history = history_deque
                resized_history.append(marker.copy())
                padded = [resized_history[0]] * (self._history_length - len(resized_history)) + list(resized_history)
                result[key] = np.stack(padded, axis=0)
        return result

    def _reset(self, session_id: str, session: _SessionState, request: dict[str, Any]) -> dict[str, Any]:
        clear_prefix_cache = request.get("clear_prefix_cache", False)
        if not isinstance(clear_prefix_cache, bool):
            raise ValueError("clear_prefix_cache must be a boolean")
        # Don't let a prefix refresh finish after reset and reactivate stale
        # cache state for the new episode.
        self._collect_prefix_refresh(session, wait=True)
        result = self._backend.reset(session_id=session_id, clear_prefix_cache=clear_prefix_cache)
        session.execution_id = 0
        session.last_prefix_execution_id = None
        session.action_queue.clear()
        session.next_action_id = 0
        session.last_acked_action_id = None
        session.pending_action_id = None
        session.pending_action_response = None
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
                self._collect_prefix_refresh(session, wait=True)
                self._backend.reset(session_id=session_id, clear_prefix_cache=False)
        except Exception:
            logger.exception("Failed to reset Runtime session %s", session_id)
        finally:
            with self._sessions_lock:
                self._sessions.pop(session_id, None)

    def close(self) -> None:
        self._prefix_refresh_executor.shutdown(wait=True, cancel_futures=False)
        self._backend.close()
