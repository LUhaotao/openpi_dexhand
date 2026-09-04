"""VLM/FM policy endpoints for the optional multi-process server."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import queue
import threading
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.serving.multiprocess import decode_prefix_cache
from openpi.serving.multiprocess import make_cache_envelope
from openpi.serving.multiprocess import validate_cache_envelope
from openpi.shared import nnx_utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _StreamingState:
    session_id: str
    execution_id: int
    action_window: Any


@dataclasses.dataclass
class MultiProcessPolicy:
    policy: _policy.Policy
    role: Literal["vlm", "fm"]
    vlm_host: str = "127.0.0.1"
    vlm_port: int | None = None

    def __post_init__(self):
        if self.role not in ("vlm", "fm"):
            raise ValueError(f"Unknown multi-process role: {self.role}")
        self._rng = jax.random.key(0)
        self._cache_id = 0
        self._cache_version: str | None = None
        self._prefix_cache: dict[str, Any] | None = None
        self._pending_cache: tuple[int, str, dict[str, Any]] | None = None
        self._cache_lock = threading.Lock()
        self._initial_ready = threading.Event()
        self._initial_error: Exception | None = None
        self._infer_lock = threading.Lock()
        self._refresh_queue: queue.Queue[tuple[int, dict[str, Any], concurrent.futures.Future | None]] = queue.Queue(maxsize=1)
        self._refresh_generation = 0
        if self.role == "fm":
            self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self._refresh_thread.start()
        # ponytail: one active stream per FM server; use a session map only for multi-robot serving.
        self._streaming_state: _StreamingState | None = None
        self._vlm_client = None
        self._encode_prefix_jit = nnx_utils.module_jit(self.policy._model.encode_prefix)  # noqa: SLF001
        self._sample_actions_from_prefix_jit = nnx_utils.module_jit(
            self.policy._model.sample_actions_from_prefix  # noqa: SLF001
        )
        self._advance_streaming_actions_from_prefix_jit = nnx_utils.module_jit(
            self.policy._model.advance_streaming_actions_from_prefix  # noqa: SLF001
        )

    @property
    def model_id(self) -> str:
        model = self.policy._model  # noqa: SLF001
        return f"{type(model).__name__}:{model.action_dim}:{model.action_horizon}"

    def _prepare(self, observation: dict[str, Any]) -> _model.Observation:
        inputs = jax.tree.map(lambda x: x, observation)
        inputs = self.policy._input_transform(inputs)  # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        return _model.Observation.from_dict(inputs)

    def _encode_prefix(self, observation: dict[str, Any]) -> dict[str, Any]:
        model_observation = self._prepare(observation)
        cache = self._encode_prefix_jit(model_observation)
        self._cache_id += 1
        self._cache_version = f"{self.model_id}:{self._cache_id}"
        self._prefix_cache = make_cache_envelope(
            cache,
            model_id=self.model_id,
            cache_id=self._cache_id,
            cache_version=self._cache_version,
        )
        return {
            "protocol_version": 1,
            "model_id": self.model_id,
            "cache_id": self._cache_id,
            "cache_version": self._cache_version,
        }

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op", "infer")
        if self.role == "vlm":
            with self._infer_lock:
                if op == "encode_prefix":
                    return self._encode_prefix(request["observation"])
                if op == "get_prefix_cache":
                    if self._prefix_cache is None:
                        raise RuntimeError("No prefix cache has been encoded")
                    if request.get("cache_id") != self._prefix_cache["cache_id"]:
                        raise ValueError("Requested prefix cache id is unavailable")
                    if request.get("cache_version") != self._prefix_cache["cache_version"]:
                        raise ValueError("Requested prefix cache version is unavailable")
                    return self._prefix_cache
                raise ValueError(f"Unsupported VLM operation: {op}")

        if op == "refresh_prefix":
            return self._refresh_prefix(request)
        with self._infer_lock:
            if op == "reset_stream":
                return self._reset_stream(request)
            if op == "stream_infer":
                return self._stream_infer(request)
            if op != "infer":
                raise ValueError(f"Unsupported FM operation: {op}")
            return self._infer_fm(request)

    def _refresh_prefix(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.vlm_port is None:
            raise ValueError("vlm_port is required for FM refresh")
        with self._cache_lock:
            self._refresh_generation += 1
            generation = self._refresh_generation
            has_active_cache = self._prefix_cache is not None
            if not has_active_cache:
                self._initial_error = None
                self._initial_ready.clear()
        future: concurrent.futures.Future | None = None
        if not has_active_cache:
            future = concurrent.futures.Future()
        self._enqueue_refresh(generation, request, future)
        if future is not None:
            try:
                result = future.result()
                self._activate_pending()
                self._initial_ready.set()
            except Exception as exc:
                self._initial_error = exc
                self._initial_ready.set()
                raise
            return {
                "protocol_version": 1,
                "cache_id": result["cache_id"],
                "cache_version": result["cache_version"],
                "active_cache_id": self._cache_id,
                "active_cache_version": self._cache_version,
                "status": "active",
            }
        result = {
            "cache_id": request["cache_id"],
            "cache_version": request["cache_version"],
        }
        return {
            "protocol_version": 1,
            # Keep the active version in the legacy fields so callers can
            # continue submitting inference while the pending cache loads.
            "cache_id": self._cache_id,
            "cache_version": self._cache_version,
            "active_cache_id": self._cache_id,
            "active_cache_version": self._cache_version,
            "pending_cache_id": result["cache_id"],
            "pending_cache_version": result["cache_version"],
            "status": "pending",
        }

    def _enqueue_refresh(
        self, generation: int, request: dict[str, Any], future: concurrent.futures.Future | None
    ) -> None:
        try:
            self._refresh_queue.put_nowait((generation, request, future))
            return
        except queue.Full:
            pass
        try:
            _, _, old_future = self._refresh_queue.get_nowait()
            if old_future is not None:
                old_future.set_exception(RuntimeError("Prefix-cache refresh superseded"))
        except queue.Empty:
            pass
        self._refresh_queue.put_nowait((generation, request, future))

    def _refresh_loop(self) -> None:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        while True:
            generation, request, future = self._refresh_queue.get()
            try:
                if self._vlm_client is None:
                    self._vlm_client = WebsocketClientPolicy(self.vlm_host, self.vlm_port)
                envelope = self._vlm_client.infer({
                    "op": "get_prefix_cache",
                    "cache_id": request["cache_id"],
                    "cache_version": request["cache_version"],
                })
                validate_cache_envelope(envelope, model_id=self.model_id)
                cache = jax.device_put(decode_prefix_cache(envelope["prefix_cache"]))
                cache = jax.tree.map(
                    lambda value: value.block_until_ready()
                    if hasattr(value, "block_until_ready")
                    else value,
                    cache,
                )
                with self._cache_lock:
                    if generation != self._refresh_generation:
                        raise RuntimeError("Prefix-cache refresh superseded")
                    self._pending_cache = (
                        int(envelope["cache_id"]),
                        envelope["cache_version"],
                        cache,
                    )
                if future is not None:
                    future.set_result({
                        "cache_id": int(envelope["cache_id"]),
                        "cache_version": envelope["cache_version"],
                    })
            except Exception as exc:
                if future is not None:
                    future.set_exception(exc)
                else:
                    logger.exception("Asynchronous prefix-cache refresh failed")

    def _activate_pending(self) -> None:
        with self._cache_lock:
            if self._pending_cache is None:
                return
            cache_id, cache_version, cache = self._pending_cache
            self._prefix_cache = cache
            self._cache_id = cache_id
            self._cache_version = cache_version
            self._pending_cache = None

    def _snapshot_cache(self) -> tuple[dict[str, Any], str, str | None]:
        previous_version = self._cache_version
        self._activate_pending()
        with self._cache_lock:
            waiting_for_initial = self._prefix_cache is None and self._refresh_generation > 0
        if waiting_for_initial:
            self._initial_ready.wait()
            with self._cache_lock:
                if self._prefix_cache is None:
                    if self._initial_error is not None:
                        raise RuntimeError("Initial prefix-cache refresh failed") from self._initial_error
                    raise RuntimeError("No active prefix cache; call refresh_prefix first")
        with self._cache_lock:
            if self._prefix_cache is None or self._cache_version is None:
                raise RuntimeError("No active prefix cache; call refresh_prefix first")
            # The caller may have read the old version just before this
            # chunk-boundary swap. Accept that one stale version transition.
            return self._prefix_cache, self._cache_version, previous_version

    def _reset_stream(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = request.get("session_id", "default")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        clear_prefix_cache = request.get("clear_prefix_cache", False)
        if not isinstance(clear_prefix_cache, bool):
            raise ValueError("clear_prefix_cache must be a boolean")
        self._streaming_state = None
        if clear_prefix_cache:
            with self._cache_lock:
                self._refresh_generation += 1
                self._prefix_cache = None
                self._pending_cache = None
                self._cache_id = 0
                self._cache_version = None
        return {
            "protocol_version": 1,
            "session_id": session_id,
            "status": "reset",
            "prefix_cache_cleared": clear_prefix_cache,
        }

    def _stream_infer(self, request: dict[str, Any]) -> dict[str, Any]:
        prefix_cache, cache_version, previous_version = self._snapshot_cache()
        expected = request.get("expected_cache_version")
        if expected is not None and expected not in (cache_version, previous_version):
            raise ValueError(
                f"Cache version mismatch: expected {expected!r}, active {cache_version!r}"
            )
        model = self.policy._model  # noqa: SLF001
        if not getattr(model, "streaming", False):
            raise ValueError("stream_infer requires a checkpoint configured with model.streaming=True")

        session_id = request.get("session_id", "default")
        execution_id = request.get("executed_action_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(execution_id, int) or isinstance(execution_id, bool) or execution_id < 0:
            raise ValueError("executed_action_id must be a non-negative integer")

        model_observation = self._prepare(request["observation"])
        if self._streaming_state is None:
            self._streaming_state = self._seed_streaming_state(
                session_id,
                execution_id,
                model_observation.state,
                prefix_cache,
                num_steps=int(request.get("num_steps", 10)),
            )
        elif self._streaming_state.session_id != session_id:
            raise RuntimeError("A different stream session is active; call reset_stream first")
        else:
            completed_actions = execution_id - self._streaming_state.execution_id
            if completed_actions < 0:
                raise ValueError("executed_action_id cannot move backwards within a stream session")
            chunk_size = int(model.streaming_chunk_size)
            if completed_actions % chunk_size:
                raise ValueError("executed_action_id must advance by whole streaming chunks")
            if completed_actions:
                self._rng, advance_rng = jax.random.split(self._rng)
                self._streaming_state.action_window = self._advance_streaming_actions_from_prefix_jit(
                    advance_rng,
                    model_observation.state,
                    prefix_cache,
                    self._streaming_state.action_window,
                    jnp.asarray(completed_actions // chunk_size, dtype=jnp.int32),
                )
                self._streaming_state.execution_id = execution_id

        chunk_size = int(model.streaming_chunk_size)
        actions = self._streaming_state.action_window[:, :chunk_size]
        result = self.policy._output_transform(  # noqa: SLF001
            {"state": np.asarray(model_observation.state[0]), "actions": np.asarray(actions[0])}
        )
        result["streaming"] = {
            "protocol_version": 1,
            "session_id": session_id,
            "execution_id": execution_id,
            "cache_version": cache_version,
            "action_count": chunk_size,
        }
        return result

    def _seed_streaming_state(
        self,
        session_id: str,
        execution_id: int,
        state: jax.Array,
        prefix_cache: dict[str, Any],
        *,
        num_steps: int,
    ) -> _StreamingState:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self._rng, sample_rng, noise_rng = jax.random.split(self._rng, 3)
        actions = self._sample_actions_from_prefix_jit(
            sample_rng,
            state,
            prefix_cache,
            num_steps=num_steps,
        )
        timestep = self.policy._model.streaming_timestep(actions.dtype)  # noqa: SLF001
        noise = jax.random.normal(noise_rng, actions.shape, dtype=actions.dtype)
        # ponytail: seed from a full sample plus the training marginal; retain an ODE trajectory only if cold starts need it.
        action_window = timestep[None, :, None] * noise + (1.0 - timestep[None, :, None]) * actions
        return _StreamingState(session_id, execution_id, action_window)

    def _infer_fm(self, request: dict[str, Any]) -> dict[str, Any]:
        prefix_cache, cache_version, previous_version = self._snapshot_cache()
        expected = request.get("expected_cache_version")
        if expected is not None and expected not in (cache_version, previous_version):
            raise ValueError(
                f"Cache version mismatch: expected {expected!r}, active {cache_version!r}"
            )
        model_observation = self._prepare(request["observation"])
        self._rng, sample_rng = jax.random.split(self._rng)
        noise_tokens = int(request.get("noise_tokens", self.policy._model.action_horizon))  # noqa: SLF001
        if noise_tokens <= 0 or noise_tokens > self.policy._model.action_horizon:  # noqa: SLF001
            raise ValueError("noise_tokens must be in [1, action_horizon]")
        noise = jax.random.normal(
            sample_rng,
            (model_observation.state.shape[0], noise_tokens, self.policy._model.action_dim),  # noqa: SLF001
        )
        actions = self._sample_actions_from_prefix_jit(
            sample_rng,
            model_observation.state,
            prefix_cache,
            num_steps=int(request.get("num_steps", 10)),
            noise=noise,
        )
        result = self.policy._output_transform(  # noqa: SLF001
            {"state": np.asarray(model_observation.state[0]), "actions": np.asarray(actions[0])}
        )
        result["cache_version"] = cache_version
        return result
