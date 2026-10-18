"""VLM/FM policy endpoints for the optional multi-process server."""

from __future__ import annotations

import atexit
import concurrent.futures
import dataclasses
import logging
import queue
import threading
from typing import Any, Literal
import uuid

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.policies import policy as _policy
from openpi.serving.multiprocess import decode_prefix_cache
from openpi.serving.multiprocess import decode_shared_memory_cache
from openpi.serving.multiprocess import make_cache_envelope
from openpi.serving.multiprocess import make_shared_memory_cache
from openpi.serving.multiprocess import validate_cache_envelope
from openpi.serving.multiprocess import validate_shared_memory_cache
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
    fm_host: str = "127.0.0.1"
    fm_port: int | None = None

    def __post_init__(self):
        if self.role not in ("vlm", "fm"):
            raise ValueError(f"Unknown multi-process role: {self.role}")
        self._rng = jax.random.key(0)
        self._cache_id = 0
        self._cache_version: str | None = None
        self._prefix_cache: dict[str, Any] | None = None
        self._pending_cache: tuple[int, str, dict[str, Any], int | None] | tuple[int, str, dict[str, Any]] | None = None
        self._active_cache_observation_chunk_id: int | None = None
        self._observation_start_chunks: dict[str, int] = {}
        self._current_chunk_index = 0
        self._cache_lock = threading.Lock()
        self._initial_ready = threading.Event()
        self._initial_error: Exception | None = None
        self._infer_lock = threading.Lock()
        self._refresh_queue: queue.Queue[tuple[int, dict[str, Any], concurrent.futures.Future | None]] = queue.Queue(maxsize=1)
        self._refresh_generation = 0
        if self.role == "fm" and self.vlm_port is not None:
            self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self._refresh_thread.start()
        # ponytail: one active stream per FM server; use a session map only for multi-robot serving.
        self._streaming_state: _StreamingState | None = None
        self._vlm_client = None
        self._fm_client = None
        self._shared_memory_artifacts: dict[str, Any] = {}
        atexit.register(self._cleanup_shared_memory_artifacts)
        self._encode_prefix_jit = nnx_utils.module_jit(self.policy._model.encode_prefix)  # noqa: SLF001
        self._encode_tactile_jit = (
            nnx_utils.module_jit(self.policy._model._marker_condition)  # noqa: SLF001
            if getattr(self.policy._model, "use_tactile", False)  # noqa: SLF001
            and getattr(self.policy._model, "use_tactile_adarms", False)  # noqa: SLF001
            else None
        )
        self._sample_actions_from_prefix_jit = nnx_utils.module_jit(
            self.policy._model.sample_actions_from_prefix  # noqa: SLF001
        )
        self._advance_streaming_actions_from_prefix_jit = nnx_utils.module_jit(
            self.policy._model.advance_streaming_actions_from_prefix  # noqa: SLF001
        )

    def _cleanup_shared_memory_artifacts(self) -> None:
        for artifact in self._shared_memory_artifacts.values():
            try:
                artifact.close()
                artifact.unlink()
            except FileNotFoundError:
                pass
        self._shared_memory_artifacts.clear()

    @property
    def model_id(self) -> str:
        model = self.policy._model  # noqa: SLF001
        return f"{type(model).__name__}:{model.action_dim}:{model.action_horizon}"

    def _prepare(
        self,
        observation: dict[str, Any],
        *,
        session_id: str = "default",
        update_tactile_history: bool = False,
    ) -> _model.Observation:
        return self.policy.prepare_observation(
            observation, session_id=session_id, update_tactile_history=update_tactile_history
        )

    def warmup(self) -> None:
        """Compile the role's first inference path before accepting clients."""
        model = self.policy._model  # noqa: SLF001
        gate_sources = tuple(getattr(model, "gate_sources", ()))
        unified_gate = getattr(model, "streaming_attention_mode", None) == "attention_gate"
        tactile_history_enabled = unified_gate and "tactile" in gate_sources
        observation = _model.Observation(
            images={
                key: jnp.ones((1, *_model.IMAGE_RESOLUTION, 3), dtype=jnp.float32)
                for key in _model.IMAGE_KEYS
            },
            image_masks={key: jnp.ones((1,), dtype=jnp.bool_) for key in _model.IMAGE_KEYS},
            state=jnp.ones((1, model.action_dim), dtype=jnp.float32),
            observation_delay_chunks=(
                jnp.zeros((1,), dtype=jnp.int32)
                if getattr(model, "observation_delay_max_chunks", 0) > 0
                else None
            ),
            torque=(
                jnp.ones((1, model.torque_dim), dtype=jnp.float32)
                if getattr(model, "use_torque", False)
                else None
            ),
            torque_history=(
                jnp.ones((1, model.gate_history_length, model.torque_dim), dtype=jnp.float32)
                if unified_gate and "torque" in gate_sources
                else None
            ),
            state_history=(
                jnp.ones((1, model.gate_history_length, model.action_dim), dtype=jnp.float32)
                if unified_gate and "state" in gate_sources
                else None
            ),
            tactile_left_marker=(
                jnp.ones((1, *_pi0_config.TACTILE_MARKER_SHAPE), dtype=jnp.float32)
                if getattr(model, "use_tactile", False)
                else None
            ),
            tactile_right_marker=(
                jnp.ones((1, *_pi0_config.TACTILE_MARKER_SHAPE), dtype=jnp.float32)
                if getattr(model, "use_tactile", False)
                else None
            ),
            tactile_left_marker_history=(
                jnp.ones(
                    (
                        1,
                        model.gate_history_length,
                        *_pi0_config.TACTILE_MARKER_SHAPE,
                    ),
                    dtype=jnp.float32,
                )
                if tactile_history_enabled
                else None
            ),
            tactile_right_marker_history=(
                jnp.ones(
                    (
                        1,
                        model.gate_history_length,
                        *_pi0_config.TACTILE_MARKER_SHAPE,
                    ),
                    dtype=jnp.float32,
                )
                if tactile_history_enabled
                else None
            ),
            tokenized_prompt=jnp.ones((1, model.max_token_len), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((1, model.max_token_len), dtype=jnp.bool_),
        )

        if self.role == "vlm":
            self._block_until_ready(self._encode_prefix_jit(observation))
            return

        # eval_shape gives the FM a correctly-shaped cache without compiling the
        # image encoder a second time in the FM process.
        prefix_spec = jax.eval_shape(model.encode_prefix, observation)
        prefix_cache = jax.tree.map(lambda value: jnp.ones(value.shape, value.dtype), prefix_spec)
        tactile_condition = self._tactile_condition(observation)
        actions = self._sample_actions_from_prefix_jit(
            jax.random.key(0),
            observation.state,
            prefix_cache,
            num_steps=10,
            tactile_condition=tactile_condition,
            tactile_left_marker_history=observation.tactile_left_marker_history,
            tactile_right_marker_history=observation.tactile_right_marker_history,
            torque_history=observation.torque_history,
            state_history=observation.state_history,
            observation_delay_chunks=observation.observation_delay_chunks,
        )
        self._block_until_ready(actions)
        if getattr(self.policy._model, "streaming", False):  # noqa: SLF001
            chunk_size = int(self.policy._model.streaming_chunk_size)  # noqa: SLF001
            action_buffer = jnp.concatenate([actions[:, :chunk_size], actions], axis=1)
            advanced = self._advance_streaming_actions_from_prefix_jit(
                jax.random.key(1),
                observation.state,
                prefix_cache,
                action_buffer,
                jnp.asarray(1, dtype=jnp.int32),
                tactile_condition=tactile_condition,
                tactile_left_marker_history=observation.tactile_left_marker_history,
                tactile_right_marker_history=observation.tactile_right_marker_history,
                torque_history=observation.torque_history,
                state_history=observation.state_history,
                observation_delay_chunks=observation.observation_delay_chunks,
            )
            self._block_until_ready(advanced)

    @staticmethod
    def _block_until_ready(value: Any) -> Any:
        return jax.tree.map(
            lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
            value,
        )

    def _tactile_condition(self, observation: _model.Observation) -> Any:
        if self._encode_tactile_jit is None:
            return None
        return self._encode_tactile_jit(observation)

    def encode_prefix_local(
        self,
        observation: dict[str, Any],
        *,
        session_id: str,
        executed_action_id: int | None = None,
    ) -> Any:
        """Compute and activate a prefix without serialization or IPC."""
        model_observation = self._prepare(observation, session_id=session_id)
        cache = self._encode_prefix_jit(model_observation)
        self._cache_id += 1
        self._cache_version = f"{self.model_id}:{self._cache_id}"
        observation_chunk_id = None
        if getattr(self.policy._model, "observation_delay_max_chunks", 0) > 0:  # noqa: SLF001
            if executed_action_id is None:
                raise ValueError("executed_action_id is required for observation-delay caches")
            chunk_size = int(self.policy._model.streaming_chunk_size)  # noqa: SLF001
            self._current_chunk_index = executed_action_id // chunk_size
            observation_chunk_id = self._current_chunk_index
        self._prefix_cache = cache
        self._active_cache_observation_chunk_id = observation_chunk_id
        self._streaming_state = None
        self._initial_error = None
        self._initial_ready.set()
        return {
            "cache_id": self._cache_id,
            "cache_version": self._cache_version,
            "cache": cache,
            "observation_chunk_id": observation_chunk_id,
        }

    def activate_local_prefix(self, cache: Any, *, cache_id: int, cache_version: str, observation_chunk_id: int | None = None) -> None:
        """Activate a JAX prefix produced by :meth:`encode_prefix_local`."""
        self._cache_id = int(cache_id)
        self._cache_version = cache_version
        self._prefix_cache = cache
        self._active_cache_observation_chunk_id = observation_chunk_id
        self._streaming_state = None
        self._initial_error = None
        self._initial_ready.set()

    def _encode_prefix(
        self,
        observation: dict[str, Any],
        *,
        observation_chunk_id: int | None = None,
        observation_request_id: str | None = None,
    ) -> dict[str, Any]:
        if (
            getattr(self.policy._model, "observation_delay_max_chunks", 0) > 0  # noqa: SLF001
            and observation_chunk_id is None
        ):
            if self.fm_port is None:
                raise ValueError("fm_port is required for observation-delay cache numbering")
            from openpi_client.websocket_client_policy import WebsocketClientPolicy

            if self._fm_client is None:
                self._fm_client = WebsocketClientPolicy(self.fm_host, self.fm_port)
            observation_request_id = uuid.uuid4().hex
            marker = self._fm_client.infer({
                "op": "observation_started",
                "request_id": observation_request_id,
            })
            observation_chunk_id = int(marker["observation_chunk_id"])
        model_observation = self._prepare(observation)
        cache = self._encode_prefix_jit(model_observation)
        self._cache_id += 1
        self._cache_version = f"{self.model_id}:{self._cache_id}"
        self._prefix_cache = make_cache_envelope(
            cache,
            model_id=self.model_id,
            cache_id=self._cache_id,
            cache_version=self._cache_version,
            observation_request_id=observation_request_id,
            observation_chunk_id=observation_chunk_id,
        )
        return {
            "protocol_version": 1,
            "model_id": self.model_id,
            "cache_id": self._cache_id,
            "cache_version": self._cache_version,
        }

    def _encode_prefix_and_get_cache(self, request: dict[str, Any]) -> dict[str, Any]:
        """Encode a prefix and return its envelope for a Runtime coordinator.

        The legacy VLM endpoint returns only cache metadata from ``encode_prefix`` and
        exposes the payload through ``get_prefix_cache``.  Runtime uses this combined
        operation so the coordinator can move the payload to FM without making the
        client participate in cache management.
        """
        self._encode_prefix(
            request["observation"],
            observation_chunk_id=request.get("observation_chunk_id"),
            observation_request_id=request.get("observation_request_id"),
        )
        if self._prefix_cache is None:
            raise RuntimeError("Prefix cache was not created")
        return self._prefix_cache

    def _encode_prefix_to_shared_memory(self, request: dict[str, Any]) -> dict[str, Any]:
        """Encode a prefix and publish its arrays through shared memory."""
        observation_request_id = request.get("observation_request_id")
        observation_chunk_id = request.get("observation_chunk_id")
        if getattr(self.policy._model, "observation_delay_max_chunks", 0) > 0 and observation_chunk_id is None:  # noqa: SLF001
            raise ValueError("Delay-conditioned shared-memory cache is missing its observation chunk marker")
        model_observation = self._prepare(request["observation"])
        cache = self._encode_prefix_jit(model_observation)
        self._cache_id += 1
        self._cache_version = f"{self.model_id}:{self._cache_id}"
        descriptor, artifact = make_shared_memory_cache(
            cache,
            model_id=self.model_id,
            cache_id=self._cache_id,
            cache_version=self._cache_version,
            observation_request_id=observation_request_id,
            observation_chunk_id=observation_chunk_id,
        )
        self._shared_memory_artifacts[descriptor["artifact_id"]] = artifact
        return descriptor

    def _release_shared_memory(self, request: dict[str, Any]) -> dict[str, Any]:
        artifact_id = request.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("release_shared_memory requires an artifact_id")
        artifact = self._shared_memory_artifacts.pop(artifact_id, None)
        if artifact is not None:
            try:
                artifact.close()
                artifact.unlink()
            except FileNotFoundError:
                pass
        return {"protocol_version": 1, "artifact_id": artifact_id, "status": "released"}

    def _activate_prefix(self, request: dict[str, Any]) -> dict[str, Any]:
        """Activate a VLM cache envelope supplied by the Runtime coordinator."""
        envelope = request.get("cache")
        if not isinstance(envelope, dict):
            raise ValueError("activate_prefix requires a cache envelope")
        validate_cache_envelope(envelope, model_id=self.model_id)
        cache = jax.device_put(decode_prefix_cache(envelope["prefix_cache"]))
        cache = jax.tree.map(
            lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
            cache,
        )
        observation_chunk_id = envelope.get("observation_chunk_id")
        if getattr(self.policy._model, "observation_delay_max_chunks", 0) > 0 and observation_chunk_id is None:  # noqa: SLF001
            raise ValueError("Delay-conditioned cache is missing its observation chunk marker")
        with self._cache_lock:
            self._refresh_generation += 1
            self._pending_cache = (
                int(envelope["cache_id"]),
                envelope["cache_version"],
                cache,
                None if observation_chunk_id is None else int(observation_chunk_id),
            )
            self._initial_error = None
            self._initial_ready.set()
        self._activate_pending()
        return {
            "protocol_version": 1,
            "cache_id": self._cache_id,
            "cache_version": self._cache_version,
            "status": "active",
        }

    def _activate_shared_memory_prefix(self, request: dict[str, Any]) -> dict[str, Any]:
        """Copy a shared-memory cache into the FM device and activate it."""
        descriptor = request.get("cache")
        if not isinstance(descriptor, dict):
            raise ValueError("activate_shared_memory_prefix requires a cache descriptor")
        validate_shared_memory_cache(descriptor, model_id=self.model_id)
        from multiprocessing import shared_memory

        artifact = shared_memory.SharedMemory(name=descriptor["shared_memory_name"], create=False)
        try:
            if artifact.size < int(descriptor["payload_size"]):
                raise ValueError("Shared-memory cache segment is smaller than its descriptor")
            host_cache = decode_shared_memory_cache(descriptor["tree"], artifact.buf)
            cache = jax.device_put(host_cache)
            cache = jax.tree.map(
                lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
                cache,
            )
        finally:
            artifact.close()
        observation_chunk_id = descriptor.get("observation_chunk_id")
        if getattr(self.policy._model, "observation_delay_max_chunks", 0) > 0 and observation_chunk_id is None:  # noqa: SLF001
            raise ValueError("Delay-conditioned cache is missing its observation chunk marker")
        with self._cache_lock:
            self._refresh_generation += 1
            self._pending_cache = (
                int(descriptor["cache_id"]),
                descriptor["cache_version"],
                cache,
                None if observation_chunk_id is None else int(observation_chunk_id),
            )
            self._initial_error = None
            self._initial_ready.set()
        self._activate_pending()
        return {
            "protocol_version": 1,
            "cache_id": self._cache_id,
            "cache_version": self._cache_version,
            "status": "active",
        }

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op", "infer")
        if self.role == "vlm":
            with self._infer_lock:
                if op == "encode_prefix":
                    return self._encode_prefix(request["observation"])
                if op == "encode_prefix_and_get_cache":
                    return self._encode_prefix_and_get_cache(request)
                if op == "encode_prefix_to_shared_memory":
                    return self._encode_prefix_to_shared_memory(request)
                if op == "release_shared_memory":
                    return self._release_shared_memory(request)
                if op == "get_prefix_cache":
                    if self._prefix_cache is None:
                        raise RuntimeError("No prefix cache has been encoded")
                    if request.get("cache_id") != self._prefix_cache["cache_id"]:
                        raise ValueError("Requested prefix cache id is unavailable")
                    if request.get("cache_version") != self._prefix_cache["cache_version"]:
                        raise ValueError("Requested prefix cache version is unavailable")
                    return self._prefix_cache
                raise ValueError(f"Unsupported VLM operation: {op}")

        if op == "activate_prefix":
            with self._infer_lock:
                return self._activate_prefix(request)
        if op == "activate_shared_memory_prefix":
            with self._infer_lock:
                return self._activate_shared_memory_prefix(request)
        if op == "refresh_prefix":
            return self._refresh_prefix(request)
        if op == "observation_started":
            with self._infer_lock:
                return self._record_observation_start(request)
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
        wait_for_activation = request.get("wait_for_activation", False)
        if not isinstance(wait_for_activation, bool):
            raise ValueError("wait_for_activation must be a boolean")
        with self._cache_lock:
            self._refresh_generation += 1
            generation = self._refresh_generation
            has_active_cache = self._prefix_cache is not None
            if not has_active_cache:
                self._initial_error = None
                self._initial_ready.clear()
        future: concurrent.futures.Future | None = None
        if wait_for_activation or not has_active_cache:
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
                observation_chunk_id = None
                if getattr(self.policy._model, "observation_delay_max_chunks", 0) > 0:  # noqa: SLF001
                    request_id = envelope.get("observation_request_id")
                    if not isinstance(request_id, str):
                        raise ValueError("Delay-conditioned cache is missing its observation request id")
                    with self._cache_lock:
                        observation_chunk_id = self._observation_start_chunks.pop(request_id, None)
                    if observation_chunk_id is None:
                        raise ValueError("No FM chunk marker found for the VLM cache request")
                    if int(envelope.get("observation_chunk_id", -1)) != observation_chunk_id:
                        raise ValueError("VLM cache chunk marker does not match the FM request marker")
                with self._cache_lock:
                    if generation != self._refresh_generation:
                        raise RuntimeError("Prefix-cache refresh superseded")
                    self._pending_cache = (
                        int(envelope["cache_id"]),
                        envelope["cache_version"],
                        cache,
                        observation_chunk_id,
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
            cache_id, cache_version, cache = self._pending_cache[:3]
            observation_chunk_id = self._pending_cache[3] if len(self._pending_cache) > 3 else None
            self._prefix_cache = cache
            self._cache_id = cache_id
            self._cache_version = cache_version
            self._active_cache_observation_chunk_id = observation_chunk_id
            self._pending_cache = None

    def _snapshot_cache(self) -> tuple[dict[str, Any], str, str | None, int | None, int]:
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
            return (
                self._prefix_cache,
                self._cache_version,
                previous_version,
                self._active_cache_observation_chunk_id,
                self._cache_id,
            )

    def _record_observation_start(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("observation_started requires a non-empty request_id")
        chunk_id = int(self._current_chunk_index)
        with self._cache_lock:
            self._observation_start_chunks[request_id] = chunk_id
            while len(self._observation_start_chunks) > 256:
                self._observation_start_chunks.pop(next(iter(self._observation_start_chunks)))
        return {"protocol_version": 1, "request_id": request_id, "observation_chunk_id": chunk_id}

    def _reset_stream(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = request.get("session_id", "default")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        clear_prefix_cache = request.get("clear_prefix_cache", False)
        if not isinstance(clear_prefix_cache, bool):
            raise ValueError("clear_prefix_cache must be a boolean")
        self._streaming_state = None
        self._current_chunk_index = 0
        reset_history = getattr(getattr(self, "policy", None), "reset_tactile_history", None)
        if callable(reset_history):
            reset_history(session_id)
        if clear_prefix_cache:
            with self._cache_lock:
                self._refresh_generation += 1
                self._prefix_cache = None
                self._pending_cache = None
                self._cache_id = 0
                self._cache_version = None
                self._active_cache_observation_chunk_id = None
                self._observation_start_chunks.clear()
        return {
            "protocol_version": 1,
            "session_id": session_id,
            "status": "reset",
            "prefix_cache_cleared": clear_prefix_cache,
        }

    def _stream_infer(self, request: dict[str, Any]) -> dict[str, Any]:
        model = self.policy._model  # noqa: SLF001
        uses_observation_delay_embedding = getattr(model, "observation_delay_max_chunks", 0) > 0
        if uses_observation_delay_embedding:
            if not getattr(model, "streaming", False):
                raise ValueError("stream_infer requires a checkpoint configured with model.streaming=True")
            session_id = request.get("session_id", "default")
            execution_id = request.get("executed_action_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("session_id must be a non-empty string")
            if not isinstance(execution_id, int) or isinstance(execution_id, bool) or execution_id < 0:
                raise ValueError("executed_action_id must be a non-negative integer")
            chunk_size = int(model.streaming_chunk_size)
            self._current_chunk_index = execution_id // chunk_size
            (
                prefix_cache,
                cache_version,
                previous_version,
                cache_observation_chunk_id,
                cache_id,
            ) = self._snapshot_cache()
            expected = request.get("expected_cache_version")
            if expected is not None and expected not in (cache_version, previous_version):
                raise ValueError(
                    f"Cache version mismatch: expected {expected!r}, active {cache_version!r}"
                )
        else:
            # Preserve the pre-delay-embedding request order. In particular,
            # snapshot the active/pending cache before validating the request.
            (
                prefix_cache,
                cache_version,
                previous_version,
                cache_observation_chunk_id,
                cache_id,
            ) = self._snapshot_cache()
            expected = request.get("expected_cache_version")
            if expected is not None and expected not in (cache_version, previous_version):
                raise ValueError(
                    f"Cache version mismatch: expected {expected!r}, active {cache_version!r}"
                )
            if not getattr(model, "streaming", False):
                raise ValueError("stream_infer requires a checkpoint configured with model.streaming=True")
            session_id = request.get("session_id", "default")
            execution_id = request.get("executed_action_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("session_id must be a non-empty string")
            if not isinstance(execution_id, int) or isinstance(execution_id, bool) or execution_id < 0:
                raise ValueError("executed_action_id must be a non-negative integer")

        model_observation = self._prepare(
            request["observation"], session_id=session_id, update_tactile_history=True
        )
        tactile_condition = self._tactile_condition(model_observation)
        observation_delay_chunks = self._cache_delay_chunks(
            self._current_chunk_index, cache_observation_chunk_id
        )
        gate_log_context = {}
        if getattr(model, "_save_gate_logs", False) and getattr(model, "_gate_log_exp_name", None):
            gate_log_context = {
                "gate_log_execution_id": jnp.asarray(execution_id, dtype=jnp.int32),
                "gate_log_cache_id": jnp.asarray(cache_id, dtype=jnp.int32),
            }
        if self._streaming_state is None:
            self._streaming_state = self._seed_streaming_state(
                session_id,
                execution_id,
                model_observation.state,
                prefix_cache,
                tactile_condition=tactile_condition,
                tactile_left_marker_history=model_observation.tactile_left_marker_history,
                tactile_right_marker_history=model_observation.tactile_right_marker_history,
                torque_history=model_observation.torque_history,
                state_history=model_observation.state_history,
                observation_delay_chunks=observation_delay_chunks,
                **gate_log_context,
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
                    tactile_condition=tactile_condition,
                    tactile_left_marker_history=model_observation.tactile_left_marker_history,
                    tactile_right_marker_history=model_observation.tactile_right_marker_history,
                    torque_history=model_observation.torque_history,
                    state_history=model_observation.state_history,
                    observation_delay_chunks=observation_delay_chunks,
                    **gate_log_context,
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
        tactile_condition: Any = None,
        tactile_left_marker_history: Any = None,
        tactile_right_marker_history: Any = None,
        torque_history: Any = None,
        state_history: Any = None,
        observation_delay_chunks: Any = None,
        gate_log_execution_id: Any = None,
        gate_log_cache_id: Any = None,
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
            tactile_condition=tactile_condition,
            tactile_left_marker_history=tactile_left_marker_history,
            tactile_right_marker_history=tactile_right_marker_history,
            torque_history=torque_history,
            state_history=state_history,
            observation_delay_chunks=observation_delay_chunks,
            gate_log_execution_id=gate_log_execution_id,
            gate_log_cache_id=gate_log_cache_id,
        )
        timestep = self.policy._model.streaming_timestep(actions.dtype)  # noqa: SLF001
        noise = jax.random.normal(noise_rng, actions.shape, dtype=actions.dtype)
        chunk_size = int(self.policy._model.streaming_chunk_size)  # noqa: SLF001
        future_timestep = timestep[:-chunk_size]
        # ponytail: seed from a full sample plus the training marginal; retain an ODE trajectory only if cold starts need it.
        future_window = (
            future_timestep[None, :, None] * noise[:, :-chunk_size]
            + (1.0 - future_timestep[None, :, None]) * actions[:, chunk_size:]
        )
        # The ready chunk stays outside the H-token future window passed to FM.
        action_window = jnp.concatenate([actions[:, :chunk_size], future_window, noise[:, -chunk_size:]], axis=1)
        return _StreamingState(session_id, execution_id, action_window)

    def _infer_fm(self, request: dict[str, Any]) -> dict[str, Any]:
        prefix_cache, cache_version, previous_version, cache_observation_chunk_id, _ = self._snapshot_cache()
        expected = request.get("expected_cache_version")
        if expected is not None and expected not in (cache_version, previous_version):
            raise ValueError(
                f"Cache version mismatch: expected {expected!r}, active {cache_version!r}"
            )
        session_id = request.get("session_id", "default")
        model_observation = self._prepare(
            request["observation"], session_id=session_id, update_tactile_history=True
        )
        tactile_condition = self._tactile_condition(model_observation)
        observation_delay_chunks = self._cache_delay_chunks(
            self._current_chunk_index, cache_observation_chunk_id
        )
        self._rng, sample_rng = jax.random.split(self._rng)
        noise_tokens = int(request.get("noise_tokens", self.policy._model.action_horizon))  # noqa: SLF001
        if noise_tokens <= 0 or noise_tokens > self.policy._model.action_horizon:  # noqa: SLF001
            raise ValueError("noise_tokens must be in [1, action_horizon]")
        if (
            getattr(self.policy._model, "streaming_attention_mode", None)  # noqa: SLF001
            == "attention_gate"
            and noise_tokens != self.policy._model.action_horizon  # noqa: SLF001
        ):
            raise ValueError("attention gates require noise_tokens == action_horizon")
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
            tactile_condition=tactile_condition,
            tactile_left_marker_history=model_observation.tactile_left_marker_history,
            tactile_right_marker_history=model_observation.tactile_right_marker_history,
            torque_history=model_observation.torque_history,
            state_history=model_observation.state_history,
            observation_delay_chunks=observation_delay_chunks,
        )
        result = self.policy._output_transform(  # noqa: SLF001
            {"state": np.asarray(model_observation.state[0]), "actions": np.asarray(actions[0])}
        )
        result["cache_version"] = cache_version
        return result

    def _cache_delay_chunks(self, current_chunk_id: int, observation_chunk_id: int | None):
        if getattr(self.policy._model, "observation_delay_max_chunks", 0) <= 0:  # noqa: SLF001
            return None
        if observation_chunk_id is None:
            raise RuntimeError("Active prefix cache is missing its observation chunk marker")
        delay = max(0, current_chunk_id - observation_chunk_id)
        return jnp.asarray([delay], dtype=jnp.int32)
