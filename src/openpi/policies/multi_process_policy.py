"""Blocking VLM/FM policy endpoints for the optional multi-process server."""

from __future__ import annotations

import dataclasses
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
        self._vlm_client = None
        self._encode_prefix_jit = nnx_utils.module_jit(self.policy._model.encode_prefix)  # noqa: SLF001
        self._sample_actions_from_prefix_jit = nnx_utils.module_jit(
            self.policy._model.sample_actions_from_prefix  # noqa: SLF001
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
        if op != "infer":
            raise ValueError(f"Unsupported FM operation: {op}")
        return self._infer_fm(request)

    def _refresh_prefix(self, request: dict[str, Any]) -> dict[str, Any]:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        if self.vlm_port is None:
            raise ValueError("vlm_port is required for FM refresh")
        if self._vlm_client is None:
            self._vlm_client = WebsocketClientPolicy(self.vlm_host, self.vlm_port)
        envelope = self._vlm_client.infer({
            "op": "get_prefix_cache",
            "cache_id": request["cache_id"],
            "cache_version": request["cache_version"],
        })
        validate_cache_envelope(envelope, model_id=self.model_id)
        cache = decode_prefix_cache(envelope["prefix_cache"])
        self._prefix_cache = jax.device_put(cache)
        self._cache_id = int(envelope["cache_id"])
        self._cache_version = envelope["cache_version"]
        return {
            "protocol_version": 1,
            "cache_id": self._cache_id,
            "cache_version": self._cache_version,
            "status": "active",
        }

    def _infer_fm(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._prefix_cache is None or self._cache_version is None:
            raise RuntimeError("No active prefix cache; call refresh_prefix first")
        expected = request.get("expected_cache_version")
        if expected is not None and expected != self._cache_version:
            raise ValueError(
                f"Cache version mismatch: expected {expected!r}, active {self._cache_version!r}"
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
            self._prefix_cache,
            num_steps=int(request.get("num_steps", 10)),
            noise=noise,
        )
        return self.policy._output_transform(  # noqa: SLF001
            {"state": np.asarray(model_observation.state[0]), "actions": np.asarray(actions[0])}
        )
