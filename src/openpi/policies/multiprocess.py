"""Two-process VLM/FM policy wrapper used by the optional server mode."""

import dataclasses
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.serving.multiprocess import decode_prefix_cache
from openpi.serving.multiprocess import encode_prefix_cache


@dataclasses.dataclass
class MultiProcessPolicy:
    policy: _policy.Policy
    role: Literal["vlm", "fm"]

    def __post_init__(self):
        if self.role not in ("vlm", "fm"):
            raise ValueError(f"Unknown multi-process role: {self.role}")
        self._rng = jax.random.key(0)

    def _prepare(self, observation: dict[str, Any]) -> _model.Observation:
        inputs = jax.tree.map(lambda x: x, observation)
        inputs = self.policy._input_transform(inputs)  # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        return _model.Observation.from_dict(inputs)

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.role == "vlm":
            observation = request.get("observation", request)
            model_observation = self._prepare(observation)
            cache = self.policy._model.encode_prefix(model_observation)  # noqa: SLF001
            cache = encode_prefix_cache(cache)
            return {"prefix_cache": cache}

        observation = request["observation"]
        cache = decode_prefix_cache(request["prefix_cache"])
        model_observation = self._prepare(observation)
        self._rng, sample_rng = jax.random.split(self._rng)
        actions = self.policy._model.sample_actions_from_prefix(  # noqa: SLF001
            sample_rng,
            model_observation.state,
            cache,
            num_steps=int(request.get("num_steps", 10)),
        )
        result = {"actions": np.asarray(actions[0])}
        return self.policy._output_transform(  # noqa: SLF001
            {"state": np.asarray(model_observation.state[0]), **result}
        )
