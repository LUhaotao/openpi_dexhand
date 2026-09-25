from collections import deque
from collections.abc import Sequence
import logging
import pathlib
import threading
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        warmup_observation: _model.Observation | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            warmup_observation: A model-format observation used to compile the JAX sampler during warmup.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._warmup_observation = warmup_observation
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._tactile_history_length = (
            int(model.tactile_history_length)
            if not is_pytorch
            and getattr(model, "streaming_attention_mode", None) == "tactile_attention_gate"
            else 0
        )
        self._tactile_histories: dict[str, tuple[deque, deque]] = {}
        self._tactile_history_lock = threading.Lock()

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    def warmup(self) -> None:
        """Compile the JAX sampler before serving requests."""
        if self._is_pytorch_model:
            return

        if self._warmup_observation is None:
            raise ValueError("warmup_observation is required to warm up a JAX policy")

        actions = self._sample_actions(jax.random.key(0), self._warmup_observation, **self._sample_kwargs)
        jax.tree.map(
            lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
            actions,
        )

    def prepare_observation(
        self, obs: dict, *, session_id: str = "default", update_tactile_history: bool = True
    ) -> _model.Observation:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if self._tactile_history_length and update_tactile_history:
            self._attach_tactile_history(inputs, session_id)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return _model.Observation.from_dict(inputs)

    def _attach_tactile_history(self, inputs: dict, session_id: str) -> None:
        left_key = "tactile_left_marker"
        right_key = "tactile_right_marker"
        left_history_key = "tactile_left_marker_history"
        right_history_key = "tactile_right_marker_history"
        if left_history_key in inputs or right_history_key in inputs:
            if left_history_key not in inputs or right_history_key not in inputs:
                raise ValueError("left and right tactile histories must be provided together")
            for key in (left_history_key, right_history_key):
                if np.asarray(inputs[key]).shape[0] != self._tactile_history_length:
                    raise ValueError(
                        f"Expected {key} length {self._tactile_history_length}, got {np.asarray(inputs[key]).shape[0]}"
                    )
            return
        if left_key not in inputs or right_key not in inputs:
            raise ValueError("tactile_attention_gate requires current left and right marker frames")

        left = np.asarray(inputs[left_key], dtype=np.float32)
        right = np.asarray(inputs[right_key], dtype=np.float32)
        with self._tactile_history_lock:
            histories = self._tactile_histories.get(session_id)
            if histories is None:
                histories = (
                    deque(maxlen=self._tactile_history_length),
                    deque(maxlen=self._tactile_history_length),
                )
                self._tactile_histories[session_id] = histories
            left_history, right_history = histories
            left_history.append(left.copy())
            right_history.append(right.copy())
            while len(left_history) < self._tactile_history_length:
                left_history.appendleft(left_history[0].copy())
                right_history.appendleft(right_history[0].copy())
            inputs[left_history_key] = np.stack(tuple(left_history), axis=0)
            inputs[right_history_key] = np.stack(tuple(right_history), axis=0)

    def reset_tactile_history(self, session_id: str) -> None:
        with self._tactile_history_lock:
            self._tactile_histories.pop(session_id, None)

    @property
    def tactile_history_enabled(self) -> bool:
        return self._tactile_history_length > 0

    def infer_with_session(self, obs: dict, *, session_id: str) -> dict:
        return self.infer(obs, session_id=session_id)

    @override
    def infer(
        self, obs: dict, *, noise: np.ndarray | None = None, session_id: str = "default"
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        uses_tactile_history = not self._is_pytorch_model and self._tactile_history_length > 0
        if uses_tactile_history:
            observation = self.prepare_observation(obs, session_id=session_id)
            inputs = {"state": observation.state}
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(lambda x: x, obs)
            inputs = self._input_transform(inputs)
        if self._is_pytorch_model:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device
        elif not uses_tactile_history:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        if not uses_tactile_history:
            observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)
        return self._record(obs, results)

    def _record(self, obs: dict, results: dict) -> dict:
        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

    def infer_with_session(self, obs: dict, *, session_id: str) -> dict:
        infer_with_session = getattr(self._policy, "infer_with_session", None)
        if callable(infer_with_session):
            results = infer_with_session(obs, session_id=session_id)
        else:
            results = self._policy.infer(obs)
        return self._record(obs, results)

    def reset_tactile_history(self, session_id: str) -> None:
        reset = getattr(self._policy, "reset_tactile_history", None)
        if callable(reset):
            reset(session_id)

    @property
    def tactile_history_enabled(self) -> bool:
        return bool(getattr(self._policy, "tactile_history_enabled", False))
