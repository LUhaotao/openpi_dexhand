from collections.abc import Mapping, Sequence
import logging
import pathlib
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
from openpi.shared import normalize as _normalize

BasePolicy: TypeAlias = _base_policy.BasePolicy


def reanchor_rtc_action_prefix(
    raw_actions: np.ndarray,
    *,
    previous_reference_state: np.ndarray,
    current_state: np.ndarray,
    action_stats: _normalize.NormStats,
    relative_action_mask: Sequence[bool],
    use_quantile_norm: bool = False,
) -> np.ndarray:
    """Express a raw RTC prefix relative to the current observation state.

    ``raw_actions`` is in the model's normalized action space. The relative
    action dimensions must be decoded against their old observation state and
    encoded again against the state for the request being served.
    """
    reanchored = np.asarray(raw_actions).copy()
    if reanchored.ndim != 2:
        raise ValueError(f"RTC prev_action_chunk must have shape (T, A), got {reanchored.shape}")

    mask = np.asarray(relative_action_mask, dtype=bool)
    dims = len(mask)
    previous_state = np.asarray(previous_reference_state)
    next_state = np.asarray(current_state)
    if previous_state.ndim != 1 or next_state.ndim != 1:
        raise ValueError("RTC reference states must be one-dimensional")
    if reanchored.shape[1] < dims or previous_state.shape[0] < dims or next_state.shape[0] < dims:
        raise ValueError("RTC prefix, previous state, and current state must cover every action dimension")
    if not np.any(mask):
        return reanchored

    if use_quantile_norm:
        if action_stats.q01 is None or action_stats.q99 is None:
            raise ValueError("RTC quantile normalization requires q01 and q99 action statistics")
        offset = np.asarray(action_stats.q01)[:dims]
        scale = np.asarray(action_stats.q99)[:dims] - offset + 1e-6
        decoded = (reanchored[:, :dims] + 1.0) / 2.0 * scale + offset
    else:
        offset = np.asarray(action_stats.mean)[:dims]
        scale = np.asarray(action_stats.std)[:dims] + 1e-6
        decoded = reanchored[:, :dims] * scale + offset

    decoded[:, mask] += previous_state[:dims][mask]
    decoded[:, mask] -= next_state[:dims][mask]
    if use_quantile_norm:
        reanchored[:, :dims] = (decoded - offset) / scale * 2.0 - 1.0
    else:
        reanchored[:, :dims] = (decoded - offset) / scale
    return reanchored


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        rtc_action_stats: _normalize.NormStats | None = None,
        rtc_relative_action_mask: Sequence[bool] | None = None,
        rtc_use_quantile_norm: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._rtc_action_stats = rtc_action_stats
        self._rtc_relative_action_mask = rtc_relative_action_mask
        self._rtc_use_quantile_norm = rtc_use_quantile_norm

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        rtc_context: Mapping[str, Any] | None = None,
    ) -> dict:  # type: ignore[misc]
        if rtc_context is not None and self._is_pytorch_model:
            raise NotImplementedError("Experimental inference-time RTC currently supports JAX OpenPI models only")

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        if rtc_context is not None:
            previous_chunk = rtc_context.get("prev_action_chunk")
            if previous_chunk is not None:
                previous_chunk = np.asarray(previous_chunk)
                if previous_chunk.ndim != 2:
                    raise ValueError(
                        f"RTC prev_action_chunk must have shape (T, A), got {previous_chunk.shape}"
                    )
                action_horizon = self._model.action_horizon
                if previous_chunk.shape[1] != self._model.action_dim:
                    raise ValueError(
                        "RTC prev_action_chunk action width does not match the policy model: "
                        f"expected {self._model.action_dim}, got {previous_chunk.shape[1]}"
                    )
                previous_reference_state = rtc_context.get("prev_action_reference_state")
                if previous_reference_state is None:
                    raise ValueError("RTC prev_action_reference_state is required with prev_action_chunk")
                if self._rtc_action_stats is None or self._rtc_relative_action_mask is None:
                    raise ValueError("This policy does not support relative-action RTC prefixes")
                previous_chunk = reanchor_rtc_action_prefix(
                    previous_chunk,
                    previous_reference_state=np.asarray(previous_reference_state),
                    current_state=np.asarray(obs["state"]),
                    action_stats=self._rtc_action_stats,
                    relative_action_mask=self._rtc_relative_action_mask,
                    use_quantile_norm=self._rtc_use_quantile_norm,
                )
                if previous_chunk.shape[0] < action_horizon:
                    padded = np.zeros((action_horizon, previous_chunk.shape[1]), dtype=previous_chunk.dtype)
                    padded[: previous_chunk.shape[0]] = previous_chunk
                    previous_chunk = padded
                else:
                    previous_chunk = previous_chunk[:action_horizon]
                sample_kwargs["previous_action_chunk"] = jnp.asarray(previous_chunk)[None, ...]
                sample_kwargs["inference_delay"] = jnp.asarray(int(rtc_context.get("inference_delay", 0)))

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

        raw_actions = np.array(outputs["actions"], copy=True)
        outputs = self._output_transform(outputs)
        if rtc_context is not None:
            outputs["raw_model_actions"] = raw_actions
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

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
