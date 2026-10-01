from typing import Any

import jax
import numpy as np

from openpi.fake_env.config import FakeEnvConfig
from openpi.models import pi0_config as _pi0_config


def make_production_wire_observation(config: FakeEnvConfig) -> dict[str, Any]:
    """Create one deterministic observation matching the client wire contract."""
    image = np.zeros((config.image_height, config.image_width, 3), dtype=np.uint8)
    observation: dict[str, Any] = {
        "image": {
            "base_0_rgb": image.copy(),
            "left_wrist_0_rgb": image.copy(),
            "right_wrist_0_rgb": image.copy(),
        },
        "image_mask": {
            "base_0_rgb": np.ones((), dtype=np.bool_),
            "left_wrist_0_rgb": np.ones((), dtype=np.bool_),
            "right_wrist_0_rgb": np.ones((), dtype=np.bool_),
        },
        "state": np.zeros((config.action_dim,), dtype=np.float32),
        "prompt": config.prompt,
    }
    if config.use_torque:
        observation["torque"] = np.zeros((config.torque_dim,), dtype=np.float32)
    if config.use_tactile:
        marker = np.zeros(_pi0_config.TACTILE_MARKER_SHAPE, dtype=np.float32)
        observation["tactile_left_marker"] = marker.copy()
        observation["tactile_right_marker"] = marker.copy()
    # The real client normally tokenizes the prompt. The fake wire fixture
    # carries tokenized fields directly so benchmark setup does not add tokenizer cost.
    observation["tokenized_prompt"] = np.zeros((config.max_token_len,), dtype=np.int32)
    return observation


def wire_to_model_dict(config: FakeEnvConfig, observation: dict[str, Any]) -> dict[str, Any]:
    """Convert production-wire leaves into the unbatched model dictionary."""
    result = {
        "image": {key: np.asarray(value) for key, value in observation["image"].items()},
        "image_mask": {
            key: np.asarray(value, dtype=np.bool_) for key, value in observation["image_mask"].items()
        },
        "state": np.asarray(observation["state"], dtype=np.float32),
        "tokenized_prompt": np.asarray(
            observation.get("tokenized_prompt", np.zeros((config.max_token_len,), dtype=np.int32)), dtype=np.int32
        ),
        "tokenized_prompt_mask": np.asarray(
            observation.get("tokenized_prompt_mask", np.ones((config.max_token_len,), dtype=np.bool_)), dtype=np.bool_
        ),
    }
    for key in (
        "torque",
        "tactile_left_marker",
        "tactile_right_marker",
        "state_history",
        "torque_history",
        "tactile_left_marker_history",
        "tactile_right_marker_history",
    ):
        if key in observation:
            result[key] = np.asarray(observation[key])
    return result


def model_dict_to_observation(model_dict: dict[str, Any]):
    """Build a model Observation from an unbatched model dictionary."""
    return _model_observation_from_dict(model_dict)


def _model_observation_from_dict(model_dict: dict[str, Any]):
    from openpi.models import model as _model

    batched = jax.tree.map(lambda value: jax.numpy.asarray(value)[None, ...], model_dict)
    return _model.Observation.from_dict(batched)


class WireToModelTransform:
    """Input transform used by the fake server's Policy wrapper."""

    def __init__(self, config: FakeEnvConfig):
        self._config = config

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return wire_to_model_dict(self._config, data)
