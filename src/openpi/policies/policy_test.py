# ruff: noqa: SLF001

import threading

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)


def test_warmup_uses_explicit_observation():
    warmup_observation = object()
    observed = []
    policy = object.__new__(_policy.Policy)
    policy._is_pytorch_model = False
    policy._warmup_observation = warmup_observation
    policy._sample_kwargs = {}

    def sample_actions(_rng, observation):
        observed.append(observation)
        return jnp.asarray(0.0)

    policy._sample_actions = sample_actions

    policy.warmup()

    assert observed == [warmup_observation]


def test_tactile_history_is_buffered_per_session_and_left_padded():
    policy = object.__new__(_policy.Policy)
    policy._tactile_history_length = 3
    policy._tactile_histories = {}
    policy._tactile_history_lock = threading.Lock()

    def inputs(value):
        return {
            "tactile_left_marker": np.asarray([value], dtype=np.float32),
            "tactile_right_marker": np.asarray([value + 10], dtype=np.float32),
        }

    first = inputs(1)
    policy._attach_tactile_history(first, "robot-a")
    np.testing.assert_array_equal(first["tactile_left_marker_history"].reshape(-1), [1, 1, 1])

    second = inputs(2)
    policy._attach_tactile_history(second, "robot-a")
    np.testing.assert_array_equal(second["tactile_left_marker_history"].reshape(-1), [1, 1, 2])

    other = inputs(9)
    policy._attach_tactile_history(other, "robot-b")
    np.testing.assert_array_equal(other["tactile_left_marker_history"].reshape(-1), [9, 9, 9])

    policy.reset_tactile_history("robot-a")
    restarted = inputs(3)
    policy._attach_tactile_history(restarted, "robot-a")
    np.testing.assert_array_equal(restarted["tactile_left_marker_history"].reshape(-1), [3, 3, 3])


def test_policy_infer_batches_a_single_frame_history_window():
    policy = object.__new__(_policy.Policy)
    policy._is_pytorch_model = False
    policy._input_transform = lambda observation: observation
    policy._output_transform = lambda output: output
    policy._sample_kwargs = {}
    policy._tactile_history_length = 2
    policy._tactile_histories = {}
    policy._tactile_history_lock = threading.Lock()
    policy._rng = jax.random.key(0)
    seen = []

    def sample_actions(_rng, observation):
        seen.append(observation)
        return jnp.zeros((1, 3, 1), dtype=jnp.float32)

    policy._sample_actions = sample_actions
    marker = np.zeros((2, 63, 2), dtype=np.float32)
    observation = {
        "image": {"camera": np.zeros((2, 2, 3), dtype=np.uint8)},
        "image_mask": {"camera": np.asarray(True)},
        "state": np.zeros((2,), dtype=np.float32),
        "tactile_left_marker": marker,
        "tactile_right_marker": marker + 1.0,
    }

    result = policy.infer(observation, session_id="client-1")

    assert result["actions"].shape == (3, 1)
    assert seen[0].tactile_left_marker_history.shape == (1, 2, *marker.shape)
    np.testing.assert_array_equal(seen[0].tactile_left_marker_history[0, :, 0, 0, 0], [0.0, 0.0])
