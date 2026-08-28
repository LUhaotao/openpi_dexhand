import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.policies.policy import reanchor_rtc_action_prefix
from openpi.shared.normalize import NormStats
from openpi.training import config as _config


def test_reanchor_rtc_action_prefix_preserves_absolute_tcp_targets():
    raw_actions = np.zeros((2, 32), dtype=np.float32)
    raw_actions[:, :6] = np.array([[1.0, -2.0, 0.5, 3.0, -1.0, 2.0]] * 2, dtype=np.float32)
    stats = NormStats(
        mean=np.arange(18, dtype=np.float32),
        std=np.full(18, 2.0, dtype=np.float32),
    )
    previous_state = np.arange(18, dtype=np.float32)
    current_state = previous_state + np.array([0.3, -0.4, 0.2, 0.1, -0.2, 0.5] + [0.0] * 12)

    reanchored = reanchor_rtc_action_prefix(
        raw_actions,
        previous_reference_state=previous_state,
        current_state=current_state,
        action_stats=stats,
        relative_action_mask=(True,) * 6 + (False,) * 12,
    )

    old_delta = raw_actions[:, :6] * (stats.std[:6] + 1e-6) + stats.mean[:6]
    old_absolute_targets = old_delta + previous_state[:6]
    new_delta = reanchored[:, :6] * (stats.std[:6] + 1e-6) + stats.mean[:6]
    new_absolute_targets = new_delta + current_state[:6]
    assert np.allclose(new_absolute_targets, old_absolute_targets)
    assert np.array_equal(reanchored[:, 6:], raw_actions[:, 6:])


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
