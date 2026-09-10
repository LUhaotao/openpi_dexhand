# ruff: noqa: SLF001

import jax.numpy as jnp
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
