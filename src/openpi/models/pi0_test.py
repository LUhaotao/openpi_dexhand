import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models.pi0 import _chunk_wise_timestep
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_chunk_wise_timestep_uses_openpi_noise_direction():
    timestep = _chunk_wise_timestep(action_horizon=8, chunk_size=2, dtype=jnp.float32)

    # OpenPI uses t=0 for clean actions and t=1 for pure noise.
    expected = jnp.asarray([0.0, 0.0, 0.125, 0.375, 0.625, 0.875, 1.0, 1.0], dtype=jnp.float32)
    assert jnp.allclose(timestep, expected)


def test_pi05_streaming_training_accepts_tokenwise_timestep():
    key = jax.random.key(0)
    config = _pi0_config.Pi0Config(pi05=True, streaming=True, action_horizon=4)
    model = config.create(key)
    obs, act = config.fake_obs(batch_size=1), config.fake_act(batch_size=1)

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (1, config.action_horizon)
