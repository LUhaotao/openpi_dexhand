import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.pi0 import Pi0
from openpi.models.pi0 import _chunk_wise_timestep
from openpi.models.pi0 import _shift_streaming_window
from openpi.models.pi0 import posemb_sincos
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


def test_shift_streaming_window_denoises_shifted_tokens_and_refills_noise():
    actions = jnp.asarray([[[0.0], [10.0], [20.0], [30.0]]])
    velocity = jnp.asarray([[[0.0], [1.0], [2.0], [3.0]]])
    timestep = jnp.asarray([0.0, 0.25, 0.75, 1.0])
    fresh_noise = jnp.asarray([[[99.0]]])

    result = _shift_streaming_window(actions, velocity, timestep, fresh_noise)

    assert jnp.allclose(result, jnp.asarray([[[9.75], [19.0], [29.25], [99.0]]]))


def test_streaming_chunk_size_must_fit_action_horizon():
    with pytest.raises(ValueError, match="must not exceed"):
        _pi0_config.Pi0Config(action_horizon=2, streaming_chunk_size=3)


def test_streaming_sampler_advances_the_window_by_completed_chunks():
    class FakeStreamingModel:
        streaming = True
        streaming_chunk_size = 1
        action_dim = 1

        @staticmethod
        def streaming_timestep(dtype):
            return jnp.asarray([0.0, 0.5, 1.0], dtype=dtype)

        @staticmethod
        def _velocity_from_prefix(state, prefix_cache, noisy_actions, timestep):
            del state, prefix_cache, timestep
            return jnp.zeros_like(noisy_actions)

    window = jnp.asarray([[[1.0], [2.0], [3.0]]])
    result = Pi0.advance_streaming_actions_from_prefix(
        FakeStreamingModel(),
        jax.random.key(0),
        jnp.zeros((1, 1)),
        {},
        window,
        jnp.asarray(2, dtype=jnp.int32),
    )

    assert result.shape == window.shape
    assert jnp.allclose(result[:, 0], jnp.asarray([[3.0]]))


def test_pi05_streaming_training_accepts_tokenwise_timestep():
    key = jax.random.key(0)
    config = _pi0_config.Pi0Config(
        pi05=True,
        streaming=True,
        action_horizon=4,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(key)
    obs, act = config.fake_obs(batch_size=1), config.fake_act(batch_size=1)

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (1, config.action_horizon)


def test_pi05_continuous_state_uses_independent_state_projection_token():
    key = jax.random.key(0)
    config = _pi0_config.Pi0Config(
        pi05=True,
        discrete_state_input=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=3,
    )
    model = config.create(key)
    observation, _ = config.fake_obs(batch_size=1), config.fake_act(batch_size=1)
    noisy_actions = jnp.zeros((1, config.action_horizon, config.action_dim), dtype=jnp.float32)
    timestep = jnp.ones((1,), dtype=jnp.float32)

    suffix_tokens, suffix_mask, _, _ = model.embed_suffix(observation, noisy_actions, timestep)

    expected_state_token = model.state_proj(observation.state)[:, None, :]
    assert model.state_proj is not model.action_in_proj
    assert suffix_tokens.shape == (1, 1 + config.action_horizon, 64)
    assert suffix_mask.shape == (1, 1 + config.action_horizon)
    assert jnp.allclose(suffix_tokens[:, :1], expected_state_token)


def test_pi05_continuous_state_uses_clean_timestep_condition():
    key = jax.random.key(0)
    config = _pi0_config.Pi0Config(
        pi05=True,
        streaming=True,
        discrete_state_input=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=3,
    )
    model = config.create(key)
    observation, actions = config.fake_obs(batch_size=1), config.fake_act(batch_size=1)
    noisy_actions = jnp.zeros((1, config.action_horizon, config.action_dim), dtype=jnp.float32)
    timestep = jnp.asarray([[0.2, 0.5, 0.8]], dtype=jnp.float32)

    _, _, _, adarms_cond = model.embed_suffix(observation, noisy_actions, timestep)

    assert adarms_cond.shape == (1, 1 + config.action_horizon, 64)
    clean_time = jnp.zeros((1,), dtype=timestep.dtype)
    clean_time_emb = posemb_sincos(
        clean_time,
        model.action_in_proj.out_features,
        min_period=4e-3,
        max_period=4.0,
    )
    clean_cond = model.time_mlp_out(jax.nn.silu(model.time_mlp_in(clean_time_emb)))
    clean_cond = jax.nn.silu(clean_cond)
    assert jnp.allclose(adarms_cond[:, 0], clean_cond)

    loss = model.compute_loss(key, observation, actions)
    assert loss.shape == (1, config.action_horizon)


def test_checkpoint_loader_keeps_new_state_projection_initialized(tmp_path, monkeypatch):
    from openpi.models import model as _model
    from openpi.training import weight_loaders

    loaded = {"action_in_proj": {"kernel": np.ones((32, 1024), dtype=np.float32)}}
    monkeypatch.setattr(_model, "restore_params", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: tmp_path)

    loader = weight_loaders.CheckpointWeightLoader("unused")
    params = {
        "action_in_proj": {"kernel": np.zeros((32, 1024), dtype=np.float32)},
        "state_proj": {"kernel": np.full((32, 1024), 7.0, dtype=np.float32)},
    }

    result = loader.load(params)

    assert np.all(result["action_in_proj"]["kernel"] == 1.0)
    assert np.all(result["state_proj"]["kernel"] == 7.0)
