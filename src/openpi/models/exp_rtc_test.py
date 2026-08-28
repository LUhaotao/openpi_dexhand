import jax.numpy as jnp

from openpi.models.exp_rtc import DEFAULT_PREFIX_ATTENTION_HORIZON, get_prefix_weights, guide_velocity


def test_default_prefix_attention_horizon_is_ten():
    assert DEFAULT_PREFIX_ATTENTION_HORIZON == 10


def test_linear_prefix_weights_match_rtc_schedule():
    weights = get_prefix_weights(2, 6, 10, "linear")
    expected = jnp.array([1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(weights, expected)


def test_guidance_uses_denoiser_vjp_to_reduce_prefix_error():
    def denoise(x_t, time):
        del time
        return x_t, jnp.zeros_like(x_t)

    x_t = jnp.zeros((1, 2, 1))
    previous = jnp.ones((1, 2, 1))
    guided = guide_velocity(
        denoise,
        x_t,
        jnp.asarray(0.5),
        previous,
        inference_delay=0,
        prefix_attention_horizon=2,
        prefix_attention_schedule="ones",
        max_guidance_weight=1.0,
    )

    assert jnp.all(guided < 0.0)
    jnp.asarray(guided).block_until_ready()
