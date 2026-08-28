from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp


PrefixAttentionSchedule = Literal["linear", "exp", "ones", "zeros"]
DEFAULT_PREFIX_ATTENTION_HORIZON = 10


def get_prefix_weights(
    start: int | jax.Array,
    end: int | jax.Array,
    total: int,
    schedule: PrefixAttentionSchedule,
) -> jax.Array:
    """Build RTC prefix guidance weights for one action chunk."""
    start = jnp.minimum(start, end)
    positions = jnp.arange(total)
    if schedule == "ones":
        weights = jnp.ones(total)
    elif schedule == "zeros":
        weights = (positions < start).astype(jnp.float32)
    elif schedule in {"linear", "exp"}:
        weights = jnp.clip((start - 1 - positions) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            weights = weights * jnp.expm1(weights) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid RTC prefix attention schedule: {schedule!r}")
    return jnp.where(positions >= end, 0, weights)


def guide_velocity(
    denoise: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    x_t: jax.Array,
    time: jax.Array,
    previous_chunk: jax.Array,
    *,
    inference_delay: int | jax.Array,
    prefix_attention_horizon: int | jax.Array,
    prefix_attention_schedule: PrefixAttentionSchedule,
    max_guidance_weight: float,
) -> jax.Array:
    """Apply inference-time RTC guidance to an OpenPI flow velocity.

    OpenPI integrates from ``t=1`` (noise) to ``t=0`` (action), so the
    denoised endpoint is ``x_t - time * v_t``.
    """
    x1_t, vjp_fn, v_t = jax.vjp(
        lambda x: denoise(x, time),
        x_t,
        has_aux=True,
    )
    horizon = jnp.minimum(prefix_attention_horizon, previous_chunk.shape[1])
    weights = get_prefix_weights(
        inference_delay,
        horizon,
        x_t.shape[1],
        prefix_attention_schedule,
    )
    error = (previous_chunk - x1_t) * weights[None, :, None]
    correction = vjp_fn(error)[0]

    tau = 1 - time
    squared_one_minus_tau = (1 - tau) ** 2
    inv_r2 = (squared_one_minus_tau + tau**2) / squared_one_minus_tau
    c = jnp.nan_to_num((1 - tau) / tau, posinf=max_guidance_weight)
    guidance_weight = jnp.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
    guidance_weight = jnp.minimum(guidance_weight, max_guidance_weight)
    return v_t - guidance_weight * correction
