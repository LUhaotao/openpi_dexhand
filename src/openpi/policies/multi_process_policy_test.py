# ruff: noqa: SLF001

import threading
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from openpi.policies.multi_process_policy import MultiProcessPolicy
from openpi.policies.multi_process_policy import _StreamingState


def test_reset_stream_can_clear_prefix_cache():
    policy = object.__new__(MultiProcessPolicy)
    policy._cache_lock = threading.Lock()
    policy._refresh_generation = 4
    policy._cache_id = 7
    policy._cache_version = "model:7"
    policy._prefix_cache = {"cache": "active"}
    policy._pending_cache = (8, "model:8", {"cache": "pending"})
    policy._streaming_state = _StreamingState("old", 12, object())

    result = policy._reset_stream({"session_id": "new", "clear_prefix_cache": True})

    assert result["prefix_cache_cleared"] is True
    assert policy._streaming_state is None
    assert policy._prefix_cache is None
    assert policy._pending_cache is None
    assert policy._cache_version is None
    assert policy._cache_id == 0
    assert policy._refresh_generation == 5


def test_seed_streaming_state_keeps_clean_chunk_before_future_window(monkeypatch):
    policy = object.__new__(MultiProcessPolicy)
    policy._rng = jax.random.key(0)
    policy._sample_actions_from_prefix_jit = lambda *args, **kwargs: jnp.asarray([[[1.0], [2.0], [3.0]]])
    policy.policy = SimpleNamespace(
        _model=SimpleNamespace(
            streaming_chunk_size=1,
            streaming_timestep=lambda dtype: jnp.asarray([1 / 3, 2 / 3, 1.0], dtype=dtype),
        )
    )
    monkeypatch.setattr(
        jax.random,
        "normal",
        lambda *args, **kwargs: jnp.asarray([[[10.0], [20.0], [30.0]]]),
    )

    result = policy._seed_streaming_state("test", 0, jnp.zeros((1, 1)), {}, num_steps=10)

    expected = jnp.asarray([[[1.0], [14 / 3], [43 / 3], [30.0]]])
    assert jnp.allclose(result.action_window, expected)
