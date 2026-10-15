# ruff: noqa: SLF001

import threading
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

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
    policy._observation_start_chunks = {}
    policy._streaming_state = _StreamingState("old", 12, object())

    result = policy._reset_stream({"session_id": "new", "clear_prefix_cache": True})

    assert result["prefix_cache_cleared"] is True
    assert policy._streaming_state is None
    assert policy._prefix_cache is None
    assert policy._pending_cache is None
    assert policy._cache_version is None
    assert policy._cache_id == 0
    assert policy._refresh_generation == 5


def test_unified_gate_warmup_uses_one_history_length_for_all_sources():
    policy = object.__new__(MultiProcessPolicy)
    policy.role = "vlm"
    policy.policy = SimpleNamespace(_model=SimpleNamespace(
        action_dim=3,
        max_token_len=4,
        streaming_attention_mode="attention_gate",
        gate_sources=("torque", "tactile", "state"),
        gate_history_length=2,
        use_torque=True,
        torque_dim=3,
        use_tactile=True,
    ))
    seen = []
    policy._encode_prefix_jit = lambda observation: seen.append(observation)

    policy.warmup()

    observation = seen[0]
    assert observation.torque_history.shape == (1, 2, 3)
    assert observation.state_history.shape == (1, 2, 3)
    assert observation.tactile_left_marker_history.shape[1] == 2
    assert observation.tactile_right_marker_history.shape[1] == 2


def test_refresh_prefix_can_wait_for_active_cache(monkeypatch):
    policy = object.__new__(MultiProcessPolicy)
    policy._cache_lock = threading.Lock()
    policy._refresh_generation = 4
    policy._cache_id = 7
    policy._cache_version = "model:7"
    policy._prefix_cache = {"cache": "active"}
    policy._pending_cache = None
    policy._initial_ready = threading.Event()
    policy._initial_error = None
    policy.vlm_port = 8001

    def enqueue_refresh(generation, request, future):
        assert generation == 5
        assert future is not None
        policy._pending_cache = (8, request["cache_version"], {"cache": "new"})
        future.set_result({"cache_id": 8, "cache_version": request["cache_version"]})

    monkeypatch.setattr(policy, "_enqueue_refresh", enqueue_refresh)

    result = policy._refresh_prefix({
        "cache_id": 8,
        "cache_version": "model:8",
        "wait_for_activation": True,
    })

    assert result["status"] == "active"
    assert result["active_cache_id"] == 8
    assert policy._cache_version == "model:8"
    assert policy._prefix_cache == {"cache": "new"}


def test_observation_start_marker_snapshots_current_fm_chunk():
    policy = object.__new__(MultiProcessPolicy)
    policy._cache_lock = threading.Lock()
    policy._current_chunk_index = 4
    policy._observation_start_chunks = {}

    marker = policy._record_observation_start({"request_id": "vlm-request-1"})

    assert marker == {
        "protocol_version": 1,
        "request_id": "vlm-request-1",
        "observation_chunk_id": 4,
    }
    assert policy._observation_start_chunks["vlm-request-1"] == 4


def test_cache_delay_advances_while_same_prefix_cache_is_active():
    policy = object.__new__(MultiProcessPolicy)
    policy.policy = SimpleNamespace(_model=SimpleNamespace(observation_delay_max_chunks=8))

    first_use = policy._cache_delay_chunks(4, 0)
    next_chunk = policy._cache_delay_chunks(5, 0)
    after_cache_switch = policy._cache_delay_chunks(5, 1)

    assert int(first_use[0]) == 4
    assert int(next_chunk[0]) == 5
    assert int(after_cache_switch[0]) == 4


def test_stream_infer_without_delay_embedding_snapshots_before_request_validation(monkeypatch):
    policy = object.__new__(MultiProcessPolicy)
    policy.policy = SimpleNamespace(
        _model=SimpleNamespace(streaming=True, streaming_chunk_size=5, observation_delay_max_chunks=0)
    )
    snapshots = []

    def snapshot():
        snapshots.append(True)
        return {}, "cache:1", None, None

    monkeypatch.setattr(policy, "_snapshot_cache", snapshot)

    with pytest.raises(ValueError, match="session_id"):
        policy._stream_infer({"session_id": "", "executed_action_id": 0})

    assert snapshots == [True]


def test_stream_infer_with_delay_embedding_sets_chunk_before_snapshot(monkeypatch):
    policy = object.__new__(MultiProcessPolicy)
    policy.policy = SimpleNamespace(
        _model=SimpleNamespace(streaming=True, streaming_chunk_size=5, observation_delay_max_chunks=6)
    )

    class SnapshotReachedError(Exception):
        pass

    def snapshot():
        assert policy._current_chunk_index == 2
        raise SnapshotReachedError

    monkeypatch.setattr(policy, "_snapshot_cache", snapshot)

    with pytest.raises(SnapshotReachedError):
        policy._stream_infer({"session_id": "test", "executed_action_id": 10})


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
