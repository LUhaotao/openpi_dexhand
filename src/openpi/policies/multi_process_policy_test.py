# ruff: noqa: SLF001

import threading

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
