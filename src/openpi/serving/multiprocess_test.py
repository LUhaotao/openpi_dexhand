import numpy as np

from openpi.serving.multiprocess import decode_prefix_cache
from openpi.serving.multiprocess import encode_prefix_cache


def test_prefix_cache_round_trip_preserves_arrays_and_metadata():
    cache = {
        "kv_cache": (
            np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1,
        ),
        "prefix_mask": np.asarray([[True, True, False]]),
    }

    payload = encode_prefix_cache(cache)
    restored = decode_prefix_cache(payload)

    assert np.array_equal(restored["kv_cache"][0], cache["kv_cache"][0])
    assert np.array_equal(restored["kv_cache"][1], cache["kv_cache"][1])
    assert np.array_equal(restored["prefix_mask"], cache["prefix_mask"])
