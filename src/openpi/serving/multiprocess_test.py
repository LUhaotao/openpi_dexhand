import numpy as np
from openpi_client import msgpack_numpy

from openpi.serving.multiprocess import decode_prefix_cache
from openpi.serving.multiprocess import encode_prefix_cache
from openpi.serving.multiprocess import make_cache_envelope
from openpi.serving.multiprocess import validate_cache_envelope


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


def test_cache_envelope_validates_manifest_and_preserves_bfloat16():
    cache = {"kv_cache": np.ones((2,), dtype=np.dtype("bfloat16"))}
    envelope = make_cache_envelope(cache, model_id="pi05", cache_id=3, cache_version="pi05:3")
    validate_cache_envelope(envelope, model_id="pi05")
    assert envelope["manifest"][0]["dtype"] == "bfloat16"


def test_decode_prefix_cache_restores_kv_cache_tuple_after_msgpack():
    cache = {"kv_cache": (np.ones((2,)), np.zeros((2,)))}
    wire = msgpack_numpy.unpackb(msgpack_numpy.packb(encode_prefix_cache(cache)))
    restored = decode_prefix_cache(wire)
    assert isinstance(restored["kv_cache"], tuple)


def test_fm_request_can_select_noise_token_count():
    from scripts.multi_process_client import build_fm_request

    request = build_fm_request({}, "v1", num_steps=4, noise_tokens=1)
    assert request["num_steps"] == 4
    assert request["noise_tokens"] == 1


def test_stream_request_only_carries_stream_identity_and_execution_clock():
    from scripts.multi_process_client import build_stream_request

    request = build_stream_request({}, "v1", session_id="episode-3", executed_action_id=12)

    assert request == {
        "op": "stream_infer",
        "observation": {},
        "expected_cache_version": "v1",
        "session_id": "episode-3",
        "executed_action_id": 12,
        "num_steps": 10,
    }
