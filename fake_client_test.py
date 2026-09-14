import numpy as np

from fake_client import build_fm_request
from fake_client import build_zero_observation
from fake_client import summarize_latencies


def test_build_zero_observation_for_franka_xhand_continuous_state():
    observation = build_zero_observation(
        "franka_xhand_continuous_state",
        state_dim=18,
        height=16,
        width=20,
        prompt="test",
    )
    assert observation["images"]["cam_side"].shape == (16, 20, 3)
    assert observation["images"]["cam_side"].dtype == np.uint8
    assert observation["state"].shape == (18,)
    assert observation["state"].dtype == np.float32
    assert np.count_nonzero(observation["state"]) == 0


def test_build_fm_request_does_not_include_cache_transfer_fields():
    observation = {"state": np.zeros(2, dtype=np.float32)}
    request = build_fm_request(observation, num_steps=10, noise_tokens=1)
    assert request == {
        "op": "infer",
        "observation": observation,
        "num_steps": 10,
        "noise_tokens": 1,
    }


def test_summarize_latencies_includes_tail_percentiles():
    result = summarize_latencies([1.0, 2.0, 3.0])
    assert result == {
        "min_ms": 1.0,
        "mean_ms": 2.0,
        "p50_ms": 2.0,
        "p95_ms": 2.9,
        "p99_ms": 3.0,
        "max_ms": 3.0,
    }
