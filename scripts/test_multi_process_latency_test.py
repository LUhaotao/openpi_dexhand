import numpy as np

from scripts.test_multi_process_latency import build_zero_observation
from scripts.test_multi_process_latency import summarize_latencies


def test_build_zero_observation_for_franka_xhand():
    observation = build_zero_observation("franka_xhand", state_dim=18, height=16, width=20)
    assert observation["images"]["cam_side"].shape == (16, 20, 3)
    assert observation["images"]["cam_side"].dtype == np.uint8
    assert np.count_nonzero(observation["state"]) == 0


def test_summarize_latencies_returns_milliseconds_percentiles():
    result = summarize_latencies([1.0, 2.0, 3.0])
    assert result == {"mean_ms": 2.0, "p50_ms": 2.0, "p95_ms": 2.9}
