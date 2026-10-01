import numpy as np
import pytest

from openpi.fake_env.config import FakeEnvConfig
from openpi.fake_env.observation import make_production_wire_observation
from openpi.fake_env.timing import format_latency_report


def test_production_wire_observation_uses_configured_views_and_modalities():
    config = FakeEnvConfig(
        pi05=True,
        action_dim=6,
        action_horizon=4,
        streaming_chunk_size=2,
        use_tactile=True,
        use_torque=True,
        torque_dim=6,
        streaming=True,
        streaming_attention_mode="attention_gate",
        gate_sources=("torque", "tactile"),
        image_height=224,
        image_width=224,
    )

    observation = make_production_wire_observation(config)

    assert set(observation["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert observation["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert observation["image"]["base_0_rgb"].dtype == np.uint8
    assert observation["state"].shape == (6,)
    assert observation["torque"].shape == (6,)
    assert observation["tactile_left_marker"].shape == (2, 63, 2)
    assert observation["tokenized_prompt"].shape == (200,)


def test_production_wire_rejects_torque_without_dimension():
    with pytest.raises(ValueError, match="torque_dim"):
        FakeEnvConfig(pi05=True, use_torque=True, torque_dim=0)


def test_latency_report_always_contains_three_chinese_sections():
    report = format_latency_report(
        {
            "inference": {"vlm_ms": 1.0, "fm_ms": 2.0},
            "server": {"vlm_ms": 3.0, "kv_refresh_ms": 4.0, "fm_ms": 5.0, "total_ms": 12.0},
            "e2e": {"total_ms": 20.0},
        }
    )

    assert "[推理层]" in report
    assert "[Server层]" in report
    assert "[端到端]" in report
