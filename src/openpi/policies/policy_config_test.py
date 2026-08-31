from pathlib import Path

from openpi.policies.policy_config import _find_checkpoint_norm_stats_dir


def test_find_checkpoint_norm_stats_dir_falls_back_to_unique_asset(tmp_path: Path):
    expected = tmp_path / "assets" / "pi05_franka_xhand_flower_zhb_right_600"
    expected.mkdir(parents=True)
    (expected / "norm_stats.json").write_text("{}")

    result = _find_checkpoint_norm_stats_dir(
        tmp_path,
        "/data/datasets/flower_4_28",
    )

    assert result == expected
