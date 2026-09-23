from pathlib import Path

from openpi.training.checkpoints import load_norm_stats


def test_load_norm_stats_reads_root_asset(tmp_path: Path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "norm_stats.json").write_text('{"norm_stats": {}}')

    assert load_norm_stats(assets_dir) == {}
