from pathlib import Path

from openpi.training import config
from openpi.training.checkpoints import load_norm_stats
from openpi.training.checkpoints import should_save_checkpoint


def test_load_norm_stats_reads_root_asset(tmp_path: Path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "norm_stats.json").write_text('{"norm_stats": {}}')

    assert load_norm_stats(assets_dir) == {}


def test_should_save_checkpoint_supports_exact_steps():
    kwargs = {
        "num_train_steps": 30_000,
        "save_interval": 5_000,
        "save_steps": (4_000, 15_000),
        "start_step": 0,
    }

    assert should_save_checkpoint(4_000, **kwargs)
    assert should_save_checkpoint(15_000, **kwargs)
    assert should_save_checkpoint(30_000, **kwargs)
    assert not should_save_checkpoint(5_000, **kwargs)


def test_should_save_checkpoint_preserves_legacy_interval_mode():
    kwargs = {
        "num_train_steps": 30_000,
        "save_interval": 5_000,
        "save_steps": (),
        "start_step": 0,
    }

    assert should_save_checkpoint(5_000, **kwargs)
    assert should_save_checkpoint(29_999, **kwargs)
    assert not should_save_checkpoint(4_000, **kwargs)


def test_cli_parses_comma_separated_save_steps(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "debug", "--save_steps=4000,15000"])

    assert config.cli().save_steps == (4_000, 15_000)
