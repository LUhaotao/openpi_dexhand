import json

import pytest

from openpi.models import pi0_config
from openpi.training import experiment_config
from openpi.training import optimizer
from openpi.training import weight_loaders


def test_load_complete_univtac_experiment():
    config = experiment_config.load_train_config("configs/experiments/pi05/univtac/pull_out_key_baseline.yaml")

    assert config.name == "pi05_UniVTAC_pull_out_key"
    assert config.exp_name == "pi05_UniVTAC_pull_out_key_yaml"
    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.pi05
    assert config.data.repo_id.endswith("/univtac/pull_out_key")
    assert isinstance(config.weight_loader, weight_loaders.CheckpointWeightLoader)
    assert isinstance(config.lr_schedule, optimizer.CosineDecaySchedule)
    assert config.resolved_config["model"]["max_token_len"] == 200


def test_expand_legacy_incremental_model_snapshot(tmp_path):
    incremental = tmp_path / "pi0_config.json"
    incremental.write_text(json.dumps({"pi05": True, "streaming": True, "streaming_attention_mode": "mask"}))

    expanded = experiment_config.load_expanded_config(incremental)

    assert expanded["model"]["streaming"] is True
    assert expanded["model"]["streaming_attention_mode"] == "mask"
    assert expanded["model"]["action_dim"] == 32
    assert expanded["batch_size"] == 32


def test_unknown_experiment_field_is_rejected(tmp_path):
    incremental = tmp_path / "bad.yaml"
    incremental.write_text("unknown_setting: true\n")

    with pytest.raises(experiment_config.ExperimentConfigError, match="unknown_setting"):
        experiment_config.load_train_config(incremental)


def test_checkpoint_train_config_round_trip(tmp_path):
    original = experiment_config.load_train_config("configs/experiments/pi05/univtac/pull_out_key_baseline.yaml")
    snapshot = tmp_path / "pi0_config.json"
    experiment_config.dump_config(original.resolved_config, snapshot)

    restored = experiment_config.load_checkpoint_train_config(snapshot)

    assert restored is not None
    assert restored.model == original.model
    assert restored.data.repo_id == original.data.repo_id
    assert restored.optimizer == original.optimizer


def test_checkpoint_train_config_ignores_legacy_model_only_snapshot(tmp_path):
    snapshot = tmp_path / "pi0_config.json"
    snapshot.write_text(json.dumps({"pi05": True, "streaming": True}))

    assert experiment_config.load_checkpoint_train_config(snapshot) is None
