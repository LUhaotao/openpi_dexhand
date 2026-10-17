import json
from pathlib import Path

import pytest

from openpi.models import pi0_config
from openpi.training import experiment_config
from openpi.training import optimizer
from openpi.training import weight_loaders


def test_load_complete_univtac_experiment():
    config = experiment_config.load_train_config("configs/examples/pi05_univtac_pull_out_key.json")

    assert config.name == "pi05_UniVTAC_pull_out_key"
    assert config.exp_name == "baseline_seed42"
    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.pi05
    assert config.data.repo_id.endswith("/univtac/pull_out_key")
    assert isinstance(config.weight_loader, weight_loaders.CheckpointWeightLoader)
    assert isinstance(config.lr_schedule, optimizer.CosineDecaySchedule)
    assert config.resolved_config["model"]["max_token_len"] == 200
    assert config.resolved_config["data"]["repack_mapping"]["images"] == {
        "cam_side": "observation.images.head",
        "cam_wrist": "observation.images.wrist",
    }


def test_expand_legacy_incremental_model_snapshot(tmp_path):
    incremental = tmp_path / "pi0_config.json"
    incremental.write_text(json.dumps({"pi05": True, "streaming": True, "streaming_attention_mode": "mask"}))

    expanded = experiment_config.load_expanded_config(incremental)

    assert expanded["model"]["streaming"] is True
    assert expanded["model"]["streaming_attention_mode"] == "mask"
    assert expanded["model"]["action_dim"] == 32
    assert expanded["batch_size"] == 32


def test_unknown_experiment_field_is_rejected(tmp_path):
    incremental = tmp_path / "bad.json"
    incremental.write_text(json.dumps({"unknown_setting": True}))

    with pytest.raises(experiment_config.ExperimentConfigError, match="unknown_setting"):
        experiment_config.load_train_config(incremental)


def test_non_json_experiment_is_rejected(tmp_path):
    incremental = tmp_path / "experiment.yaml"
    incremental.write_text("name: test\n")

    with pytest.raises(experiment_config.ExperimentConfigError, match="must be JSON"):
        experiment_config.load_expanded_config(incremental)


def test_null_data_type_explains_flower_mapping(tmp_path):
    value = json.loads(Path("configs/experiments/pi05/real/flower/streaming_state_attention_gate_delay_6.json").read_text())
    value["data"]["type"] = None

    with pytest.raises(experiment_config.ExperimentConfigError, match="Flower uses 'franka_xhand'"):
        experiment_config.train_config_from_dict(value)


def test_tactile_univtac_snapshot_contains_resolved_marker_mapping():
    value = json.loads(Path("configs/examples/pi05_univtac_pull_out_key.json").read_text())
    value["model"]["use_tactile"] = True
    value = experiment_config.expand_config_dict(value)

    config = experiment_config.train_config_from_dict(value)
    mapping = config.resolved_config["data"]["repack_mapping"]
    data_config = config.data.create(config.assets_dirs, config.model)

    assert mapping["left_marker"] == "observation.tactile.left_marker"
    assert mapping["right_marker"] == "observation.tactile.right_marker"
    assert data_config.repack_transforms.inputs[0].structure == mapping


def test_checkpoint_train_config_round_trip(tmp_path):
    original = experiment_config.load_train_config("configs/examples/pi05_univtac_pull_out_key.json")
    snapshot = tmp_path / "pi0_config.json"
    experiment_config.dump_config(original.resolved_config, snapshot)

    restored = experiment_config.load_checkpoint_train_config(snapshot)

    assert restored is not None
    assert restored.model == original.model
    assert restored.data.repo_id == original.data.repo_id
    assert restored.optimizer == original.optimizer


def test_checkpoint_snapshot_before_repack_mapping_is_backfilled(tmp_path):
    original = experiment_config.load_expanded_config("configs/examples/pi05_univtac_pull_out_key.json")
    del original["data"]["repack_mapping"]
    snapshot = tmp_path / "pi0_config.json"
    snapshot.write_text(json.dumps(original))

    restored = experiment_config.load_checkpoint_train_config(snapshot)

    assert restored is not None
    assert restored.resolved_config["data"]["repack_mapping"]["images"]["cam_side"] == ("observation.images.head")


def test_checkpoint_train_config_ignores_legacy_model_only_snapshot(tmp_path):
    snapshot = tmp_path / "pi0_config.json"
    snapshot.write_text(json.dumps({"pi05": True, "streaming": True}))

    assert experiment_config.load_checkpoint_train_config(snapshot) is None
