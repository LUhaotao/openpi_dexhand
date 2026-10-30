"""File-backed experiment configuration for JAX Pi0.5 training."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import dataclasses
import json
import pathlib
from typing import Any

from openpi.models import pi0_config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

SCHEMA_VERSION = 1
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.json"


class ExperimentConfigError(ValueError):
    """Raised when an experiment file cannot be converted to a TrainConfig."""


def _load_mapping(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".json":
        raise ExperimentConfigError(f"Config must be JSON: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"Config root must be an object: {path}")
    return value


def _deep_merge(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in defaults.items():
        result[key] = _deep_merge(value, {}) if isinstance(value, Mapping) else value
    for key, value in overrides.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            if "type" in value and value["type"] != result[key].get("type"):
                result[key] = _deep_merge({}, value)
            else:
                result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ExperimentConfigError(f"Unknown {context} field(s): {', '.join(unknown)}")


def _resolve_derived_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize fields that are implied by other JSON settings."""
    result = copy.deepcopy(dict(value))
    model = result.get("model", {})
    data = result.get("data", {})
    if data.get("type") == "univtac" and model.get("use_tactile", False):
        mapping = data.setdefault("repack_mapping", {})
        mapping.setdefault("left_marker", "observation.tactile.left_marker")
        mapping.setdefault("right_marker", "observation.tactile.right_marker")
    return result


def expand_config_dict(incremental: Mapping[str, Any], defaults: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Expand an experiment fragment, including legacy flat Pi0Config snapshots."""
    if defaults is None:
        defaults = _load_mapping(DEFAULT_CONFIG_PATH)

    model_fields = {field.name for field in dataclasses.fields(pi0_config.Pi0Config)}
    root_fields = set(defaults)
    # Old checkpoints stored a flat, incremental Pi0Config object. Treat it as
    # the model section so it can be upgraded with the same command as an
    # incremental experiment config.
    if incremental and set(incremental) <= model_fields and not (set(incremental) & root_fields):
        incremental = {"model": dict(incremental)}
    return _resolve_derived_fields(_deep_merge(defaults, incremental))


def load_expanded_config(
    path: str | pathlib.Path, *, defaults_path: str | pathlib.Path = DEFAULT_CONFIG_PATH
) -> dict[str, Any]:
    path = pathlib.Path(path)
    defaults = _load_mapping(pathlib.Path(defaults_path))
    return expand_config_dict(_load_mapping(path), defaults)


def dump_config(config: Mapping[str, Any], path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".json":
        raise ExperimentConfigError(f"Output must be JSON: {path}")
    path.write_text(json.dumps(config, indent=2, sort_keys=False) + "\n")
    return path


def _model_from_dict(value: Mapping[str, Any]):
    from openpi.training import config as training_config

    value = dict(value)
    model_type = value.pop("type")
    fields = {field.name for field in dataclasses.fields(pi0_config.Pi0Config)}
    _reject_unknown(value, fields, "model")
    cls = {
        "pi0": pi0_config.Pi0Config,
        "pi05": pi0_config.Pi0Config,
        "bench2dex_pi05": training_config.Bench2DexPi05Config,
    }.get(model_type)
    if cls is None:
        raise ExperimentConfigError(f"Unsupported JAX model type: {model_type!r}")
    model = cls(**value)
    if not model.pi05:
        raise ExperimentConfigError("Experiment JSON currently supports JAX Pi0.5 only (model.pi05 must be true).")
    return model


def _assets_from_dict(value: Mapping[str, Any]):
    from openpi.training import config as training_config

    _reject_unknown(value, {"assets_dir", "asset_id"}, "data.assets")
    return training_config.AssetsConfig(**value)


def _data_from_dict(value: Mapping[str, Any]):
    from openpi.training import config as training_config

    value = dict(value)
    data_type = value.pop("type")
    if data_type is None:
        raise ExperimentConfigError(
            "data.type is null. Set it to a registered JAX data adapter, for example "
            "'univtac', 'franka_xhand', 'bench2dex', or 'fake'. "
            "Flower uses 'franka_xhand'."
        )
    assets = _assets_from_dict(value.pop("assets"))
    prompt_from_task = bool(value.pop("prompt_from_task"))
    base_config = training_config.DataConfig(prompt_from_task=prompt_from_task)

    common = {"repo_id": value.pop("repo_id"), "assets": assets, "base_config": base_config}
    if data_type == "univtac":
        _reject_unknown(value, {"action_dim", "action_sequence_keys", "repack_mapping"}, "UniVTAC data")
        action_dim = value.pop("action_dim")
        if action_dim != 9:
            raise ExperimentConfigError("UniVTAC action_dim is fixed at 9 by the current JAX data adapter.")
        return training_config.LeRobotUniVTACDataConfig(
            **common,
            repack_mapping=value.pop("repack_mapping"),
            action_sequence_keys=tuple(value.pop("action_sequence_keys")),
        )
    if data_type == "franka_xhand":
        _reject_unknown(
            value,
            {
                "action_dim",
                "action_sequence_keys",
                "default_prompt",
                "repack_mapping",
                "use_delta_ee_actions",
                "use_delta_hand_actions",
            },
            "Franka XHand data",
        )
        return training_config.LeRobotFrankaXHandDataConfig(
            **common,
            repack_mapping=value.pop("repack_mapping"),
            action_sequence_keys=tuple(value.pop("action_sequence_keys")),
            **value,
        )
    if data_type == "bench2dex":
        _reject_unknown(
            value,
            {"action_dim", "action_sequence_keys", "repack_mapping", "state_dim", "use_active_dof"},
            "Bench2Dex data",
        )
        return training_config.Bench2DexDataConfig(
            **common,
            repack_mapping=value.pop("repack_mapping"),
            action_sequence_keys=tuple(value.pop("action_sequence_keys")),
            **value,
        )
    if data_type == "fake":
        _reject_unknown(value, {"action_dim", "action_sequence_keys"}, "fake data")
        return training_config.FakeDataConfig(repo_id=common["repo_id"], assets=assets, base_config=base_config)
    raise ExperimentConfigError(f"Unsupported JAX data type: {data_type!r}")


def _weight_loader_from_dict(value: Mapping[str, Any]):
    value = dict(value)
    loader_type = value.pop("type")
    classes = {
        "none": weight_loaders.NoOpWeightLoader,
        "checkpoint": weight_loaders.CheckpointWeightLoader,
        "lenient_checkpoint": weight_loaders.LenientCheckpointWeightLoader,
        "paligemma": weight_loaders.PaliGemmaWeightLoader,
    }
    cls = classes.get(loader_type)
    if cls is None:
        raise ExperimentConfigError(f"Unsupported JAX weight loader: {loader_type!r}")
    fields = {field.name for field in dataclasses.fields(cls)}
    _reject_unknown(value, fields, "weight_loader")
    return cls(**value)


def _lr_schedule_from_dict(value: Mapping[str, Any]):
    value = dict(value)
    schedule_type = value.pop("type")
    cls = {"cosine_decay": _optimizer.CosineDecaySchedule, "rsqrt_decay": _optimizer.RsqrtDecaySchedule}.get(
        schedule_type
    )
    if cls is None:
        raise ExperimentConfigError(f"Unsupported JAX lr schedule: {schedule_type!r}")
    _reject_unknown(value, {field.name for field in dataclasses.fields(cls)}, "lr_schedule")
    return cls(**value)


def _optimizer_from_dict(value: Mapping[str, Any]):
    value = dict(value)
    optimizer_type = value.pop("type")
    cls = {"adamw": _optimizer.AdamW, "sgd": _optimizer.SGD}.get(optimizer_type)
    if cls is None:
        raise ExperimentConfigError(f"Unsupported JAX optimizer: {optimizer_type!r}")
    _reject_unknown(value, {field.name for field in dataclasses.fields(cls)}, "optimizer")
    return cls(**value)


def train_config_from_dict(value: Mapping[str, Any]):
    """Build the existing runtime TrainConfig from a fully expanded mapping."""
    from openpi.training import config as training_config

    resolved_config = copy.deepcopy(dict(value))
    value = dict(value)
    _reject_unknown(
        value,
        {
            "schema_version",
            "name",
            "project_name",
            "exp_name",
            "model",
            "data",
            "weight_loader",
            "lr_schedule",
            "optimizer",
            "ema_decay",
            "freeze_filter",
            "freeze_vlm",
            "assets_base_dir",
            "checkpoint_base_dir",
            "seed",
            "batch_size",
            "num_workers",
            "num_train_steps",
            "log_interval",
            "save_interval",
            "save_steps",
            "keep_period",
            "overwrite",
            "resume",
            "wandb_enabled",
            "save_gate_logs",
            "gate_log_path",
            "policy_metadata",
            "fsdp_devices",
        },
        "experiment",
    )
    if value.pop("schema_version") != SCHEMA_VERSION:
        raise ExperimentConfigError(f"Only experiment schema_version={SCHEMA_VERSION} is supported.")

    model_value = value.pop("model")
    model = _model_from_dict(model_value)
    resolved_config["model"] = {"type": model_value["type"], **dataclasses.asdict(model)}
    freeze_filter_name = value.pop("freeze_filter")
    if freeze_filter_name == "model":
        freeze_filter = model.get_freeze_filter()
    elif freeze_filter_name == "none":
        import flax.nnx as nnx

        freeze_filter = nnx.Nothing
    else:
        raise ExperimentConfigError("freeze_filter must be 'none' or 'model'.")

    name = value.pop("name")
    exp_name = value.pop("exp_name")
    if not name or not exp_name:
        raise ExperimentConfigError("name and exp_name must be non-empty strings.")

    data_value = value.pop("data")
    weight_loader_value = value.pop("weight_loader")
    lr_schedule_value = value.pop("lr_schedule")
    optimizer_value = value.pop("optimizer")
    data = _data_from_dict(data_value)
    weight_loader = _weight_loader_from_dict(weight_loader_value)
    lr_schedule = _lr_schedule_from_dict(lr_schedule_value)
    optimizer = _optimizer_from_dict(optimizer_value)
    resolved_config["weight_loader"] = {
        "type": weight_loader_value["type"],
        **dataclasses.asdict(weight_loader),
    }
    resolved_config["lr_schedule"] = {
        "type": lr_schedule_value["type"],
        **dataclasses.asdict(lr_schedule),
    }
    resolved_config["optimizer"] = {
        "type": optimizer_value["type"],
        **dataclasses.asdict(optimizer),
    }

    return training_config.TrainConfig(
        name=name,
        exp_name=exp_name,
        model=model,
        data=data,
        weight_loader=weight_loader,
        lr_schedule=lr_schedule,
        optimizer=optimizer,
        freeze_filter=freeze_filter,
        save_steps=tuple(value.pop("save_steps")),
        resolved_config=resolved_config,
        **value,
    )


def load_train_config(path: str | pathlib.Path):
    """Load a JSON experiment, fill defaults, and return TrainConfig."""
    return train_config_from_dict(load_expanded_config(path))


def load_checkpoint_train_config(path: str | pathlib.Path):
    """Load a complete checkpoint experiment snapshot, or return None for legacy model-only files."""
    value = _load_mapping(pathlib.Path(path))
    required = {"schema_version", "name", "exp_name", "model", "data"}
    if not required <= set(value):
        return None
    return train_config_from_dict(expand_config_dict(value))
