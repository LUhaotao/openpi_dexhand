#!/usr/bin/env python3
"""Prepare an active-DOF LeRobot dataset from Bench2Dex replay HDF5 files.

The source HDF5 files remain full-width. This script applies the official
valid-frame and homing cutoffs, maps joints by name into the runtime order,
selects active DOFs, and writes a new active-width LeRobot dataset.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import yaml
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

_OPENPI_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_SRC))
from openpi.shared import normalize  # noqa: E402


DEFAULT_SOURCE = Path(
    "/public/node01/users/lvrui/datasets/hdf5/bench2dex/teleopdata/teleopdata/dataset/"
    "34_fridge_wine_interhand_pour/replay-generalization"
)
DEFAULT_OUTPUT = Path(
    "/public/node01/users/lvrui/datasets/lerobot/bench2dex/"
    "34_fridge_wine_interhand_pour_active38"
)
DEFAULT_MAP = Path(__file__).resolve().parent.parent / "src/openpi/bench2dex_active_dof_maps.yml"

CAMERAS: dict[str, str] = {
    "stereo_left": "cam_stereo_left",
    "stereo_right": "cam_stereo_right",
    "wrist_left": "cam_wrist_left",
    "wrist_right": "cam_wrist_right",
}
TACTILE_SITES = (
    "left_index_pad",
    "left_little_pad",
    "left_middle_pad",
    "left_ring_pad",
    "left_thumb_pad",
    "right_index_pad",
    "right_little_pad",
    "right_middle_pad",
    "right_ring_pad",
    "right_thumb_pad",
)
EPISODE_RE = re.compile(r"^episode_(\d{6})(?:_1)?\.hdf5$")


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_scalar(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_text(ep: h5py.File, path: str, fallback: str = "") -> str:
    if path not in ep:
        return fallback
    value = _decode_scalar(ep[path][()])
    return str(value).strip() if value is not None else fallback


def _episode_paths(source_dir: Path) -> list[Path]:
    paths: list[tuple[int, int, Path]] = []
    for path in source_dir.glob("episode_*.hdf5"):
        match = EPISODE_RE.match(path.name)
        if match is None:
            continue
        variant = 1 if path.stem.endswith("_1") else 0
        paths.append((int(match.group(1)), variant, path))
    paths.sort(key=lambda item: (item[0], item[1]))
    if not paths:
        raise FileNotFoundError(f"No episode_XXXXXX[[_1]].hdf5 files found in {source_dir}")
    return [path for _, _, path in paths]


def _decode_rgb(value: Any, source: str) -> np.ndarray:
    encoded = np.asarray(value, dtype=np.uint8).reshape(-1)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode JPEG at {source}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _tacmap_rgb(value: Any, source: str) -> np.ndarray:
    image = np.asarray(value, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D TacMap at {source}, got {image.shape}")
    return np.repeat(image[..., None], 3, axis=2)


def _robot_key(ep: h5py.File) -> str:
    return _read_text(ep, "meta/robot_key", _read_text(ep, "robot/tactile/meta/robot_key"))


def _runtime_active_indices(ep: h5py.File, robot_map: dict[str, Any]) -> tuple[list[int], list[str]]:
    runtime_names = [_decode_scalar(name) for name in ep["robot/joint_names"][:]]
    runtime_names = [str(name) for name in runtime_names]
    full_names = [str(name) for name in robot_map["full_joint_names"]]
    active_names = [str(name) for name in robot_map["active_joint_names"]]
    if len(runtime_names) != int(robot_map["full_dof"]) or len(runtime_names) != len(full_names):
        raise ValueError(f"HDF5 joint count {len(runtime_names)} does not match map full_dof {robot_map['full_dof']}")
    if set(runtime_names) != set(full_names):
        raise ValueError("HDF5 robot/joint_names do not match active_dof_maps full_joint_names")
    active_indices = sorted(runtime_names.index(name) for name in active_names)
    return active_indices, [runtime_names[index] for index in active_indices]


def _valid_indices(ep: h5py.File) -> np.ndarray:
    total = int(ep["robot/qpos"].shape[0])
    frame_valid = np.asarray(ep["frame_valid"][:], dtype=bool)
    action_valid = (
        np.asarray(ep["action/action_valid"][:], dtype=bool)
        if "action/action_valid" in ep
        else np.ones(total, dtype=bool)
    )
    qpos = np.asarray(ep["robot/qpos"][:], dtype=np.float32)
    qvel = np.asarray(ep["robot/qvel"][:], dtype=np.float32)
    qeffort = np.asarray(ep["robot/qeffort"][:], dtype=np.float32)
    action = np.asarray(ep["action/commanded"][:], dtype=np.float32)
    finite = (
        np.isfinite(qpos).all(axis=1)
        & np.isfinite(qvel).all(axis=1)
        & np.isfinite(qeffort).all(axis=1)
        & np.isfinite(action).all(axis=1)
    )
    return np.flatnonzero(frame_valid & action_valid & finite)


def _homing_cutoff(ep: h5py.File) -> int | None:
    if "meta/homing_start_sim_step" not in ep or "time/sim_step" not in ep:
        return None
    marker = int(ep["meta/homing_start_sim_step"][()])
    sim_steps = np.asarray(ep["time/sim_step"][:])
    cutoff = int(np.searchsorted(sim_steps, marker))
    return cutoff if 0 < cutoff < len(sim_steps) else None


def _feature_defs(ep: h5py.File, active_names: list[str]) -> dict[str, dict[str, Any]]:
    first_rgb = _decode_rgb(ep[f"cameras/{next(iter(CAMERAS.values()))}/rgb"][0], "first camera")
    first_tactile = _tacmap_rgb(ep[f"robot/tactile/tacmap/{TACTILE_SITES[0]}"][0], "first TacMap")
    state_dim = len(active_names)
    features: dict[str, dict[str, Any]] = {
        key: {"dtype": "float32", "shape": (state_dim,), "names": [active_names]}
        for key in ("observation.state", "observation.velocity", "observation.torque", "action")
    }
    for output_name in CAMERAS:
        features[f"observation.images.{output_name}"] = {
            "dtype": "video",
            "shape": (3, *first_rgb.shape[:2]),
            "names": ["channels", "height", "width"],
        }
    for site in TACTILE_SITES:
        features[f"observation.images.tactile_{site}"] = {
            "dtype": "video",
            "shape": (3, *first_tactile.shape[:2]),
            "names": ["channels", "height", "width"],
        }
    return features


def prepare(source_dir: Path, output_dir: Path, map_path: Path, *, overwrite: bool, fps: int | None) -> None:
    maps = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    episodes = _episode_paths(source_dir)
    with h5py.File(episodes[0], "r") as first:
        robot_key = _robot_key(first)
        if robot_key not in maps["robots"]:
            raise KeyError(f"Robot {robot_key!r} is not in {map_path}")
        robot_map = maps["robots"][robot_key]
        active_indices, active_names = _runtime_active_indices(first, robot_map)
        full_dof = int(robot_map["full_dof"])
        active_dof = int(robot_map["active_dof"])
        if len(active_indices) != active_dof:
            raise ValueError(f"Map active_dof={active_dof} but runtime mapping produced {len(active_indices)} indices")
        features = _feature_defs(first, active_names)
        output_fps = fps or int(round(float(_decode_scalar(first["meta/effective_fps"][()]))))
        robot_type = robot_key

    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise FileExistsError(f"{output_dir} is not empty; pass --overwrite")
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=output_dir.name,
        root=output_dir,
        robot_type=robot_type,
        fps=output_fps,
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    total_frames = 0
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    for episode_number, episode_path in enumerate(episodes):
        with h5py.File(episode_path, "r") as ep:
            episode_robot_key = _robot_key(ep)
            if episode_robot_key != robot_key:
                raise ValueError(f"Mixed robot keys in one dataset: {robot_key!r} and {episode_robot_key!r}")
            episode_active_indices, _ = _runtime_active_indices(ep, robot_map)
            if episode_active_indices != active_indices:
                raise ValueError(f"Runtime joint order changed in {episode_path}")
            for key in ("robot/qpos", "robot/qvel", "robot/qeffort", "action/commanded"):
                if int(ep[key].shape[1]) != full_dof:
                    raise ValueError(f"{episode_path}: {key} width {ep[key].shape[1]} != map full_dof {full_dof}")
            keep = _valid_indices(ep)
            cutoff = _homing_cutoff(ep)
            if cutoff is not None:
                keep = keep[keep < cutoff]
            if keep.size == 0:
                raise ValueError(f"No usable frames remain after filtering: {episode_path}")
            qpos = np.asarray(ep["robot/qpos"][:], dtype=np.float32)
            qvel = np.asarray(ep["robot/qvel"][:], dtype=np.float32)
            qeffort = np.asarray(ep["robot/qeffort"][:], dtype=np.float32)
            action = np.asarray(ep["action/commanded"][:], dtype=np.float32)
            instruction = _read_text(ep, "meta/instruction", episode_path.stem)
            active_state = qpos[keep][:, active_indices]
            active_actions = action[keep][:, active_indices]
            for start in range(0, len(keep), 512):
                stats["state"].update(active_state[start : start + 512])
                stats["actions"].update(active_actions[start : start + 512])
            for source_index in keep:
                i = int(source_index)
                frame: dict[str, Any] = {
                    "observation.state": qpos[i, active_indices],
                    "observation.velocity": qvel[i, active_indices],
                    "observation.torque": qeffort[i, active_indices],
                    "action": action[i, active_indices],
                    "task": instruction,
                }
                for output_name, hdf5_name in CAMERAS.items():
                    source = f"cameras/{hdf5_name}/rgb"
                    frame[f"observation.images.{output_name}"] = _decode_rgb(ep[source][i], source)
                for site in TACTILE_SITES:
                    source = f"robot/tactile/tacmap/{site}"
                    frame[f"observation.images.tactile_{site}"] = _tacmap_rgb(ep[source][i], source)
                dataset.add_frame(frame)
            dataset.save_episode()
            total_frames += int(keep.size)
        print(f"prepared {episode_number:03d}: {episode_path.name} ({keep.size} frames)", flush=True)

    normalize.save(output_dir, {key: value.get_statistics() for key, value in stats.items()})

    manifest = {
        "format_version": 1,
        "source_dir": str(source_dir),
        "source_map": str(map_path),
        "robot_key": robot_key,
        "full_dof": full_dof,
        "active_dof": active_dof,
        "full_joint_names": [str(name) for name in robot_map["full_joint_names"]],
        "active_joint_names": active_names,
        "inactive_joint_names": [str(name) for name in robot_map.get("inactive_joint_names", [])],
        "inactive_indices_urdf": [int(index) for index in robot_map.get("inactive_indices", [])],
        "runtime_active_indices": active_indices,
        "runtime_active_joint_names": active_names,
        "homing_cutoff": True,
        "episodes": len(episodes),
        "frames": total_frames,
        "norm_stats": "norm_stats.json",
    }
    (output_dir / "active_dof_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(episodes)} episodes / {total_frames} frames to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP, dest="map_path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fps", type=int, default=None)
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir, args.map_path, overwrite=args.overwrite, fps=args.fps)


if __name__ == "__main__":
    main()
