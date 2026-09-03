#!/usr/bin/env python3
"""Convert UniVTAC episode HDF5 files to a LeRobot v2 dataset.

UniVTAC does not store an action field. As in its native data loader, this
uses joint[t] as the observation and joint[t + 1] as the action.

Tactile forms selectable with ``--tactile``:
  - ``rgb``: raw GelSight RGB image (JPEG bytes)
  - ``rgb_marker``: RGB image with tracked markers drawn on it (JPEG bytes)
  - ``depth``: float32 depth/deformation map
  - ``marker``: float32 marker coordinates
  - ``pose``: float32 tactile sensor pose
  - ``all``: all five forms; repeat ``--tactile`` to choose a subset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


CAMERAS = {
    "head": "observation/head/rgb",
    "wrist": "observation/wrist/rgb",
}

TACTILE_FORMS = {
    "rgb": ("tactile_left", "tactile/left_gsmini/rgb", "image"),
    "rgb_marker": ("tactile_left_rgb_marker", "tactile/left_gsmini/rgb_marker", "image"),
    "depth": ("tactile_left_depth", "tactile/left_gsmini/depth", "numeric"),
    "marker": ("tactile_left_marker", "tactile/left_gsmini/marker", "numeric"),
    "pose": ("tactile_left_pose", "tactile/left_gsmini/pose", "numeric"),
}
TACTILE_SIDES = (("left", "left_gsmini"), ("right", "right_gsmini"))


def decode_rgb(encoded: bytes, name: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode JPEG from {name}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def selected_features(tactile_forms: list[str]) -> list[tuple[str, str, str]]:
    features = [(f"observation.images.{name}", path, "image") for name, path in CAMERAS.items()]
    tactile_forms = list(TACTILE_FORMS) if "all" in tactile_forms else list(dict.fromkeys(tactile_forms))
    for form in tactile_forms:
        _, _, kind = TACTILE_FORMS[form]
        for side, sensor in TACTILE_SIDES:
            name, _, _ = TACTILE_FORMS[form]
            name = name.replace("left", side)
            path = f"tactile/{sensor}/{form}"
            feature_name = f"observation.images.{name}" if kind == "image" else f"observation.tactile.{name}"
            features.append((feature_name, path, kind))
    return features


def episode_paths(source_dir: Path) -> list[Path]:
    paths = list(source_dir.glob("*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No .hdf5 files in {source_dir}")
    return sorted(paths, key=lambda path: int(path.stem))


def create_dataset(
    output_dir: Path, fps: int, first_episode: Path, features: list[tuple[str, str, str]], mode: str
) -> LeRobotDataset:
    with h5py.File(first_episode, "r") as h5:
        state_dim = h5["embodiment/joint"].shape[1]
        feature_defs = {
            "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
            "action": {"dtype": "float32", "shape": (state_dim,), "names": None},
        }
        for feature_name, path, kind in features:
            if kind == "image":
                image = decode_rgb(h5[path][0], path)
                feature_defs[feature_name] = {
                    "dtype": mode,
                    "shape": (3, *image.shape[:2]),
                    "names": ["channels", "height", "width"],
                }
            else:
                value = np.asarray(h5[path][0])
                feature_defs[feature_name] = {
                    "dtype": "float32",
                    "shape": value.shape,
                    "names": None,
                }

    return LeRobotDataset.create(
        repo_id=output_dir.name,
        root=output_dir,
        robot_type="univtac_franka_gripper",
        fps=fps,
        features=feature_defs,
        use_videos=mode == "video",
    )


def convert(
    source_dir: Path,
    output_dir: Path,
    task: str,
    fps: int,
    *,
    resume: bool,
    max_episodes: int | None,
    no_tactile: bool,
    tactile: list[str] | None,
    mode: str,
) -> None:
    episodes = episode_paths(source_dir)
    if no_tactile and tactile:
        raise ValueError("--no-tactile cannot be combined with --tactile")
    tactile_forms = [] if no_tactile else (tactile or ["rgb"])
    features = selected_features(tactile_forms)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    has_output = output_dir.exists() and any(output_dir.iterdir())
    if has_output:
        metadata_files = [
            output_dir / "meta" / "info.json",
            output_dir / "meta" / "tasks.jsonl",
            output_dir / "meta" / "episodes.jsonl",
            output_dir / "meta" / "episodes_stats.jsonl",
        ]
        if not all(path.is_file() for path in metadata_files):
            raise ValueError(
                f"{output_dir} is not a complete local LeRobot dataset; "
                "use a new --output-dir or remove this incomplete directory"
            )
        if not resume:
            raise FileExistsError(f"{output_dir} exists; pass --resume to continue it")
        dataset = LeRobotDataset(output_dir.name, root=output_dir)
        expected_features = {name for name, _, _ in features}
        existing_features = {
            name
            for name in dataset.meta.features
            if name.startswith(("observation.images.", "observation.tactile."))
        }
        if existing_features != expected_features or dataset.fps != fps:
            raise ValueError("Existing dataset modalities or fps differ; use a separate output directory")
        expected_images = {name for name, _, kind in features if kind == "image"}
        if any(dataset.meta.features[name]["dtype"] != mode for name in expected_images):
            raise ValueError("Existing dataset storage mode differs; use a separate output directory")
        start = dataset.num_episodes
    else:
        if output_dir.exists():
            output_dir.rmdir()
        dataset = create_dataset(output_dir, fps, episodes[0], features, mode)
        start = 0

    for episode_path in episodes[start:]:
        with h5py.File(episode_path, "r") as h5:
            joints = h5["embodiment/joint"][:].astype(np.float32, copy=False)
            if len(joints) < 2:
                raise ValueError(f"{episode_path} has fewer than two frames")
            for _, path, _ in features:
                if len(h5[path]) != len(joints):
                    raise ValueError(f"{episode_path}: {path} length does not match embodiment/joint")

            # UniVTAC's loader defines the next joint state as the action.
            for frame_index in range(len(joints) - 1):
                frame = {
                    "observation.state": joints[frame_index],
                    "action": joints[frame_index + 1],
                    "task": task,
                }
                for feature_name, path, kind in features:
                    value = h5[path][frame_index]
                    frame[feature_name] = (
                        decode_rgb(value, path) if kind == "image" else np.asarray(value, dtype=np.float32)
                    )
                dataset.add_frame(frame)
            dataset.save_episode()
        print(f"converted {episode_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="Empty")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--mode", choices=("video", "image"), default="video")
    parser.add_argument(
        "--tactile",
        choices=(*TACTILE_FORMS, "all"),
        action="append",
        help="Tactile form to keep; repeat for multiple forms. Default: rgb. Use --tactile all for all forms.",
    )
    parser.add_argument("--no-tactile", action="store_true", help="Keep only head and wrist cameras")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    main()
