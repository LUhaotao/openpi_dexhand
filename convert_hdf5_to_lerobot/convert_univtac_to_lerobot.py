#!/usr/bin/env python3
"""Convert UniVTAC episode HDF5 files to a LeRobot v2 dataset.

UniVTAC does not store an action field. As in its native data loader, this
uses joint[t] as the observation and joint[t + 1] as the action.
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
    "tactile_left": "tactile/left_gsmini/rgb",
    "tactile_right": "tactile/right_gsmini/rgb",
}


def decode_rgb(encoded: bytes, name: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode JPEG from {name}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def episode_paths(source_dir: Path) -> list[Path]:
    paths = list(source_dir.glob("*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No .hdf5 files in {source_dir}")
    return sorted(paths, key=lambda path: int(path.stem))


def create_dataset(
    output_dir: Path, fps: int, first_episode: Path, cameras: dict[str, str], mode: str
) -> LeRobotDataset:
    with h5py.File(first_episode, "r") as h5:
        state_dim = h5["embodiment/joint"].shape[1]
        features = {
            "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
            "action": {"dtype": "float32", "shape": (state_dim,), "names": None},
        }
        for name, path in cameras.items():
            image = decode_rgb(h5[path][0], path)
            features[f"observation.images.{name}"] = {
                "dtype": mode,
                "shape": (3, *image.shape[:2]),
                "names": ["channels", "height", "width"],
            }

    return LeRobotDataset.create(
        repo_id=output_dir.name,
        root=output_dir,
        robot_type="univtac_franka_gripper",
        fps=fps,
        features=features,
        use_videos=mode == "video",
    )


def convert(
    source_dir: Path,
    output_dir: Path,
    task: str,
    fps: int,
    resume: bool,
    max_episodes: int | None,
    no_tactile: bool,
    mode: str,
) -> None:
    episodes = episode_paths(source_dir)
    cameras = CAMERAS if not no_tactile else {"head": CAMERAS["head"], "wrist": CAMERAS["wrist"]}
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"{output_dir} exists; pass --resume to continue it")
        dataset = LeRobotDataset(output_dir.name, root=output_dir)
        expected_images = {f"observation.images.{name}" for name in cameras}
        existing_images = {name for name in dataset.meta.features if name.startswith("observation.images.")}
        if existing_images != expected_images or dataset.fps != fps:
            raise ValueError("Existing dataset cameras or fps differ; use a separate output directory")
        if any(dataset.meta.features[name]["dtype"] != mode for name in expected_images):
            raise ValueError("Existing dataset storage mode differs; use a separate output directory")
        start = dataset.num_episodes
    else:
        dataset = create_dataset(output_dir, fps, episodes[0], cameras, mode)
        start = 0

    for episode_path in episodes[start:]:
        with h5py.File(episode_path, "r") as h5:
            joints = h5["embodiment/joint"][:].astype(np.float32, copy=False)
            if len(joints) < 2:
                raise ValueError(f"{episode_path} has fewer than two frames")
            for path in cameras.values():
                if len(h5[path]) != len(joints):
                    raise ValueError(f"{episode_path}: {path} length does not match embodiment/joint")

            # UniVTAC's loader defines the next joint state as the action.
            for frame_index in range(len(joints) - 1):
                frame = {
                    "observation.state": joints[frame_index],
                    "action": joints[frame_index + 1],
                    "task": task,
                }
                for name, path in cameras.items():
                    frame[f"observation.images.{name}"] = decode_rgb(h5[path][frame_index], path)
                dataset.add_frame(frame)
            dataset.save_episode()
        print(f"converted {episode_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="grasp and classify an object by tactile feedback")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--mode", choices=("video", "image"), default="video")
    parser.add_argument("--no-tactile", action="store_true", help="Keep only head and wrist cameras")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    main()
