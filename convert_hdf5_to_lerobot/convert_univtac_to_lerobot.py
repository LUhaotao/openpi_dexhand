#!/usr/bin/env python3
"""Convert UniVTAC episode HDF5 files to a LeRobot v2 dataset.

UniVTAC does not store an action field. As in its native data loader, this
uses joint[t] as the observation and joint[t + 1] as the action.

Tactile forms selectable with ``--tactile``:
  - ``rgb``: raw GelSight RGB image (JPEG bytes)
  - ``rgb_marker``: RGB image with tracked markers drawn on it (JPEG bytes)
  - ``depth``: float32 depth/deformation map, normalized to uint8 grayscale video
  - ``marker``: float32 marker coordinates
  - ``pose``: float32 tactile sensor pose
  - ``all``: all five forms; repeat ``--tactile`` to choose a subset

With ``--augment``, selected tactile data is added to the standard LeRobot
``data/``, ``videos/`` and ``meta/`` paths without changing state/action data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import av
import cv2
import datasets
import h5py
import numpy as np
import pyarrow.parquet as pq

from lerobot.common.datasets.compute_stats import sample_indices
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_hf_features_from_features
from lerobot.common.datasets.video_utils import get_video_info


CAMERAS = {
    "head": "observation/head/rgb",
    "wrist": "observation/wrist/rgb",
}

TACTILE_FORMS = {
    "rgb": ("tactile_left", "tactile/left_gsmini/rgb", "image"),
    "rgb_marker": ("tactile_left_rgb_marker", "tactile/left_gsmini/rgb_marker", "image"),
    "depth": ("tactile_left_depth", "tactile/left_gsmini/depth", "depth_video"),
    "marker": ("tactile_left_marker", "tactile/left_gsmini/marker", "numeric"),
    "pose": ("tactile_left_pose", "tactile/left_gsmini/pose", "numeric"),
}
TACTILE_SIDES = (("left", "left_gsmini"), ("right", "right_gsmini"))
DEPTH_RANGE = (24.0, 34.0)


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
            feature_name = (
                f"observation.images.{name}"
                if kind in ("image", "depth_video")
                else f"observation.tactile.{name}"
            )
            features.append((feature_name, path, kind))
    return features


def normalized_tactile_forms(tactile_forms: list[str] | None) -> list[str]:
    forms = tactile_forms or ["rgb"]
    return list(TACTILE_FORMS) if "all" in forms else list(dict.fromkeys(forms))


def standard_tactile_entries(tactile_forms: list[str]) -> list[tuple[str, str, str, str]]:
    """Return (LeRobot feature, HDF5 path, storage kind, tactile key)."""
    entries = []
    for form in normalized_tactile_forms(tactile_forms):
        for side, sensor in TACTILE_SIDES:
            source = f"tactile/{sensor}/{form}"
            if form == "rgb":
                feature = f"observation.images.tactile_{side}"
                kind = "video"
            elif form == "rgb_marker":
                feature = f"observation.images.tactile_{side}_rgb_marker"
                kind = "video"
            elif form == "depth":
                feature = f"observation.images.tactile_{side}_depth"
                kind = "depth_video"
            else:
                feature = f"observation.tactile.{side}_{form}"
                kind = "numeric"
            entries.append((feature, source, kind, f"{side}_{form}"))
    return entries


class TactileVideoWriter:
    def __init__(self, path: Path, fps: int, first_frame: np.ndarray) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), "w")
        height, width = first_frame.shape[:2]
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"

    def write(self, image: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def episode_paths(source_dir: Path) -> list[Path]:
    paths = list(source_dir.glob("*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No .hdf5 files in {source_dir}")
    return sorted(paths, key=lambda path: int(path.stem))


def depth_to_video_frame(depth: np.ndarray) -> np.ndarray:
    """Encode the float32 height map as an RGB-compatible uint8 video frame."""
    lo, hi = DEPTH_RANGE
    gray = np.rint(np.clip((np.asarray(depth, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0) * 255).astype(
        np.uint8
    )
    return np.repeat(gray[..., None], 3, axis=-1)


def standard_video_frame(value: bytes | np.ndarray, source: str, kind: str) -> np.ndarray:
    return depth_to_video_frame(value) if kind == "depth_video" else decode_rgb(value, source)


def array_stats(values: np.ndarray) -> dict[str, list]:
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [len(values)],
    }


def image_stats(frames: list[np.ndarray]) -> dict[str, list]:
    chw = np.moveaxis(np.stack(frames, axis=0), -1, 1).astype(np.float32) / 255.0
    return {
        "min": np.min(chw, axis=(0, 2, 3), keepdims=True).reshape(3, 1, 1).tolist(),
        "max": np.max(chw, axis=(0, 2, 3), keepdims=True).reshape(3, 1, 1).tolist(),
        "mean": np.mean(chw, axis=(0, 2, 3), keepdims=True).reshape(3, 1, 1).tolist(),
        "std": np.std(chw, axis=(0, 2, 3), keepdims=True).reshape(3, 1, 1).tolist(),
        "count": [len(frames)],
    }


def standard_video_path(output_dir: Path, info: dict, episode_index: int, feature_name: str) -> Path:
    chunk = episode_index // int(info.get("chunks_size", 1000))
    video_template = info.get("video_path") or "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    relative = video_template.format(
        episode_chunk=chunk,
        video_key=feature_name,
        episode_index=episode_index,
    )
    return output_dir / relative


def rewrite_parquet_with_numeric_features(
    parquet_path: Path,
    numeric_columns: dict[str, np.ndarray],
    info: dict,
) -> None:
    table = pq.read_table(parquet_path)
    data = {name: table[name].to_pylist() for name in table.column_names}
    data.update({name: values for name, values in numeric_columns.items()})
    feature_defs = {
        name: {**info["features"][name], "shape": tuple(info["features"][name]["shape"])}
        for name in data
        if name in info["features"] and info["features"][name]["dtype"] != "video"
    }
    hf_features = get_hf_features_from_features(feature_defs)
    dataset = datasets.Dataset.from_dict(data, features=hf_features)
    temporary = parquet_path.with_name(parquet_path.stem + ".part.parquet")
    dataset.to_parquet(temporary)
    os.replace(temporary, parquet_path)


def augment_tactile(
    source_dir: Path,
    output_dir: Path,
    fps: int,
    *,
    resume: bool,
    max_episodes: int | None,
    no_tactile: bool,
    tactile: list[str] | None,
    mode: str,
) -> None:
    """Add tactile data to the standard LeRobot Parquet/videos layout."""
    if no_tactile:
        raise ValueError("--augment requires at least one --tactile form")
    if mode != "video":
        raise ValueError("--augment writes RGB and depth to standard videos; use --mode video")
    metadata_dir = output_dir / "meta"
    if not (metadata_dir / "info.json").is_file() or not (metadata_dir / "episodes.jsonl").is_file():
        raise ValueError(f"Existing LeRobot metadata not found under {metadata_dir}")

    info = json.loads((metadata_dir / "info.json").read_text())
    if int(info.get("fps", fps)) != fps:
        raise ValueError(f"fps={fps} does not match existing LeRobot dataset fps={info.get('fps')}")
    episode_metadata = [
        json.loads(line) for line in (metadata_dir / "episodes.jsonl").read_text().splitlines() if line.strip()
    ]
    episodes = episode_paths(source_dir)
    episode_count = int(info["total_episodes"])
    if len(episode_metadata) != episode_count or len(episodes) != episode_count:
        raise ValueError("HDF5 and LeRobot episode counts do not match")
    if max_episodes is not None and max_episodes != episode_count:
        raise ValueError("--augment must process all episodes so metadata stays consistent")

    forms = normalized_tactile_forms(tactile)
    entries = standard_tactile_entries(forms)
    if any(kind != "numeric" for _, _, kind, _ in entries):
        info["video_path"] = info.get("video_path") or (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
    with h5py.File(episodes[0], "r") as h5:
        for feature_name, source, kind, _ in entries:
            value = standard_video_frame(h5[source][0], source, kind) if kind != "numeric" else np.asarray(h5[source][0])
            if feature_name not in info["features"]:
                info["features"][feature_name] = {
                    "dtype": "video" if kind != "numeric" else "float32",
                    "shape": list(value.shape) if kind == "numeric" else [3, *value.shape[:2]],
                    "names": None if kind == "numeric" else ["channels", "height", "width"],
                }

    episode_stats = [
        json.loads(line)
        for line in (metadata_dir / "episodes_stats.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(episode_stats) != episode_count:
        raise ValueError("LeRobot episodes_stats.jsonl does not match info.json total_episodes")

    for episode_index, episode_path in enumerate(episodes):
        with h5py.File(episode_path, "r") as h5:
            transition_count = len(h5["embodiment/joint"]) - 1
            if transition_count != episode_metadata[episode_index]["length"]:
                raise ValueError(f"{episode_path}: frame count does not match LeRobot episode")
            numeric_columns = {}
            sampled_stats = {}
            sample_positions = set(sample_indices(transition_count))
            for feature_name, source, kind, _ in entries:
                if len(h5[source]) != transition_count + 1:
                    raise ValueError(f"{episode_path}: {source} length does not match embodiment/joint")
                if kind == "numeric":
                    values = np.asarray(h5[source][:transition_count], dtype=np.float32)
                    numeric_columns[feature_name] = values
                    episode_stats[episode_index]["stats"][feature_name] = array_stats(values)
                    continue

                target = standard_video_path(output_dir, info, episode_index, feature_name)
                target.parent.mkdir(parents=True, exist_ok=True)
                frames = []
                if not (target.exists() and resume):
                    first = standard_video_frame(h5[source][0], source, kind)
                    temporary = target.with_name(target.stem + ".part.mp4")
                    writer = TactileVideoWriter(temporary, fps, first)
                    try:
                        for frame_index in range(transition_count):
                            frame = standard_video_frame(h5[source][frame_index], source, kind)
                            writer.write(frame)
                            if frame_index in sample_positions:
                                frames.append(frame)
                    finally:
                        writer.close()
                    os.replace(temporary, target)
                else:
                    for frame_index in sample_positions:
                        frames.append(standard_video_frame(h5[source][frame_index], source, kind))
                sampled_stats[feature_name] = image_stats(frames)

            if numeric_columns:
                parquet_path = output_dir / info["data_path"].format(
                    episode_chunk=episode_index // int(info.get("chunks_size", 1000)),
                    episode_index=episode_index,
                )
                if not parquet_path.is_file():
                    raise FileNotFoundError(parquet_path)
                rewrite_parquet_with_numeric_features(parquet_path, numeric_columns, info)
            episode_stats[episode_index]["stats"].update(sampled_stats)
        print(f"augmented standard tactile {episode_path.name}")

    for feature_name, _, kind, _ in entries:
        if kind != "numeric":
            video_path = standard_video_path(output_dir, info, 0, feature_name)
            video_info = get_video_info(video_path)
            if kind == "depth_video":
                video_info.update(
                    {
                        "video.is_depth_map": True,
                        "video.depth_min": DEPTH_RANGE[0],
                        "video.depth_max": DEPTH_RANGE[1],
                        "video.depth_encoding": "uint8 normalized from float32 height map",
                    }
                )
            info["features"][feature_name]["info"] = video_info
    info["total_videos"] = len(
        [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    ) * episode_count
    temporary_info = metadata_dir / "info.part.json"
    temporary_info.write_text(json.dumps(info, indent=2) + "\n")
    os.replace(temporary_info, metadata_dir / "info.json")
    temporary_stats = metadata_dir / "episodes_stats.part.jsonl"
    temporary_stats.write_text("".join(json.dumps(item) + "\n" for item in episode_stats))
    os.replace(temporary_stats, metadata_dir / "episodes_stats.jsonl")


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
            if kind in ("image", "video", "depth_video"):
                image = standard_video_frame(h5[path][0], path, kind)
                feature_defs[feature_name] = {
                    "dtype": "video" if kind == "depth_video" else mode,
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
        use_videos=mode == "video" or any(kind == "depth_video" for _, _, kind in features),
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
                        standard_video_frame(value, path, kind)
                        if kind in ("image", "video", "depth_video")
                        else np.asarray(value, dtype=np.float32)
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
        "--augment",
        action="store_true",
        help="Add selected tactile data to standard LeRobot Parquet/videos and update metadata",
    )
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
    options = vars(args)
    augment = options.pop("augment")
    if augment:
        options.pop("task")
        augment_tactile(**options)
    else:
        convert(**options)


if __name__ == "__main__":
    main()
