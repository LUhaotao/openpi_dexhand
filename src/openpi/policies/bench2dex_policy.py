"""LeRobot transforms for the Bench2Dex RGB/action policies."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

BENCH2DEX_CAMERAS: tuple[str, ...] = (
    "stereo_left",
    "stereo_right",
    "wrist_left",
    "wrist_right",
)

BENCH2DEX_MODEL_IMAGES: tuple[str, ...] = (
    "base_0_rgb",
    "base_1_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

BENCH2DEX_ACTIVE_INDICES: tuple[int, ...] = (
    # Keep the dataset joint order and drop the ten mimic/locked channels. This
    # matches the active38 ordering used by the dataset's norm_stats.json.
    *range(29),
    32, 33, 34, 35, 36, 39, 40, 41, 47,
)


@dataclasses.dataclass(frozen=True)
class Bench2DexInputs(transforms.DataTransformFn):
    """Map the four Bench2Dex RGB streams to Pi0.5 image slots."""

    state_dim: int = 48
    action_dim: int = 48
    use_active_dof: bool = False
    expected_cameras: ClassVar[tuple[str, ...]] = BENCH2DEX_CAMERAS

    def __post_init__(self) -> None:
        expected_dim = len(BENCH2DEX_ACTIVE_INDICES) if self.use_active_dof else 48
        if self.state_dim != expected_dim or self.action_dim != expected_dim:
            raise ValueError(f"Bench2Dex dimensions must be {expected_dim} for this layout.")

    def __call__(self, data: dict) -> dict:
        images = data.get("images")
        if not isinstance(images, dict):
            raise TypeError(f"Expected 'images' to be a dict, got {type(images)}")
        missing = set(self.expected_cameras) - set(images)
        if missing:
            raise KeyError(f"Missing Bench2Dex cameras: {sorted(missing)}")

        output_images = {
            model_key: _to_hwc_uint8(images[source])
            for model_key, source in zip(BENCH2DEX_MODEL_IMAGES, self.expected_cameras, strict=True)
        }
        output = {
            "image": output_images,
            "image_mask": dict.fromkeys(output_images, np.True_),
            "state": _select_dof(data["state"], self.state_dim, self.use_active_dof, "state"),
        }
        if "actions" in data:
            output["actions"] = _select_dof(data["actions"], self.action_dim, self.use_active_dof, "actions")
        if "prompt" in data:
            output["prompt"] = data["prompt"]
        return output


@dataclasses.dataclass(frozen=True)
class Bench2DexOutputs(transforms.DataTransformFn):
    """Keep the model's full 48-D absolute action output."""

    action_dim: int = 48

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim], dtype=np.float32)}


def _require_dim(value: np.ndarray, dim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] != dim:
        raise ValueError(f"Bench2Dex {name} has dim {array.shape[-1]}, expected {dim}.")
    return array


def _select_dof(value: np.ndarray, output_dim: int, active: bool, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not active:
        return _require_dim(array, 48, name)
    if output_dim != len(BENCH2DEX_ACTIVE_INDICES):
        raise ValueError(f"Expected active Bench2Dex {name} dim {len(BENCH2DEX_ACTIVE_INDICES)}")
    if array.shape[-1] != output_dim:
        raise ValueError(
            f"Active Bench2Dex {name} must be offline-converted to dim {output_dim}; got {array.shape[-1]}"
        )
    return array


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected an RGB image with 3 dimensions, got {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = einops.rearrange(array, "c h w -> h w c")
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.asarray(array, dtype=np.uint8)
