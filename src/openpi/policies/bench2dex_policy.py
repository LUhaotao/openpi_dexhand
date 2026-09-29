"""LeRobot transforms for the Bench2Dex RGB/full-joint policy.

The first OpenPI Bench2Dex baseline keeps the dataset's 48-D full joint layout
and trains absolute joint-position actions. Mimic-joint reduction is left for
a separate active-DOF experiment so this adapter stays lossless.
"""

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


@dataclasses.dataclass(frozen=True)
class Bench2DexInputs(transforms.DataTransformFn):
    """Map the four Bench2Dex RGB streams to Pi0.5 image slots."""

    state_dim: int = 48
    action_dim: int = 48
    expected_cameras: ClassVar[tuple[str, ...]] = BENCH2DEX_CAMERAS

    def __post_init__(self) -> None:
        if self.state_dim != 48 or self.action_dim != 48:
            raise ValueError("The initial Bench2Dex baseline requires state_dim=action_dim=48.")

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
            "image_mask": {key: np.True_ for key in output_images},
            "state": _require_dim(data["state"], self.state_dim, "state"),
        }
        if "actions" in data:
            output["actions"] = _require_dim(data["actions"], self.action_dim, "actions")
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


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected an RGB image with 3 dimensions, got {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = einops.rearrange(array, "c h w -> h w c")
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.asarray(array, dtype=np.uint8)
