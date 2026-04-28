import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class FrankaXHandInputs(transforms.DataTransformFn):
    """Inputs for franka TCP pose + xhand joint-position policies.

    Expected data after repack:
    - images: cam_side and optionally cam_wrist, in CHW or HWC format
    - state: [18] = franka absolute TCP pose [6] + xhand absolute joints [12]
    - actions: [horizon, 18] during training
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_side", "cam_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")
        if "cam_side" not in in_images:
            raise ValueError("cam_side is required and is used as base_0_rgb.")

        base_image = _to_hwc_uint8(in_images["cam_side"])
        wrist_image = _to_hwc_uint8(in_images["cam_wrist"]) if "cam_wrist" in in_images else np.zeros_like(base_image)

        inputs = {
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.asarray("cam_wrist" in in_images),
            },
            "state": np.asarray(data["state"], dtype=np.float32),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaXHandOutputs(transforms.DataTransformFn):
    action_dim: int = 18

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim], dtype=np.float32)}


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)

    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0)
        img = (255 * img).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if img.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {img.shape}")
    if img.shape[0] == 3:
        return einops.rearrange(img, "c h w -> h w c")
    if img.shape[-1] == 3:
        return img
    raise ValueError(f"Expected CHW or HWC RGB image, got shape {img.shape}")
