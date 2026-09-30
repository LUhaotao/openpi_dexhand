import numpy as np

from openpi.policies.bench2dex_policy import Bench2DexInputs


def test_bench2dex_inputs_map_four_rgb_views_and_full_actions():
    transform = Bench2DexInputs()
    sample = {
        "images": {
            "stereo_left": np.zeros((3, 8, 9), dtype=np.uint8),
            "stereo_right": np.ones((8, 9, 3), dtype=np.uint8),
            "wrist_left": np.full((8, 9, 3), 2, dtype=np.uint8),
            "wrist_right": np.full((8, 9, 3), 3, dtype=np.uint8),
        },
        "state": np.zeros(48, dtype=np.float32),
        "actions": np.zeros((50, 48), dtype=np.float32),
        "prompt": "pour wine",
    }

    output = transform(sample)

    assert tuple(output["image"]) == (
        "base_0_rgb",
        "base_1_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    assert output["image"]["base_0_rgb"].shape == (8, 9, 3)
    assert output["state"].shape == (48,)
    assert output["actions"].shape == (50, 48)
    assert output["prompt"] == "pour wine"


def test_bench2dex_inputs_select_active_38d_actions():
    transform = Bench2DexInputs(state_dim=38, action_dim=38, use_active_dof=True)
    sample = {
        "images": {
            "stereo_left": np.zeros((8, 9, 3), dtype=np.uint8),
            "stereo_right": np.zeros((8, 9, 3), dtype=np.uint8),
            "wrist_left": np.zeros((8, 9, 3), dtype=np.uint8),
            "wrist_right": np.zeros((8, 9, 3), dtype=np.uint8),
        },
        "state": np.arange(38, dtype=np.float32),
        "actions": np.broadcast_to(np.arange(38, dtype=np.float32), (50, 38)),
    }

    output = transform(sample)

    assert output["state"].shape == (38,)
    assert output["actions"].shape == (50, 38)
    np.testing.assert_array_equal(output["state"], sample["state"])
    np.testing.assert_array_equal(output["actions"], sample["actions"])
