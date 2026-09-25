import numpy as np
import pytest

from openpi.models import pi0_config as _pi0_config
from openpi.policies.franka_xhand_policy import FrankaXHandInputs
from openpi.policies.franka_xhand_policy import _to_marker


def test_to_marker_accepts_univtac_marker_shape():
    marker = np.zeros(_pi0_config.TACTILE_MARKER_SHAPE, dtype=np.float32)

    result = _to_marker(marker, "marker")

    assert result.shape == _pi0_config.TACTILE_MARKER_SHAPE
    assert result.dtype == np.float32


def test_to_marker_rejects_padded_marker_shape():
    marker = np.zeros((2, 1200, 2), dtype=np.float32)

    with pytest.raises(ValueError, match=r"\(2, 63, 2\)"):
        _to_marker(marker, "marker")


def test_franka_inputs_preserve_marker_history_and_current_frame():
    history = np.stack(
        [np.full(_pi0_config.TACTILE_MARKER_SHAPE, index, dtype=np.float32) for index in range(4)], axis=0
    )
    transform = FrankaXHandInputs()

    result = transform(
        {
            "images": {"cam_side": np.zeros((4, 4, 3), dtype=np.uint8)},
            "state": np.zeros((18,), dtype=np.float32),
            "left_marker": history,
            "right_marker": history + 10,
        }
    )

    assert result["tactile_left_marker_history"].shape == (4, *_pi0_config.TACTILE_MARKER_SHAPE)
    np.testing.assert_array_equal(result["tactile_left_marker"], history[-1])
    np.testing.assert_array_equal(result["tactile_right_marker"], history[-1] + 10)
