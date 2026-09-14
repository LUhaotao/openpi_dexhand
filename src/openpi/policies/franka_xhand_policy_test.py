import numpy as np
import pytest

from openpi.models import pi0_config as _pi0_config
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
