import math

import numpy as np

from holosoma.utils.viser_helper import ViserHelper


def test_build_heading_arrow_segments_points_along_positive_x_for_identity_quaternion():
    segments = ViserHelper._build_heading_arrow_segments(
        position=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        shaft_length=0.24,
        head_length=0.08,
        head_width=0.06,
        vertical_offset=0.06,
    )

    origin = segments[0, 0]
    tip = segments[0, 1]

    assert np.allclose(origin, np.array([1.0, 2.0, 3.06], dtype=np.float32))
    assert np.allclose(tip, np.array([1.24, 2.0, 3.06], dtype=np.float32))


def test_build_heading_arrow_segments_uses_yaw_heading_from_quaternion():
    yaw_quat_xyzw = np.array([0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)], dtype=np.float32)

    segments = ViserHelper._build_heading_arrow_segments(
        position=np.zeros(3, dtype=np.float32),
        quat_xyzw=yaw_quat_xyzw,
        shaft_length=0.24,
        head_length=0.08,
        head_width=0.06,
        vertical_offset=0.0,
    )

    tip = segments[0, 1]

    assert np.allclose(tip, np.array([0.0, 0.24, 0.0], dtype=np.float32), atol=1.0e-5)
