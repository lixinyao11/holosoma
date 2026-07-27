from __future__ import annotations

import numpy as np
import pytest

from holosoma.utils.motion_trim import resolve_trim_bounds, trim_motion_dict


def _make_motion(num_frames: int = 8) -> dict[str, np.ndarray]:
    body_names = np.array([f"body_{idx}" for idx in range(num_frames)], dtype="<U16")
    joint_names = np.array(["joint_0", "joint_1", "joint_2"], dtype="<U16")
    return {
        "fps": np.array([50.0], dtype=np.float32),
        "source_fps": np.array([100.0], dtype=np.float32),
        "joint_pos": np.arange(num_frames * 10, dtype=np.float32).reshape(num_frames, 10),
        "joint_vel": np.arange(num_frames * 9, dtype=np.float32).reshape(num_frames, 9),
        "body_pos_w": np.arange(num_frames * num_frames * 3, dtype=np.float32).reshape(num_frames, num_frames, 3),
        "body_quat_w": np.arange(num_frames * num_frames * 4, dtype=np.float32).reshape(num_frames, num_frames, 4),
        "body_lin_vel_w": np.arange(num_frames * num_frames * 3, dtype=np.float32).reshape(num_frames, num_frames, 3),
        "body_ang_vel_w": np.arange(num_frames * num_frames * 3, dtype=np.float32).reshape(num_frames, num_frames, 3),
        "joint_names": joint_names,
        "body_names": body_names,
        "object_pos_w": np.arange(num_frames * 3, dtype=np.float32).reshape(num_frames, 3),
        "object_quat_w": np.arange(num_frames * 4, dtype=np.float32).reshape(num_frames, 4),
        "object_lin_vel_w": np.arange(num_frames * 3, dtype=np.float32).reshape(num_frames, 3),
        "object_ang_vel_w": np.arange(num_frames * 3, dtype=np.float32).reshape(num_frames, 3),
        "object_geom_type": np.array("mesh"),
        "object_mesh_vertices": np.arange(num_frames * 3, dtype=np.float32).reshape(num_frames, 3),
        "object_mesh_faces": np.arange(num_frames * 3, dtype=np.int32).reshape(num_frames, 3),
        "fixed_scene_object_names": body_names.copy(),
        "fixed_scene_object_geom_types": np.array(["box"] * num_frames, dtype="<U16"),
        "fixed_scene_object_pos_w": np.arange(num_frames * 3, dtype=np.float32).reshape(num_frames, 3),
        "fixed_scene_object_quat_w": np.arange(num_frames * 4, dtype=np.float32).reshape(num_frames, 4),
        "fixed_scene_object_size": np.arange(num_frames * 3, dtype=np.float32).reshape(num_frames, 3),
        "custom_frame_signal": np.arange(num_frames * 2, dtype=np.float32).reshape(num_frames, 2),
    }


def test_resolve_trim_bounds_supports_defaults_positive_bounds_and_negative_end():
    assert resolve_trim_bounds(757) == (0, 757)
    assert resolve_trim_bounds(757, start=10, end=700) == (10, 700)
    assert resolve_trim_bounds(757, start=10, end=-10) == (10, 747)
    assert resolve_trim_bounds(757, start=-20, end=-10) == (737, 747)


def test_trim_motion_slices_frame_arrays_and_preserves_metadata():
    motion = _make_motion(num_frames=8)
    trimmed = trim_motion_dict(motion, start=2, end=-1)

    assert trimmed["joint_pos"].shape[0] == 5
    np.testing.assert_array_equal(trimmed["joint_pos"], motion["joint_pos"][2:7])
    np.testing.assert_array_equal(trimmed["body_quat_w"], motion["body_quat_w"][2:7])
    np.testing.assert_array_equal(trimmed["object_pos_w"], motion["object_pos_w"][2:7])
    np.testing.assert_array_equal(trimmed["custom_frame_signal"], motion["custom_frame_signal"][2:7])

    np.testing.assert_array_equal(trimmed["body_names"], motion["body_names"])
    np.testing.assert_array_equal(trimmed["fixed_scene_object_names"], motion["fixed_scene_object_names"])
    np.testing.assert_array_equal(trimmed["fixed_scene_object_pos_w"], motion["fixed_scene_object_pos_w"])
    np.testing.assert_array_equal(trimmed["object_mesh_vertices"], motion["object_mesh_vertices"])
    np.testing.assert_array_equal(trimmed["object_mesh_faces"], motion["object_mesh_faces"])


def test_trim_motion_default_keeps_full_motion():
    motion = _make_motion(num_frames=6)
    trimmed = trim_motion_dict(motion)

    for key, value in motion.items():
        np.testing.assert_array_equal(trimmed[key], value)


def test_trim_motion_rejects_empty_or_out_of_range_bounds():
    with pytest.raises(ValueError, match="at least one frame"):
        resolve_trim_bounds(10, start=5, end=5)
    with pytest.raises(ValueError, match="Trim end"):
        resolve_trim_bounds(10, end=-11)
