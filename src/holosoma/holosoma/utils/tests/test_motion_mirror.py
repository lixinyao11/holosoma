import numpy as np
import pytest

from holosoma.utils.motion_mirror import mirror_motion_dict


def _minimal_motion_with_fixed_scene() -> dict[str, np.ndarray]:
    return {
        "fps": np.array(50.0, dtype=np.float32),
        "joint_pos": np.array(
            [
                [1.0, 0.3, 0.9, 1.0, 0.0, 0.0, 0.0, 0.2, -0.4],
            ],
            dtype=np.float32,
        ),
        "joint_vel": np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.1, -1.2],
            ],
            dtype=np.float32,
        ),
        "body_pos_w": np.array(
            [
                [
                    [1.0, 0.5, 0.8],
                    [1.2, 0.7, 0.9],
                    [0.8, -0.7, 0.9],
                ]
            ],
            dtype=np.float32,
        ),
        "body_quat_w": np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.9238795, 0.0, 0.0, 0.3826834],
                    [0.9238795, 0.0, 0.0, -0.3826834],
                ]
            ],
            dtype=np.float32,
        ),
        "body_lin_vel_w": np.array(
            [
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, -0.5, 0.9],
                ]
            ],
            dtype=np.float32,
        ),
        "body_ang_vel_w": np.array(
            [
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, -0.5, 0.9],
                ]
            ],
            dtype=np.float32,
        ),
        "body_names": np.array(["pelvis", "left_hand", "right_hand"], dtype="<U32"),
        "joint_names": np.array(["left_knee_joint", "right_knee_joint"], dtype="<U32"),
        "fixed_scene_object_names": np.array(["climb_box"], dtype="<U32"),
        "fixed_scene_object_geom_types": np.array(["box"], dtype="<U16"),
        "fixed_scene_object_pos_w": np.array([[2.7, 0.4, 0.25]], dtype=np.float32),
        "fixed_scene_object_quat_w": np.array([[0.9238795, 0.0, 0.0, 0.3826834]], dtype=np.float32),
        "fixed_scene_object_size": np.array([[0.8, 2.0, 0.5]], dtype=np.float32),
    }


def test_mirror_motion_dict_reflects_fixed_scene_metadata():
    mirrored = mirror_motion_dict(_minimal_motion_with_fixed_scene(), motion_path=None)

    assert np.allclose(mirrored["fixed_scene_object_pos_w"], [[2.7, -0.4, 0.25]])
    assert np.allclose(mirrored["fixed_scene_object_quat_w"], [[0.9238795, -0.0, 0.0, -0.3826834]], atol=1e-6)
    assert np.allclose(mirrored["fixed_scene_object_size"], [[0.8, 2.0, 0.5]])
    assert mirrored["fixed_scene_object_names"].tolist() == ["climb_box"]
    assert mirrored["fixed_scene_object_geom_types"].tolist() == ["box"]


def test_mirror_motion_dict_rejects_incomplete_fixed_scene_metadata():
    motion = _minimal_motion_with_fixed_scene()
    motion.pop("fixed_scene_object_quat_w")

    with pytest.raises(ValueError, match="Fixed scene object metadata is incomplete"):
        mirror_motion_dict(motion, motion_path=None)
