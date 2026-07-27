from __future__ import annotations

import numpy as np

from holosoma.utils.motion_mirror import _infer_object_body_index, extract_heading_yaw_wxyz
from holosoma.utils.motion_reframe import (
    compute_reframe_transform,
    reframe_motion_dict,
)
from holosoma.utils.path import resolve_data_file_path


def test_reframe_motion_starts_at_origin_facing_x_and_preserves_object_local_frame():
    motion_path = resolve_data_file_path(
        "holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
    )

    with np.load(motion_path, allow_pickle=False) as motion_data:
        root_start_xy, _, rotation, _ = compute_reframe_transform(motion_data)
        reframed = reframe_motion_dict(motion_data)
        object_body_index = _infer_object_body_index(motion_data)
        assert object_body_index is not None

        np.testing.assert_allclose(reframed["joint_pos"][0, :2], np.zeros(2), atol=1e-6)
        np.testing.assert_allclose(extract_heading_yaw_wxyz(reframed["joint_pos"][0, 3:7]), 0.0, atol=1e-6)

        np.testing.assert_allclose(reframed["joint_pos"][:, 7:], motion_data["joint_pos"][:, 7:])
        np.testing.assert_allclose(reframed["joint_vel"][:, 3:6], motion_data["joint_vel"][:, 3:6])
        np.testing.assert_allclose(reframed["joint_vel"][:, 6:], motion_data["joint_vel"][:, 6:])

        expected_root_pos = motion_data["joint_pos"][:, :3].copy()
        expected_root_pos[:, :2] -= root_start_xy[None, :]
        expected_root_pos = np.einsum("ij,tj->ti", rotation, expected_root_pos)
        np.testing.assert_allclose(reframed["joint_pos"][:, :3], expected_root_pos)
        np.testing.assert_allclose(
            reframed["joint_vel"][:, :3],
            np.einsum("ij,tj->ti", rotation, motion_data["joint_vel"][:, :3]),
        )

        expected_body_pos = motion_data["body_pos_w"].copy()
        expected_body_pos[:, :, :2] -= root_start_xy[None, None, :]
        expected_body_pos = np.einsum("ij,tbj->tbi", rotation, expected_body_pos)
        np.testing.assert_allclose(reframed["body_pos_w"][:, 1:, :], expected_body_pos[:, 1:, :])
        np.testing.assert_allclose(reframed["body_pos_w"][:, 0, :], np.zeros(3), atol=1e-8)
        np.testing.assert_allclose(reframed["body_quat_w"][:, 0, :], np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-8)
        np.testing.assert_allclose(reframed["body_lin_vel_w"][:, 0, :], np.zeros(3), atol=1e-8)
        np.testing.assert_allclose(reframed["body_ang_vel_w"][:, 0, :], np.zeros(3), atol=1e-8)
        np.testing.assert_allclose(
            reframed["body_lin_vel_w"][:, 1:, :],
            np.einsum("ij,tbj->tbi", rotation, motion_data["body_lin_vel_w"][:, 1:, :]),
        )
        np.testing.assert_allclose(
            reframed["body_ang_vel_w"][:, 1:, :],
            np.einsum("ij,tbj->tbi", rotation, motion_data["body_ang_vel_w"][:, 1:, :]),
        )

        expected_object_pos = motion_data["object_pos_w"].copy()
        expected_object_pos[:, :2] -= root_start_xy[None, :]
        expected_object_pos = np.einsum("ij,tj->ti", rotation, expected_object_pos)
        np.testing.assert_allclose(reframed["object_pos_w"], expected_object_pos)
        np.testing.assert_allclose(
            reframed["object_lin_vel_w"],
            np.einsum("ij,tj->ti", rotation, motion_data["object_lin_vel_w"]),
        )
        np.testing.assert_allclose(reframed["object_ang_vel_w"], motion_data["object_ang_vel_w"])

        np.testing.assert_allclose(reframed["body_pos_w"][:, object_body_index, :], reframed["object_pos_w"])
        np.testing.assert_allclose(reframed["body_quat_w"][:, object_body_index, :], reframed["object_quat_w"])
