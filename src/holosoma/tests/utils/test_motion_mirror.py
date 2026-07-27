from __future__ import annotations

import numpy as np
import pytest

from holosoma.utils.motion_mirror import (
    _build_body_mirror_map,
    _infer_object_body_index,
    _quaternion_wxyz_to_matrix,
    extract_heading_yaw_wxyz,
    infer_object_urdf_path_from_motion_path,
    infer_urdf_path_from_motion_path,
    infer_retargeting_mjcf_path_from_motion_path,
    load_object_canonical_to_local_rotation_from_urdf,
    mirror_heading_angular_velocity,
    mirror_heading_quaternion_wxyz,
    mirror_motion_dict,
    orient_object_canonical_frame_from_motion,
    reflect_local_pseudovector_in_calibrated_frame,
    reflect_pseudovector,
    reflect_quaternion_wxyz,
    reflect_quaternion_wxyz_in_calibrated_frame,
    reflect_vector,
)
from holosoma.utils.path import resolve_data_file_path


def test_reflection_helpers_for_y_axis():
    vectors = np.array([[1.0, 2.0, 3.0]])
    quaternions = np.array([[0.5, 0.1, 0.2, 0.3]])

    np.testing.assert_allclose(reflect_vector(vectors, 1), np.array([[1.0, -2.0, 3.0]]))
    np.testing.assert_allclose(reflect_pseudovector(vectors, 1), np.array([[-1.0, 2.0, -3.0]]))
    np.testing.assert_allclose(reflect_quaternion_wxyz(quaternions, 1), np.array([[0.5, -0.1, 0.2, -0.3]]))


def test_heading_mirror_negates_z_yaw_and_yaw_rate():
    yaw = 0.4
    quat = np.array([[np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]])
    mirrored_quat = mirror_heading_quaternion_wxyz(quat)
    mirrored_yaw = extract_heading_yaw_wxyz(mirrored_quat)

    np.testing.assert_allclose(mirrored_yaw, np.array([-yaw]))
    np.testing.assert_allclose(
        mirror_heading_angular_velocity(np.array([[0.2, -0.3, 1.5]])),
        np.array([[0.2, -0.3, -1.5]]),
    )


def test_mirror_motion_swaps_joint_columns_and_reflects_object_track():
    motion_path = resolve_data_file_path(
        "holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
    )
    urdf_path = infer_urdf_path_from_motion_path(motion_path)

    with np.load(motion_path, allow_pickle=False) as motion_data:
        mirrored = mirror_motion_dict(motion_data, urdf_path=urdf_path, motion_path=motion_path)
        joint_names = motion_data["joint_names"].tolist()
        body_names = motion_data["body_names"].tolist()

        left_hip_pitch = joint_names.index("left_hip_pitch_joint")
        right_hip_pitch = joint_names.index("right_hip_pitch_joint")
        left_hip_roll = joint_names.index("left_hip_roll_joint")
        right_hip_roll = joint_names.index("right_hip_roll_joint")
        waist_roll = joint_names.index("waist_roll_joint")
        waist_pitch = joint_names.index("waist_pitch_joint")
        left_shoulder = body_names.index("left_shoulder_pitch_link")
        right_shoulder = body_names.index("right_shoulder_pitch_link")
        left_hand = body_names.index("left_rubber_hand_link")
        right_hand = body_names.index("right_rubber_hand_link")
        object_urdf_path = infer_object_urdf_path_from_motion_path(motion_path, motion_data)
        assert object_urdf_path is not None
        object_canonical_to_local = load_object_canonical_to_local_rotation_from_urdf(object_urdf_path)
        assert object_canonical_to_local is not None
        object_canonical_to_local = orient_object_canonical_frame_from_motion(object_canonical_to_local, motion_data)

        np.testing.assert_array_equal(mirrored["joint_names"], motion_data["joint_names"])
        np.testing.assert_array_equal(mirrored["body_names"], motion_data["body_names"])
        np.testing.assert_allclose(mirrored["joint_pos"][:, 1], -motion_data["joint_pos"][:, 1])
        np.testing.assert_allclose(mirrored["joint_vel"][:, :3], reflect_vector(motion_data["joint_vel"][:, :3], 1))
        np.testing.assert_allclose(mirrored["joint_vel"][:, 3:6], reflect_pseudovector(motion_data["joint_vel"][:, 3:6], 1))
        np.testing.assert_allclose(mirrored["joint_pos"][:, 7 + left_hip_pitch], motion_data["joint_pos"][:, 7 + right_hip_pitch])
        np.testing.assert_allclose(mirrored["joint_pos"][:, 7 + left_hip_roll], -motion_data["joint_pos"][:, 7 + right_hip_roll])
        np.testing.assert_allclose(mirrored["joint_pos"][:, 7 + waist_roll], -motion_data["joint_pos"][:, 7 + waist_roll])
        np.testing.assert_allclose(mirrored["joint_pos"][:, 7 + waist_pitch], motion_data["joint_pos"][:, 7 + waist_pitch])
        np.testing.assert_allclose(mirrored["joint_vel"][:, 6 + left_hip_pitch], motion_data["joint_vel"][:, 6 + right_hip_pitch])
        np.testing.assert_allclose(mirrored["joint_vel"][:, 6 + left_hip_roll], -motion_data["joint_vel"][:, 6 + right_hip_roll])
        np.testing.assert_allclose(mirrored["object_pos_w"][:, 1], -motion_data["object_pos_w"][:, 1])
        np.testing.assert_allclose(mirrored["object_lin_vel_w"], reflect_vector(motion_data["object_lin_vel_w"], 1))
        np.testing.assert_allclose(
            mirrored["object_quat_w"],
            reflect_quaternion_wxyz_in_calibrated_frame(
                motion_data["object_quat_w"],
                1,
                object_canonical_to_local,
            ),
        )
        np.testing.assert_allclose(
            mirrored["object_ang_vel_w"],
            reflect_local_pseudovector_in_calibrated_frame(
                motion_data["object_ang_vel_w"],
                object_canonical_to_local,
            ),
        )
        np.testing.assert_allclose(
            mirrored["body_pos_w"][:, left_shoulder, :],
            reflect_vector(motion_data["body_pos_w"][:, right_shoulder, :], 1),
        )

        canonical_to_world_orig = np.einsum(
            "tij,jk->tik",
            np.stack([_quaternion_wxyz_to_matrix(quat) for quat in motion_data["object_quat_w"]]),
            object_canonical_to_local,
        )
        canonical_to_world_mir = np.einsum(
            "tij,jk->tik",
            np.stack([_quaternion_wxyz_to_matrix(quat) for quat in mirrored["object_quat_w"]]),
            object_canonical_to_local,
        )
        local_left_right_flip = np.diag([-1.0, 1.0, 1.0])
        expected_left_hand_local = np.einsum(
            "ij,tj->ti",
            local_left_right_flip,
            np.einsum(
                "tji,tj->ti",
                canonical_to_world_orig,
                motion_data["body_pos_w"][:, right_hand, :] - motion_data["object_pos_w"],
            ),
        )
        actual_left_hand_local = np.einsum(
            "tji,tj->ti",
            canonical_to_world_mir,
            mirrored["body_pos_w"][:, left_hand, :] - mirrored["object_pos_w"],
        )
        np.testing.assert_allclose(actual_left_hand_local, expected_left_hand_local, atol=1e-5)

        object_to_robot_orig = motion_data["body_pos_w"][:, body_names.index("pelvis"), :] - motion_data["object_pos_w"]
        object_to_robot_mir = mirrored["body_pos_w"][:, body_names.index("pelvis"), :] - mirrored["object_pos_w"]
        object_to_robot_orig[:, 2] = 0.0
        object_to_robot_mir[:, 2] = 0.0
        object_to_robot_orig /= np.maximum(np.linalg.norm(object_to_robot_orig, axis=-1, keepdims=True), 1e-8)
        object_to_robot_mir /= np.maximum(np.linalg.norm(object_to_robot_mir, axis=-1, keepdims=True), 1e-8)
        front_orig = canonical_to_world_orig[:, :, 1]
        front_mir = canonical_to_world_mir[:, :, 1]
        np.testing.assert_allclose(
            np.einsum("ti,ti->t", front_mir, object_to_robot_mir),
            np.einsum("ti,ti->t", front_orig, object_to_robot_orig),
            atol=1e-5,
        )

        mujoco = pytest.importorskip("mujoco")
        mjcf_path = infer_retargeting_mjcf_path_from_motion_path(motion_path, motion_data)
        assert mjcf_path is not None
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        data = mujoco.MjData(model)
        object_body_index = _infer_object_body_index(motion_data)
        assert object_body_index is not None

        for frame_idx in (0, mirrored["joint_pos"].shape[0] // 2, mirrored["joint_pos"].shape[0] - 1):
            data.qpos[:] = np.concatenate(
                [mirrored["joint_pos"][frame_idx], mirrored["object_pos_w"][frame_idx], mirrored["object_quat_w"][frame_idx]]
            )
            data.qvel[:] = np.concatenate(
                [mirrored["joint_vel"][frame_idx], mirrored["object_lin_vel_w"][frame_idx], mirrored["object_ang_vel_w"][frame_idx]]
            )
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(mirrored["body_pos_w"][frame_idx], data.xpos)
            np.testing.assert_allclose(mirrored["body_quat_w"][frame_idx], data.xquat)

            velocities = np.zeros((model.nbody, 6), dtype=np.float64)
            for body_idx in range(model.nbody):
                mujoco.mj_objectVelocity(
                    model,
                    data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_idx,
                    velocities[body_idx],
                    0,
                )

            np.testing.assert_allclose(mirrored["body_lin_vel_w"][frame_idx], velocities[:, 3:6])
            np.testing.assert_allclose(mirrored["body_ang_vel_w"][frame_idx], velocities[:, 0:3])

            object_rotation = _quaternion_wxyz_to_matrix(mirrored["object_quat_w"][frame_idx])
            np.testing.assert_allclose(
                mirrored["object_lin_vel_w"][frame_idx],
                mirrored["body_lin_vel_w"][frame_idx, object_body_index],
            )
            np.testing.assert_allclose(
                object_rotation @ mirrored["object_ang_vel_w"][frame_idx],
                mirrored["body_ang_vel_w"][frame_idx, object_body_index],
                atol=1e-5,
            )


def test_mirror_motion_plain_dict_fallback_preserves_robot_velocity_reflection():
    motion_path = resolve_data_file_path(
        "holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
    )
    urdf_path = infer_urdf_path_from_motion_path(motion_path)

    with np.load(motion_path, allow_pickle=False) as motion_data:
        plain_motion = {key: np.array(motion_data[key], copy=True) for key in motion_data.files}
        body_source_indices = _build_body_mirror_map(motion_data["body_names"].tolist())
        object_body_index = _infer_object_body_index(motion_data)
        mirrored = mirror_motion_dict(plain_motion, urdf_path=urdf_path)

        assert object_body_index is not None
        body_mask = np.ones(len(motion_data["body_names"]), dtype=bool)
        body_mask[object_body_index] = False

        np.testing.assert_allclose(mirrored["joint_vel"][:, :3], reflect_vector(motion_data["joint_vel"][:, :3], 1))
        np.testing.assert_allclose(mirrored["joint_vel"][:, 3:6], reflect_pseudovector(motion_data["joint_vel"][:, 3:6], 1))
        np.testing.assert_allclose(
            mirrored["body_lin_vel_w"][:, body_mask, :],
            reflect_vector(motion_data["body_lin_vel_w"][:, body_source_indices, :], 1)[:, body_mask, :],
        )
        np.testing.assert_allclose(
            mirrored["body_ang_vel_w"][:, body_mask, :],
            reflect_pseudovector(motion_data["body_ang_vel_w"][:, body_source_indices, :], 1)[:, body_mask, :],
        )


def test_mirror_motion_infers_source_path_from_loaded_npz():
    motion_path = resolve_data_file_path(
        "holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
    )
    urdf_path = infer_urdf_path_from_motion_path(motion_path)

    with np.load(motion_path, allow_pickle=False) as motion_data:
        mirrored_from_loaded_npz = mirror_motion_dict(motion_data, urdf_path=urdf_path)

    with np.load(motion_path, allow_pickle=False) as motion_data:
        mirrored_with_explicit_path = mirror_motion_dict(
            motion_data,
            urdf_path=urdf_path,
            motion_path=motion_path,
        )

    np.testing.assert_allclose(mirrored_from_loaded_npz["joint_pos"], mirrored_with_explicit_path["joint_pos"])
    np.testing.assert_allclose(mirrored_from_loaded_npz["joint_vel"], mirrored_with_explicit_path["joint_vel"])
    np.testing.assert_allclose(mirrored_from_loaded_npz["body_pos_w"], mirrored_with_explicit_path["body_pos_w"])
    np.testing.assert_allclose(mirrored_from_loaded_npz["body_quat_w"], mirrored_with_explicit_path["body_quat_w"])
    np.testing.assert_allclose(mirrored_from_loaded_npz["body_lin_vel_w"], mirrored_with_explicit_path["body_lin_vel_w"])
    np.testing.assert_allclose(mirrored_from_loaded_npz["body_ang_vel_w"], mirrored_with_explicit_path["body_ang_vel_w"])
    np.testing.assert_allclose(mirrored_from_loaded_npz["object_lin_vel_w"], mirrored_with_explicit_path["object_lin_vel_w"])
    np.testing.assert_allclose(mirrored_from_loaded_npz["object_ang_vel_w"], mirrored_with_explicit_path["object_ang_vel_w"])
