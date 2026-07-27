"""Helpers for reframing holosoma-format motion NPZ files in the world XY/yaw plane."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from holosoma.utils.motion_mirror import REQUIRED_MOTION_KEYS, extract_heading_yaw_wxyz
from holosoma.utils.path import resolve_data_file_path


def _rotation_matrix_from_yaw(yaw: float) -> np.ndarray:
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _yaw_quaternion_wxyz(yaw: float) -> np.ndarray:
    half_yaw = 0.5 * float(yaw)
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _rotate_world_vectors(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...j->...i", rotation, np.asarray(values, dtype=np.float64)).astype(
        np.asarray(values).dtype,
        copy=False,
    )


def _left_multiply_world_quaternions(quaternions: np.ndarray, left_quaternion: np.ndarray) -> np.ndarray:
    quat_array = np.asarray(quaternions)
    left = np.broadcast_to(np.asarray(left_quaternion, dtype=np.float64), quat_array.shape)
    reframed = _quat_mul_wxyz(left, quat_array.astype(np.float64, copy=False))
    return reframed.astype(quat_array.dtype, copy=False)


def compute_reframe_transform(
    motion_data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Compute the rigid world transform that sends frame-0 root xy to the origin and yaw to +x."""
    root_start_xy = np.asarray(motion_data["joint_pos"][0, :2], dtype=np.float64)
    root_start_yaw = float(extract_heading_yaw_wxyz(motion_data["joint_pos"][0, 3:7]))
    yaw_offset = -root_start_yaw
    rotation = _rotation_matrix_from_yaw(yaw_offset)
    yaw_quaternion = _yaw_quaternion_wxyz(yaw_offset)
    return root_start_xy, yaw_offset, rotation, yaw_quaternion


def reframe_motion_dict(
    motion_data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
) -> dict[str, np.ndarray]:
    """Rigidly reframe a motion so the initial root stands at xy=(0, 0) and faces +x."""
    keys = set(motion_data.files if hasattr(motion_data, "files") else motion_data.keys())
    missing = REQUIRED_MOTION_KEYS - keys
    if missing:
        raise ValueError(f"Unsupported motion format: missing required keys {sorted(missing)}.")

    root_start_xy, _, rotation, yaw_quaternion = compute_reframe_transform(motion_data)
    body_names = motion_data["body_names"].tolist()

    reframed: dict[str, np.ndarray] = {
        key: np.array(motion_data[key], copy=True) for key in (motion_data.files if hasattr(motion_data, "files") else motion_data)
    }

    joint_pos = np.array(motion_data["joint_pos"], copy=True)
    joint_pos[:, :2] -= root_start_xy[None, :]
    joint_pos[:, :3] = _rotate_world_vectors(joint_pos[:, :3], rotation)
    joint_pos[:, 3:7] = _left_multiply_world_quaternions(motion_data["joint_pos"][:, 3:7], yaw_quaternion)
    reframed["joint_pos"] = joint_pos

    joint_vel = np.array(motion_data["joint_vel"], copy=True)
    joint_vel[:, :3] = _rotate_world_vectors(motion_data["joint_vel"][:, :3], rotation)
    reframed["joint_vel"] = joint_vel

    body_pos_w = np.array(motion_data["body_pos_w"], copy=True)
    body_pos_w[:, :, :2] -= root_start_xy[None, None, :]
    body_pos_w = _rotate_world_vectors(body_pos_w, rotation)
    body_quat_w = _left_multiply_world_quaternions(motion_data["body_quat_w"], yaw_quaternion)
    body_lin_vel_w = _rotate_world_vectors(motion_data["body_lin_vel_w"], rotation)
    body_ang_vel_w = _rotate_world_vectors(motion_data["body_ang_vel_w"], rotation)

    if body_names and body_names[0] == "world":
        body_pos_w[:, 0, :] = 0.0
        body_quat_w[:, 0, :] = np.array([1.0, 0.0, 0.0, 0.0], dtype=body_quat_w.dtype)
        body_lin_vel_w[:, 0, :] = 0.0
        body_ang_vel_w[:, 0, :] = 0.0

    reframed["body_pos_w"] = body_pos_w
    reframed["body_quat_w"] = body_quat_w
    reframed["body_lin_vel_w"] = body_lin_vel_w
    reframed["body_ang_vel_w"] = body_ang_vel_w

    if "object_pos_w" in motion_data:
        object_pos_w = np.array(motion_data["object_pos_w"], copy=True)
        object_pos_w[:, :2] -= root_start_xy[None, :]
        reframed["object_pos_w"] = _rotate_world_vectors(object_pos_w, rotation)

    if "object_quat_w" in motion_data:
        reframed["object_quat_w"] = _left_multiply_world_quaternions(motion_data["object_quat_w"], yaw_quaternion)

    if "object_lin_vel_w" in motion_data:
        reframed["object_lin_vel_w"] = _rotate_world_vectors(motion_data["object_lin_vel_w"], rotation)

    if "object_ang_vel_w" in motion_data:
        # Object angular velocity is stored in the object's local frame; a pure world-frame
        # reframe does not change that local representation.
        reframed["object_ang_vel_w"] = np.array(motion_data["object_ang_vel_w"], copy=True)

    return reframed


def default_reframed_output_path(input_path: str) -> Path:
    resolved_input = Path(resolve_data_file_path(input_path))
    return resolved_input.with_name(f"{resolved_input.stem}_reframed{resolved_input.suffix}")


__all__ = [
    "compute_reframe_transform",
    "default_reframed_output_path",
    "reframe_motion_dict",
]
