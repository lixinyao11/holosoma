"""Helpers for mirroring holosoma-format motion NPZ files."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from holosoma.utils.fixed_scene_motion import (
    FIXED_SCENE_KEYS,
    FIXED_SCENE_OBJECT_POS_W_KEY,
    FIXED_SCENE_OBJECT_QUAT_W_KEY,
)
from holosoma.utils.path import resolve_data_file_path

REQUIRED_MOTION_KEYS = {
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "body_names",
    "joint_names",
}

_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def _axis_to_index(mirror_axis: str) -> int:
    normalized = mirror_axis.lower()
    if normalized != "y":
        raise ValueError("Only mirror_axis='y' is currently supported.")
    return _AXIS_TO_INDEX[normalized]


def reflect_vector(values: np.ndarray, axis_index: int) -> np.ndarray:
    """Reflect an ordinary vector across a coordinate plane."""
    mirrored = np.array(values, copy=True)
    mirrored[..., axis_index] *= -1
    return mirrored


def reflect_pseudovector(values: np.ndarray, axis_index: int) -> np.ndarray:
    """Reflect an axial vector such as angular velocity across a coordinate plane."""
    return -reflect_vector(values, axis_index)


def _reflection_matrix(axis_index: int) -> np.ndarray:
    reflection = np.eye(3, dtype=np.float64)
    reflection[axis_index, axis_index] = -1.0
    return reflection


def reflect_quaternion_wxyz(quaternions: np.ndarray, axis_index: int) -> np.ndarray:
    """Reflect a quaternion stored in wxyz order across a coordinate plane."""
    mirrored = np.array(quaternions, copy=True)
    mirrored[..., 1:] = reflect_pseudovector(mirrored[..., 1:], axis_index)
    return mirrored


def reflect_quaternion_wxyz_in_calibrated_frame(
    quaternions: np.ndarray,
    axis_index: int,
    canonical_to_local: np.ndarray,
    local_axis_to_flip: int = 0,
) -> np.ndarray:
    """Reflect a quaternion in a physically meaningful local frame before mapping back."""
    quat_array = np.asarray(quaternions)
    mirrored = np.empty_like(quat_array)
    quat_rows = quat_array.reshape(-1, 4)
    mirrored_rows = mirrored.reshape(-1, 4)

    reflection = _reflection_matrix(axis_index)
    calibration = np.asarray(canonical_to_local, dtype=np.float64)
    local_reflection = np.eye(3, dtype=np.float64)
    local_reflection[local_axis_to_flip, local_axis_to_flip] = -1.0

    for idx, quaternion in enumerate(quat_rows):
        rotation = _quaternion_wxyz_to_matrix(np.asarray(quaternion, dtype=np.float64))
        mirrored_rotation = reflection @ rotation @ calibration @ local_reflection @ calibration.T
        mirrored_rows[idx] = _matrix_to_quaternion_wxyz(mirrored_rotation).astype(quat_array.dtype, copy=False)

    return mirrored


def reflect_local_pseudovector_in_calibrated_frame(
    values: np.ndarray,
    canonical_to_local: np.ndarray,
    local_axis_to_flip: int = 0,
) -> np.ndarray:
    """Reflect a local-frame axial vector through the same calibrated object symmetry used for pose."""
    calibration = np.asarray(canonical_to_local, dtype=np.float64)
    local_reflection = np.eye(3, dtype=np.float64)
    local_reflection[local_axis_to_flip, local_axis_to_flip] = -1.0
    reflected_local_basis = calibration @ local_reflection @ calibration.T

    mirrored = np.einsum("ij,...j->...i", -reflected_local_basis, np.asarray(values, dtype=np.float64))
    return mirrored.astype(np.asarray(values).dtype, copy=False)


def _quat_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
    conjugate = np.array(quaternion, copy=True)
    conjugate[..., 1:] *= -1
    return conjugate


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


def _quat_to_rotvec_wxyz(quaternion: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = np.where(quat[..., :1] < 0.0, -quat, quat)
    w = np.clip(quat[..., 0], -1.0, 1.0)
    xyz = quat[..., 1:]
    mag = np.linalg.norm(xyz, axis=-1)
    half_angle = np.arctan2(mag, w)
    angle = 2.0 * half_angle
    denom = np.where(np.abs(angle) > eps, np.sin(half_angle) / angle, 0.5 - angle * angle / 48.0)
    return xyz / denom[..., None]


def derive_angular_velocity_from_quat_sequence_wxyz(quaternions: np.ndarray, *, fps: float) -> np.ndarray:
    """Estimate angular velocity from a quaternion sequence in wxyz order."""
    quat = np.asarray(quaternions, dtype=np.float64)
    if quat.ndim != 2 or quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion sequence with shape (T, 4), got {quat.shape}.")
    if quat.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if quat.shape[0] == 1:
        return np.zeros((1, 3), dtype=np.float64)

    q_prev = quat[:-2]
    q_next = quat[2:]
    q_rel = _quat_mul_wxyz(q_next, _quat_conjugate_wxyz(q_prev))
    rotvec = _quat_to_rotvec_wxyz(q_rel)
    omega = rotvec / (2.0 / float(fps))
    return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)


def _gradient(sequence: np.ndarray, dt: float) -> np.ndarray:
    if sequence.shape[0] == 0:
        return np.array(sequence, copy=True)
    if sequence.shape[0] == 1:
        return np.zeros_like(sequence)
    return np.gradient(sequence, dt, axis=0)


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        diag = np.diag(matrix)
        idx = int(np.argmax(diag))
        if idx == 0:
            s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif idx == 1:
            s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s

    quaternion = np.array([w, x, y, z], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion


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


def extract_heading_yaw_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """Extract world-frame heading (yaw about +z) from wxyz quaternions."""
    quat_flat = np.asarray(quaternions)
    flat_shape = quat_flat.shape[:-1]
    yaws = np.empty(int(np.prod(flat_shape, dtype=np.int64)) if flat_shape else 1, dtype=np.float64)
    quat_rows = quat_flat.reshape(-1, 4)

    for idx, quaternion in enumerate(quat_rows):
        rotation = _quaternion_wxyz_to_matrix(np.asarray(quaternion, dtype=np.float64))
        yaws[idx] = float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    return yaws.reshape(flat_shape) if flat_shape else yaws.reshape(())


def mirror_heading_quaternion_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """Mirror only the world-frame heading while preserving the non-yaw tilt."""
    quat_array = np.asarray(quaternions)
    mirrored = np.empty_like(quat_array)
    quat_rows = quat_array.reshape(-1, 4)
    mirrored_rows = mirrored.reshape(-1, 4)

    for idx, quaternion in enumerate(quat_rows):
        rotation = _quaternion_wxyz_to_matrix(np.asarray(quaternion, dtype=np.float64))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        tilt = _rotation_matrix_from_yaw(-yaw) @ rotation
        mirrored_rotation = _rotation_matrix_from_yaw(-yaw) @ tilt
        mirrored_rows[idx] = _matrix_to_quaternion_wxyz(mirrored_rotation).astype(quat_array.dtype, copy=False)

    return mirrored


def mirror_heading_angular_velocity(values: np.ndarray) -> np.ndarray:
    """Mirror only the yaw-rate sign for z-up world angular velocities."""
    mirrored = np.array(values, copy=True)
    mirrored[..., 2] *= -1
    return mirrored


def _swap_left_right_name(name: str, available_names: set[str]) -> str:
    if name.startswith("left_"):
        counterpart = f"right_{name[len('left_'):]}"
        return counterpart if counterpart in available_names else name
    if name.startswith("right_"):
        counterpart = f"left_{name[len('right_'):]}"
        return counterpart if counterpart in available_names else name
    return name


def _load_joint_axes_from_urdf(urdf_path: str | None) -> dict[str, np.ndarray]:
    if urdf_path is None:
        return {}

    resolved_path = Path(resolve_data_file_path(urdf_path))
    root = ET.parse(resolved_path).getroot()
    joint_axes: dict[str, np.ndarray] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        axis = joint.find("axis")
        if name is None or axis is None:
            continue
        xyz = axis.attrib.get("xyz")
        if xyz is None:
            continue
        joint_axes[name] = np.fromstring(xyz, sep=" ", dtype=np.float64)
    return joint_axes


def infer_urdf_path_from_motion_path(motion_path: str) -> str | None:
    """Infer a package-data URDF path from a holosoma motion path."""
    resolved_motion = Path(resolve_data_file_path(motion_path))
    parts = list(resolved_motion.parts)
    if "motions" not in parts:
        return None

    motions_idx = parts.index("motions")
    if motions_idx + 1 >= len(parts):
        return None

    robot_name = parts[motions_idx + 1]
    robot_family = robot_name.split("_", 1)[0]
    candidate = f"holosoma/data/robots/{robot_family}/{robot_name}.urdf"
    resolved_candidate = Path(resolve_data_file_path(candidate))
    return candidate if resolved_candidate.exists() else None


def infer_object_urdf_path_from_motion_path(
    motion_path: str,
    motion_data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
) -> str | None:
    """Infer the URDF that defines the dynamic object visual mesh for a motion clip."""
    object_body_index = _infer_object_body_index(motion_data)
    if object_body_index is None:
        return None

    object_body_name = str(motion_data["body_names"][object_body_index])
    object_name = object_body_name[:-5] if object_body_name.endswith("_link") else object_body_name
    resolved_motion = Path(resolve_data_file_path(motion_path))
    candidate = resolved_motion.parent / f"objects_{object_name}.urdf"
    return str(candidate) if candidate.exists() else None


def _infer_motion_path_from_loaded_motion(
    motion_data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
) -> str | None:
    """Recover the source NPZ path from a loaded ``np.load`` result when possible."""
    zip_file = getattr(motion_data, "zip", None)
    zip_filename = getattr(zip_file, "filename", None)
    if isinstance(zip_filename, str) and zip_filename:
        return zip_filename

    fid = getattr(motion_data, "fid", None)
    fid_name = getattr(fid, "name", None)
    if isinstance(fid_name, str) and fid_name:
        return fid_name

    return None


def load_object_canonical_to_local_rotation_from_urdf(urdf_path: str) -> np.ndarray | None:
    """Estimate a canonical physical frame from the first visual mesh in an object URDF."""
    try:
        import trimesh
    except Exception:
        return None

    resolved_urdf = Path(resolve_data_file_path(urdf_path))
    root = ET.parse(resolved_urdf).getroot()
    visual = root.find(".//visual")
    if visual is None:
        return None

    origin = visual.find("origin")
    visual_rotation = np.eye(3, dtype=np.float64)
    if origin is not None:
        rpy = origin.attrib.get("rpy")
        if rpy:
            roll, pitch, yaw = [float(v) for v in rpy.split()]
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            visual_rotation = np.array(
                [
                    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                    [-sp, cp * sr, cp * cr],
                ],
                dtype=np.float64,
            )

    geometry = visual.find("geometry")
    if geometry is None:
        return None
    mesh_node = geometry.find("mesh")
    if mesh_node is None:
        return None

    mesh_filename = mesh_node.attrib.get("filename")
    if not mesh_filename:
        return None

    mesh_path = (resolved_urdf.parent / mesh_filename).resolve()
    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh is None:
        return None

    canonical_to_mesh = np.asarray(mesh.bounding_box_oriented.primitive.transform[:3, :3], dtype=np.float64)
    canonical_to_local = visual_rotation @ canonical_to_mesh
    u, _, vt = np.linalg.svd(canonical_to_local)
    canonical_to_local = u @ vt
    if np.linalg.det(canonical_to_local) < 0.0:
        canonical_to_local[:, -1] *= -1.0
    return canonical_to_local


def orient_object_canonical_frame_from_motion(
    canonical_to_local: np.ndarray,
    motion_data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
) -> np.ndarray:
    """Orient the canonical object frame so x=left/right, y=toward robot, z=up."""
    if "object_quat_w" not in motion_data:
        return canonical_to_local

    object_rotations = np.stack(
        [_quaternion_wxyz_to_matrix(np.asarray(quaternion, dtype=np.float64)) for quaternion in motion_data["object_quat_w"]],
        axis=0,
    )
    canonical_world = object_rotations @ np.asarray(canonical_to_local, dtype=np.float64)

    up_axis_index = int(np.argmax(np.mean(np.abs(canonical_world[:, 2, :]), axis=0)))
    up_local = np.array(canonical_to_local[:, up_axis_index], copy=True)
    up_world = canonical_world[:, :, up_axis_index]
    if float(np.mean(up_world[:, 2])) < 0.0:
        up_local *= -1.0

    remaining_indices = [idx for idx in range(3) if idx != up_axis_index]
    if "body_names" in motion_data:
        body_names = motion_data["body_names"].tolist()
        if "pelvis" in body_names:
            robot_reference = motion_data["body_pos_w"][:, body_names.index("pelvis"), :]
        else:
            robot_reference = motion_data["joint_pos"][:, :3]
    else:
        robot_reference = motion_data["joint_pos"][:, :3]

    object_to_robot = np.array(robot_reference - motion_data["object_pos_w"], copy=True)
    object_to_robot[:, 2] = 0.0
    robot_distance = np.linalg.norm(object_to_robot, axis=-1, keepdims=True)
    object_to_robot = object_to_robot / np.maximum(robot_distance, 1e-8)

    front_scores = [
        float(np.mean(np.abs(np.einsum("ti,ti->t", canonical_world[:, :, idx], object_to_robot))))
        for idx in remaining_indices
    ]
    front_axis_index = remaining_indices[int(np.argmax(front_scores))]
    front_local = np.array(canonical_to_local[:, front_axis_index], copy=True)
    front_world = canonical_world[:, :, front_axis_index]
    if float(np.mean(np.einsum("ti,ti->t", front_world, object_to_robot))) < 0.0:
        front_local *= -1.0

    side_local = np.cross(front_local, up_local)
    side_norm = float(np.linalg.norm(side_local))
    if side_norm < 1e-8:
        return canonical_to_local
    side_local /= side_norm

    oriented = np.stack([side_local, front_local, up_local], axis=1)
    u, _, vt = np.linalg.svd(oriented)
    oriented = u @ vt
    if np.linalg.det(oriented) < 0.0:
        oriented[:, 0] *= -1.0
    return oriented


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def infer_retargeting_mjcf_path_from_motion_path(
    motion_path: str,
    motion_data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
) -> str | None:
    """Infer the retargeting MuJoCo XML used to generate holosoma motion clips."""
    resolved_motion = Path(resolve_data_file_path(motion_path))
    parts = list(resolved_motion.parts)
    if "motions" not in parts:
        return None

    motions_idx = parts.index("motions")
    if motions_idx + 1 >= len(parts):
        return None

    robot_name = parts[motions_idx + 1]
    robot_family = robot_name.split("_", 1)[0]
    retargeting_root = _repo_root() / "src/holosoma_retargeting/holosoma_retargeting/models" / robot_family

    if "object_pos_w" in motion_data and len(motion_data["body_names"]) > 0:
        object_body_name = str(motion_data["body_names"][-1])
        object_name = object_body_name[:-5] if object_body_name.endswith("_link") else object_body_name
        object_candidate = retargeting_root / f"{robot_name}_w_{object_name}.xml"
        if object_candidate.exists():
            return str(object_candidate)

    base_candidate = retargeting_root / f"{robot_name}.xml"
    if base_candidate.exists():
        return str(base_candidate)

    return None


def _heuristic_joint_sign(name: str) -> float:
    lowered = name.lower()
    if "roll" in lowered or "yaw" in lowered:
        return -1.0
    return 1.0


def _normalize_axis(axis: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        raise ValueError(f"Expected a non-zero joint axis, got {axis}.")
    return axis / norm


def _infer_joint_sign(
    target_name: str,
    source_name: str,
    joint_axes: dict[str, np.ndarray],
    axis_index: int,
) -> float:
    target_axis = joint_axes.get(target_name)
    source_axis = joint_axes.get(source_name)
    if target_axis is None or source_axis is None:
        return _heuristic_joint_sign(target_name)

    mirrored_source_axis = reflect_pseudovector(_normalize_axis(source_axis), axis_index)
    target_axis = _normalize_axis(target_axis)
    dot = float(np.dot(target_axis, mirrored_source_axis))
    return 1.0 if dot >= 0.0 else -1.0


def _build_joint_mirror_map(
    joint_names: list[str],
    joint_axes: dict[str, np.ndarray],
    axis_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    joint_name_set = set(joint_names)
    joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names)}

    source_indices = np.empty(len(joint_names), dtype=np.int64)
    signs = np.empty(len(joint_names), dtype=np.float64)

    for idx, name in enumerate(joint_names):
        source_name = _swap_left_right_name(name, joint_name_set)
        source_indices[idx] = joint_name_to_idx[source_name]
        signs[idx] = _infer_joint_sign(name, source_name, joint_axes, axis_index)

    return source_indices, signs


def _build_body_mirror_map(body_names: list[str]) -> np.ndarray:
    body_name_set = set(body_names)
    body_name_to_idx = {name: idx for idx, name in enumerate(body_names)}
    return np.array(
        [body_name_to_idx[_swap_left_right_name(name, body_name_set)] for name in body_names],
        dtype=np.int64,
    )


def _infer_object_body_index(motion_data: dict[str, np.ndarray] | np.lib.npyio.NpzFile) -> int | None:
    if "object_pos_w" not in motion_data or "object_quat_w" not in motion_data:
        return None

    body_pos = motion_data["body_pos_w"]
    body_quat = motion_data["body_quat_w"]
    object_pos = motion_data["object_pos_w"]
    object_quat = motion_data["object_quat_w"]

    for idx in range(body_pos.shape[1]):
        if np.allclose(body_pos[:, idx, :], object_pos) and np.allclose(body_quat[:, idx, :], object_quat):
            return idx
    return None


def _mirror_fixed_scene_metadata(
    motion_data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    mirrored: dict[str, np.ndarray],
    *,
    axis_index: int,
) -> None:
    keys = set(motion_data.files if hasattr(motion_data, "files") else motion_data.keys())
    present_fixed_scene_keys = FIXED_SCENE_KEYS & keys
    if not present_fixed_scene_keys:
        return
    if present_fixed_scene_keys != FIXED_SCENE_KEYS:
        missing_fixed_scene_keys = sorted(FIXED_SCENE_KEYS - present_fixed_scene_keys)
        raise ValueError(f"Fixed scene object metadata is incomplete; missing keys: {missing_fixed_scene_keys}.")

    mirrored[FIXED_SCENE_OBJECT_POS_W_KEY] = reflect_vector(motion_data[FIXED_SCENE_OBJECT_POS_W_KEY], axis_index)
    mirrored[FIXED_SCENE_OBJECT_QUAT_W_KEY] = reflect_quaternion_wxyz(
        motion_data[FIXED_SCENE_OBJECT_QUAT_W_KEY],
        axis_index,
    )


def _world_body_velocities(model, data, mujoco_module) -> tuple[np.ndarray, np.ndarray]:
    velocities = np.zeros((model.nbody, 6), dtype=np.float64)
    for body_idx in range(model.nbody):
        mujoco_module.mj_objectVelocity(
            model,
            data,
            mujoco_module.mjtObj.mjOBJ_BODY,
            body_idx,
            velocities[body_idx],
            0,
        )
    return velocities[:, 3:6], velocities[:, 0:3]


def recompute_body_tracks_with_mujoco(
    mirrored_motion: dict[str, np.ndarray],
    *,
    motion_path: str,
) -> dict[str, np.ndarray]:
    """Recompute body-space tracks from mirrored qpos/qvel to keep the NPZ internally consistent."""
    mjcf_path = infer_retargeting_mjcf_path_from_motion_path(motion_path, mirrored_motion)
    if mjcf_path is None:
        return mirrored_motion

    try:
        import mujoco
    except Exception:
        return mirrored_motion

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    model_body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, idx) for idx in range(model.nbody)]
    motion_body_names = mirrored_motion["body_names"].tolist()
    if model_body_names != motion_body_names:
        raise ValueError(
            "Retargeting MJCF body names do not match the motion NPZ body names: "
            f"{mjcf_path}"
        )

    has_object = "object_pos_w" in mirrored_motion
    num_frames = mirrored_motion["joint_pos"].shape[0]
    body_pos_w = np.empty((num_frames, model.nbody, 3), dtype=mirrored_motion["body_pos_w"].dtype)
    body_quat_w = np.empty((num_frames, model.nbody, 4), dtype=mirrored_motion["body_quat_w"].dtype)
    body_lin_vel_w = np.empty((num_frames, model.nbody, 3), dtype=mirrored_motion["body_lin_vel_w"].dtype)
    body_ang_vel_w = np.empty((num_frames, model.nbody, 3), dtype=mirrored_motion["body_ang_vel_w"].dtype)

    for frame_idx in range(num_frames):
        if has_object:
            qpos = np.concatenate(
                [
                    mirrored_motion["joint_pos"][frame_idx],
                    mirrored_motion["object_pos_w"][frame_idx],
                    mirrored_motion["object_quat_w"][frame_idx],
                ]
            )
            qvel = np.concatenate(
                [
                    mirrored_motion["joint_vel"][frame_idx],
                    mirrored_motion["object_lin_vel_w"][frame_idx],
                    mirrored_motion["object_ang_vel_w"][frame_idx],
                ]
            )
        else:
            qpos = mirrored_motion["joint_pos"][frame_idx]
            qvel = mirrored_motion["joint_vel"][frame_idx]

        if qpos.shape[0] != model.nq or qvel.shape[0] != model.nv:
            raise ValueError(
                f"Mirrored motion dimensions do not match MJCF {mjcf_path}: "
                f"qpos {qpos.shape[0]} vs model.nq {model.nq}, "
                f"qvel {qvel.shape[0]} vs model.nv {model.nv}"
            )

        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)

        body_pos_w[frame_idx] = data.xpos
        body_quat_w[frame_idx] = data.xquat
        lin_vel_w, ang_vel_w = _world_body_velocities(model, data, mujoco)
        body_lin_vel_w[frame_idx] = lin_vel_w
        body_ang_vel_w[frame_idx] = ang_vel_w

    mirrored_motion["body_pos_w"] = body_pos_w
    mirrored_motion["body_quat_w"] = body_quat_w
    mirrored_motion["body_lin_vel_w"] = body_lin_vel_w
    mirrored_motion["body_ang_vel_w"] = body_ang_vel_w
    return mirrored_motion


def mirror_motion_dict(
    motion_data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    *,
    mirror_axis: str = "y",
    urdf_path: str | None = None,
    motion_path: str | None = None,
) -> dict[str, np.ndarray]:
    """Mirror a holosoma-format motion dictionary while keeping the same name order."""
    if motion_path is None:
        motion_path = _infer_motion_path_from_loaded_motion(motion_data)

    axis_index = _axis_to_index(mirror_axis)
    keys = set(motion_data.files if hasattr(motion_data, "files") else motion_data.keys())
    missing = REQUIRED_MOTION_KEYS - keys
    if missing:
        raise ValueError(f"Unsupported motion format: missing required keys {sorted(missing)}.")

    joint_names = motion_data["joint_names"].tolist()
    body_names = motion_data["body_names"].tolist()
    joint_axes = _load_joint_axes_from_urdf(urdf_path)
    joint_source_indices, joint_signs = _build_joint_mirror_map(joint_names, joint_axes, axis_index)
    body_source_indices = _build_body_mirror_map(body_names)
    object_body_index = _infer_object_body_index(motion_data)
    object_canonical_to_local = None
    if motion_path is not None and object_body_index is not None:
        object_urdf_path = infer_object_urdf_path_from_motion_path(motion_path, motion_data)
        if object_urdf_path is not None:
            object_canonical_to_local = load_object_canonical_to_local_rotation_from_urdf(object_urdf_path)
            if object_canonical_to_local is not None:
                object_canonical_to_local = orient_object_canonical_frame_from_motion(
                    object_canonical_to_local,
                    motion_data,
                )

    mirrored: dict[str, np.ndarray] = {
        key: np.array(motion_data[key], copy=True) for key in (motion_data.files if hasattr(motion_data, "files") else motion_data)
    }

    joint_pos = np.array(motion_data["joint_pos"], copy=True)
    joint_signs_joint_pos = joint_signs.astype(joint_pos.dtype, copy=False)

    joint_pos[:, :3] = reflect_vector(joint_pos[:, :3], axis_index)
    joint_pos[:, 3:7] = reflect_quaternion_wxyz(motion_data["joint_pos"][:, 3:7], axis_index)
    joint_pos[:, 7:] = motion_data["joint_pos"][:, 7:][:, joint_source_indices] * joint_signs_joint_pos[None, :]

    joint_vel = np.empty_like(motion_data["joint_vel"])
    joint_signs_joint_vel = joint_signs.astype(joint_vel.dtype, copy=False)
    joint_vel[:, :3] = reflect_vector(motion_data["joint_vel"][:, :3], axis_index)
    joint_vel[:, 3:6] = reflect_pseudovector(motion_data["joint_vel"][:, 3:6], axis_index)
    joint_vel[:, 6:] = motion_data["joint_vel"][:, 6:][:, joint_source_indices] * joint_signs_joint_vel[None, :]

    mirrored["joint_pos"] = joint_pos
    mirrored["joint_vel"] = joint_vel
    mirrored["body_pos_w"] = reflect_vector(motion_data["body_pos_w"][:, body_source_indices, :], axis_index)
    mirrored["body_quat_w"] = reflect_quaternion_wxyz(motion_data["body_quat_w"][:, body_source_indices, :], axis_index)
    mirrored["body_lin_vel_w"] = reflect_vector(motion_data["body_lin_vel_w"][:, body_source_indices, :], axis_index)
    mirrored["body_ang_vel_w"] = reflect_pseudovector(
        motion_data["body_ang_vel_w"][:, body_source_indices, :], axis_index
    )

    if "object_pos_w" in motion_data:
        mirrored["object_pos_w"] = reflect_vector(motion_data["object_pos_w"], axis_index)
    if "object_quat_w" in motion_data:
        if object_canonical_to_local is None:
            mirrored["object_quat_w"] = reflect_quaternion_wxyz(motion_data["object_quat_w"], axis_index)
        else:
            mirrored["object_quat_w"] = reflect_quaternion_wxyz_in_calibrated_frame(
                motion_data["object_quat_w"],
                axis_index,
                object_canonical_to_local,
            )
    if "object_lin_vel_w" in motion_data:
        mirrored["object_lin_vel_w"] = reflect_vector(motion_data["object_lin_vel_w"], axis_index)
    if "object_ang_vel_w" in motion_data:
        if object_canonical_to_local is None:
            mirrored["object_ang_vel_w"] = reflect_pseudovector(motion_data["object_ang_vel_w"], axis_index)
        else:
            mirrored["object_ang_vel_w"] = reflect_local_pseudovector_in_calibrated_frame(
                motion_data["object_ang_vel_w"],
                object_canonical_to_local,
            )

    _mirror_fixed_scene_metadata(motion_data, mirrored, axis_index=axis_index)

    if object_body_index is not None:
        mirrored_object_body_index = int(np.where(body_source_indices == object_body_index)[0][0])
        mirrored["body_quat_w"][:, mirrored_object_body_index, :] = mirrored["object_quat_w"]

    if motion_path is not None:
        mirrored = recompute_body_tracks_with_mujoco(mirrored, motion_path=motion_path)

    return mirrored


def default_mirrored_output_path(input_path: str) -> Path:
    resolved_input = Path(resolve_data_file_path(input_path))
    return resolved_input.with_name(f"{resolved_input.stem}_mirrored{resolved_input.suffix}")


__all__ = [
    "REQUIRED_MOTION_KEYS",
    "default_mirrored_output_path",
    "infer_urdf_path_from_motion_path",
    "infer_object_urdf_path_from_motion_path",
    "extract_heading_yaw_wxyz",
    "infer_retargeting_mjcf_path_from_motion_path",
    "load_object_canonical_to_local_rotation_from_urdf",
    "orient_object_canonical_frame_from_motion",
    "mirror_heading_angular_velocity",
    "mirror_heading_quaternion_wxyz",
    "mirror_motion_dict",
    "recompute_body_tracks_with_mujoco",
    "reflect_local_pseudovector_in_calibrated_frame",
    "reflect_pseudovector",
    "reflect_quaternion_wxyz",
    "reflect_quaternion_wxyz_in_calibrated_frame",
    "reflect_vector",
]
