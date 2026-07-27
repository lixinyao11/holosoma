"""Helpers for trimming holosoma-format motion NPZ files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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

FIXED_SCENE_KEYS = {
    "fixed_scene_object_names",
    "fixed_scene_object_geom_types",
    "fixed_scene_object_pos_w",
    "fixed_scene_object_quat_w",
    "fixed_scene_object_size",
}

MOTION_OBJECT_GEOM_TYPE_KEY = "object_geom_type"
MOTION_OBJECT_USD_PATH_KEY = "object_usd_path"
MOTION_OBJECT_URDF_PATH_KEY = "object_urdf_path"
MOTION_OBJECT_MESH_PATH_KEY = "object_mesh_path"
MOTION_OBJECT_MESH_VERTICES_KEY = "object_mesh_vertices"
MOTION_OBJECT_MESH_FACES_KEY = "object_mesh_faces"
MOTION_OBJECT_BOX_SIZE_KEY = "object_box_size"


FRAME_INDEXED_KEYS = {
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "object_pos_w",
    "object_quat_w",
    "object_lin_vel_w",
    "object_ang_vel_w",
}

NON_FRAME_INDEXED_KEYS = {
    "fps",
    "source_fps",
    "joint_names",
    "body_names",
    "source_format",
    "source_motion_key",
    MOTION_OBJECT_GEOM_TYPE_KEY,
    MOTION_OBJECT_BOX_SIZE_KEY,
    MOTION_OBJECT_MESH_VERTICES_KEY,
    MOTION_OBJECT_MESH_FACES_KEY,
    MOTION_OBJECT_MESH_PATH_KEY,
    MOTION_OBJECT_USD_PATH_KEY,
    MOTION_OBJECT_URDF_PATH_KEY,
    *FIXED_SCENE_KEYS,
}


def resolve_trim_bounds(num_frames: int, start: int | None = 0, end: int | None = None) -> tuple[int, int]:
    """Resolve user trim bounds to a non-empty ``[start, end)`` frame interval."""
    if num_frames <= 0:
        raise ValueError(f"Cannot trim a motion with non-positive frame count: {num_frames}.")

    start_idx = 0 if start is None else int(start)
    end_idx = num_frames if end is None else int(end)

    if start_idx < 0:
        start_idx += num_frames
    if end_idx < 0:
        end_idx += num_frames

    if not 0 <= start_idx <= num_frames:
        raise ValueError(f"Trim start must resolve into [0, {num_frames}], got {start}.")
    if not 0 <= end_idx <= num_frames:
        raise ValueError(f"Trim end must resolve into [0, {num_frames}], got {end}.")
    if start_idx >= end_idx:
        raise ValueError(
            f"Trim range must keep at least one frame; resolved range was [{start_idx}, {end_idx}) "
            f"for a {num_frames}-frame motion."
        )

    return start_idx, end_idx


def _motion_keys(motion_data: dict[str, np.ndarray] | np.lib.npyio.NpzFile) -> list[str]:
    return list(motion_data.files if hasattr(motion_data, "files") else motion_data.keys())


def _is_frame_indexed_key(key: str, value: np.ndarray, num_frames: int) -> bool:
    if key in FRAME_INDEXED_KEYS:
        return True
    if key in NON_FRAME_INDEXED_KEYS or key.startswith("fixed_scene_"):
        return False
    return value.ndim > 0 and value.shape[0] == num_frames


def _validate_frame_indexed_arrays(
    motion_data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    num_frames: int,
) -> None:
    keys = set(_motion_keys(motion_data))
    for key in sorted(FRAME_INDEXED_KEYS & keys):
        value = np.asarray(motion_data[key])
        if value.ndim == 0 or value.shape[0] != num_frames:
            raise ValueError(
                f"Frame-indexed key '{key}' must have first dimension {num_frames}, got shape {value.shape}."
            )


def trim_motion_dict(
    motion_data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    *,
    start: int | None = 0,
    end: int | None = None,
) -> dict[str, np.ndarray]:
    """Trim a holosoma-format motion dictionary to ``[start, end)`` frames.

    Negative ``start`` or ``end`` values are resolved relative to the total frame
    count, so ``end=-10`` drops the last ten frames.
    """
    keys = _motion_keys(motion_data)
    missing = REQUIRED_MOTION_KEYS - set(keys)
    if missing:
        raise ValueError(f"Unsupported motion format: missing required keys {sorted(missing)}.")

    joint_pos = np.asarray(motion_data["joint_pos"])
    if joint_pos.ndim == 0:
        raise ValueError("Motion key 'joint_pos' must be frame-indexed.")

    num_frames = int(joint_pos.shape[0])
    start_idx, end_idx = resolve_trim_bounds(num_frames, start=start, end=end)
    _validate_frame_indexed_arrays(motion_data, num_frames)

    trimmed: dict[str, np.ndarray] = {}
    for key in keys:
        value = np.array(motion_data[key], copy=True)
        if _is_frame_indexed_key(key, value, num_frames):
            trimmed[key] = value[start_idx:end_idx].copy()
        else:
            trimmed[key] = value

    return trimmed


def default_trimmed_output_path(input_path: str) -> Path:
    resolved_input = Path(resolve_data_file_path(input_path))
    return resolved_input.with_name(f"{resolved_input.stem}_trim{resolved_input.suffix}")


__all__ = [
    "FRAME_INDEXED_KEYS",
    "NON_FRAME_INDEXED_KEYS",
    "default_trimmed_output_path",
    "resolve_trim_bounds",
    "trim_motion_dict",
]
