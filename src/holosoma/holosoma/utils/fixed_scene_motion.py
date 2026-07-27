from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from holosoma.utils.file_cache import cached_open
from holosoma.utils.path import resolve_data_file_path

FIXED_SCENE_OBJECT_NAMES_KEY = "fixed_scene_object_names"
FIXED_SCENE_OBJECT_GEOM_TYPES_KEY = "fixed_scene_object_geom_types"
FIXED_SCENE_OBJECT_POS_W_KEY = "fixed_scene_object_pos_w"
FIXED_SCENE_OBJECT_QUAT_W_KEY = "fixed_scene_object_quat_w"
FIXED_SCENE_OBJECT_SIZE_KEY = "fixed_scene_object_size"

MOTION_OBJECT_NAME = "object"
MOTION_OBJECT_GEOM_TYPE_KEY = "object_geom_type"
MOTION_OBJECT_USD_PATH_KEY = "object_usd_path"
MOTION_OBJECT_URDF_PATH_KEY = "object_urdf_path"
MOTION_OBJECT_MESH_PATH_KEY = "object_mesh_path"
MOTION_OBJECT_MESH_VERTICES_KEY = "object_mesh_vertices"
MOTION_OBJECT_MESH_FACES_KEY = "object_mesh_faces"
MOTION_OBJECT_BOX_SIZE_KEY = "object_box_size"
MOTION_OBJECT_POS_W_KEY = "object_pos_w"
MOTION_OBJECT_QUAT_W_KEY = "object_quat_w"

FIXED_SCENE_KEYS = {
    FIXED_SCENE_OBJECT_NAMES_KEY,
    FIXED_SCENE_OBJECT_GEOM_TYPES_KEY,
    FIXED_SCENE_OBJECT_POS_W_KEY,
    FIXED_SCENE_OBJECT_QUAT_W_KEY,
    FIXED_SCENE_OBJECT_SIZE_KEY,
}


@dataclass(frozen=True)
class FixedSceneObjectSpec:
    """Object metadata embedded in a motion NPZ."""

    name: str
    geom_type: str
    position_w: np.ndarray
    quat_wxyz: np.ndarray
    size: np.ndarray | None = None
    usd_path: str | None = None
    urdf_path: str | None = None
    mesh_path: str | None = None
    mesh_vertices: np.ndarray | None = None
    mesh_faces: np.ndarray | None = None
    is_motion_object: bool = False


def _decode_string_array(values: np.ndarray, *, key: str) -> list[str]:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"Expected '{key}' to have shape (N,), got {tuple(values.shape)}.")

    decoded: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode())
        else:
            decoded.append(str(value))
    return decoded


def _decode_optional_string(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data.files:
        return None

    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"Expected '{key}' to be a scalar string, got shape {tuple(value.shape)}.")

    scalar = value.item()
    if isinstance(scalar, bytes):
        decoded = scalar.decode()
    else:
        decoded = str(scalar)
    decoded = decoded.strip()
    return decoded or None


def _decode_optional_geom_type(data: np.lib.npyio.NpzFile) -> str | None:
    geom_type = _decode_optional_string(data, MOTION_OBJECT_GEOM_TYPE_KEY)
    return geom_type.lower() if geom_type else None


def _first_pose(data: np.lib.npyio.NpzFile, *, key: str, width: int) -> np.ndarray:
    values = np.asarray(data[key], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != width or values.shape[0] < 1:
        raise ValueError(f"Expected '{key}' to have shape (T, {width}) with T >= 1, got {tuple(values.shape)}.")
    return values[0].copy()


def _path_suffix(path: str | None) -> str:
    if path is None:
        return ""
    return Path(path).suffix.lower()


def _motion_object_asset_from_npz(
    data: np.lib.npyio.NpzFile,
) -> tuple[str, str | None, str | None, str | None]:
    """Return (geom_type, usd_path, urdf_path, mesh_path) for the tracked object asset."""
    usd_path = _decode_optional_string(data, MOTION_OBJECT_USD_PATH_KEY)
    urdf_path = _decode_optional_string(data, MOTION_OBJECT_URDF_PATH_KEY)
    mesh_path = _decode_optional_string(data, MOTION_OBJECT_MESH_PATH_KEY)

    if usd_path:
        return "usd", usd_path, None, mesh_path
    if urdf_path:
        return "urdf", None, urdf_path, mesh_path

    mesh_suffix = _path_suffix(mesh_path)
    if mesh_suffix in {".usd", ".usda", ".usdc"}:
        return "usd", mesh_path, None, mesh_path
    if mesh_suffix == ".urdf":
        return "urdf", None, mesh_path, mesh_path

    geom_type = _decode_optional_geom_type(data)
    if geom_type == "box" and MOTION_OBJECT_BOX_SIZE_KEY in data.files:
        return "box", None, None, mesh_path

    if geom_type == "mesh" or mesh_path is not None:
        hint = f" from '{mesh_path}'" if mesh_path else ""
        raise ValueError(
            "Motion object metadata describes a mesh object but does not provide a supported simulator asset"
            f"{hint}. Provide object_usd_path/object_urdf_path, or store a .usd/.urdf in object_mesh_path."
        )

    return "", None, None, mesh_path


def _motion_object_mesh_arrays_from_npz(
    data: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    has_vertices = MOTION_OBJECT_MESH_VERTICES_KEY in data.files
    has_faces = MOTION_OBJECT_MESH_FACES_KEY in data.files
    if not has_vertices and not has_faces:
        return None, None
    if has_vertices != has_faces:
        missing = MOTION_OBJECT_MESH_FACES_KEY if has_vertices else MOTION_OBJECT_MESH_VERTICES_KEY
        raise ValueError(f"Motion object mesh metadata is incomplete; missing '{missing}'.")

    vertices = np.asarray(data[MOTION_OBJECT_MESH_VERTICES_KEY], dtype=np.float32)
    faces = np.asarray(data[MOTION_OBJECT_MESH_FACES_KEY], dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"Expected '{MOTION_OBJECT_MESH_VERTICES_KEY}' to have shape (N, 3), got {tuple(vertices.shape)}."
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected '{MOTION_OBJECT_MESH_FACES_KEY}' to have shape (M, 3), got {tuple(faces.shape)}.")
    return vertices.copy(), faces.copy()


def _load_motion_object_from_npz(data: np.lib.npyio.NpzFile) -> list[FixedSceneObjectSpec]:
    keys = set(data.files)
    if MOTION_OBJECT_POS_W_KEY not in keys and MOTION_OBJECT_QUAT_W_KEY not in keys:
        return []
    if MOTION_OBJECT_POS_W_KEY not in keys or MOTION_OBJECT_QUAT_W_KEY not in keys:
        raise ValueError(
            "Motion object metadata is incomplete; expected both "
            f"'{MOTION_OBJECT_POS_W_KEY}' and '{MOTION_OBJECT_QUAT_W_KEY}'."
        )

    geom_type, usd_path, urdf_path, mesh_path = _motion_object_asset_from_npz(data)
    if not geom_type:
        return []

    size = None
    if MOTION_OBJECT_BOX_SIZE_KEY in data.files:
        size = np.asarray(data[MOTION_OBJECT_BOX_SIZE_KEY], dtype=np.float32)
        if size.shape != (3,):
            raise ValueError(f"Expected '{MOTION_OBJECT_BOX_SIZE_KEY}' to have shape (3,), got {tuple(size.shape)}.")

    mesh_vertices, mesh_faces = _motion_object_mesh_arrays_from_npz(data)

    return [
        FixedSceneObjectSpec(
            name=MOTION_OBJECT_NAME,
            geom_type=geom_type,
            position_w=_first_pose(data, key=MOTION_OBJECT_POS_W_KEY, width=3),
            quat_wxyz=_first_pose(data, key=MOTION_OBJECT_QUAT_W_KEY, width=4),
            size=size.copy() if size is not None else None,
            usd_path=usd_path,
            urdf_path=urdf_path,
            mesh_path=mesh_path,
            mesh_vertices=mesh_vertices,
            mesh_faces=mesh_faces,
            is_motion_object=True,
        )
    ]


def load_fixed_scene_objects_from_npz(data: np.lib.npyio.NpzFile) -> list[FixedSceneObjectSpec]:
    """Parse scene/object metadata from an opened motion NPZ."""
    keys = set(data.files)
    present_keys = FIXED_SCENE_KEYS & keys
    if not present_keys:
        return _load_motion_object_from_npz(data)
    if present_keys != FIXED_SCENE_KEYS:
        missing = sorted(FIXED_SCENE_KEYS - present_keys)
        raise ValueError(f"Fixed scene object metadata is incomplete; missing keys: {missing}.")

    names = _decode_string_array(data[FIXED_SCENE_OBJECT_NAMES_KEY], key=FIXED_SCENE_OBJECT_NAMES_KEY)
    geom_types = _decode_string_array(data[FIXED_SCENE_OBJECT_GEOM_TYPES_KEY], key=FIXED_SCENE_OBJECT_GEOM_TYPES_KEY)
    positions = np.asarray(data[FIXED_SCENE_OBJECT_POS_W_KEY], dtype=np.float32)
    quats = np.asarray(data[FIXED_SCENE_OBJECT_QUAT_W_KEY], dtype=np.float32)
    sizes = np.asarray(data[FIXED_SCENE_OBJECT_SIZE_KEY], dtype=np.float32)

    num_objects = len(names)
    if len(set(names)) != num_objects:
        raise ValueError(f"Fixed scene object names must be unique, got: {names}.")
    if len(geom_types) != num_objects:
        raise ValueError(
            f"Expected '{FIXED_SCENE_OBJECT_GEOM_TYPES_KEY}' to have {num_objects} entries, got {len(geom_types)}."
        )
    if positions.shape != (num_objects, 3):
        raise ValueError(
            f"Expected '{FIXED_SCENE_OBJECT_POS_W_KEY}' to have shape ({num_objects}, 3), got {tuple(positions.shape)}."
        )
    if quats.shape != (num_objects, 4):
        raise ValueError(
            f"Expected '{FIXED_SCENE_OBJECT_QUAT_W_KEY}' to have shape ({num_objects}, 4), got {tuple(quats.shape)}."
        )
    if sizes.shape != (num_objects, 3):
        raise ValueError(
            f"Expected '{FIXED_SCENE_OBJECT_SIZE_KEY}' to have shape ({num_objects}, 3), got {tuple(sizes.shape)}."
        )

    fixed_scene_specs = [
        FixedSceneObjectSpec(
            name=name,
            geom_type=geom_type,
            position_w=positions[index].copy(),
            quat_wxyz=quats[index].copy(),
            size=sizes[index].copy(),
            is_motion_object=False,
        )
        for index, (name, geom_type) in enumerate(zip(names, geom_types))
    ]
    return fixed_scene_specs + _load_motion_object_from_npz(data)


def load_fixed_scene_objects_from_motion_npz(motion_path: str | Path) -> list[FixedSceneObjectSpec]:
    """Load fixed scene object metadata from a motion NPZ path."""
    resolved = resolve_data_file_path(str(motion_path))
    with cached_open(resolved, "rb") as file_handle, np.load(file_handle, allow_pickle=False) as data:
        return load_fixed_scene_objects_from_npz(data)


def list_motion_npz_files(motion_dir: str | Path, *, recursive: bool = False) -> list[Path]:
    """List motion NPZ files from one or more directories.

    The input may be a single directory or a comma-separated list of directories.
    Paths are resolved with the same semantics as motion loading.
    """
    directories = [entry.strip() for entry in str(motion_dir).split(",") if entry.strip()]
    motion_files: list[Path] = []
    for directory in directories:
        expanded = resolve_data_file_path(os.path.expanduser(directory))
        path = Path(expanded)
        files = path.rglob("*.npz") if recursive else path.glob("*.npz")
        motion_files.extend(sorted(files))
    return motion_files


def _fixed_scene_specs_match(
    reference_specs: list[FixedSceneObjectSpec], candidate_specs: list[FixedSceneObjectSpec], *, atol: float = 1.0e-6
) -> bool:
    if len(reference_specs) != len(candidate_specs):
        return False

    reference_by_name = {spec.name: spec for spec in reference_specs}
    candidate_by_name = {spec.name: spec for spec in candidate_specs}
    if reference_by_name.keys() != candidate_by_name.keys():
        return False

    for name, reference_spec in reference_by_name.items():
        candidate_spec = candidate_by_name[name]
        if not _single_scene_spec_matches(reference_spec, candidate_spec, atol=atol):
            return False

    return True


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _optional_arrays_close(first: np.ndarray | None, second: np.ndarray | None, *, atol: float) -> bool:
    if first is None or second is None:
        return first is None and second is None
    return np.allclose(first, second, atol=atol)


def _single_scene_spec_matches(
    reference_spec: FixedSceneObjectSpec, candidate_spec: FixedSceneObjectSpec, *, atol: float = 1.0e-6
) -> bool:
    shared_asset_matches = (
        reference_spec.geom_type == candidate_spec.geom_type
        and reference_spec.usd_path == candidate_spec.usd_path
        and reference_spec.urdf_path == candidate_spec.urdf_path
        and reference_spec.mesh_path == candidate_spec.mesh_path
        and reference_spec.is_motion_object == candidate_spec.is_motion_object
        and _optional_arrays_close(reference_spec.size, candidate_spec.size, atol=atol)
    )
    if not shared_asset_matches:
        return False
    if reference_spec.is_motion_object:
        return True
    return (
        np.allclose(reference_spec.position_w, candidate_spec.position_w, atol=atol)
        and np.allclose(reference_spec.quat_wxyz, candidate_spec.quat_wxyz, atol=atol)
    )


def load_fixed_scene_objects_from_motion_source(
    *, motion_file: str | Path = "", motion_dir: str | Path = "", recursive: bool = False
) -> list[FixedSceneObjectSpec]:
    """Load scene/object metadata from a motion file or a motion directory.

    When ``motion_dir`` is provided, all motion NPZs must carry identical fixed-scene
    and tracked-object asset metadata (or all omit it). This mirrors the current
    simulator assumption that the instantiated scene is shared across all environments
    for a given task.
    """
    if motion_dir:
        motion_files = list_motion_npz_files(motion_dir, recursive=recursive)
        if not motion_files:
            raise ValueError(f"No .npz files found in motion_dir '{motion_dir}'.")

        reference_path: Path | None = None
        reference_specs: list[FixedSceneObjectSpec] | None = None
        for candidate_path in motion_files:
            candidate_specs = load_fixed_scene_objects_from_motion_npz(candidate_path)
            if reference_specs is None:
                reference_specs = candidate_specs
                reference_path = candidate_path
                continue
            if not _fixed_scene_specs_match(reference_specs, candidate_specs):
                raise ValueError(
                    "Fixed scene metadata must match across all motion files when "
                    f"load_fixed_scene_from_motion is enabled. Mismatch between "
                    f"'{reference_path}' and '{candidate_path}'."
                )
        return reference_specs or []

    if motion_file:
        return load_fixed_scene_objects_from_motion_npz(motion_file)

    return []


def load_fixed_scene_objects_from_motion_config(motion_cfg: Any) -> list[FixedSceneObjectSpec]:
    """Load scene/object metadata from a motion config."""
    if motion_cfg is None:
        return []

    motion_file = str(_cfg_get(motion_cfg, "motion_file", "") or "")
    motion_dir = str(_cfg_get(motion_cfg, "motion_dir", "") or "")
    if not motion_file and not motion_dir:
        return []
    return load_fixed_scene_objects_from_motion_source(
        motion_file=motion_file,
        motion_dir=motion_dir,
        recursive=False,
    )
