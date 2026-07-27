"""Shared helpers for fixed hybrid environment partitioning and size randomization."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from holosoma.config_types.command import HybridSizeRandomizationConfig
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import quat_apply


def build_fixed_hybrid_track_mask(num_envs: int, track_ratio: float, partition_seed: int) -> torch.Tensor:
    """Return the canonical CPU track mask shared by setup-time and runtime code."""
    if num_envs < 0:
        raise ValueError(f"num_envs must be non-negative, got {num_envs}.")
    if not 0.0 <= track_ratio <= 1.0:
        raise ValueError(f"Hybrid track_ratio must be in [0, 1], got {track_ratio}.")

    num_track_envs = max(0, min(num_envs, int(round(num_envs * track_ratio))))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(partition_seed))
    shuffled_env_ids = torch.randperm(num_envs, generator=generator, device="cpu")
    track_mask = torch.zeros(num_envs, dtype=torch.bool, device="cpu")
    track_mask[shuffled_env_ids[:num_track_envs]] = True
    return track_mask


def sample_fixed_hybrid_size_scales(
    track_mask: torch.Tensor,
    cfg: HybridSizeRandomizationConfig,
) -> torch.Tensor:
    """Sample one reproducible scalar per environment from a dedicated CPU stream."""
    track_mask = track_mask.to(device="cpu", dtype=torch.bool)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.sampling_seed))
    unit_samples = torch.rand(track_mask.numel(), generator=generator, dtype=torch.float32, device="cpu")

    track_min, track_max = (float(value) for value in cfg.track_scale_range)
    gen_min, gen_max = (float(value) for value in cfg.gen_scale_range)
    track_samples = track_min + unit_samples * (track_max - track_min)
    gen_samples = gen_min + unit_samples * (gen_max - gen_min)
    return torch.where(track_mask, track_samples, gen_samples)


def expand_size_scalars(size_scalars: torch.Tensor, mode: str) -> torch.Tensor:
    """Convert scalar samples to per-axis USD scale vectors."""
    if mode == "uniform_xyz":
        return size_scalars.unsqueeze(-1).repeat(1, 3)
    if mode == "height_only":
        scales = torch.ones(size_scalars.numel(), 3, dtype=size_scalars.dtype, device=size_scalars.device)
        scales[:, 2] = size_scalars
        return scales
    raise ValueError(f"Unsupported hybrid size randomization mode: {mode!r}.")


def _parse_xyz(raw_value: str | None, *, default: tuple[float, float, float]) -> torch.Tensor:
    if raw_value is None:
        return torch.tensor(default, dtype=torch.float64)
    values = [float(value) for value in raw_value.split()]
    if len(values) != 3:
        raise ValueError(f"Expected three values, got {raw_value!r}.")
    return torch.tensor(values, dtype=torch.float64)


def _rpy_rotation_matrix(rpy: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float64,
    )


def _obj_vertices(mesh_path: Path) -> torch.Tensor:
    vertices: list[list[float]] = []
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as mesh_file:
        for line in mesh_file:
            line = line.lstrip()
            if not line.startswith("v "):
                continue
            values = line.split()
            if len(values) >= 4:
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
    if not vertices:
        raise ValueError(f"Collision mesh contains no OBJ vertices: {mesh_path}")
    return torch.tensor(vertices, dtype=torch.float64)


def load_urdf_collision_bounds(urdf_path: str) -> tuple[list[float], list[float]]:
    """Compute the canonical AABB from OBJ meshes or boxes in a URDF's collision elements."""
    resolved_urdf_path = Path(resolve_data_file_path(urdf_path))
    root = ET.parse(resolved_urdf_path).getroot()
    collision_vertices: list[torch.Tensor] = []

    for collision in root.findall(".//collision"):
        origin = collision.find("origin")
        translation = _parse_xyz(origin.get("xyz") if origin is not None else None, default=(0.0, 0.0, 0.0))
        rpy = _parse_xyz(origin.get("rpy") if origin is not None else None, default=(0.0, 0.0, 0.0))
        rotation = _rpy_rotation_matrix(rpy)
        geometry = collision.find("geometry")
        if geometry is None:
            continue

        mesh = geometry.find("mesh")
        box = geometry.find("box")
        if mesh is not None:
            filename = mesh.get("filename")
            if not filename:
                raise ValueError(f"Collision mesh in {resolved_urdf_path} is missing filename.")
            if filename.startswith("package://"):
                raise ValueError(f"Package URI collision meshes are not supported for size bounds: {filename}")
            mesh_path = Path(filename)
            if not mesh_path.is_absolute():
                mesh_path = resolved_urdf_path.parent / mesh_path
            vertices = _obj_vertices(mesh_path)
            mesh_scale = _parse_xyz(mesh.get("scale"), default=(1.0, 1.0, 1.0))
            vertices = vertices * mesh_scale
        elif box is not None:
            half_size = 0.5 * _parse_xyz(box.get("size"), default=(0.0, 0.0, 0.0))
            signs = torch.tensor(
                [
                    [-1.0, -1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, -1.0, -1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, -1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=torch.float64,
            )
            vertices = signs * half_size
        else:
            raise ValueError(
                f"Only OBJ mesh and box collision geometry is supported for hybrid size bounds: {resolved_urdf_path}"
            )
        collision_vertices.append(vertices @ rotation.T + translation)

    if not collision_vertices:
        raise ValueError(f"URDF contains no supported collision geometry: {resolved_urdf_path}")
    vertices = torch.cat(collision_vertices, dim=0)
    return vertices.amin(dim=0).tolist(), vertices.amax(dim=0).tolist()


def collision_bounds_corners(bounds: torch.Tensor) -> torch.Tensor:
    """Expand a ``[min, max]`` AABB tensor to its eight corner points."""
    if bounds.shape != (2, 3):
        raise ValueError(f"Expected collision bounds with shape (2, 3), got {tuple(bounds.shape)}.")
    lower, upper = bounds[0], bounds[1]
    return torch.stack(
        [
            torch.stack([x, y, z])
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ]
    )


def rotated_scaled_lower_bound_z(
    collision_corners: torch.Tensor,
    scales_xyz: torch.Tensor,
    quaternions_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Return each pose's lowest scaled, rotated collision point relative to its center."""
    scaled_corners = collision_corners.unsqueeze(0) * scales_xyz.unsqueeze(1)
    repeated_quaternions = quaternions_xyzw.unsqueeze(1).expand(-1, collision_corners.shape[0], -1)
    rotated_corners = quat_apply(
        repeated_quaternions.reshape(-1, 4),
        scaled_corners.reshape(-1, 3),
        w_last=True,
    ).reshape(scales_xyz.shape[0], collision_corners.shape[0], 3)
    return rotated_corners[..., 2].amin(dim=1)


def correct_object_positions_for_ground(
    candidate_positions_w: torch.Tensor,
    candidate_quaternions_xyzw: torch.Tensor,
    reference_positions_w: torch.Tensor,
    reference_quaternions_xyzw: torch.Tensor,
    scales_xyz: torch.Tensor,
    collision_corners: torch.Tensor,
    ground_z: torch.Tensor,
    *,
    clearance: float,
    contact_tolerance: float,
) -> torch.Tensor:
    """Correct ground-contact poses and clamp only penetrating airborne poses."""
    canonical_scales = torch.ones_like(scales_xyz)
    reference_lower = rotated_scaled_lower_bound_z(
        collision_corners,
        canonical_scales,
        reference_quaternions_xyzw,
    )
    candidate_lower = rotated_scaled_lower_bound_z(
        collision_corners,
        scales_xyz,
        candidate_quaternions_xyzw,
    )
    reference_bottom = reference_positions_w[:, 2] + reference_lower
    candidate_bottom = candidate_positions_w[:, 2] + candidate_lower
    required_shift = ground_z + float(clearance) - candidate_bottom
    is_ground_contact = reference_bottom <= ground_z + float(contact_tolerance)
    shift = torch.where(is_ground_contact, required_shift, torch.clamp_min(required_shift, 0.0))
    corrected = candidate_positions_w.clone()
    corrected[:, 2] += shift
    return corrected
