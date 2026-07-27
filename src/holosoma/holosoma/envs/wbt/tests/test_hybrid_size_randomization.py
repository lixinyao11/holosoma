from __future__ import annotations

from dataclasses import replace

import torch

from holosoma.config_types.command import HybridSizeRandomizationConfig
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt_hybrid_climb,
    g1_29dof_wbt_hybrid_object,
)
from holosoma.envs.wbt.hybrid_size_randomization import (
    build_fixed_hybrid_track_mask,
    collision_bounds_corners,
    correct_object_positions_for_ground,
    expand_size_scalars,
    load_urdf_collision_bounds,
    rotated_scaled_lower_bound_z,
    sample_fixed_hybrid_size_scales,
)
from holosoma.envs.wbt.wbt_manager import (
    _inject_fixed_scene_objects_from_motion,
    _prepare_hybrid_size_randomization,
)
from holosoma.utils.rotations import quat_from_euler_xyz


def _enable_size_randomization(experiment_cfg, *, seed: int = 0):
    motion_term = experiment_cfg.command.setup_terms["motion_command"]
    motion_cfg = motion_term.params["motion_config"]
    size_cfg = replace(motion_cfg.hybrid.size_randomization, enabled=True, sampling_seed=seed)
    updated_motion_cfg = replace(motion_cfg, hybrid=replace(motion_cfg.hybrid, size_randomization=size_cfg))
    updated_motion_term = replace(motion_term, params={**motion_term.params, "motion_config": updated_motion_cfg})
    updated_command = replace(
        experiment_cfg.command,
        setup_terms={**experiment_cfg.command.setup_terms, "motion_command": updated_motion_term},
    )
    return replace(experiment_cfg, command=updated_command, training=replace(experiment_cfg.training, num_envs=32))


def test_partition_and_scale_sampling_are_reproducible_and_independent():
    cfg = HybridSizeRandomizationConfig(enabled=True, sampling_seed=11)
    track_mask = build_fixed_hybrid_track_mask(4096, 0.5, 7)
    same_partition = build_fixed_hybrid_track_mask(4096, 0.5, 7)
    different_partition = build_fixed_hybrid_track_mask(4096, 0.5, 8)

    samples = sample_fixed_hybrid_size_scales(track_mask, cfg)
    repeated = sample_fixed_hybrid_size_scales(same_partition, cfg)
    changed_seed = sample_fixed_hybrid_size_scales(
        track_mask,
        replace(cfg, sampling_seed=12),
    )

    assert torch.equal(track_mask, same_partition)
    assert not torch.equal(track_mask, different_partition)
    assert torch.equal(samples, repeated)
    assert not torch.equal(samples, changed_seed)
    assert torch.all((samples[track_mask] >= 0.95) & (samples[track_mask] <= 1.05))
    assert torch.all((samples[~track_mask] >= 0.85) & (samples[~track_mask] <= 1.15))
    assert torch.unique(samples).numel() > 4000


def test_disabled_size_randomization_is_an_exact_noop():
    assert _prepare_hybrid_size_randomization(g1_29dof_wbt_hybrid_object) is g1_29dof_wbt_hybrid_object
    assert _prepare_hybrid_size_randomization(g1_29dof_wbt_hybrid_climb) is g1_29dof_wbt_hybrid_climb


def test_object_size_randomization_converts_legacy_actor_once():
    experiment_cfg = _enable_size_randomization(g1_29dof_wbt_hybrid_object, seed=5)
    injected_cfg = _inject_fixed_scene_objects_from_motion(experiment_cfg)
    updated_cfg = _prepare_hybrid_size_randomization(injected_cfg)

    rigid_objects = updated_cfg.simulator.config.scene.rigid_objects
    assert rigid_objects is not None
    assert [obj.name for obj in rigid_objects].count("object") == 1
    assert updated_cfg.robot.object.object_urdf_path is None
    assert updated_cfg.simulator.config.scene.replicate_physics is False

    object_cfg = next(obj for obj in rigid_objects if obj.name == "object")
    scales = torch.tensor(object_cfg.per_env_scales)
    assert scales.shape == (32, 3)
    assert torch.equal(scales[:, 0], scales[:, 1])
    assert torch.equal(scales[:, 1], scales[:, 2])
    assert object_cfg.per_env_positions == [[0.0, 0.0, 0.5] for _ in range(32)]
    assert object_cfg.local_collision_bounds is not None
    assert object_cfg.physics is not None
    assert object_cfg.physics.mass is None
    assert object_cfg.physics.density is None


def test_climb_size_randomization_changes_height_and_preserves_ground_clearance():
    experiment_cfg = _enable_size_randomization(g1_29dof_wbt_hybrid_climb, seed=3)
    injected_cfg = _inject_fixed_scene_objects_from_motion(experiment_cfg)
    updated_cfg = _prepare_hybrid_size_randomization(injected_cfg)

    rigid_objects = updated_cfg.simulator.config.scene.rigid_objects
    assert rigid_objects is not None
    assert [obj.name for obj in rigid_objects].count("climb_box") == 1
    climb_cfg = next(obj for obj in rigid_objects if obj.name == "climb_box")
    scales = torch.tensor(climb_cfg.per_env_scales)
    positions = torch.tensor(climb_cfg.per_env_positions)

    assert updated_cfg.simulator.config.scene.replicate_physics is False
    assert climb_cfg.physics is not None and climb_cfg.physics.kinematic_enabled is True
    assert torch.equal(scales[:, :2], torch.ones(32, 2))
    assert torch.allclose(positions[:, 2] - 0.25 * scales[:, 2], torch.full((32,), 0.001))
    assert torch.allclose(positions[:, :2], torch.tensor([2.7, 0.0]).repeat(32, 1))

    endpoint_scales = torch.tensor([0.85, 1.15])
    endpoint_centers = 0.001 + 0.25 * endpoint_scales
    assert torch.allclose(endpoint_centers - 0.25 * endpoint_scales, torch.full((2,), 0.001))


def test_object_collision_ground_correction_handles_scale_rotation_and_noise():
    collision_min, collision_max = load_urdf_collision_bounds(
        "holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
    )
    corners = collision_bounds_corners(torch.tensor([collision_min, collision_max], dtype=torch.float32))
    scales = expand_size_scalars(torch.tensor([0.85, 1.0, 1.15]), "uniform_xyz")
    reference_quat = quat_from_euler_xyz(
        torch.tensor([0.0, 0.7, 1.1]),
        torch.tensor([0.0, -0.4, 0.2]),
        torch.tensor([0.0, 0.3, -0.6]),
    )
    canonical_lower = rotated_scaled_lower_bound_z(corners, torch.ones_like(scales), reference_quat)
    reference_pos = torch.zeros(3, 3)
    reference_pos[:, 2] = -canonical_lower

    noisy_quat = quat_from_euler_xyz(
        torch.tensor([0.1, 0.6, 1.0]),
        torch.tensor([0.0, -0.3, 0.1]),
        torch.tensor([0.2, 0.4, -0.5]),
    )
    candidate_pos = reference_pos + torch.tensor([[0.2, -0.1, 0.01]])
    corrected = correct_object_positions_for_ground(
        candidate_pos,
        noisy_quat,
        reference_pos,
        reference_quat,
        scales,
        corners,
        torch.zeros(3),
        clearance=0.001,
        contact_tolerance=0.02,
    )
    corrected_bottom = corrected[:, 2] + rotated_scaled_lower_bound_z(corners, scales, noisy_quat)

    assert torch.allclose(corrected[:, :2], candidate_pos[:, :2])
    assert torch.allclose(corrected_bottom, torch.full((3,), 0.001), atol=1.0e-6)


def test_airborne_object_center_is_unchanged_unless_scaled_geometry_penetrates():
    bounds = torch.tensor([[-0.2, -0.2, -0.2], [0.2, 0.2, 0.2]], dtype=torch.float32)
    corners = collision_bounds_corners(bounds)
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    scales = torch.tensor([[1.15, 1.15, 1.15]], dtype=torch.float32)
    reference_pos = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32)

    safe_candidate = torch.tensor([[0.0, 0.0, 0.4]], dtype=torch.float32)
    safe_corrected = correct_object_positions_for_ground(
        safe_candidate,
        quat,
        reference_pos,
        quat,
        scales,
        corners,
        torch.zeros(1),
        clearance=0.001,
        contact_tolerance=0.02,
    )
    assert torch.equal(safe_corrected, safe_candidate)

    penetrating_candidate = torch.tensor([[0.0, 0.0, 0.1]], dtype=torch.float32)
    penetrating_corrected = correct_object_positions_for_ground(
        penetrating_candidate,
        quat,
        reference_pos,
        quat,
        scales,
        corners,
        torch.zeros(1),
        clearance=0.001,
        contact_tolerance=0.02,
    )
    bottom = penetrating_corrected[:, 2] + rotated_scaled_lower_bound_z(corners, scales, quat)
    assert torch.allclose(bottom, torch.tensor([0.001]), atol=1.0e-6)
