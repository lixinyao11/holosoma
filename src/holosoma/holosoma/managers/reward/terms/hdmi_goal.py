"""HDMI-inspired rewards for the opt-in hybrid goal preset.

No actor reference input, no simulator-specific contact claims. Stable arrival is
not a release/support detector and must not be reported as placement success.
"""
from __future__ import annotations

import math
import re
import torch
from holosoma.managers.reward.base import RewardTermBase
from holosoma.managers.reward.terms import wbt
from holosoma.utils.rotations import quat_error_magnitude, quat_apply, quat_inverse
from holosoma.envs.wbt.hybrid_size_randomization import load_urdf_collision_vertices


def track_only(env, reference_term: str, sigma: float):
    """Reuse original reward math, but remove the gen group's constant baseline."""
    allowed = {
        "motion_global_ref_position_error_exp", "motion_global_ref_orientation_error_exp",
        "motion_relative_body_position_error_exp", "motion_relative_body_orientation_error_exp",
        "motion_global_body_lin_vel", "motion_global_body_ang_vel",
        "object_global_ref_position_error_exp", "object_global_ref_orientation_error_exp",
    }
    if reference_term not in allowed:
        raise ValueError(f"Unsupported reference reward: {reference_term}")
    cmd = wbt._get_motion_command_and_assert_type(env)
    values = getattr(wbt, reference_term)(env, sigma)
    return torch.where(cmd.track_env_mask, values, torch.zeros_like(values))


def joint_tracking(env, sigma: float, velocity: bool = False,
                   joint_pattern: str = r"(waist|left_hip|right_hip|left_knee|right_knee|left_shoulder|right_shoulder|left_elbow|right_elbow|left_wrist|right_wrist).*_joint"):
    """HDMI's exp(-mean(abs(joint error))/sigma), in simulator joint order."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    cmd = wbt._get_motion_command_and_assert_type(env)
    indices = [i for i, name in enumerate(env.simulator.dof_names) if re.fullmatch(joint_pattern, name)]
    if not indices:
        raise ValueError(f"No joints match {joint_pattern}")
    target = cmd.joint_vel if velocity else cmd.joint_pos
    actual = cmd.robot_joint_vel if velocity else cmd.robot_joint_pos
    error = (target[:, indices] - actual[:, indices]).abs().mean(-1)
    values = torch.exp(-error / sigma)
    return torch.where(cmd.track_env_mask, values, torch.zeros_like(values))


def joint_speed_penalty(env):
    return torch.sum(env.simulator.dof_vel.square(), dim=-1)


def goal_errors(cmd):
    position = torch.linalg.vector_norm(cmd.goal_object_pos_w - cmd.simulator_object_pos_w, dim=-1)
    orientation = quat_error_magnitude(cmd.goal_object_quat_w, wbt._get_goal_conditioning_object_quat_w(cmd))
    return position, orientation


class GoalProgress(RewardTermBase):
    """Reward only new best goal proximity, never repeated approach/retreat cycles.

    Dimensionless error combines position/sigma and orientation/sigma. First call
    establishes a baseline. Reset and goal-version changes cannot earn progress.
    This is shaping, not a claim of potential-based policy invariance.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.pos_scale = float(cfg.params.get("pos_scale", 1.0))
        self.ori_scale = float(cfg.params.get("ori_scale", math.pi))
        if min(self.pos_scale, self.ori_scale, float(env.dt)) <= 0:
            raise ValueError("Scales and dt must be positive")
        self.best = torch.full((env.num_envs,), float('inf'), device=env.device)
        self.version = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.best[ids] = float('inf')
        self.version[ids] = -1

    def __call__(self, env, **kwargs):
        cmd = wbt._get_motion_command_and_assert_type(env)
        pos, ori = goal_errors(cmd)
        error = pos / self.pos_scale + ori / self.ori_scale
        active = cmd.gen_env_mask
        continuing = (self.version == cmd.goal_version) & torch.isfinite(self.best) & active
        progress = torch.where(continuing, (self.best - error).clamp_min(0), torch.zeros_like(error))
        self.best.copy_(torch.where(continuing, torch.minimum(self.best, error), error).detach())
        self.version.copy_(cmd.goal_version)
        return progress / float(env.dt)  # reward manager multiplies by dt once


class ObjectGroundClearance:
    """Lowest transformed collision vertex above the flat environment ground.

    Single rigid object only. Uses collision vertices, not rotated AABB corners.
    Geometry is loaded once; chunked dot products avoid N*V*3 temporaries.
    This is a geometric height test, not a support/contact-force test.
    """
    def __init__(self, env):
        obj = env.simulator.robot_config.object
        if obj.object_urdf_paths:
            raise ValueError("Ground-height reward does not yet support multiple object variants.")
        self.vertices = load_urdf_collision_vertices(obj.object_urdf_path, single_link=True).to(
            device=env.device, dtype=torch.float32).unique(dim=0)
        if not torch.isfinite(self.vertices).all():
            raise ValueError("Non-finite collision vertices")
        # Reward terms are constructed before CommandManager. Resolve scales on
        # first evaluation, after command setup and startup randomization.
        self.scales = None
        self.ground_z = env.simulator.scene.env_origins[:, 2]

    def __call__(self, env, cmd):
        if self.scales is None:
            size_cfg = cmd.hybrid_cfg.size_randomization
            if size_cfg.enabled:
                self.scales = env.simulator.get_actor_scale_factors("object").to(env.device)
            else:
                self.scales = torch.ones(env.num_envs, 3, device=env.device)
        up = torch.zeros_like(cmd.simulator_object_pos_w)
        up[:, 2] = 1.
        # z(R * (scale * vertex)) = (R^-1 * up * scale) dot vertex.
        local_up = quat_apply(quat_inverse(cmd.simulator_object_quat_w, w_last=True), up, w_last=True)
        direction = local_up * self.scales
        bottom = torch.full((env.num_envs,), float('inf'), device=env.device)
        for vertices in self.vertices.split(1024):
            bottom = torch.minimum(bottom, (direction @ vertices.T).amin(dim=1))
        return cmd.simulator_object_pos_w[:, 2] + bottom - self.ground_z


class GoalSettledOnce(RewardTermBase):
    """Goal XY/orientation + bottom <= 5cm above ground + low speeds for consecutive steps.

    One bonus per goal, gen only. No hand-release or ground-support assertion.
    Kept under goal_object_reached_once in the preset so the existing curriculum
    uses this stricter event rather than the old pose-only success criterion.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        self.xy_threshold = float(p.get("xy_threshold", 0.2))
        self.max_bottom_height = float(p.get("max_bottom_height", 0.05))
        self.ground_clearance = ObjectGroundClearance(env)
        self.ori_threshold = float(p.get("ori_threshold", math.pi / 6))
        self.lin_vel_threshold = float(p.get("lin_vel_threshold", 0.05))
        self.ang_vel_threshold = float(p.get("ang_vel_threshold", 0.2))
        self.hold_seconds = float(p.get("hold_seconds", 0.5))
        self.bonus = float(p.get("bonus", 3.0))
        if min(self.xy_threshold, self.max_bottom_height, self.ori_threshold, self.lin_vel_threshold,
               self.ang_vel_threshold, self.hold_seconds, self.bonus, float(env.dt)) <= 0:
            raise ValueError("Goal-settling thresholds, bonus, and dt must be positive")
        self.required_steps = math.ceil(self.hold_seconds / float(env.dt) - 1e-9)
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.version = torch.full_like(self.count, -1)
        self.awarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.count[ids] = 0
        self.version[ids] = -1
        self.awarded[ids] = False

    def __call__(self, env, **kwargs):
        cmd = wbt._get_motion_command_and_assert_type(env)
        changed = self.version != cmd.goal_version
        self.count[changed] = 0
        self.awarded[changed] = False
        self.version.copy_(cmd.goal_version)
        _, ori = goal_errors(cmd)
        pos = torch.linalg.vector_norm(cmd.goal_object_pos_w[:, :2] - cmd.simulator_object_pos_w[:, :2], dim=-1)
        bottom_height = self.ground_clearance(env, cmd)
        stable = (cmd.gen_env_mask & (pos <= self.xy_threshold)
                  & torch.isfinite(bottom_height) & (bottom_height <= self.max_bottom_height) & (ori <= self.ori_threshold)
                  & (torch.linalg.vector_norm(cmd.simulator_object_lin_vel_w, dim=-1) <= self.lin_vel_threshold)
                  & (torch.linalg.vector_norm(cmd.simulator_object_ang_vel_w, dim=-1) <= self.ang_vel_threshold))
        # A terminal failure must not be labeled a successful stable arrival.
        if hasattr(env, 'reset_buf') and hasattr(env, 'time_out_buf'):
            stable = stable & ~(env.reset_buf.bool() & ~env.time_out_buf.bool())
        self.count.copy_(torch.where(stable, self.count + 1, torch.zeros_like(self.count)))
        reached = (self.count >= self.required_steps) & ~self.awarded
        self.awarded |= reached
        return reached.to(pos.dtype) * self.bonus / float(env.dt)
