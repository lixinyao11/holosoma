"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_error_magnitude, yaw_quat

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


def _get_hybrid_masks(motion_command: MotionCommand) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(motion_command, "track_env_mask") or not hasattr(motion_command, "gen_env_mask"):
        raise RuntimeError("Hybrid reward term requires HybridMotionCommand with track/gen masks.")
    return motion_command.track_env_mask, motion_command.gen_env_mask


def _get_goal_orientation_cfg(motion_command: MotionCommand):
    motion_cfg = getattr(motion_command, "motion_cfg", None)
    return getattr(motion_cfg, "goal_orientation", None)


def _robot_yaw_only_enabled(motion_command: MotionCommand) -> bool:
    goal_orientation = _get_goal_orientation_cfg(motion_command)
    return bool(getattr(goal_orientation, "robot_yaw_only", False))


def _object_yaw_only_enabled(motion_command: MotionCommand) -> bool:
    goal_orientation = _get_goal_orientation_cfg(motion_command)
    return bool(getattr(goal_orientation, "object_yaw_only", False))


def _get_goal_conditioning_ref_quat_w(motion_command: MotionCommand) -> torch.Tensor:
    if hasattr(motion_command, "goal_conditioning_ref_quat_w"):
        return motion_command.goal_conditioning_ref_quat_w

    robot_ref_quat_w = motion_command.robot_ref_quat_w
    if _robot_yaw_only_enabled(motion_command):
        return yaw_quat(robot_ref_quat_w, w_last=True)
    return robot_ref_quat_w


def _get_goal_conditioning_object_quat_w(motion_command: MotionCommand) -> torch.Tensor:
    if hasattr(motion_command, "goal_conditioning_object_quat_w"):
        return motion_command.goal_conditioning_object_quat_w

    simulator_object_quat_w = motion_command.simulator_object_quat_w
    if _object_yaw_only_enabled(motion_command):
        return yaw_quat(simulator_object_quat_w, w_last=True)
    return simulator_object_quat_w


def _mask_by_env(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values * mask.to(dtype=values.dtype)


def _mask_track_envs(
    values: torch.Tensor,
    motion_command: MotionCommand,
    gen_constant: float = 0.5,
) -> torch.Tensor:
    track_env_mask, gen_env_mask = _get_hybrid_masks(motion_command)
    masked = _mask_by_env(values, track_env_mask)
    if gen_constant == 0.0:
        return masked
    return masked + gen_env_mask.to(dtype=values.dtype) * float(gen_constant)


#########################################################################################################
## terms same to managers/reward/terms/locomotion.py
#########################################################################################################


def penalty_action_rate(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def limits_dof_pos(env: WholeBodyTrackingManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################

# ================================================================================================
# Robot Tracking Rewards
# ================================================================================================


def motion_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.ref_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def hybrid_motion_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return _mask_track_envs(motion_global_ref_position_error_exp(env, sigma), motion_command)


def motion_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def hybrid_motion_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return _mask_track_envs(motion_global_ref_orientation_error_exp(env, sigma), motion_command)


def motion_relative_body_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def hybrid_motion_relative_body_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return _mask_track_envs(motion_relative_body_position_error_exp(env, sigma), motion_command)


def motion_relative_body_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w) ** 2
    return torch.exp(-error.mean(-1) / sigma**2)


def hybrid_motion_relative_body_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return _mask_track_envs(motion_relative_body_orientation_error_exp(env, sigma), motion_command)


def motion_global_body_lin_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def hybrid_motion_global_body_lin_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return _mask_track_envs(motion_global_body_lin_vel(env, sigma), motion_command)


def motion_global_body_ang_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def hybrid_motion_global_body_ang_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return _mask_track_envs(motion_global_body_ang_vel(env, sigma), motion_command)


# ================================================================================================
# Object Tracking Rewards
# ================================================================================================


def object_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def hybrid_object_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return _mask_track_envs(object_global_ref_position_error_exp(env, sigma), motion_command)


def object_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def hybrid_object_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return _mask_track_envs(object_global_ref_orientation_error_exp(env, sigma), motion_command)


class _HybridGoalReachedOnceBase(RewardTermBase):
    """One-time sparse goal reward for gen envs in hybrid WBT tasks."""

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.pos_threshold = float(cfg.params["pos_threshold"])
        self.ori_threshold = float(cfg.params["ori_threshold"])
        self.bonus = float(cfg.params.get("bonus", 10.0))
        self._awarded_goal_version = torch.full(
            (self.env.num_envs,),
            -1,
            dtype=torch.long,
            device=self.env.device,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._awarded_goal_version.fill_(-1)
            return
        env_ids = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long).flatten()
        self._awarded_goal_version[env_ids] = -1

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        motion_command = _get_motion_command_and_assert_type(env)
        _, gen_env_mask = _get_hybrid_masks(motion_command)
        reached = self._get_reached_mask(motion_command) & gen_env_mask
        new_reached = reached & (self._awarded_goal_version != motion_command.goal_version)
        self._awarded_goal_version[new_reached] = motion_command.goal_version[new_reached]

        bonus_per_step = self.bonus / max(float(self.env.dt), 1.0e-6)
        return new_reached.to(dtype=torch.float32) * bonus_per_step

    def _get_reached_mask(self, motion_command: MotionCommand) -> torch.Tensor:
        raise NotImplementedError


class HybridGoalRefReachedOnce(_HybridGoalReachedOnceBase):
    """One-time robot goal success reward for hybrid gen envs."""

    def _get_reached_mask(self, motion_command: MotionCommand) -> torch.Tensor:
        pos_error = torch.norm(motion_command.goal_ref_pos_w - motion_command.robot_ref_pos_w, dim=-1)
        ori_error = quat_error_magnitude(
            motion_command.goal_ref_quat_w,
            _get_goal_conditioning_ref_quat_w(motion_command),
        )
        return (pos_error <= self.pos_threshold) & (ori_error <= self.ori_threshold)


class HybridGoalObjectReachedOnce(_HybridGoalReachedOnceBase):
    """One-time object goal success reward for hybrid gen envs."""

    def _get_reached_mask(self, motion_command: MotionCommand) -> torch.Tensor:
        pos_error = torch.norm(motion_command.goal_object_pos_w - motion_command.simulator_object_pos_w, dim=-1)
        ori_error = quat_error_magnitude(
            motion_command.goal_object_quat_w,
            _get_goal_conditioning_object_quat_w(motion_command),
        )
        return (pos_error <= self.pos_threshold) & (ori_error <= self.ori_threshold)


# ================================================================================================
# Undesired Contacts Rewards
# ================================================================================================


class UndesiredContacts(RewardTermBase):
    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        undesired_contacts_body_names = [
            body_name
            for body_name in self.env.simulator.body_names  # type: ignore[attr-defined]
            if re.match(cfg.params.get("undesired_contacts_body_names", ""), body_name)
        ]
        self.undesired_contacts_body_indexes = self._get_index_of_a_in_b(
            undesired_contacts_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # (num_envs, history_length, num_bodies, 3)
        net_contact_forces = self.env.simulator.contact_forces_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contacts_body_indexes], dim=-1), dim=1)[0]
            > self.threshold
        )
        return torch.sum(is_contact, dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
