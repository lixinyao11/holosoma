"""Whole body tracking observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import quat_rotate_inverse, quaternion_to_matrix, subtract_frame_transforms, yaw_quat
from holosoma.utils.torch_utils import get_axis_params, to_torch

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


#########################################################################################################
## terms same to managers/observation/terms/locomotion.py
#########################################################################################################
def _base_quat(env: WholeBodyTrackingManager) -> torch.Tensor:
    return env.base_quat


def gravity_vector(env: WholeBodyTrackingManager, up_axis_idx: int = 2) -> torch.Tensor:
    axis = to_torch(get_axis_params(-1.0, up_axis_idx), device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def base_forward_vector(env: WholeBodyTrackingManager) -> torch.Tensor:
    axis = to_torch([1.0, 0.0, 0.0], device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def get_base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    root_states = env.simulator.robot_root_states
    lin_vel_world = root_states[:, 7:10]
    return quat_rotate_inverse(_base_quat(env), lin_vel_world, w_last=True)


def get_base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(_base_quat(env), ang_vel_world, w_last=True)


def get_projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    return quat_rotate_inverse(_base_quat(env), gravity_vector(env), w_last=True)


def base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base linear velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_lin_vel()
    """
    return get_base_lin_vel(env)


def base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base angular velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_ang_vel()
    """
    return get_base_ang_vel(env)


def projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Gravity vector projected into base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_projected_gravity()
    """
    return get_projected_gravity(env)


def dof_pos(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint positions relative to default positions.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_pos()
    """
    return env.simulator.dof_pos - env.default_dof_pos


def dof_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint velocities.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_vel()
    """
    return env.simulator.dof_vel


def actions(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Last actions taken by the policy.

    Returns:
        Tensor of shape [num_envs, num_actions]

    Equivalent to:
        env._get_obs_actions()
    """
    return env.action_manager.action


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


def _get_hybrid_masks(motion_command: MotionCommand) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(motion_command, "track_env_mask") or not hasattr(motion_command, "gen_env_mask"):
        raise RuntimeError("Hybrid observation term requires HybridMotionCommand with track/gen masks.")
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


def _get_goal_conditioning_motion_object_quat_w(motion_command: MotionCommand) -> torch.Tensor:
    object_quat_w = motion_command.object_quat_w
    if _object_yaw_only_enabled(motion_command):
        return yaw_quat(object_quat_w, w_last=True)
    return object_quat_w


def _mask_gen_rows(values: torch.Tensor, motion_command: MotionCommand) -> torch.Tensor:
    _, gen_env_mask = _get_hybrid_masks(motion_command)
    keep_mask = (~gen_env_mask).to(dtype=values.dtype)
    return values * keep_mask.view(values.shape[0], *([1] * (values.ndim - 1)))


def motion_command(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.command


def hybrid_motion_command(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return _mask_gen_rows(motion_command.command, motion_command)


def motion_goal_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        _get_goal_conditioning_ref_quat_w(motion_command),
        motion_command.goal_ref_pos_w,
        motion_command.goal_ref_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_goal_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        _get_goal_conditioning_ref_quat_w(motion_command),
        motion_command.goal_ref_pos_w,
        motion_command.goal_ref_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_ref_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    return pos.view(env.num_envs, -1)


def hybrid_motion_ref_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return _mask_gen_rows(motion_ref_pos_b(env), motion_command)


def motion_ref_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def hybrid_motion_ref_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return _mask_gen_rows(motion_ref_ori_b(env), motion_command)


def robot_body_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    _, ori_b = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )
    mat = quaternion_to_matrix(ori_b, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        _get_goal_conditioning_ref_quat_w(motion_command),
        motion_command.simulator_object_pos_w,
        _get_goal_conditioning_object_quat_w(motion_command),
    )
    return pos.view(env.num_envs, -1)


def obj_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        _get_goal_conditioning_ref_quat_w(motion_command),
        motion_command.simulator_object_pos_w,
        _get_goal_conditioning_object_quat_w(motion_command),
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_ref_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        _get_goal_conditioning_ref_quat_w(motion_command),
        motion_command.object_pos_w,
        _get_goal_conditioning_motion_object_quat_w(motion_command),
    )
    return pos.view(env.num_envs, -1)


def hybrid_obj_ref_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return _mask_gen_rows(obj_ref_pos_b(env), motion_command)


def obj_ref_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        _get_goal_conditioning_ref_quat_w(motion_command),
        motion_command.object_pos_w,
        _get_goal_conditioning_motion_object_quat_w(motion_command),
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def hybrid_obj_ref_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return _mask_gen_rows(obj_ref_ori_b(env), motion_command)


def obj_goal_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        _get_goal_conditioning_ref_quat_w(motion_command),
        motion_command.goal_object_pos_w,
        motion_command.goal_object_quat_w,
    )
    return pos.view(env.num_envs, -1)


def obj_goal_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        _get_goal_conditioning_ref_quat_w(motion_command),
        motion_command.goal_object_pos_w,
        motion_command.goal_object_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_lin_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    vel_b = quat_rotate_inverse(
        _get_goal_conditioning_ref_quat_w(motion_command),
        motion_command.simulator_object_lin_vel_w,
        w_last=True,
    )
    return vel_b.view(env.num_envs, -1)
