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


def object_group_one_hot(env: WholeBodyTrackingManager) -> torch.Tensor:
    """One-hot object group of each environment (``motion_dir_object_groups``); a constant 1 without groups."""
    motion_command = _get_motion_command_and_assert_type(env)
    return torch.nn.functional.one_hot(motion_command.env_object_group, motion_command.num_object_groups).float()


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


def _obj_future_poses_b(env: WholeBodyTrackingManager, frames: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference object poses ``frames`` steps ahead of each environment's time step (held at the clip's last frame),
    in the goal-conditioning robot frame of ``obj_goal_pos_b``: positions [num_envs, K, 3], quaternions [num_envs, K,
    4]."""
    motion_command = _get_motion_command_and_assert_type(env)
    offsets = torch.as_tensor(frames, dtype=torch.long, device=env.device)
    last = motion_command.motion.motion_end_idx[motion_command.motion_ids] - 1
    steps = torch.minimum(motion_command.time_steps[:, None] + offsets[None, :], last[:, None])  # [num_envs, K]
    pos_w = motion_command.motion.object_pos_w[steps] + env.simulator.scene.env_origins[:, None, :]
    quat_w = motion_command.motion.object_quat_w[steps]
    if _object_yaw_only_enabled(motion_command):
        quat_w = yaw_quat(quat_w, w_last=True).view(quat_w.shape)
    num_frames = len(frames)
    pos_b, quat_b = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_frames, 1),
        _get_goal_conditioning_ref_quat_w(motion_command)[:, None, :].repeat(1, num_frames, 1),
        pos_w,
        quat_w,
    )
    return pos_b, quat_b


def obj_future_pos_b(env: WholeBodyTrackingManager, frames: list[int]) -> torch.Tensor:
    """Reference object positions ``frames`` steps ahead, in the robot frame (3 values per frame)."""
    pos_b, _ = _obj_future_poses_b(env, frames)
    return pos_b.reshape(env.num_envs, -1)


def obj_future_ori_b(env: WholeBodyTrackingManager, frames: list[int]) -> torch.Tensor:
    """Reference object orientations ``frames`` steps ahead, in the robot frame (first two rotation-matrix columns,
    6 values per frame)."""
    _, quat_b = _obj_future_poses_b(env, frames)
    mat = quaternion_to_matrix(quat_b, w_last=True)
    return mat[..., :2].reshape(env.num_envs, -1)


def _hands_in_object_frame(motion_command: MotionCommand, body_names: list[str], reference: bool) -> torch.Tensor:
    names = motion_command.motion_cfg.body_names_to_track
    idx = [names.index(n) for n in body_names]
    if reference:
        hand, obj_pos, obj_quat = motion_command.body_pos_w[:, idx], motion_command.object_pos_w, motion_command.object_quat_w
    else:
        hand = motion_command.robot_body_pos_w[:, idx]
        obj_pos, obj_quat = motion_command.simulator_object_pos_w, motion_command.simulator_object_quat_w
    n = len(idx)
    local = quat_rotate_inverse(
        obj_quat[:, None].expand(-1, n, -1).reshape(-1, 4), (hand - obj_pos[:, None]).reshape(-1, 3), w_last=True
    )
    return local.view(-1, n, 3)


def hand_obj_pos(env: WholeBodyTrackingManager, body_names: list[str]) -> torch.Tensor:
    """The hands' positions in the simulated object's frame."""
    motion_command = _get_motion_command_and_assert_type(env)
    return _hands_in_object_frame(motion_command, body_names, reference=False).reshape(env.num_envs, -1)


def hybrid_hand_obj_pos_error(env: WholeBodyTrackingManager, body_names: list[str]) -> torch.Tensor:
    """Track envs: the hands in the simulated object's frame minus the reference hands in the reference object's
    frame (what ``hand_object_position_error_exp`` rewards); zeros for gen envs, which have no reference to follow."""
    motion_command = _get_motion_command_and_assert_type(env)
    sim = _hands_in_object_frame(motion_command, body_names, reference=False)
    ref = _hands_in_object_frame(motion_command, body_names, reference=True)
    return _mask_gen_rows((sim - ref).reshape(env.num_envs, -1), motion_command)


def obj_lin_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    vel_b = quat_rotate_inverse(
        _get_goal_conditioning_ref_quat_w(motion_command),
        motion_command.simulator_object_lin_vel_w,
        w_last=True,
    )
    return vel_b.view(env.num_envs, -1)
