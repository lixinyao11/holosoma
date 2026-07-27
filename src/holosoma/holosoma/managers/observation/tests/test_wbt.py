from types import SimpleNamespace

import torch

from holosoma.config_types.command import GoalOrientationConfig
from holosoma.config_values.wbt.g1.observation import g1_29dof_wbt_observation_w_object
from holosoma.managers.observation.terms import wbt as wbt_obs
from holosoma.utils.rotations import quat_from_euler_xyz


def test_wbt_object_actor_observation_includes_object_terms():
    actor_terms = g1_29dof_wbt_observation_w_object.groups["actor_obs"].terms
    critic_terms = g1_29dof_wbt_observation_w_object.groups["critic_obs"].terms

    assert "obj_pos_b" in actor_terms
    assert "obj_ori_b" in actor_terms
    assert "obj_lin_vel_b" in actor_terms
    assert "obj_ref_pos_b" in actor_terms
    assert "obj_ref_ori_b" in actor_terms
    assert "obj_ref_pos_b" in critic_terms
    assert "obj_ref_ori_b" in critic_terms


def test_obj_ref_pos_b_uses_reference_object_pose_relative_to_robot_ref(monkeypatch):
    env = SimpleNamespace(num_envs=1, device="cpu")
    motion_command = SimpleNamespace(
        robot_ref_pos_w=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        object_pos_w=torch.tensor([[4.0, 8.0, 13.0]], dtype=torch.float32),
        object_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
    )

    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)

    pos_b = wbt_obs.obj_ref_pos_b(env)

    assert torch.allclose(pos_b, torch.tensor([[3.0, 6.0, 10.0]], dtype=torch.float32), atol=1e-5)


def test_motion_goal_pos_b_uses_terminal_goal_pose_relative_to_robot_ref(monkeypatch):
    env = SimpleNamespace(num_envs=1, device="cpu")
    motion_command = SimpleNamespace(
        robot_ref_pos_w=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        goal_ref_pos_w=torch.tensor([[4.0, 8.0, 13.0]], dtype=torch.float32),
        goal_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
    )

    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)

    pos_b = wbt_obs.motion_goal_pos_b(env)

    assert torch.allclose(pos_b, torch.tensor([[3.0, 6.0, 10.0]], dtype=torch.float32), atol=1e-5)


def test_motion_goal_ori_b_uses_terminal_goal_orientation_relative_to_robot_ref(monkeypatch):
    env = SimpleNamespace(num_envs=1, device="cpu")
    motion_command = SimpleNamespace(
        robot_ref_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        goal_ref_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        goal_ref_quat_w=torch.tensor([[0.0, 0.0, 0.70710678, 0.70710678]], dtype=torch.float32),
    )

    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)

    ori_b = wbt_obs.motion_goal_ori_b(env)

    expected = torch.tensor([[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(ori_b, expected, atol=1e-5)

def test_motion_goal_pos_b_uses_yaw_only_robot_frame_when_enabled(monkeypatch):
    env = SimpleNamespace(num_envs=1, device="cpu")
    motion_command = SimpleNamespace(
        robot_ref_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        robot_ref_quat_w=quat_from_euler_xyz(
            torch.tensor([1.5707964], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
        ),
        goal_ref_pos_w=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
        goal_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        motion_cfg=SimpleNamespace(goal_orientation=GoalOrientationConfig(robot_yaw_only=True)),
    )

    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)

    pos_b = wbt_obs.motion_goal_pos_b(env)

    assert torch.allclose(pos_b, torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32), atol=1e-5)


def test_motion_goal_ori_b_ignores_current_roll_pitch_when_robot_yaw_only_enabled(monkeypatch):
    env = SimpleNamespace(num_envs=1, device="cpu")
    motion_command = SimpleNamespace(
        robot_ref_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        robot_ref_quat_w=quat_from_euler_xyz(
            torch.tensor([1.5707964], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
        ),
        goal_ref_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        goal_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        motion_cfg=SimpleNamespace(goal_orientation=GoalOrientationConfig(robot_yaw_only=True)),
    )

    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)

    ori_b = wbt_obs.motion_goal_ori_b(env)

    expected = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(ori_b, expected, atol=1e-5)


def test_obj_lin_vel_b_rotates_velocity_into_reference_frame(monkeypatch):
    env = SimpleNamespace(num_envs=1, device="cpu")
    motion_command = SimpleNamespace(
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.70710678, 0.70710678]], dtype=torch.float32),
        simulator_object_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
    )

    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)

    vel_b = wbt_obs.obj_lin_vel_b(env)

    assert torch.allclose(vel_b, torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float32), atol=1e-5)


def test_obj_ori_b_uses_yaw_only_robot_frame_but_keeps_full_object_orientation(monkeypatch):
    env = SimpleNamespace(num_envs=1, device="cpu")
    roll_quat = quat_from_euler_xyz(
        torch.tensor([1.5707964], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
    )
    motion_command = SimpleNamespace(
        robot_ref_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        robot_ref_quat_w=roll_quat,
        simulator_object_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        simulator_object_quat_w=roll_quat,
        motion_cfg=SimpleNamespace(
            goal_orientation=GoalOrientationConfig(robot_yaw_only=True, object_yaw_only=False)
        ),
    )

    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)

    ori_b = wbt_obs.obj_ori_b(env)

    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(ori_b, expected, atol=1e-5)


def test_obj_ori_b_projects_object_orientation_to_yaw_when_enabled(monkeypatch):
    env = SimpleNamespace(num_envs=1, device="cpu")
    motion_command = SimpleNamespace(
        robot_ref_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        simulator_object_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        simulator_object_quat_w=quat_from_euler_xyz(
            torch.tensor([1.5707964], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
        ),
        motion_cfg=SimpleNamespace(goal_orientation=GoalOrientationConfig(object_yaw_only=True)),
    )

    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)

    ori_b = wbt_obs.obj_ori_b(env)

    expected = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(ori_b, expected, atol=1e-5)


def test_obj_lin_vel_b_uses_yaw_only_robot_frame_when_enabled(monkeypatch):
    env = SimpleNamespace(num_envs=1, device="cpu")
    motion_command = SimpleNamespace(
        robot_ref_quat_w=quat_from_euler_xyz(
            torch.tensor([1.5707964], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
        ),
        simulator_object_lin_vel_w=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
        motion_cfg=SimpleNamespace(goal_orientation=GoalOrientationConfig(robot_yaw_only=True)),
    )

    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)

    vel_b = wbt_obs.obj_lin_vel_b(env)

    assert torch.allclose(vel_b, torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32), atol=1e-5)


def test_obj_goal_pos_b_uses_terminal_goal_object_pose_relative_to_robot_ref(monkeypatch):
    env = SimpleNamespace(num_envs=1, device="cpu")
    motion_command = SimpleNamespace(
        robot_ref_pos_w=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        goal_object_pos_w=torch.tensor([[6.0, 7.0, 9.0]], dtype=torch.float32),
        goal_object_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
    )

    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)

    pos_b = wbt_obs.obj_goal_pos_b(env)

    assert torch.allclose(pos_b, torch.tensor([[5.0, 5.0, 6.0]], dtype=torch.float32), atol=1e-5)


def test_obj_goal_ori_b_uses_terminal_goal_object_orientation_relative_to_robot_ref(monkeypatch):
    env = SimpleNamespace(num_envs=1, device="cpu")
    motion_command = SimpleNamespace(
        robot_ref_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        goal_object_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        goal_object_quat_w=torch.tensor([[0.0, 0.0, 0.70710678, 0.70710678]], dtype=torch.float32),
    )

    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)

    ori_b = wbt_obs.obj_goal_ori_b(env)

    expected = torch.tensor([[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(ori_b, expected, atol=1e-5)
