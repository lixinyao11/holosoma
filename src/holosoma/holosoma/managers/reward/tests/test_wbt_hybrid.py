from types import SimpleNamespace

import torch

from holosoma.config_types.command import GoalOrientationConfig
from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.reward.terms import wbt as wbt_reward
from holosoma.utils.rotations import quat_from_euler_xyz


def test_hybrid_track_rewards_are_zeroed_on_gen_envs(monkeypatch):
    motion_command = SimpleNamespace(
        ref_pos_w=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        robot_ref_pos_w=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32),
        track_env_mask=torch.tensor([True, False]),
        gen_env_mask=torch.tensor([False, True]),
    )
    env = SimpleNamespace()
    monkeypatch.setattr(wbt_reward, "_get_motion_command_and_assert_type", lambda _: motion_command)

    reward = wbt_reward.hybrid_motion_global_ref_position_error_exp(env, sigma=1.0)

    assert torch.allclose(reward, torch.tensor([1.0, 0.5], dtype=torch.float32))


def test_hybrid_goal_ref_reached_once_rewards_once_per_goal_version(monkeypatch):
    motion_command = SimpleNamespace(
        goal_ref_pos_w=torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=torch.float32),
        robot_ref_pos_w=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
        goal_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        goal_version=torch.tensor([1, 1], dtype=torch.long),
        track_env_mask=torch.tensor([True, False]),
        gen_env_mask=torch.tensor([False, True]),
    )
    env = SimpleNamespace(num_envs=2, device="cpu", dt=0.02)
    monkeypatch.setattr(wbt_reward, "_get_motion_command_and_assert_type", lambda _: motion_command)

    term = wbt_reward.HybridGoalRefReachedOnce(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:HybridGoalRefReachedOnce",
            params={"bonus": 10.0, "pos_threshold": 0.2, "ori_threshold": 0.5235987755982988},
        ),
        env,
    )

    reward_first = term(env)
    reward_second = term(env)
    motion_command.goal_version = torch.tensor([1, 2], dtype=torch.long)
    reward_after_goal_refresh = term(env)
    term.reset(torch.tensor([1], dtype=torch.long))
    reward_after_reset = term(env)

    assert torch.allclose(reward_first, torch.tensor([0.0, 500.0], dtype=torch.float32))
    assert torch.allclose(reward_second, torch.zeros(2, dtype=torch.float32))
    assert torch.allclose(reward_after_goal_refresh, torch.tensor([0.0, 500.0], dtype=torch.float32))
    assert torch.allclose(reward_after_reset, torch.tensor([0.0, 500.0], dtype=torch.float32))


def test_hybrid_goal_ref_reached_once_uses_yaw_only_robot_orientation_when_enabled(monkeypatch):
    motion_command = SimpleNamespace(
        goal_ref_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        robot_ref_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        goal_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        robot_ref_quat_w=quat_from_euler_xyz(
            torch.tensor([1.5707964], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
        ),
        motion_cfg=SimpleNamespace(goal_orientation=GoalOrientationConfig(robot_yaw_only=True)),
        goal_version=torch.tensor([1], dtype=torch.long),
        track_env_mask=torch.tensor([False]),
        gen_env_mask=torch.tensor([True]),
    )
    env = SimpleNamespace(num_envs=1, device="cpu", dt=0.02)
    monkeypatch.setattr(wbt_reward, "_get_motion_command_and_assert_type", lambda _: motion_command)

    term = wbt_reward.HybridGoalRefReachedOnce(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:HybridGoalRefReachedOnce",
            params={"bonus": 10.0, "pos_threshold": 0.2, "ori_threshold": 0.1},
        ),
        env,
    )

    reward = term(env)

    assert torch.allclose(reward, torch.tensor([500.0], dtype=torch.float32))


def test_hybrid_goal_object_reached_once_rewards_once_per_goal_version(monkeypatch):
    motion_command = SimpleNamespace(
        goal_object_pos_w=torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=torch.float32),
        simulator_object_pos_w=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
        goal_object_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        simulator_object_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        goal_version=torch.tensor([1, 1], dtype=torch.long),
        track_env_mask=torch.tensor([True, False]),
        gen_env_mask=torch.tensor([False, True]),
    )
    env = SimpleNamespace(num_envs=2, device="cpu", dt=0.02)
    monkeypatch.setattr(wbt_reward, "_get_motion_command_and_assert_type", lambda _: motion_command)

    term = wbt_reward.HybridGoalObjectReachedOnce(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:HybridGoalObjectReachedOnce",
            params={"bonus": 10.0, "pos_threshold": 0.2, "ori_threshold": 0.5235987755982988},
        ),
        env,
    )

    reward_first = term(env)
    reward_second = term(env)
    motion_command.goal_version = torch.tensor([1, 2], dtype=torch.long)
    reward_after_goal_refresh = term(env)
    term.reset(torch.tensor([1], dtype=torch.long))
    reward_after_reset = term(env)

    assert torch.allclose(reward_first, torch.tensor([0.0, 500.0], dtype=torch.float32))
    assert torch.allclose(reward_second, torch.zeros(2, dtype=torch.float32))
    assert torch.allclose(reward_after_goal_refresh, torch.tensor([0.0, 500.0], dtype=torch.float32))
    assert torch.allclose(reward_after_reset, torch.tensor([0.0, 500.0], dtype=torch.float32))


def test_hybrid_goal_object_reached_once_uses_yaw_only_object_orientation_when_enabled(monkeypatch):
    motion_command = SimpleNamespace(
        goal_object_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        simulator_object_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        goal_object_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        simulator_object_quat_w=quat_from_euler_xyz(
            torch.tensor([1.5707964], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
        ),
        motion_cfg=SimpleNamespace(goal_orientation=GoalOrientationConfig(object_yaw_only=True)),
        goal_version=torch.tensor([1], dtype=torch.long),
        track_env_mask=torch.tensor([False]),
        gen_env_mask=torch.tensor([True]),
    )
    env = SimpleNamespace(num_envs=1, device="cpu", dt=0.02)
    monkeypatch.setattr(wbt_reward, "_get_motion_command_and_assert_type", lambda _: motion_command)

    term = wbt_reward.HybridGoalObjectReachedOnce(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:HybridGoalObjectReachedOnce",
            params={"bonus": 10.0, "pos_threshold": 0.2, "ori_threshold": 0.1},
        ),
        env,
    )

    reward = term(env)

    assert torch.allclose(reward, torch.tensor([500.0], dtype=torch.float32))
