from types import SimpleNamespace

import torch

from holosoma.config_types.command import (
    GoalPoseNoiseConfig,
    HybridMotionConfig,
    HybridSizeRandomizationConfig,
    StepZeroRootNoiseConfig,
)
from holosoma.envs.wbt.hybrid_size_randomization import collision_bounds_corners
from holosoma.managers.command.terms.wbt import AdaptiveTimestepsSampler, HybridMotionCommand, MotionCommand
from holosoma.utils.rotations import quat_from_euler_xyz, quat_mul, quat_normalize


def _make_hybrid_command(num_envs: int = 4) -> HybridMotionCommand:
    command = object.__new__(HybridMotionCommand)
    command.device = torch.device("cpu")
    command.num_envs = num_envs
    command.hybrid_cfg = HybridMotionConfig(
        track_ratio=0.5,
        env_partition_seed=0,
        robot_goal_noise=GoalPoseNoiseConfig(),
        object_goal_noise=GoalPoseNoiseConfig(),
    )
    command.track_env_mask = torch.zeros(num_envs, dtype=torch.bool)
    command.gen_env_mask = torch.zeros(num_envs, dtype=torch.bool)
    command._goal_time_steps = torch.zeros(num_envs, dtype=torch.long)
    command._goal_version = torch.zeros(num_envs, dtype=torch.long)
    command._exact_goal_ref_pos_w = torch.zeros(num_envs, 3, dtype=torch.float32)
    command._exact_goal_ref_quat_w = torch.zeros(num_envs, 4, dtype=torch.float32)
    command._exact_goal_ref_quat_w[:, 3] = 1.0
    command._active_goal_ref_pos_w = torch.zeros(num_envs, 3, dtype=torch.float32)
    command._active_goal_ref_quat_w = torch.zeros(num_envs, 4, dtype=torch.float32)
    command._active_goal_ref_quat_w[:, 3] = 1.0
    command._exact_goal_object_pos_w = torch.zeros(num_envs, 3, dtype=torch.float32)
    command._exact_goal_object_quat_w = torch.zeros(num_envs, 4, dtype=torch.float32)
    command._exact_goal_object_quat_w[:, 3] = 1.0
    command._active_goal_object_pos_w = torch.zeros(num_envs, 3, dtype=torch.float32)
    command._active_goal_object_quat_w = torch.zeros(num_envs, 4, dtype=torch.float32)
    command._active_goal_object_quat_w[:, 3] = 1.0
    return command


def test_hybrid_partition_is_seeded_and_stable():
    command_a = _make_hybrid_command()
    command_b = _make_hybrid_command()

    command_a._build_fixed_env_partition()
    command_b._build_fixed_env_partition()

    assert torch.equal(command_a.track_env_mask, command_b.track_env_mask)
    assert torch.equal(command_a.gen_env_mask, command_b.gen_env_mask)
    assert int(command_a.track_env_mask.sum().item()) == 2
    assert int(command_a.gen_env_mask.sum().item()) == 2

    saved_track_mask = command_a.track_env_mask.clone()
    saved_gen_mask = command_a.gen_env_mask.clone()
    command_a._build_fixed_env_partition()
    assert torch.equal(command_a.track_env_mask, saved_track_mask)
    assert torch.equal(command_a.gen_env_mask, saved_gen_mask)


def test_scaled_object_reference_velocity_adds_only_ground_correction_velocity():
    command = _make_hybrid_command(num_envs=1)
    command.hybrid_cfg = HybridMotionConfig(
        size_randomization=HybridSizeRandomizationConfig(enabled=True, mode="uniform_xyz")
    )
    command.time_steps = torch.tensor([0], dtype=torch.long)
    command.motion_ids = torch.tensor([0], dtype=torch.long)
    identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    command.motion = SimpleNamespace(
        has_object=True,
        fps=10.0,
        motion_start_idx=torch.tensor([0], dtype=torch.long),
        motion_end_idx=torch.tensor([2], dtype=torch.long),
        object_pos_w=torch.tensor([[0.0, 0.0, 0.20], [0.0, 0.0, 0.21]], dtype=torch.float32),
        object_quat_w=identity_quat,
        object_lin_vel_w=torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float32),
    )
    command._env = SimpleNamespace(
        simulator=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(1, 3, dtype=torch.float32)))
    )
    command._fixed_actor_scales_xyz = torch.tensor([[1.15, 1.15, 1.15]], dtype=torch.float32)
    command._fixed_size_scalars = command._fixed_actor_scales_xyz[:, 0]
    command._fixed_object_collision_corners = collision_bounds_corners(
        torch.tensor([[-0.2, -0.2, -0.2], [0.2, 0.2, 0.2]], dtype=torch.float32)
    )

    velocity = command.object_lin_vel_w

    assert torch.allclose(velocity[:, :2], torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(velocity[:, 2], torch.tensor([2.9]), atol=1.0e-6)


def test_hybrid_init_buffers_preserves_existing_partition():
    command = _make_hybrid_command()
    command.track_env_mask[:] = torch.tensor([True, False, True, False])
    command.gen_env_mask[:] = ~command.track_env_mask
    command.set_object_goal_noise_curriculum_scale(0.4)
    command.motion_cfg = SimpleNamespace(body_names_to_track=["torso_link"], use_adaptive_timesteps_sampler=False)
    command.num_envs = 4
    command.device = torch.device("cpu")
    command.time_steps = torch.ones(4, dtype=torch.long)
    command.motion_ids = torch.ones(4, dtype=torch.long)
    command.last_clip_end_env_ids = torch.ones(1, dtype=torch.long)
    command.body_pos_relative_w = torch.ones(4, 1, 3)
    command.body_quat_relative_w = torch.ones(4, 1, 4)

    HybridMotionCommand.init_buffers(command)

    assert torch.equal(command.track_env_mask, torch.tensor([True, False, True, False]))
    assert torch.equal(command.gen_env_mask, torch.tensor([False, True, False, True]))
    assert torch.equal(command.time_steps, torch.zeros(4, dtype=torch.long))
    assert torch.allclose(command.object_goal_noise_curriculum_scale, torch.tensor(0.4, dtype=torch.float32))


def test_motion_command_start_at_timestep_zero_probs_use_motion_default():
    command = object.__new__(MotionCommand)
    command.device = torch.device("cpu")
    command.motion_cfg = SimpleNamespace(start_at_timestep_zero_prob=0.25)

    probs = command._get_start_at_timestep_zero_probs(torch.tensor([0, 2], dtype=torch.long))

    assert torch.allclose(probs, torch.tensor([0.25, 0.25], dtype=torch.float32))


def test_hybrid_start_at_timestep_zero_probs_override_gen_envs_only():
    command = _make_hybrid_command()
    command.motion_cfg = SimpleNamespace(start_at_timestep_zero_prob=0.2)
    command.track_env_mask[:] = torch.tensor([True, False, True, False])
    command.gen_env_mask[:] = ~command.track_env_mask
    command.hybrid_cfg = HybridMotionConfig(
        track_ratio=0.5,
        env_partition_seed=0,
        gen_start_at_timestep_zero_prob=0.8,
        robot_goal_noise=GoalPoseNoiseConfig(),
        object_goal_noise=GoalPoseNoiseConfig(),
    )

    probs = command._get_start_at_timestep_zero_probs(torch.arange(4, dtype=torch.long))

    assert torch.allclose(probs, torch.tensor([0.2, 0.8, 0.2, 0.8], dtype=torch.float32))


def test_hybrid_start_at_timestep_zero_probs_fall_back_to_motion_default_when_explicit_none():
    command = _make_hybrid_command()
    command.motion_cfg = SimpleNamespace(start_at_timestep_zero_prob=0.2)
    command.track_env_mask[:] = torch.tensor([True, False, True, False])
    command.gen_env_mask[:] = ~command.track_env_mask
    command.hybrid_cfg = HybridMotionConfig(
        track_ratio=0.5,
        env_partition_seed=0,
        gen_start_at_timestep_zero_prob=None,
        robot_goal_noise=GoalPoseNoiseConfig(),
        object_goal_noise=GoalPoseNoiseConfig(),
    )

    probs = command._get_start_at_timestep_zero_probs(torch.arange(4, dtype=torch.long))

    assert torch.allclose(probs, torch.full((4,), 0.2, dtype=torch.float32))


def test_hybrid_step_zero_root_noise_overrides_distinguish_track_and_gen():
    command = _make_hybrid_command()
    command.track_env_mask[:] = torch.tensor([True, False, True, False])
    command.gen_env_mask[:] = ~command.track_env_mask
    command.init_pose_cfg = SimpleNamespace(
        step_zero_root_noise=StepZeroRootNoiseConfig(
            pos_std_xyz=[2.0, 2.0, 2.0],
            pos_clip_xyz=[2.5, 2.5, 2.5],
            rpy_std=[0.2, 0.2, 0.2],
            rpy_clip=[0.25, 0.25, 0.25],
        ),
    )
    command.hybrid_cfg = HybridMotionConfig(
        track_ratio=0.5,
        env_partition_seed=0,
        gen_step_zero_root_noise=StepZeroRootNoiseConfig(
            pos_std_xyz=[4.0, 4.0, 4.0],
            pos_clip_xyz=[4.5, 4.5, 4.5],
            rpy_std=[0.4, 0.4, 0.4],
            rpy_clip=[0.45, 0.45, 0.45],
        ),
        robot_goal_noise=GoalPoseNoiseConfig(),
        object_goal_noise=GoalPoseNoiseConfig(),
    )
    command.time_steps = torch.tensor([0, 0, 6, 0], dtype=torch.long)
    global_quat = quat_from_euler_xyz(
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.1], dtype=torch.float32),
    )
    gen_quat = quat_from_euler_xyz(
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.3], dtype=torch.float32),
    )

    def _fake_step_zero_noise(num_envs, noise_cfg, *, pos_axis_scale_xyz=None, rpy_axis_scale=None):
        if noise_cfg is command.init_pose_cfg.step_zero_root_noise:
            return torch.full((num_envs, 3), 1.0, dtype=torch.float32), global_quat.repeat(num_envs, 1)
        if noise_cfg is command.hybrid_cfg.gen_step_zero_root_noise:
            return torch.full((num_envs, 3), 2.0, dtype=torch.float32), gen_quat.repeat(num_envs, 1)
        raise AssertionError("Unexpected step-zero root noise config")

    command._sample_step_zero_root_noise = _fake_step_zero_noise

    env_ids = torch.arange(4, dtype=torch.long)
    start_idx = torch.zeros(4, dtype=torch.long)
    root_pos = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    root_rot = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    target_root_pos = root_pos.clone()
    target_root_rot = root_rot.clone()

    command._apply_step_zero_root_pose_overrides(
        env_ids,
        start_idx,
        root_pos,
        root_rot,
        target_root_pos,
        target_root_rot,
    )

    assert torch.allclose(target_root_pos[0], torch.tensor([11.0, 1.0, 1.0], dtype=torch.float32))
    assert torch.allclose(target_root_rot[0], global_quat[0])
    assert torch.allclose(target_root_pos[1], torch.tensor([22.0, 2.0, 2.0], dtype=torch.float32))
    assert torch.allclose(target_root_rot[1], gen_quat[0])
    assert torch.allclose(target_root_pos[2], root_pos[2])
    assert torch.allclose(target_root_rot[2], root_rot[2])
    assert torch.allclose(target_root_pos[3], torch.tensor([42.0, 2.0, 2.0], dtype=torch.float32))
    assert torch.allclose(target_root_rot[3], gen_quat[0])


def test_hybrid_adaptive_sampler_keeps_track_and_gen_groups_isolated():
    sampler = AdaptiveTimestepsSampler(
        motion_start_idx=torch.tensor([0, 10], dtype=torch.long),
        motion_end_idx=torch.tensor([10, 20], dtype=torch.long),
        device="cpu",
        env_dt=1.0,
        group_names=["track", "gen"],
        bin_width_s=10.0,
        ema_alpha=0.5,
        bin_uniform_mix=1.0,
        motion_uniform_mix=0.0,
    )
    sampler.bin_fail_ema[:] = 0.0
    sampler.bin_fail_ema[1, 1, 0] = 1.0

    track_probs = sampler.motion_probabilities(0, torch.tensor([0, 1], dtype=torch.long))
    gen_probs = sampler.motion_probabilities(1, torch.tensor([0, 1], dtype=torch.long))

    assert torch.allclose(track_probs, torch.tensor([0.5, 0.5], dtype=torch.float32))
    assert gen_probs[1] > gen_probs[0]


def test_hybrid_adaptive_sampler_stats_ignore_inactive_groups():
    sampler = AdaptiveTimestepsSampler(
        motion_start_idx=torch.tensor([0], dtype=torch.long),
        motion_end_idx=torch.tensor([10], dtype=torch.long),
        device="cpu",
        env_dt=1.0,
        group_names=["track", "gen"],
        bin_width_s=10.0,
        ema_alpha=0.5,
        bin_uniform_mix=1.0,
        motion_uniform_mix=1.0,
    )
    sampler.bin_fail_ema[0, 0, 0] = 0.25
    sampler.bin_fail_ema[1, 0, 0] = 1.0

    sampler.get_stats(
        group_priors=torch.tensor([1.0, 0.0], dtype=torch.float32),
        candidate_motion_ids_by_group={
            0: torch.tensor([0], dtype=torch.long),
            1: torch.tensor([0], dtype=torch.long),
        },
        group_first_frame_probs=torch.tensor([0.0, 0.0], dtype=torch.float32),
    )

    assert torch.allclose(sampler.metrics["fail_ema_mean"], torch.tensor(0.25, dtype=torch.float32))


def test_hybrid_gen_envs_bypass_adaptive_reset_sampling():
    command = _make_hybrid_command()
    command.track_env_mask[:] = torch.tensor([True, False, True, False])
    command.gen_env_mask[:] = ~command.track_env_mask

    def _fake_adaptive(env_ids, group_ids):
        assert env_ids.tolist() == [0, 2]
        assert group_ids.tolist() == [0, 0]
        return (
            torch.tensor([10, 12], dtype=torch.long),
            torch.tensor([1, 1], dtype=torch.long),
            torch.tensor([100, 120], dtype=torch.long),
        )

    def _fake_uniform(env_ids):
        assert env_ids.tolist() == [1, 3]
        return (
            torch.tensor([21, 23], dtype=torch.long),
            torch.tensor([210, 230], dtype=torch.long),
        )

    command._sample_adaptive_reset_states = _fake_adaptive
    command._sample_uniform_reset_states = _fake_uniform

    env_ids = torch.arange(4, dtype=torch.long)
    motion_ids, time_steps = command._sample_reset_states_with_adaptive_fallback(
        env_ids,
        command._adaptive_sampler_group_ids(env_ids),
    )

    assert torch.equal(motion_ids, torch.tensor([10, 21, 12, 23], dtype=torch.long))
    assert torch.equal(time_steps, torch.tensor([100, 210, 120, 230], dtype=torch.long))


def test_hybrid_gen_envs_do_not_update_adaptive_sampler():
    command = _make_hybrid_command(num_envs=2)
    command.track_env_mask[:] = torch.tensor([True, False])
    command.gen_env_mask[:] = ~command.track_env_mask
    command.motion_cfg = SimpleNamespace(
        use_adaptive_timesteps_sampler=True,
        adaptive_timestep_sampler=SimpleNamespace(update_interval=1),
    )
    command.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
        motion_start_idx=torch.tensor([0], dtype=torch.long),
        motion_end_idx=torch.tensor([10], dtype=torch.long),
        device="cpu",
        env_dt=1.0,
        group_names=["track", "gen"],
        bin_width_s=10.0,
        ema_alpha=0.5,
        min_segment_steps=0,
    )
    command.adaptive_sampler_start_group_ids = torch.tensor([0, 1], dtype=torch.long)
    command.adaptive_sampler_start_motion_ids = torch.tensor([0, 0], dtype=torch.long)
    command.adaptive_sampler_start_bins = torch.tensor([0, 0], dtype=torch.long)
    command.adaptive_sampler_segment_start_env_step = torch.tensor([0, 0], dtype=torch.long)
    command.adaptive_sampler_segment_active_mask = torch.tensor([True, False], dtype=torch.bool)
    command._adaptive_sampler_global_env_step = 1
    command._adaptive_sampler_collect_rollout = False
    command.time_steps = torch.tensor([0, 0], dtype=torch.long)
    command._env = SimpleNamespace(
        termination_manager=SimpleNamespace(
            terminated=torch.tensor([False, False], dtype=torch.bool),
            time_outs=torch.tensor([True, True], dtype=torch.bool),
        )
    )

    command._record_adaptive_sampler_episode_resets(torch.tensor([0, 1], dtype=torch.long))

    assert torch.allclose(command.adaptive_timesteps_sampler.bin_fail_ema[0, 0, 0], torch.tensor(0.5))
    assert torch.allclose(command.adaptive_timesteps_sampler.bin_fail_ema[1, 0, 0], torch.tensor(1.0))


def test_hybrid_global_env_step_advances_once_per_rollout_step():
    command = _make_hybrid_command(num_envs=2)
    command.motion_cfg = SimpleNamespace(use_adaptive_timesteps_sampler=True)
    command._adaptive_sampler_global_env_step = 3

    MotionCommand._increment_adaptive_sampler_global_env_step(command)

    assert command._adaptive_sampler_global_env_step == 4


def test_hybrid_refresh_active_goals_uses_sampled_goal_frames_and_gen_noise():
    command = _make_hybrid_command()
    command.track_env_mask[:] = torch.tensor([True, False, True, False])
    command.gen_env_mask[:] = ~command.track_env_mask
    command.motion_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    command._goal_time_steps[:] = torch.tensor([1, 4, 1, 4], dtype=torch.long)
    command.ref_body_index = 0
    command.motion = SimpleNamespace(
        motion_end_idx=torch.tensor([2, 5], dtype=torch.long),
        body_pos_w=torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0]],
                [[10.0, 0.0, 0.0]],
                [[20.0, 0.0, 0.0]],
                [[30.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        body_quat_w=torch.tensor(
            [
                [[0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        ),
        has_object=True,
        object_pos_w=torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [10.0, 10.0, 10.0],
                [20.0, 20.0, 20.0],
                [30.0, 30.0, 30.0],
            ],
            dtype=torch.float32,
        ),
        object_quat_w=torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )
    command._env = SimpleNamespace(
        simulator=SimpleNamespace(
            scene=SimpleNamespace(
                env_origins=torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
            )
        )
    )

    robot_pos_noise = torch.tensor([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]], dtype=torch.float32)
    object_pos_noise = torch.tensor([[0.0, 0.25, 0.0], [0.0, -0.25, 0.0]], dtype=torch.float32)
    robot_quat_noise = quat_normalize(
        quat_from_euler_xyz(torch.zeros(2), torch.zeros(2), torch.tensor([0.1, -0.2], dtype=torch.float32))
    )
    object_quat_noise = quat_normalize(
        quat_from_euler_xyz(torch.tensor([0.1, 0.0], dtype=torch.float32), torch.zeros(2), torch.zeros(2))
    )
    sampled = [
        (robot_pos_noise, robot_quat_noise),
        (object_pos_noise, object_quat_noise),
    ]
    command._sample_goal_noise = lambda num_envs, noise_cfg: sampled.pop(0)

    command._refresh_active_goals(torch.arange(4, dtype=torch.long))

    expected_exact_robot = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [31.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [33.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    expected_exact_object = torch.tensor(
        [
            [2.0, 2.0, 2.0],
            [31.0, 30.0, 30.0],
            [4.0, 2.0, 2.0],
            [33.0, 30.0, 30.0],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(command.exact_goal_ref_pos_w, expected_exact_robot)
    assert torch.allclose(command.exact_goal_object_pos_w, expected_exact_object)
    assert torch.allclose(command.goal_ref_pos_w[command.track_env_mask], expected_exact_robot[command.track_env_mask])
    assert torch.allclose(
        command.goal_object_pos_w[command.track_env_mask], expected_exact_object[command.track_env_mask]
    )
    assert torch.allclose(
        command.goal_ref_pos_w[command.gen_env_mask],
        expected_exact_robot[command.gen_env_mask] + robot_pos_noise,
    )
    assert torch.allclose(
        command.goal_object_pos_w[command.gen_env_mask],
        expected_exact_object[command.gen_env_mask] + object_pos_noise,
    )
    assert torch.allclose(
        command.goal_ref_quat_w[command.gen_env_mask],
        quat_normalize(quat_mul(robot_quat_noise, command.exact_goal_ref_quat_w[command.gen_env_mask], w_last=True)),
    )
    assert torch.allclose(
        command.goal_object_quat_w[command.gen_env_mask],
        quat_normalize(quat_mul(object_quat_noise, command.exact_goal_object_quat_w[command.gen_env_mask], w_last=True)),
    )

    assert torch.equal(command.track_env_mask, torch.tensor([True, False, True, False]))
    assert torch.equal(command.gen_env_mask, torch.tensor([False, True, False, True]))


def test_hybrid_goal_refresh_updates_only_refreshed_env_buffers_and_resamples_gen_noise():
    command = _make_hybrid_command()
    command.track_env_mask[:] = torch.tensor([True, False, True, False])
    command.gen_env_mask[:] = ~command.track_env_mask
    command.ref_body_index = 0
    command._goal_time_steps[:] = torch.tensor([0, 3, 2, 4], dtype=torch.long)
    command.motion = SimpleNamespace(
        body_pos_w=torch.tensor(
            [
                [[10.0, 0.0, 0.0]],
                [[20.0, 0.0, 0.0]],
                [[30.0, 0.0, 0.0]],
                [[40.0, 0.0, 0.0]],
                [[50.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        body_quat_w=torch.tensor(
            [
                [[0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        ),
        has_object=True,
        object_pos_w=torch.tensor(
            [
                [11.0, 1.0, 1.0],
                [21.0, 1.0, 1.0],
                [31.0, 1.0, 1.0],
                [41.0, 1.0, 1.0],
                [51.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        object_quat_w=torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )
    command._env = SimpleNamespace(
        simulator=SimpleNamespace(
            scene=SimpleNamespace(
                env_origins=torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [100.0, 0.0, 0.0],
                        [200.0, 0.0, 0.0],
                        [300.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
            )
        )
    )
    robot_pos_noise = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    object_pos_noise = torch.tensor([[0.0, 0.25, 0.0]], dtype=torch.float32)
    identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    sampled = [
        (robot_pos_noise, identity_quat),
        (object_pos_noise, identity_quat),
    ]
    command._sample_goal_noise = lambda num_envs, noise_cfg: sampled.pop(0)

    command._on_goal_time_steps_updated(torch.tensor([1, 2], dtype=torch.long))

    assert torch.allclose(command.exact_goal_ref_pos_w[0], torch.zeros(3, dtype=torch.float32))
    assert torch.allclose(command.exact_goal_ref_pos_w[3], torch.zeros(3, dtype=torch.float32))
    assert torch.allclose(command.exact_goal_ref_pos_w[1], torch.tensor([140.0, 0.0, 0.0], dtype=torch.float32))
    assert torch.allclose(command.exact_goal_ref_pos_w[2], torch.tensor([230.0, 0.0, 0.0], dtype=torch.float32))
    assert torch.allclose(command.goal_ref_pos_w[1], torch.tensor([140.5, 0.0, 0.0], dtype=torch.float32))
    assert torch.allclose(command.goal_ref_pos_w[2], torch.tensor([230.0, 0.0, 0.0], dtype=torch.float32))
    assert torch.allclose(command.exact_goal_object_pos_w[1], torch.tensor([141.0, 1.0, 1.0], dtype=torch.float32))
    assert torch.allclose(command.exact_goal_object_pos_w[2], torch.tensor([231.0, 1.0, 1.0], dtype=torch.float32))
    assert torch.allclose(command.goal_object_pos_w[1], torch.tensor([141.0, 1.25, 1.0], dtype=torch.float32))
    assert torch.allclose(command.goal_object_pos_w[2], torch.tensor([231.0, 1.0, 1.0], dtype=torch.float32))


def test_hybrid_sample_goal_noise_clips_each_axis(monkeypatch):
    command = _make_hybrid_command(num_envs=2)
    noise_cfg = GoalPoseNoiseConfig(
        pos_std_xyz=[0.5, 0.25, 0.1],
        pos_clip_xyz=[0.2, 0.1, 0.05],
        rpy_std=[0.5, 0.25, 0.1],
        rpy_clip=[0.2, 0.1, 0.05],
    )
    samples = [
        torch.tensor([[2.0, -2.0, 1.0], [0.1, -0.1, 0.2]], dtype=torch.float32),
        torch.tensor([[2.0, -2.0, 1.0], [0.1, -0.1, 0.2]], dtype=torch.float32),
    ]

    def _fake_randn(*shape, **kwargs):
        return samples.pop(0)

    monkeypatch.setattr(torch, "randn", _fake_randn)

    pos_noise, quat_noise = command._sample_goal_noise(2, noise_cfg)

    assert torch.allclose(
        pos_noise,
        torch.tensor([[0.2, -0.1, 0.05], [0.05, -0.025, 0.02]], dtype=torch.float32),
    )
    expected_rpy = torch.tensor([[0.2, -0.1, 0.05], [0.05, -0.025, 0.02]], dtype=torch.float32)
    expected_quat = quat_normalize(quat_from_euler_xyz(expected_rpy[:, 0], expected_rpy[:, 1], expected_rpy[:, 2]))
    assert torch.allclose(quat_noise, expected_quat)


def test_hybrid_object_goal_noise_curriculum_scales_only_xy_and_yaw():
    command = _make_hybrid_command(num_envs=2)
    command.set_object_goal_noise_curriculum_scale(0.5)
    noise_cfg = GoalPoseNoiseConfig(
        pos_std_xyz=[0.2, 0.4, 0.6],
        pos_clip_xyz=[0.3, 0.5, 0.7],
        rpy_std=[0.1, 0.2, 0.3],
        rpy_clip=[0.4, 0.5, 0.6],
    )
    captured: dict[str, GoalPoseNoiseConfig] = {}
    identity_quat = torch.zeros(2, 4, dtype=torch.float32)
    identity_quat[:, 3] = 1.0

    def _fake_sample(num_envs, forwarded_cfg):
        captured["cfg"] = forwarded_cfg
        return torch.zeros(num_envs, 3, dtype=torch.float32), identity_quat[:num_envs].clone()

    command._sample_goal_noise = _fake_sample

    command._sample_goal_noise_with_axis_scale(
        2,
        noise_cfg,
        pos_axis_scale_xyz=command.object_goal_noise_pos_axis_scale,
        rpy_axis_scale=command.object_goal_noise_rpy_axis_scale,
    )

    scaled_cfg = captured["cfg"]
    assert torch.allclose(torch.tensor(scaled_cfg.pos_std_xyz), torch.tensor([0.1, 0.2, 0.6]))
    assert torch.allclose(torch.tensor(scaled_cfg.pos_clip_xyz), torch.tensor([0.15, 0.25, 0.7]))
    assert torch.allclose(torch.tensor(scaled_cfg.rpy_std), torch.tensor([0.1, 0.2, 0.15]))
    assert torch.allclose(torch.tensor(scaled_cfg.rpy_clip), torch.tensor([0.4, 0.5, 0.3]))


def test_hybrid_goal_noise_sampling_without_curriculum_uses_original_config():
    command = _make_hybrid_command(num_envs=1)
    noise_cfg = GoalPoseNoiseConfig(
        pos_std_xyz=[0.3, 0.2, 0.1],
        pos_clip_xyz=[0.4, 0.3, 0.2],
        rpy_std=[0.5, 0.4, 0.3],
        rpy_clip=[0.6, 0.5, 0.4],
    )
    captured: dict[str, GoalPoseNoiseConfig] = {}
    active_goal_pos_w = torch.zeros(1, 3, dtype=torch.float32)
    active_goal_quat_w = torch.zeros(1, 4, dtype=torch.float32)
    active_goal_quat_w[:, 3] = 1.0

    def _fake_sample(num_envs, forwarded_cfg):
        captured["cfg"] = forwarded_cfg
        quat = torch.zeros(num_envs, 4, dtype=torch.float32)
        quat[:, 3] = 1.0
        return torch.zeros(num_envs, 3, dtype=torch.float32), quat

    command._sample_goal_noise = _fake_sample

    command._apply_gen_goal_noise(
        torch.tensor([0], dtype=torch.long),
        exact_goal_pos_w=torch.zeros(1, 3, dtype=torch.float32),
        exact_goal_quat_w=active_goal_quat_w.clone(),
        noise_cfg=noise_cfg,
        active_goal_pos_w=active_goal_pos_w,
        active_goal_quat_w=active_goal_quat_w,
    )

    assert captured["cfg"] is noise_cfg


def test_hybrid_reapply_gen_object_goal_noise_updates_only_gen_object_goals():
    command = _make_hybrid_command()
    command.track_env_mask[:] = torch.tensor([True, False, True, False])
    command.gen_env_mask[:] = ~command.track_env_mask
    command.hybrid_cfg = HybridMotionConfig(
        track_ratio=0.5,
        env_partition_seed=0,
        robot_goal_noise=GoalPoseNoiseConfig(),
        object_goal_noise=GoalPoseNoiseConfig(),
    )
    command.motion = SimpleNamespace(has_object=True)
    command.set_object_goal_noise_curriculum_scale(0.5)
    command._exact_goal_object_pos_w = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=torch.float32,
    )
    command._exact_goal_object_quat_w = torch.zeros(4, 4, dtype=torch.float32)
    command._exact_goal_object_quat_w[:, 3] = 1.0
    command._active_goal_object_pos_w = command._exact_goal_object_pos_w.clone()
    command._active_goal_object_quat_w = command._exact_goal_object_quat_w.clone()
    command._active_goal_ref_pos_w = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    object_pos_noise = torch.tensor([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]], dtype=torch.float32)
    identity_quat = torch.zeros(2, 4, dtype=torch.float32)
    identity_quat[:, 3] = 1.0
    command._sample_goal_noise = lambda num_envs, noise_cfg: (object_pos_noise, identity_quat)

    command.reapply_gen_object_goal_noise(torch.arange(4, dtype=torch.long))

    assert torch.allclose(command.goal_object_pos_w[command.track_env_mask], command.exact_goal_object_pos_w[command.track_env_mask])
    assert torch.allclose(
        command.goal_object_pos_w[command.gen_env_mask],
        command.exact_goal_object_pos_w[command.gen_env_mask] + object_pos_noise,
    )
    assert torch.allclose(
        command._active_goal_ref_pos_w,
        torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
                [40.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
