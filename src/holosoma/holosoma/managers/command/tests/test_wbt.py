from types import SimpleNamespace

import numpy as np
import torch

from holosoma.config_types.command import EvalFixedResetConfig, StepZeroRootNoiseConfig
from holosoma.managers.command.terms.wbt import (
    AdaptiveTimestepsSampler,
    MotionCommand,
    MotionLoader,
    MultiMotionLoader,
    derive_angular_velocity_from_quat_sequence,
)
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import quat_from_euler_xyz


def _z_axis_quat_sequence(angles: torch.Tensor) -> torch.Tensor:
    half_angles = angles * 0.5
    zeros = torch.zeros_like(half_angles)
    return torch.stack([zeros, zeros, torch.sin(half_angles), torch.cos(half_angles)], dim=-1)


def _make_minimal_motion_command_for_reset(
    *,
    use_adaptive: bool,
    evaluating: bool,
    start_prob: float,
    freeze_prob: float,
    eval_sample_random_timestep: bool = False,
    eval_fixed_reset: EvalFixedResetConfig | None = None,
    update_interval: int = 10,
    dump_interval: int = 200,
) -> MotionCommand:
    command = object.__new__(MotionCommand)
    command.device = torch.device("cpu")
    command.num_envs = 1
    command.motion_cfg = SimpleNamespace(
        use_adaptive_timesteps_sampler=use_adaptive,
        adaptive_timestep_sampler=SimpleNamespace(
            bin_width_s=1.0,
            ema_alpha=0.5,
            bin_uniform_mix=0.1,
            motion_uniform_mix=0.1,
            score_power=1.0,
            min_segment_steps=0,
            update_interval=update_interval,
            dump_interval=dump_interval,
            dump_file="sampleweight.txt",
        ),
        start_at_timestep_zero_prob=start_prob,
        eval_sample_random_timestep=eval_sample_random_timestep,
        eval_fixed_reset=eval_fixed_reset or EvalFixedResetConfig(),
        freeze_at_timestep_zero_prob=freeze_prob,
        body_names_to_track=["torso_link"],
    )
    command.init_pose_cfg = SimpleNamespace(
        overall_noise_scale=0.0,
        dof_pos=0.0,
        root_pos=[0.0, 0.0, 0.0],
        root_rot=[0.0, 0.0, 0.0],
        root_lin_vel=[0.0, 0.0, 0.0],
        root_ang_vel=[0.0, 0.0, 0.0],
        object_pos=[0.0, 0.0, 0.0],
        step_zero_root_noise=None,
    )

    identity_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32).repeat(5, 1, 1)
    command.motion = SimpleNamespace(
        num_motions=1,
        motion_start_idx=torch.tensor([0], dtype=torch.long),
        motion_end_idx=torch.tensor([5], dtype=torch.long),
        fps=10.0,
        has_object=False,
        joint_pos=torch.zeros(5, 1, dtype=torch.float32),
        joint_vel=torch.zeros(5, 1, dtype=torch.float32),
        body_pos_w=torch.zeros(5, 1, 3, dtype=torch.float32),
        body_quat_w=identity_quat,
        body_lin_vel_w=torch.zeros(5, 1, 3, dtype=torch.float32),
        body_ang_vel_w=torch.zeros(5, 1, 3, dtype=torch.float32),
    )
    command.ref_body_index = 0
    command.tracked_body_indexes = torch.tensor([0], dtype=torch.long)
    command._env = SimpleNamespace(
        is_evaluating=evaluating,
        dt=0.1,
        episode_length_buf=torch.tensor([0], dtype=torch.long),
        termination_manager=SimpleNamespace(
            terminated=torch.tensor([False], dtype=torch.bool),
            time_outs=torch.tensor([False], dtype=torch.bool),
        ),
        simulator=SimpleNamespace(
            scene=SimpleNamespace(env_origins=torch.zeros(1, 3, dtype=torch.float32)),
            dof_pos_limits=torch.tensor([[-1.0, 1.0]], dtype=torch.float32),
            dof_pos=torch.zeros(1, 1, dtype=torch.float32),
            dof_vel=torch.zeros(1, 1, dtype=torch.float32),
            robot_root_states=torch.zeros(1, 13, dtype=torch.float32),
            _rigid_body_pos=torch.zeros(1, 1, 3, dtype=torch.float32),
            _rigid_body_rot=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32),
            _rigid_body_vel=torch.zeros(1, 1, 3, dtype=torch.float32),
            _rigid_body_ang_vel=torch.zeros(1, 1, 3, dtype=torch.float32),
        ),
    )

    if use_adaptive:
        command.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
            command.motion.motion_start_idx,
            command.motion.motion_end_idx,
            device=command.device,
            env_dt=0.1,
            group_names=["default"],
            ema_alpha=0.5,
            min_segment_steps=command.motion_cfg.adaptive_timestep_sampler.min_segment_steps,
        )

    MotionCommand.init_buffers(command)
    if use_adaptive:
        command._adaptive_sampler_collect_rollout = False
        command._refresh_adaptive_sampler_runtime_cache()
        command._refresh_adaptive_sampler_metrics()
    return command


def test_derive_angular_velocity_from_quat_sequence_matches_constant_spin():
    fps = 20.0
    angular_speed = 1.25
    time = torch.arange(6, dtype=torch.float32) / fps
    quat_sequence = _z_axis_quat_sequence(angular_speed * time)

    angular_velocity = derive_angular_velocity_from_quat_sequence(quat_sequence, fps=fps)

    expected = torch.zeros_like(angular_velocity)
    expected[:, 2] = angular_speed
    assert torch.allclose(angular_velocity, expected, atol=1.0e-4)


def test_derive_angular_velocity_from_quat_sequence_returns_zero_for_single_frame():
    quat_sequence = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

    angular_velocity = derive_angular_velocity_from_quat_sequence(quat_sequence, fps=60.0)

    assert torch.equal(angular_velocity, torch.zeros(1, 3, dtype=torch.float32))


def test_build_default_pose_state_keeps_object_pose_but_zeros_object_velocities():
    motion_command = object.__new__(MotionCommand)
    motion_command.device = torch.device("cpu")
    motion_command._env = SimpleNamespace(
        robot_config=SimpleNamespace(
            init_state=SimpleNamespace(
                rot=(0.0, 0.0, 0.0, 1.0),
                pos=(0.0, 0.0, 0.8),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            )
        ),
        default_dof_pos_base=torch.zeros(1, 2, dtype=torch.float32),
    )
    motion_command.motion = SimpleNamespace(
        has_object=True,
        body_pos_w=torch.zeros(2, 1, 3, dtype=torch.float32),
        body_quat_w=torch.tensor(
            [
                [[0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        ),
        _object_pos_w=torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.float32,
        ),
        _object_quat_w=torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.70710678, 0.70710678],
            ],
            dtype=torch.float32,
        ),
        _object_lin_vel_w=torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [1.0, 2.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        _object_ang_vel_w=torch.tensor(
            [
                [0.4, 0.5, 0.6],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.float32,
        ),
    )
    motion_command._capture_body_states = lambda *args, **kwargs: {
        "pos": torch.zeros(1, 3, dtype=torch.float32),
        "quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        "lin_vel": torch.zeros(1, 3, dtype=torch.float32),
        "ang_vel": torch.zeros(1, 3, dtype=torch.float32),
    }
    motion_command._map_robot_bodies_to_motion_order = lambda tensor: tensor

    default_state = motion_command._build_default_pose_state(use_motion_end=True)

    assert torch.equal(default_state["object_pos"], torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32))
    assert torch.equal(
        default_state["object_quat"],
        torch.tensor([0.0, 0.0, 0.70710678, 0.70710678], dtype=torch.float32),
    )
    assert torch.equal(default_state["object_lin_vel"], torch.zeros(3, dtype=torch.float32))
    assert torch.equal(default_state["object_ang_vel"], torch.zeros(3, dtype=torch.float32))


def test_motion_loader_append_terminal_pose_hold_repeats_last_pose_and_zeros_velocities():
    loader = object.__new__(MotionLoader)
    loader.has_object = True
    loader._joint_pos = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    loader._joint_vel = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    loader._body_pos_w = torch.tensor(
        [
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    loader._body_quat_w = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.70710678, 0.70710678]],
        ],
        dtype=torch.float32,
    )
    loader._body_lin_vel_w = torch.tensor(
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
        ],
        dtype=torch.float32,
    )
    loader._body_ang_vel_w = torch.tensor(
        [
            [[7.0, 8.0, 9.0]],
            [[10.0, 11.0, 12.0]],
        ],
        dtype=torch.float32,
    )
    loader._object_pos_w = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float32,
    )
    loader._object_quat_w = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.70710678, 0.70710678],
        ],
        dtype=torch.float32,
    )
    loader._object_lin_vel_w = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [1.0, 2.0, 3.0],
        ],
        dtype=torch.float32,
    )
    loader._object_ang_vel_w = torch.tensor(
        [
            [0.4, 0.5, 0.6],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float32,
    )
    loader.time_step_total = 2

    loader.append_terminal_pose_hold(2)

    assert loader.time_step_total == 4
    assert torch.equal(
        loader._joint_pos,
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [3.0, 4.0],
                [3.0, 4.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(loader._joint_vel[-2:], torch.zeros(2, 2, dtype=torch.float32))
    assert torch.equal(
        loader._body_pos_w[-2:],
        torch.tensor(
            [
                [[2.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        loader._body_quat_w[-2:],
        torch.tensor(
            [
                [[0.0, 0.0, 0.70710678, 0.70710678]],
                [[0.0, 0.0, 0.70710678, 0.70710678]],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(loader._body_lin_vel_w[-2:], torch.zeros(2, 1, 3, dtype=torch.float32))
    assert torch.equal(loader._body_ang_vel_w[-2:], torch.zeros(2, 1, 3, dtype=torch.float32))
    assert torch.equal(
        loader._object_pos_w[-2:],
        torch.tensor(
            [
                [4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        loader._object_quat_w[-2:],
        torch.tensor(
            [
                [0.0, 0.0, 0.70710678, 0.70710678],
                [0.0, 0.0, 0.70710678, 0.70710678],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(loader._object_lin_vel_w[-2:], torch.zeros(2, 3, dtype=torch.float32))
    assert torch.equal(loader._object_ang_vel_w[-2:], torch.zeros(2, 3, dtype=torch.float32))


def test_multi_motion_loader_append_terminal_pose_hold_extends_each_motion_clip():
    loader = object.__new__(MultiMotionLoader)
    loader.has_object = True
    loader._motion_start_idx = torch.tensor([0, 2], dtype=torch.long)
    loader._motion_end_idx = torch.tensor([2, 5], dtype=torch.long)
    loader._num_motions = 2
    loader._joint_pos = torch.tensor([[1.0], [2.0], [10.0], [20.0], [30.0]], dtype=torch.float32)
    loader._joint_vel = torch.tensor([[0.1], [0.2], [1.0], [2.0], [3.0]], dtype=torch.float32)
    loader._body_pos_w = torch.tensor(
        [
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0]],
            [[20.0, 0.0, 0.0]],
            [[30.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    loader._body_quat_w = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.70710678, 0.70710678]],
        ],
        dtype=torch.float32,
    )
    loader._body_lin_vel_w = torch.tensor(
        [
            [[0.1, 0.0, 0.0]],
            [[0.2, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    loader._body_ang_vel_w = torch.tensor(
        [
            [[0.0, 0.1, 0.0]],
            [[0.0, 0.2, 0.0]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 2.0, 0.0]],
            [[0.0, 3.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    loader._object_pos_w = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    loader._object_quat_w = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.70710678, 0.70710678],
        ],
        dtype=torch.float32,
    )
    loader._object_lin_vel_w = torch.tensor(
        [
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    loader._object_ang_vel_w = torch.tensor(
        [
            [0.0, 0.1, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=torch.float32,
    )
    loader.time_step_total = 5

    loader.append_terminal_pose_hold(2)

    assert loader.time_step_total == 9
    assert loader._num_motions == 2
    assert torch.equal(loader._motion_start_idx, torch.tensor([0, 4], dtype=torch.long))
    assert torch.equal(loader._motion_end_idx, torch.tensor([4, 9], dtype=torch.long))
    assert torch.equal(
        loader._joint_pos.squeeze(-1),
        torch.tensor([1.0, 2.0, 2.0, 2.0, 10.0, 20.0, 30.0, 30.0, 30.0], dtype=torch.float32),
    )
    assert torch.equal(
        loader._joint_vel.squeeze(-1),
        torch.tensor([0.1, 0.2, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0], dtype=torch.float32),
    )
    assert torch.equal(loader._body_pos_w[2:4], torch.tensor([[[2.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]], dtype=torch.float32))
    assert torch.equal(loader._body_lin_vel_w[2:4], torch.zeros(2, 1, 3, dtype=torch.float32))
    assert torch.equal(
        loader._object_pos_w[7:9],
        torch.tensor(
            [
                [30.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(loader._object_lin_vel_w[7:9], torch.zeros(2, 3, dtype=torch.float32))


def test_motion_command_maybe_add_terminal_pose_hold_uses_duration_and_dt():
    motion_command = object.__new__(MotionCommand)
    motion_command.motion_cfg = SimpleNamespace(enable_terminal_pose_hold=True, terminal_pose_hold_duration_s=1.0)
    motion_command._env = SimpleNamespace(dt=0.2)

    calls: list[int] = []

    class DummyMotion:
        def append_terminal_pose_hold(self, num_steps: int):
            calls.append(num_steps)
            return self

    motion_command.motion = DummyMotion()

    motion_command._maybe_add_terminal_pose_hold()

    assert calls == [5]


def test_adaptive_timestep_sampler_updates_fail_ema_with_failure_and_success_credit():
    sampler = AdaptiveTimestepsSampler(
        motion_start_idx=torch.tensor([0], dtype=torch.long),
        motion_end_idx=torch.tensor([9], dtype=torch.long),
        device="cpu",
        env_dt=1.0,
        group_names=["default"],
        bin_width_s=3.0,
        ema_alpha=0.5,
        bin_uniform_mix=0.0,
        motion_uniform_mix=0.0,
        min_segment_steps=0,
    )

    sampler.record_segment_end(
        group_ids=torch.tensor([0], dtype=torch.long),
        motion_ids=torch.tensor([0], dtype=torch.long),
        start_bins=torch.tensor([0], dtype=torch.long),
        segment_step_counts=torch.tensor([3], dtype=torch.long),
        terminal_time_steps=torch.tensor([7], dtype=torch.long),
        failure_mask=torch.tensor([True], dtype=torch.bool),
    )

    assert torch.allclose(sampler.bin_fail_ema[0, 0, :3], torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32))

    sampler.record_segment_end(
        group_ids=torch.tensor([0], dtype=torch.long),
        motion_ids=torch.tensor([0], dtype=torch.long),
        start_bins=torch.tensor([0], dtype=torch.long),
        segment_step_counts=torch.tensor([2], dtype=torch.long),
        terminal_time_steps=torch.tensor([4], dtype=torch.long),
        failure_mask=torch.tensor([False], dtype=torch.bool),
    )

    assert torch.allclose(sampler.bin_fail_ema[0, 0, :3], torch.tensor([0.25, 0.25, 1.0], dtype=torch.float32))

    sampler.record_segment_end(
        group_ids=torch.tensor([0], dtype=torch.long),
        motion_ids=torch.tensor([0], dtype=torch.long),
        start_bins=torch.tensor([2], dtype=torch.long),
        segment_step_counts=torch.tensor([1], dtype=torch.long),
        terminal_time_steps=torch.tensor([8], dtype=torch.long),
        failure_mask=torch.tensor([False], dtype=torch.bool),
    )

    assert torch.allclose(sampler.bin_fail_ema[0, 0, :3], torch.tensor([0.25, 0.25, 0.5], dtype=torch.float32))


def test_adaptive_timestep_sampler_weights_motion_and_bin_difficulty_locally():
    sampler = AdaptiveTimestepsSampler(
        motion_start_idx=torch.tensor([0, 10], dtype=torch.long),
        motion_end_idx=torch.tensor([10, 20], dtype=torch.long),
        device="cpu",
        env_dt=0.5,
        group_names=["default"],
        bin_width_s=1.0,
        ema_alpha=0.5,
        bin_uniform_mix=0.0,
        motion_uniform_mix=0.0,
    )
    sampler.bin_fail_ema[:] = 0.1
    sampler.bin_fail_ema[0, 0, 4] = 1.0

    motion_probs = sampler.motion_probabilities(0, torch.tensor([0, 1], dtype=torch.long))
    bin_probs_motion0 = sampler.bin_probabilities(0, 0)
    bin_probs_motion1 = sampler.bin_probabilities(0, 1)

    assert motion_probs[0] > motion_probs[1]
    assert bin_probs_motion0[-1] > bin_probs_motion0[0]
    assert torch.allclose(bin_probs_motion1, torch.full_like(bin_probs_motion1, 0.2))


def test_adaptive_timestep_sampler_merges_short_final_bin_into_previous_bin():
    sampler = AdaptiveTimestepsSampler(
        motion_start_idx=torch.tensor([0], dtype=torch.long),
        motion_end_idx=torch.tensor([807], dtype=torch.long),
        device="cpu",
        env_dt=0.02,
        group_names=["default"],
        bin_width_s=2.0,
        ema_alpha=0.5,
    )

    assert sampler.bin_width_steps == 100
    assert int(sampler.motion_num_bins[0].item()) == 8
    assert torch.equal(
        sampler.bin_local_start_steps[0, :8],
        torch.tensor([0, 100, 200, 300, 400, 500, 600, 700], dtype=torch.long),
    )
    assert torch.equal(
        sampler.bin_local_end_steps[0, :8],
        torch.tensor([100, 200, 300, 400, 500, 600, 700, 807], dtype=torch.long),
    )

    motion_ids = torch.zeros(3, dtype=torch.long)
    time_steps = torch.tensor([700, 799, 806], dtype=torch.long)
    assert torch.equal(sampler.time_step_to_bin(motion_ids, time_steps), torch.full((3,), 7, dtype=torch.long))


def test_adaptive_timestep_sampler_zero_group_priors_yield_zeroed_stats():
    sampler = AdaptiveTimestepsSampler(
        motion_start_idx=torch.tensor([0], dtype=torch.long),
        motion_end_idx=torch.tensor([10], dtype=torch.long),
        device="cpu",
        env_dt=1.0,
        group_names=["default"],
        bin_width_s=10.0,
        ema_alpha=0.5,
    )

    sampler.get_stats(
        group_priors=torch.tensor([0.0], dtype=torch.float32),
        candidate_motion_ids_by_group={0: torch.tensor([0], dtype=torch.long)},
        group_first_frame_probs=torch.tensor([0.0], dtype=torch.float32),
    )

    assert torch.allclose(sampler.metrics["fail_ema_mean"], torch.tensor(0.0, dtype=torch.float32))
    assert torch.allclose(sampler.metrics["sampling_top1_prob"], torch.tensor(0.0, dtype=torch.float32))


def test_motion_command_step_zero_root_noise_override_applies_only_at_clip_start():
    command = object.__new__(MotionCommand)
    command.device = torch.device("cpu")
    command.init_pose_cfg = SimpleNamespace(
        step_zero_root_noise=StepZeroRootNoiseConfig(
            pos_std_xyz=[4.0, 5.0, 6.0],
            pos_clip_xyz=[7.0, 8.0, 9.0],
            rpy_std=[0.4, 0.5, 0.6],
            rpy_clip=[0.7, 0.8, 0.9],
        ),
    )
    command.time_steps = torch.tensor([0, 7], dtype=torch.long)
    quat_noise = quat_from_euler_xyz(
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.2], dtype=torch.float32),
    )
    command._sample_step_zero_root_noise = lambda num_envs, noise_cfg, **_: (
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        quat_noise,
    )

    env_ids = torch.tensor([0, 1], dtype=torch.long)
    start_idx = torch.tensor([0, 3], dtype=torch.long)
    root_pos = torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=torch.float32)
    root_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
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

    assert torch.allclose(target_root_pos[0], torch.tensor([11.0, 2.0, 3.0], dtype=torch.float32))
    assert torch.allclose(target_root_rot[0], quat_noise[0])
    assert torch.allclose(target_root_pos[1], root_pos[1])
    assert torch.allclose(target_root_rot[1], root_rot[1])


def test_motion_command_reset_uses_frame_zero_during_evaluation_with_adaptive_sampling():
    command = _make_minimal_motion_command_for_reset(
        use_adaptive=True,
        evaluating=True,
        start_prob=0.0,
        freeze_prob=0.0,
    )

    command.reset(torch.tensor([0], dtype=torch.long))

    assert int(command.motion_ids[0].item()) == 0
    assert int(command.time_steps[0].item()) == 0


def test_motion_command_reset_can_sample_random_timestep_during_evaluation():
    command = _make_minimal_motion_command_for_reset(
        use_adaptive=True,
        evaluating=True,
        start_prob=0.0,
        freeze_prob=0.0,
        eval_sample_random_timestep=True,
    )

    def fake_sample_uniform_reset_states(env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert torch.equal(env_ids, torch.tensor([0], dtype=torch.long))
        return torch.tensor([0], dtype=torch.long), torch.tensor([3], dtype=torch.long)

    command._sample_uniform_reset_states = fake_sample_uniform_reset_states

    command.reset(torch.tensor([0], dtype=torch.long))

    assert int(command.motion_ids[0].item()) == 0
    assert int(command.time_steps[0].item()) == 3


def test_motion_command_eval_fixed_reset_samples_requested_bin_start():
    command = object.__new__(MotionCommand)
    command.device = torch.device("cpu")
    command.motion_cfg = SimpleNamespace(
        adaptive_timestep_sampler=SimpleNamespace(bin_width_s=1.0),
        eval_fixed_reset=EvalFixedResetConfig(
            enabled=True,
            motion_id=0,
            bin_id=12,
            randomize_within_bin=False,
        ),
    )
    command._env = SimpleNamespace(dt=0.02)
    command.motion = SimpleNamespace(
        num_motions=1,
        motion_start_idx=torch.tensor([0], dtype=torch.long),
        motion_end_idx=torch.tensor([807], dtype=torch.long),
    )

    motion_ids, time_steps = command._sample_eval_fixed_reset_states(torch.arange(4, dtype=torch.long))

    assert torch.equal(motion_ids, torch.zeros(4, dtype=torch.long))
    assert torch.equal(time_steps, torch.full((4,), 600, dtype=torch.long))


def test_motion_command_eval_fixed_reset_uses_merged_short_final_bin():
    command = object.__new__(MotionCommand)
    command.device = torch.device("cpu")
    command.motion_cfg = SimpleNamespace(
        adaptive_timestep_sampler=SimpleNamespace(bin_width_s=2.0),
        eval_fixed_reset=EvalFixedResetConfig(
            enabled=True,
            motion_id=0,
            bin_id=7,
            randomize_within_bin=False,
        ),
    )
    command._env = SimpleNamespace(dt=0.02)
    command.motion = SimpleNamespace(
        num_motions=1,
        motion_start_idx=torch.tensor([0], dtype=torch.long),
        motion_end_idx=torch.tensor([807], dtype=torch.long),
    )

    motion_ids, time_steps = command._sample_eval_fixed_reset_states(torch.arange(4, dtype=torch.long))

    assert torch.equal(motion_ids, torch.zeros(4, dtype=torch.long))
    assert torch.equal(time_steps, torch.full((4,), 700, dtype=torch.long))


def test_motion_command_eval_fixed_reset_takes_priority_over_random_eval_sampling():
    command = _make_minimal_motion_command_for_reset(
        use_adaptive=False,
        evaluating=True,
        start_prob=1.0,
        freeze_prob=0.0,
        eval_sample_random_timestep=True,
        eval_fixed_reset=EvalFixedResetConfig(enabled=True, motion_id=0, local_frame=2),
    )

    def fake_sample_uniform_reset_states(env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("eval_fixed_reset should take priority over eval_sample_random_timestep")

    command._sample_uniform_reset_states = fake_sample_uniform_reset_states

    command.reset(torch.tensor([0], dtype=torch.long))

    assert int(command.motion_ids[0].item()) == 0
    assert int(command.time_steps[0].item()) == 2


def test_motion_command_reset_honors_start_at_timestep_zero_prob():
    command = _make_minimal_motion_command_for_reset(
        use_adaptive=False,
        evaluating=False,
        start_prob=1.0,
        freeze_prob=0.0,
    )

    command.reset(torch.tensor([0], dtype=torch.long))

    assert int(command.time_steps[0].item()) == 0


def test_motion_command_step_honors_freeze_at_timestep_zero_prob():
    command = _make_minimal_motion_command_for_reset(
        use_adaptive=False,
        evaluating=False,
        start_prob=1.0,
        freeze_prob=1.0,
    )
    command.reset(torch.tensor([0], dtype=torch.long))

    command.step()

    assert int(command.time_steps[0].item()) == 0


def test_motion_command_on_training_step_writes_adaptive_sampler_dump(tmp_path):
    command = _make_minimal_motion_command_for_reset(
        use_adaptive=True,
        evaluating=False,
        start_prob=0.0,
        freeze_prob=0.0,
    )
    command.adaptive_timesteps_sampler.bin_fail_ema[:] = 0.25
    command.adaptive_timesteps_sampler.bin_fail_ema[0, 0, 0] = 0.75

    command.on_training_step(training_step=200, log_dir=str(tmp_path), is_main_process=True)

    dump_path = tmp_path / "sampleweight.txt"
    assert dump_path.exists()
    contents = dump_path.read_text(encoding="utf-8")
    assert "step 200" in contents
    assert "group default" in contents
    assert "first_frame_prob 0.00000000" in contents
    assert "motion 0" in contents
    assert "columns group motion_id bin start_frame end_frame fail_ema" in contents


def test_motion_command_update_metrics_uses_cached_adaptive_sampler_stats(monkeypatch):
    command = _make_minimal_motion_command_for_reset(
        use_adaptive=True,
        evaluating=False,
        start_prob=0.0,
        freeze_prob=0.0,
    )
    command.metrics = {}
    calls = []

    monkeypatch.setattr(
        command.adaptive_timesteps_sampler,
        "get_stats",
        lambda **kwargs: calls.append(kwargs),
    )

    command.update_metrics()

    assert calls == []
    assert "motion/adaptive_timesteps_sampler_entropy" in command.metrics


def test_motion_command_on_training_step_refreshes_adaptive_sampler_stats_without_dump(monkeypatch):
    command = _make_minimal_motion_command_for_reset(
        use_adaptive=True,
        evaluating=False,
        start_prob=0.0,
        freeze_prob=0.0,
    )
    calls = []

    monkeypatch.setattr(
        command.adaptive_timesteps_sampler,
        "get_stats",
        lambda **kwargs: calls.append(kwargs),
    )

    command.on_training_step(training_step=50, log_dir=None, is_main_process=False)

    assert len(calls) == 1


def test_motion_command_on_training_step_refreshes_adaptive_sampler_stats_each_iteration(monkeypatch):
    command = _make_minimal_motion_command_for_reset(
        use_adaptive=True,
        evaluating=False,
        start_prob=0.0,
        freeze_prob=0.0,
    )
    calls = []

    monkeypatch.setattr(
        command.adaptive_timesteps_sampler,
        "get_stats",
        lambda **kwargs: calls.append(kwargs),
    )

    command.on_training_step(training_step=1, log_dir=None, is_main_process=False)

    assert len(calls) == 1


def test_motion_command_on_training_step_refreshes_adaptive_sampler_stats_before_dump(tmp_path, monkeypatch):
    command = _make_minimal_motion_command_for_reset(
        use_adaptive=True,
        evaluating=False,
        start_prob=0.0,
        freeze_prob=0.0,
        dump_interval=75,
    )
    calls = []

    monkeypatch.setattr(
        command.adaptive_timesteps_sampler,
        "get_stats",
        lambda **kwargs: calls.append(kwargs),
    )

    command.on_training_step(training_step=75, log_dir=str(tmp_path), is_main_process=True)

    assert len(calls) == 1
    assert (tmp_path / "sampleweight.txt").exists()


def test_motion_command_on_rollout_end_applies_batched_adaptive_sampler_update():
    command = _make_minimal_motion_command_for_reset(
        use_adaptive=True,
        evaluating=False,
        start_prob=0.0,
        freeze_prob=0.0,
        update_interval=10,
    )
    command._adaptive_sampler_collect_rollout = True
    command.adaptive_sampler_start_group_ids[:] = 0
    command.adaptive_sampler_start_motion_ids[:] = 0
    command.adaptive_sampler_start_bins[:] = 0
    command.adaptive_sampler_segment_start_env_step[:] = 0
    command.adaptive_sampler_segment_active_mask[:] = True
    command._adaptive_sampler_global_env_step = 3
    command.time_steps[:] = 3
    command._env.termination_manager.terminated[:] = False
    command._env.termination_manager.time_outs[:] = True

    command._record_adaptive_sampler_episode_resets(torch.tensor([0], dtype=torch.long))
    assert command.adaptive_timesteps_sampler.has_pending_updates()
    assert torch.allclose(command.adaptive_timesteps_sampler.bin_fail_ema[0, 0, 0], torch.tensor(1.0))

    command.on_rollout_end(training_step=9, log_dir=None, is_main_process=False)

    assert not command.adaptive_timesteps_sampler.has_pending_updates()
    assert torch.allclose(command.adaptive_timesteps_sampler.bin_fail_ema[0, 0, 0], torch.tensor(0.5))


def test_motion_loader_treats_fixed_scene_metadata_as_robot_only_motion():
    motion_file = "holosoma/data/motions/g1_29dof/whole_body_tracking/climbox1/walk_climb1_reframed.npz"
    resolved_motion_file = resolve_data_file_path(motion_file)
    with np.load(resolved_motion_file, allow_pickle=False) as data:
        body_names = data["body_names"].tolist()
        joint_names = data["joint_names"].tolist()

    loader = MotionLoader(motion_file, body_names, joint_names, device="cpu")

    assert loader.has_object is False
