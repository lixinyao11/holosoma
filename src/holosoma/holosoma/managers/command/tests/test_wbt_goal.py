from types import SimpleNamespace

import torch

from holosoma.config_types.command import GoalOrientationConfig
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import quat_from_euler_xyz, yaw_quat


def test_motion_command_uses_each_active_motion_terminal_frame_as_goal():
    motion_command = object.__new__(MotionCommand)
    motion_command.device = torch.device("cpu")
    motion_command.motion_ids = torch.tensor([0, 1], dtype=torch.long)
    motion_command._goal_time_steps = torch.zeros(2, dtype=torch.long)
    motion_command._goal_version = torch.zeros(2, dtype=torch.long)
    motion_command.motion = SimpleNamespace(motion_end_idx=torch.tensor([3, 6], dtype=torch.long))

    motion_command._assign_goal_time_steps(torch.tensor([0, 1], dtype=torch.long))

    assert torch.equal(motion_command.goal_time_steps, torch.tensor([2, 5], dtype=torch.long))
    assert torch.equal(motion_command.goal_version, torch.tensor([1, 1], dtype=torch.long))


def test_motion_command_parses_goal_orientation_dict_from_cli_style_motion_config():
    motion_command = MotionCommand(
        SimpleNamespace(
            params={
                "motion_config": {
                    "motion_file": "holosoma/data/motions/dummy.npz",
                    "body_name_ref": ["torso_link"],
                    "body_names_to_track": ["torso_link"],
                    "goal_orientation": {
                        "robot_yaw_only": True,
                        "object_yaw_only": True,
                    },
                }
            }
        ),
        SimpleNamespace(),
    )

    assert isinstance(motion_command.motion_cfg.goal_orientation, GoalOrientationConfig)
    assert motion_command.motion_cfg.goal_orientation.robot_yaw_only is True
    assert motion_command.motion_cfg.goal_orientation.object_yaw_only is True


def test_motion_command_goal_orientation_properties_project_to_yaw_when_enabled():
    motion_command = object.__new__(MotionCommand)
    motion_command.motion_cfg = SimpleNamespace(
        goal_orientation=GoalOrientationConfig(robot_yaw_only=True, object_yaw_only=True)
    )
    motion_command.motion_ids = torch.tensor([0], dtype=torch.long)
    motion_command._goal_time_steps = torch.tensor([1], dtype=torch.long)
    motion_command.ref_body_index = 0
    motion_command.object_indices_in_simulator = torch.tensor([0], dtype=torch.long)

    goal_ref_quat_w = quat_from_euler_xyz(
        torch.tensor([0.3], dtype=torch.float32),
        torch.tensor([-0.2], dtype=torch.float32),
        torch.tensor([0.6], dtype=torch.float32),
    )
    goal_object_quat_w = quat_from_euler_xyz(
        torch.tensor([0.1], dtype=torch.float32),
        torch.tensor([0.4], dtype=torch.float32),
        torch.tensor([-0.5], dtype=torch.float32),
    )
    robot_ref_quat_w = quat_from_euler_xyz(
        torch.tensor([0.2], dtype=torch.float32),
        torch.tensor([0.1], dtype=torch.float32),
        torch.tensor([-0.4], dtype=torch.float32),
    )
    simulator_object_quat_w = quat_from_euler_xyz(
        torch.tensor([-0.3], dtype=torch.float32),
        torch.tensor([0.2], dtype=torch.float32),
        torch.tensor([0.7], dtype=torch.float32),
    )

    body_quat_w = torch.zeros(2, 1, 4, dtype=torch.float32)
    body_quat_w[..., 3] = 1.0
    body_quat_w[1, 0] = goal_ref_quat_w[0]

    object_quat_w = torch.zeros(2, 4, dtype=torch.float32)
    object_quat_w[:, 3] = 1.0
    object_quat_w[1] = goal_object_quat_w[0]

    all_root_states = torch.zeros(1, 13, dtype=torch.float32)
    all_root_states[0, 3:7] = simulator_object_quat_w[0]

    motion_command._env = SimpleNamespace(
        simulator=SimpleNamespace(
            _rigid_body_rot=robot_ref_quat_w.unsqueeze(1),
            all_root_states=all_root_states,
        )
    )
    motion_command.motion = SimpleNamespace(
        body_quat_w=body_quat_w,
        object_quat_w=object_quat_w,
        has_object=True,
    )

    assert torch.allclose(motion_command.goal_ref_quat_w, yaw_quat(goal_ref_quat_w, w_last=True), atol=1.0e-5)
    assert torch.allclose(
        motion_command.goal_conditioning_ref_quat_w,
        yaw_quat(robot_ref_quat_w, w_last=True),
        atol=1.0e-5,
    )
    assert torch.allclose(motion_command.goal_object_quat_w, yaw_quat(goal_object_quat_w, w_last=True), atol=1.0e-5)
    assert torch.allclose(
        motion_command.goal_conditioning_object_quat_w,
        yaw_quat(simulator_object_quat_w, w_last=True),
        atol=1.0e-5,
    )


def test_motion_command_goal_properties_use_buffered_sampled_goal_frame():
    motion_command = object.__new__(MotionCommand)
    motion_command.motion_ids = torch.tensor([0, 1], dtype=torch.long)
    motion_command._goal_time_steps = torch.tensor([2, 5], dtype=torch.long)
    motion_command.ref_body_index = 1
    motion_command._env = SimpleNamespace(
        simulator=SimpleNamespace(
            scene=SimpleNamespace(
                env_origins=torch.tensor([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]], dtype=torch.float32)
            )
        )
    )

    body_pos_w = torch.zeros(6, 2, 3, dtype=torch.float32)
    body_pos_w[2, 1] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    body_pos_w[5, 1] = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)

    body_quat_w = torch.zeros(6, 2, 4, dtype=torch.float32)
    body_quat_w[..., 3] = 1.0
    body_quat_w[5, 1] = torch.tensor([0.0, 0.0, 0.70710678, 0.70710678], dtype=torch.float32)

    object_pos_w = torch.zeros(6, 3, dtype=torch.float32)
    object_pos_w[2] = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float32)
    object_pos_w[5] = torch.tensor([10.0, 11.0, 12.0], dtype=torch.float32)

    object_quat_w = torch.zeros(6, 4, dtype=torch.float32)
    object_quat_w[:, 3] = 1.0
    object_quat_w[5] = torch.tensor([0.0, 0.0, 0.70710678, 0.70710678], dtype=torch.float32)

    motion_command.motion = SimpleNamespace(
        motion_end_idx=torch.tensor([3, 6], dtype=torch.long),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        object_pos_w=object_pos_w,
        object_quat_w=object_quat_w,
        has_object=True,
    )

    assert torch.equal(motion_command.goal_time_steps, torch.tensor([2, 5], dtype=torch.long))
    assert torch.allclose(
        motion_command.goal_ref_pos_w,
        torch.tensor([[11.0, 2.0, 3.0], [4.0, 25.0, 6.0]], dtype=torch.float32),
    )
    assert torch.allclose(
        motion_command.goal_ref_quat_w,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.70710678, 0.70710678],
            ],
            dtype=torch.float32,
        ),
        atol=1.0e-5,
    )
    assert torch.allclose(
        motion_command.goal_object_pos_w,
        torch.tensor([[17.0, 8.0, 9.0], [10.0, 31.0, 12.0]], dtype=torch.float32),
    )
    assert torch.allclose(
        motion_command.goal_object_quat_w,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.70710678, 0.70710678],
            ],
            dtype=torch.float32,
        ),
        atol=1.0e-5,
    )


def test_motion_command_update_metrics_logs_ref_body_speed_terms():
    motion_command = object.__new__(MotionCommand)
    motion_command.metrics = {}
    motion_command.motion_cfg = SimpleNamespace(
        goal_orientation=GoalOrientationConfig(),
        use_adaptive_timesteps_sampler=False,
    )
    motion_command.time_steps = torch.tensor([0, 1], dtype=torch.long)
    motion_command.motion_ids = torch.tensor([0, 0], dtype=torch.long)
    motion_command.ref_body_index = 0
    motion_command.tracked_body_indexes = [0]

    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    yaw_quat_90 = quat_from_euler_xyz(
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([torch.pi / 2], dtype=torch.float32),
    )[0]

    body_pos_w = torch.tensor(
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
        ],
        dtype=torch.float32,
    )
    body_quat_w = torch.stack([identity_quat, yaw_quat_90], dim=0).unsqueeze(1)
    body_lin_vel_w = torch.tensor(
        [
            [[1.0, 2.0, 0.5]],
            [[1.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    body_ang_vel_w = torch.zeros(2, 1, 3, dtype=torch.float32)

    motion_command.motion = SimpleNamespace(
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_pos=torch.zeros(2, 1, dtype=torch.float32),
        joint_vel=torch.zeros(2, 1, dtype=torch.float32),
    )
    motion_command.body_pos_relative_w = body_pos_w[motion_command.time_steps][:, motion_command.tracked_body_indexes]
    motion_command.body_quat_relative_w = body_quat_w[motion_command.time_steps][:, motion_command.tracked_body_indexes]
    motion_command._env = SimpleNamespace(
        simulator=SimpleNamespace(
            scene=SimpleNamespace(env_origins=torch.zeros(2, 3, dtype=torch.float32)),
            _rigid_body_pos=body_pos_w[motion_command.time_steps].clone(),
            _rigid_body_rot=body_quat_w[motion_command.time_steps].clone(),
            _rigid_body_vel=body_lin_vel_w[motion_command.time_steps].clone(),
            _rigid_body_ang_vel=body_ang_vel_w[motion_command.time_steps].clone(),
            dof_pos=torch.zeros(2, 1, dtype=torch.float32),
            dof_vel=torch.zeros(2, 1, dtype=torch.float32),
        )
    )

    MotionCommand.update_metrics(motion_command)

    assert torch.allclose(
        motion_command.metrics["motion/ref_body_speed_world"],
        torch.tensor([2.2912879, 1.0], dtype=torch.float32),
        atol=1.0e-6,
    )
    assert torch.allclose(
        motion_command.metrics["motion/ref_body_vel_local_x"],
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        atol=1.0e-5,
    )
    assert torch.allclose(
        motion_command.metrics["motion/ref_body_vel_local_y"],
        torch.tensor([2.0, 1.0], dtype=torch.float32),
        atol=1.0e-5,
    )
