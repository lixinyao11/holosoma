from types import SimpleNamespace

import torch

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.managers.termination.terms.wbt import HybridBadTracking


def test_hybrid_bad_tracking_uses_full_tracking_for_track_envs_and_root_height_for_gen_envs():
    motion_command = SimpleNamespace(
        motion_cfg=SimpleNamespace(body_names_to_track=["pelvis", "left_wrist_yaw_link"]),
        motion=SimpleNamespace(has_object=True),
        track_env_mask=torch.tensor([True, False, False]),
        gen_env_mask=torch.tensor([False, True, True]),
        ref_pos_w=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0]], dtype=torch.float32),
        robot_ref_pos_w=torch.tensor([[0.8, 0.0, 1.0], [0.0, 0.0, 1.6], [0.0, 0.0, 1.9]], dtype=torch.float32),
        ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=torch.float32),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=torch.float32),
        body_pos_relative_w=torch.zeros(3, 2, 3, dtype=torch.float32),
        robot_body_pos_w=torch.zeros(3, 2, 3, dtype=torch.float32),
        object_pos_w=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 0.0, 2.0]], dtype=torch.float32),
        simulator_object_pos_w=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [0.0, 0.0, 1.9]], dtype=torch.float32),
        object_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=torch.float32),
        simulator_object_quat_w=torch.tensor(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.7071068, 0.7071068]],
            dtype=torch.float32,
        ),
        robot_root_pos_w=torch.tensor([[0.0, 0.0, 0.9], [0.0, 0.0, 0.8], [0.0, 0.0, 0.6]], dtype=torch.float32),
    )
    env = SimpleNamespace(
        device="cpu",
        command_manager=SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None),
        num_envs=3,
    )
    cfg = TerminationTermCfg(
        func="holosoma.managers.termination.terms.wbt:HybridBadTracking",
        params={
            "bad_ref_pos_threshold": 0.5,
            "bad_ref_ori_threshold": 0.8,
            "bad_motion_body_pos_threshold": 0.25,
            "body_names_to_track": motion_command.motion_cfg.body_names_to_track,
            "bad_motion_body_pos_body_names": ["left_wrist_yaw_link"],
            "bad_object_pos_threshold": 0.3,
            "bad_object_ori_threshold": 0.8,
            "min_root_height": 0.45,
            "gen_bad_ref_pos_z_threshold": 0.3,
            "gen_bad_ref_pos_xyz_threshold": 100.0,
            "gen_bad_object_pos_z_threshold": 0.3,
            "gen_bad_object_ori_threshold": 0.3,
        },
    )

    term = HybridBadTracking(cfg, env)
    terminated = term(env)
    breakdown = term.get_last_breakdown()

    assert torch.equal(terminated, torch.tensor([True, True, True]))
    assert torch.equal(breakdown["track_ref_pos"], torch.tensor([True, False, False]))
    assert torch.equal(breakdown["track_ref_ori"], torch.tensor([False, False, False]))
    assert torch.equal(breakdown["track_body_pos"], torch.tensor([False, False, False]))
    assert torch.equal(breakdown["track_object_pos"], torch.tensor([False, False, False]))
    assert torch.equal(breakdown["track_object_ori"], torch.tensor([False, False, False]))
    assert torch.equal(breakdown["gen_low_root_height"], torch.tensor([False, False, False]))
    assert torch.equal(breakdown["gen_ref_pos_z"], torch.tensor([False, True, False]))
    assert torch.equal(breakdown["gen_ref_pos_xyz"], torch.tensor([False, False, False]))
    assert torch.equal(breakdown["gen_object_pos_z"], torch.tensor([False, True, False]))
    assert torch.equal(breakdown["gen_object_ori"], torch.tensor([False, False, True]))


def test_hybrid_bad_tracking_can_terminate_gen_envs_on_xyz_ref_position_error():
    motion_command = SimpleNamespace(
        motion_cfg=SimpleNamespace(body_names_to_track=["pelvis"]),
        motion=SimpleNamespace(has_object=False),
        track_env_mask=torch.tensor([True, False, False]),
        gen_env_mask=torch.tensor([False, True, True]),
        ref_pos_w=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        robot_ref_pos_w=torch.tensor(
            [[0.0, 0.0, 1.0], [1.1, 0.0, 1.0], [0.1, 0.1, 1.0]],
            dtype=torch.float32,
        ),
        ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=torch.float32),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=torch.float32),
        body_pos_relative_w=torch.zeros(3, 1, 3, dtype=torch.float32),
        robot_body_pos_w=torch.zeros(3, 1, 3, dtype=torch.float32),
        robot_root_pos_w=torch.tensor([[0.0, 0.0, 0.8], [0.0, 0.0, 0.8], [0.0, 0.0, 0.8]], dtype=torch.float32),
    )
    env = SimpleNamespace(
        device="cpu",
        command_manager=SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None),
        num_envs=3,
    )

    cfg = TerminationTermCfg(
        func="holosoma.managers.termination.terms.wbt:HybridBadTracking",
        params={
            "bad_ref_pos_threshold": 0.5,
            "bad_ref_ori_threshold": 0.8,
            "bad_motion_body_pos_threshold": 0.25,
            "body_names_to_track": motion_command.motion_cfg.body_names_to_track,
            "bad_motion_body_pos_body_names": ["pelvis"],
            "bad_object_pos_threshold": 0.3,
            "bad_object_ori_threshold": 0.8,
            "min_root_height": 0.45,
            "gen_bad_ref_pos_z_threshold": 0.3,
            "gen_bad_ref_pos_xyz_threshold": 1.0,
        },
    )

    term = HybridBadTracking(cfg, env)
    terminated = term(env)
    breakdown = term.get_last_breakdown()

    assert torch.equal(terminated, torch.tensor([False, True, False]))
    assert torch.equal(breakdown["gen_ref_pos_z"], torch.tensor([False, False, False]))
    assert torch.equal(breakdown["gen_ref_pos_xyz"], torch.tensor([False, True, False]))
