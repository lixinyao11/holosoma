import torch

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.managers.termination.manager import TerminationManager


class _CommandManager:
    def __init__(self, motion_command):
        self._motion_command = motion_command

    def get_state(self, name: str):
        if name != "motion_command":
            return None
        return self._motion_command


class _MotionCfg:
    def __init__(self, body_names_to_track):
        self.body_names_to_track = body_names_to_track


class _MotionState:
    def __init__(self, has_object: bool):
        self.has_object = has_object


class _MotionCommand:
    def __init__(self):
        tracked = [
            "pelvis",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]
        self.motion_cfg = _MotionCfg(tracked)
        self.motion = _MotionState(has_object=True)

        self.ref_pos_w = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        self.robot_ref_pos_w = torch.tensor([[0.0, 0.0, 1.7], [0.0, 0.0, 1.0]])
        self.ref_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        self.robot_ref_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])

        self.body_pos_relative_w = torch.zeros(2, len(tracked), 3)
        self.robot_body_pos_w = torch.zeros_like(self.body_pos_relative_w)

        self.object_pos_w = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.simulator_object_pos_w = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
        self.object_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        self.simulator_object_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])


class _FakeEnv:
    def __init__(self):
        self.device = "cpu"
        self.num_envs = 2
        self.episode_length_buf = torch.tensor([0, 5], dtype=torch.long)
        self.max_episode_length = 4
        self.command_manager = _CommandManager(_MotionCommand())


def test_termination_manager_tracks_term_results_and_breakdowns():
    env = _FakeEnv()
    cfg = TerminationManagerCfg(
        terms={
            "timeout": TerminationTermCfg(
                func="holosoma.managers.termination.terms.common:timeout_exceeded",
                is_timeout=True,
            ),
            "bad_tracking": TerminationTermCfg(
                func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
                params={
                    "bad_ref_pos_threshold": 0.5,
                    "bad_ref_ori_threshold": 0.8,
                    "bad_motion_body_pos_threshold": 0.25,
                    "body_names_to_track": env.command_manager.get_state("motion_command").motion_cfg.body_names_to_track,
                    "bad_motion_body_pos_body_names": [
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                        "left_wrist_yaw_link",
                        "right_wrist_yaw_link",
                    ],
                    "bad_object_pos_threshold": 0.25,
                    "bad_object_ori_threshold": 0.8,
                },
            ),
        }
    )

    manager = TerminationManager(cfg, env, device="cpu")
    reset_flags, timeout_flags = manager.check()

    assert torch.equal(timeout_flags, torch.tensor([False, True]))
    assert torch.equal(reset_flags, torch.tensor([True, True]))
    assert torch.equal(manager.last_term_results["timeout"], torch.tensor([False, True]))
    assert torch.equal(manager.last_term_results["bad_tracking"], torch.tensor([True, True]))

    breakdown = manager.last_term_breakdowns["bad_tracking"]
    assert torch.equal(breakdown["ref_pos_z"], torch.tensor([True, False]))
    assert torch.equal(breakdown["ref_ori"], torch.tensor([False, False]))
    assert torch.equal(breakdown["body_pos_z"], torch.tensor([False, False]))
    assert torch.equal(breakdown["object_pos"], torch.tensor([False, True]))
    assert torch.equal(breakdown["object_ori"], torch.tensor([False, False]))


def test_termination_manager_applies_non_timeout_reset_grace_window():
    env = _FakeEnv()
    env.episode_length_buf = torch.tensor([2, 5], dtype=torch.long)
    env.max_episode_length = 99
    cfg = TerminationManagerCfg(
        terms={
            "bad_tracking": TerminationTermCfg(
                func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
                grace_steps_after_reset=3,
                params={
                    "bad_ref_pos_threshold": 0.5,
                    "bad_ref_ori_threshold": 0.8,
                    "bad_motion_body_pos_threshold": 0.25,
                    "body_names_to_track": env.command_manager.get_state("motion_command").motion_cfg.body_names_to_track,
                    "bad_motion_body_pos_body_names": [
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                        "left_wrist_yaw_link",
                        "right_wrist_yaw_link",
                    ],
                    "bad_object_pos_threshold": 0.25,
                    "bad_object_ori_threshold": 0.8,
                },
            ),
        }
    )

    manager = TerminationManager(cfg, env, device="cpu")
    reset_flags, timeout_flags = manager.check()

    assert torch.equal(timeout_flags, torch.tensor([False, False]))
    assert torch.equal(reset_flags, torch.tensor([False, True]))
    assert torch.equal(manager.last_term_results["bad_tracking"], torch.tensor([False, True]))

    breakdown = manager.last_term_breakdowns["bad_tracking"]
    assert torch.equal(breakdown["ref_pos_z"], torch.tensor([False, False]))
    assert torch.equal(breakdown["object_pos"], torch.tensor([False, True]))
