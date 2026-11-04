"""User-selected upstream joint reward and hand observation contract."""
from dataclasses import asdict
from types import SimpleNamespace as NS
import math

import pytest
import torch

from holosoma.config_values.wbt.g1.hdmi_goal import make_legacy_m15_config
from holosoma.config_values.wbt.g1.observation import hand_obs_terms
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward_hybrid_object_hands
from holosoma.managers.observation.terms import wbt as obs
from holosoma.managers.reward.terms import wbt as reward


def test_config_reuses_upstream_terms_and_removes_old_joint_terms():
    cfg = make_legacy_m15_config()
    for group in ("actor_obs", "critic_obs"):
        terms = cfg.observation.groups[group].terms
        assert "object_to_hands_b" not in terms
        for name, term in hand_obs_terms.items():
            assert asdict(terms[name]) == asdict(term)
    terms = cfg.reward.terms
    assert "joint_pos_tracking" not in terms and "joint_vel_tracking" not in terms
    assert asdict(terms["motion_joint_pos_error_exp"]) == asdict(
        g1_29dof_wbt_reward_hybrid_object_hands.terms["motion_joint_pos_error_exp"])
    assert "hand_object_position_error_exp" not in terms  # no extra reward bundled in
    assert cfg.training.name.endswith("_upstream_hands")


def test_upstream_joint_reward_includes_every_joint_and_preserves_gen_semantics(monkeypatch):
    cfg = make_legacy_m15_config()
    cmd = NS(joint_pos=torch.zeros(2, 29),
             track_env_mask=torch.tensor([True, False]),
             gen_env_mask=torch.tensor([False, True]))
    actual = torch.zeros(2, 29)
    ankle = next(i for i, name in enumerate(cfg.robot.dof_names) if "ankle" in name)
    actual[0, ankle] = .29
    env = NS(simulator=NS(dof_pos=actual))
    monkeypatch.setattr(reward, "_get_motion_command_and_assert_type", lambda _: cmd)
    value = reward.hybrid_motion_joint_pos_error_exp(env, sigma=.25)
    assert value[0].item() == pytest.approx(math.exp(-.29/29/.25))
    assert value[1].item() == .5


def test_hands_use_object_frame_and_mask_reference_error_only_in_gen(monkeypatch):
    identity = torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1)
    yaw90 = torch.tensor([[0., 0., math.sin(math.pi/4), math.cos(math.pi/4)]]).repeat(2, 1)
    cmd = NS(
        motion_cfg=NS(body_names_to_track=["left_wrist_yaw_link", "right_wrist_yaw_link"]),
        robot_body_pos_w=torch.tensor([[[1.,0.,0.],[0.,1.,0.]]]).repeat(2,1,1),
        body_pos_w=torch.zeros(2,2,3),
        simulator_object_pos_w=torch.zeros(2,3), object_pos_w=torch.zeros(2,3),
        simulator_object_quat_w=yaw90, object_quat_w=identity,
        track_env_mask=torch.tensor([True,False]), gen_env_mask=torch.tensor([False,True]))
    env = NS(num_envs=2)
    monkeypatch.setattr(obs, "_get_motion_command_and_assert_type", lambda _:cmd)
    names = cmd.motion_cfg.body_names_to_track
    expected = torch.tensor([[0.,-1.,0.,1.,0.,0.]]).repeat(2,1)
    torch.testing.assert_close(obs.hand_obj_pos(env,names), expected)
    error = obs.hybrid_hand_obj_pos_error(env,names)
    torch.testing.assert_close(error[0],expected[0])
    assert torch.equal(error[1],torch.zeros(6))
    # Common world translation cannot alter these object-relative features.
    cmd.robot_body_pos_w += 3
    cmd.simulator_object_pos_w += 3
    cmd.body_pos_w += 3
    cmd.object_pos_w += 3
    torch.testing.assert_close(obs.hand_obj_pos(env,names),expected)
    torch.testing.assert_close(obs.hybrid_hand_obj_pos_error(env,names),error)
