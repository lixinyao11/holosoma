"""Only the newly added phase behavior and configuration delta are tested."""
from dataclasses import asdict
from types import SimpleNamespace as NS
import torch
from holosoma.managers.observation.terms.track_phase import track_phase
from holosoma.config_values.wbt.g1.hdmi_friction_dr import g1_29dof_wbt_hybrid_object_hdmi_friction_dr as base
from holosoma.config_values.wbt.g1.hdmi_track_phase import g1_29dof_wbt_hybrid_object_hdmi_friction_dr_track_phase as phase_cfg


def fixture():
    cmd = NS(motion=NS(motion_start_idx=torch.tensor([0, 151]),
                      motion_end_idx=torch.tensor([151, 252])),
             motion_ids=torch.tensor([0, 0, 0, 0, 1, 1]),
             time_steps=torch.tensor([0, 50, 100, 130, 176, 201]),
             track_env_mask=torch.tensor([True, True, True, True, True, False]),
             motion_cfg=NS(enable_default_pose_prepend=False, enable_default_pose_append=False,
                           enable_terminal_pose_hold=True, terminal_pose_hold_duration_s=1.))
    env = NS(dt=.02, command_manager=NS(get_state=lambda _: cmd))
    return env, cmd


def test_midclip_reset_clip_offsets_hold_and_gen():
    env, cmd = fixture()
    expected = torch.tensor([[0.], [.5], [1.], [1.], [.5], [0.]])
    assert torch.equal(track_phase(env), expected)
    # Partial reset immediately changes phase, independently of episode clock.
    env.episode_length_buf = torch.full((6,), 9999)
    cmd.time_steps[1] = 25
    expected[1] = .25
    assert torch.equal(track_phase(env), expected)
    cmd.time_steps[1] += 1
    assert abs(track_phase(env)[1].item() - .26) < 1e-7


def test_no_hold_and_single_frame():
    env, cmd = fixture()
    cmd.motion_cfg.enable_terminal_pose_hold = False
    cmd.time_steps[1] = 75
    assert track_phase(env)[1].item() == .5
    cmd.motion.motion_end_idx[0] = 1
    cmd.time_steps[:4] = 0
    assert torch.equal(track_phase(env)[:4], torch.ones(4, 1))


def test_only_actor_scalar_added():
    before, after = asdict(base), asdict(phase_cfg)
    term = after['observation']['groups']['actor_obs']['terms'].pop('track_phase')
    assert before == after
    assert 'track_phase' not in base.observation.groups['actor_obs'].terms
    assert term['noise'] == 0 and not term['history_offsets']
    assert phase_cfg.observation.groups['critic_obs'] is base.observation.groups['critic_obs']
