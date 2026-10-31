"""Test actual HybridBadTracking dispatch with the new preset, without simulation."""
from types import SimpleNamespace
import pytest
import torch
from holosoma.config_values.wbt.g1.hdmi_goal import g1_29dof_wbt_hybrid_object_hdmi as adapted
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_hybrid_object as original
from holosoma.managers.termination.terms.wbt import HybridBadTracking


def fixture():
    names=adapted.termination.terms['bad_tracking'].params['body_names_to_track']
    quat=torch.tensor([[0.,0.,0.,1.]]).repeat(2,1)
    pos=torch.tensor([[0.,0.,.8]]).repeat(2,1)
    cmd=SimpleNamespace(motion_cfg=SimpleNamespace(body_names_to_track=names),
        motion=SimpleNamespace(has_object=True),track_env_mask=torch.tensor([True,False]),
        gen_env_mask=torch.tensor([False,True]),ref_pos_w=pos.clone(),robot_ref_pos_w=pos.clone(),
        robot_root_pos_w=pos.clone(),ref_quat_w=quat.clone(),robot_ref_quat_w=quat.clone(),
        body_pos_relative_w=torch.zeros(2,len(names),3),robot_body_pos_w=torch.zeros(2,len(names),3),
        object_pos_w=pos.clone(),simulator_object_pos_w=pos.clone(),
        object_quat_w=quat.clone(),simulator_object_quat_w=quat.clone())
    env=SimpleNamespace(num_envs=2,device='cpu',
        command_manager=SimpleNamespace(get_state=lambda name:cmd))
    return env,cmd


@pytest.mark.parametrize('difference',['torso_z','torso_xy','object_z','object_orientation','wrist_z'])
def test_reference_deviation_terminates_tracking_only(difference):
    env,c=fixture()
    if difference=='torso_z':c.ref_pos_w[:,2]+=1.
    if difference=='torso_xy':c.ref_pos_w[:,0]+=101.
    if difference=='object_z':c.object_pos_w[:,2]+=1.
    if difference=='object_orientation':c.object_quat_w[:]=torch.tensor([0.,0.,1.,0.])
    if difference=='wrist_z':c.body_pos_relative_w[:,c.motion_cfg.body_names_to_track.index('left_wrist_yaw_link'),2]=1.
    new=HybridBadTracking(adapted.termination.terms['bad_tracking'],env)
    old=HybridBadTracking(original.termination.terms['bad_tracking'],env)
    assert torch.equal(new(env),torch.tensor([True,False]))
    assert bool(old(env)[0])
    if difference!='wrist_z':assert bool(old(env)[1])
    for key in ('gen_ref_pos_z','gen_ref_pos_xyz','gen_object_pos_z','gen_object_ori'):
        assert not new.get_last_breakdown()[key].any()


def test_gen_low_root_height_still_terminates():
    env,c=fixture();c.robot_root_pos_w[1,2]=.44
    term=HybridBadTracking(adapted.termination.terms['bad_tracking'],env)
    assert torch.equal(term(env),torch.tensor([False,True]))
    assert bool(term.get_last_breakdown()['gen_low_root_height'][1])
