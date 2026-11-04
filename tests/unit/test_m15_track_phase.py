"""C vs D isolation and native phase edge cases; no simulator or policy."""
from dataclasses import asdict, replace
from types import SimpleNamespace as NS
import pytest
import torch
from test_m15_multi40 import prepared
from holosoma.config_values.wbt.g1.hdmi_goal043 import make_config,make_eval_config
from holosoma.config_values.wbt.g1.legacy_track_phase import with_track_phase
from holosoma.managers.observation.terms.track_phase import track_phase
from holosoma.managers.observation import ObservationManager,ObservationManagerCfg,ObsGroupCfg,ObsTermCfg


def phase_fixture():
    cmd=NS(motion=NS(motion_start_idx=torch.tensor([0,151]),motion_end_idx=torch.tensor([151,252])),
           motion_ids=torch.tensor([0,0,0,0,1,1]),time_steps=torch.tensor([0,50,100,130,176,201]),
           track_env_mask=torch.tensor([True,True,True,True,True,False]),
           motion_cfg=NS(enable_default_pose_prepend=False,enable_default_pose_append=False,
                         enable_terminal_pose_hold=True,terminal_pose_hold_duration_s=1.))
    return NS(dt=.02,device='cpu',num_envs=6,command_manager=NS(get_state=lambda _:cmd)),cmd


def test_local_clip_phase_reset_hold_and_gen():
    env,cmd=phase_fixture();expected=torch.tensor([[0.],[.5],[1.],[1.],[.5],[0.]])
    assert torch.equal(track_phase(env),expected)
    env.episode_length_buf=torch.full((6,),99999)
    cmd.time_steps[1]=25;expected[1]=.25
    assert torch.equal(track_phase(env),expected)
    cmd.time_steps[1]+=1
    assert track_phase(env)[1].item()==pytest.approx(.26)
    cmd.motion_ids[1]=1;cmd.time_steps[1]=151
    assert track_phase(env)[1].item()==0  # reset into another clip, not global time


def test_no_hold_single_frame_and_clamping():
    env,cmd=phase_fixture();cmd.motion_cfg.enable_terminal_pose_hold=False
    cmd.time_steps[1]=75;assert track_phase(env)[1].item()==.5
    cmd.motion.motion_end_idx[0]=1;cmd.time_steps[:4]=0
    assert torch.equal(track_phase(env)[:4],torch.ones(4,1))
    cmd.time_steps[4]=1000;assert track_phase(env)[4].item()==1
    cmd.time_steps[4]=0;assert track_phase(env)[4].item()==0


@pytest.mark.parametrize('field',['enable_default_pose_prepend','enable_default_pose_append'])
def test_rejects_undefined_transition_phase(field):
    env,cmd=phase_fixture();setattr(cmd.motion_cfg,field,True)
    with pytest.raises(ValueError,match='default-pose'):track_phase(env)


@pytest.mark.parametrize('split',['train','eval'])
@pytest.mark.parametrize('extra_dr',[False,True])
def test_D_adds_only_one_actor_scalar_and_name(prepared,split,extra_dr):
    root,_,_=prepared
    kwargs=dict(extra_dr=extra_dr,motion_set='multi40',motion_split=split)
    c=make_config(root,**kwargs);d=make_config(root,track_phase=True,**kwargs)
    before,after=asdict(c),asdict(d)
    phase=after['observation']['groups']['actor_obs']['terms'].pop('track_phase')
    assert after['training']['name']==before['training']['name']+'_track_phase'
    after['training']['name']=before['training']['name']
    assert after==before
    assert asdict(make_config(root,**kwargs))==before
    assert 'track_phase' not in c.observation.groups['actor_obs'].terms
    assert 'track_phase' not in d.observation.groups['critic_obs'].terms
    assert phase==asdict(ObsTermCfg(func='holosoma.managers.observation.terms.track_phase:track_phase'))
    if split=='eval':assert asdict(make_eval_config(root,extra_dr=extra_dr,track_phase=True))==asdict(d)


def test_phase_uses_native_observation_manager_current_scalar_only():
    env,cmd=phase_fixture()
    cfg=ObservationManagerCfg(groups={'actor_obs':ObsGroupCfg(terms={
        'track_phase':ObsTermCfg(func='holosoma.managers.observation.terms.track_phase:track_phase')},enable_noise=True)})
    manager=ObservationManager(cfg,env,'cpu')
    assert manager.get_obs_dims()=={'actor_obs':1}
    assert torch.equal(manager.compute()['actor_obs'],track_phase(env))
    cmd.time_steps[1]=0
    assert torch.equal(manager.compute()['actor_obs'],track_phase(env))
    assert manager.compute()['actor_obs'].shape==(6,1)


def test_config_rejects_default_pose_transition(prepared):
    root,_,_=prepared;cfg=make_config(root,motion_set='multi40')
    terms=dict(cfg.command.setup_terms);term=terms['motion_command'];params=dict(term.params)
    params['motion_config']=replace(params['motion_config'],enable_default_pose_prepend=True)
    terms['motion_command']=replace(term,params=params)
    cfg=replace(cfg,command=replace(cfg.command,setup_terms=terms))
    with pytest.raises(ValueError,match='default-pose'):with_track_phase(cfg)
