"""CPU-only behavioral tests for the opt-in HDMI goal teacher."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import math
import pytest
import torch

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.observation.manager import ObservationManager
from holosoma.managers.reward.terms import hdmi_goal as rewards
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_hybrid_object as original
from holosoma.config_values.wbt.g1.hdmi_goal import g1_29dof_wbt_hybrid_object_hdmi as adapted
from holosoma.config_values.wbt.g1.hdmi_goal import _hdmi_original as teacher_before_m15


def value(env):
    return env.value


def history_manager(env, offsets=(0,1,2,3,4,8,16), concatenate=True):
    return ObservationManager(ObservationManagerCfg(groups={'a':ObsGroupCfg(
        concatenate=concatenate, terms={'x':ObsTermCfg(func=f'{__name__}:value',history_offsets=offsets)})}),env,'cpu')


def test_history_alignment_reset_and_nonmutating_final_observation():
    env=SimpleNamespace(num_envs=2,device='cpu',value=torch.zeros(2,1))
    m=history_manager(env)
    for t in range(17):
        env.value=torch.tensor([[float(t)],[float(100+t)]])
        result=m.compute_group('a')
    assert torch.equal(result[0],torch.tensor([16.,15.,14.,13.,12.,8.,0.]))
    assert m.get_obs_dims()['a']==7
    env.value=torch.tensor([[17.],[117.]])
    before=[x.clone() for x in m._history_buffers['a']['x']]
    preview=m.compute_group('a',modify_history=False)
    assert all(torch.equal(a,b) for a,b in zip(before,m._history_buffers['a']['x']))
    assert torch.equal(preview,m.compute_group('a'))
    m.reset(torch.tensor([0]))
    env.value=torch.tensor([[18.],[118.]])
    result=m.compute_group('a')
    assert torch.equal(result[0],torch.tensor([18.,0.,0.,0.,0.,0.,0.]))
    assert torch.equal(result[1],torch.tensor([118.,117.,116.,115.,114.,110.,102.]))
    m.reset()
    assert torch.equal(m.compute_group('a')[:,1:],torch.zeros(2,6))


@pytest.mark.parametrize('offsets',[(0,-1),(0,0)])
def test_bad_history_offsets_rejected(offsets):
    with pytest.raises(ValueError):history_manager(SimpleNamespace(num_envs=1,device='cpu'),offsets)


def test_history_dimensions_for_dict_output_and_current_only():
    env=SimpleNamespace(num_envs=1,device='cpu',value=torch.tensor([[3.]]))
    m=history_manager(env,(0,1,2),False)
    assert m.get_obs_dims()=={'a':{'x':3}}
    assert torch.equal(m.compute_group('a')['x'],torch.tensor([[3.,0.,0.]]))
    current=history_manager(env,())
    assert torch.equal(current.compute_group('a'),env.value)


def test_existing_base_preset_unmodified_and_upstream_hand_reference_enabled():
    assert original.observation.groups['actor_obs'].terms['dof_pos'].history_offsets==()
    assert 'projected_gravity' not in original.observation.groups['actor_obs'].terms
    assert original.reward.terms['goal_object_reached_once'].func.endswith(':HybridGoalObjectReachedOnce')
    assert asdict(original.algo)==asdict(adapted.algo)
    # The unenhanced teacher preserves physics; the public preset now adopts M15.
    assert asdict(original.robot)==asdict(teacher_before_m15.robot)
    assert asdict(original.command)==asdict(adapted.command)
    expected_termination=asdict(original.termination)
    for key in ('gen_bad_ref_pos_z_threshold','gen_bad_ref_pos_xyz_threshold',
                'gen_bad_object_pos_z_threshold','gen_bad_object_ori_threshold'):
        assert expected_termination['terms']['bad_tracking']['params'][key] is not None
        expected_termination['terms']['bad_tracking']['params'][key]=None
    assert expected_termination==asdict(adapted.termination)
    assert asdict(original.randomization)==asdict(teacher_before_m15.randomization)
    for field in ("algo", "command", "observation", "reward", "termination"):
        assert asdict(getattr(teacher_before_m15, field)) == asdict(getattr(adapted, field))
    actor=adapted.observation.groups['actor_obs']
    assert 'hybrid_hand_obj_pos_error' in actor.terms
    assert 'object_to_hands_b' not in actor.terms
    assert 'track_phase' not in actor.terms
    assert actor.terms['actions'].history_offsets==(0,1,2)
    assert actor.terms['projected_gravity'].history_offsets==(0,1,2,3,4,8,16)
    assert all('reference_term' in adapted.reward.terms[k].params for k in (
        'motion_global_ref_position_error_exp','object_global_ref_position_error_exp'))
    assert adapted.reward.terms['goal_object_reached_once'].func.endswith(':GoalSettledOnce')


@pytest.fixture
def env(monkeypatch, tmp_path):
    e=SimpleNamespace(num_envs=2,device='cpu',dt=0.1,reset_buf=torch.zeros(2,dtype=torch.bool),time_out_buf=torch.zeros(2,dtype=torch.bool))
    e.cmd=SimpleNamespace(gen_env_mask=torch.tensor([True,False]),track_env_mask=torch.tensor([False,True]),
        goal_version=torch.zeros(2,dtype=torch.long),goal_object_pos_w=torch.zeros(2,3),
        simulator_object_pos_w=torch.zeros(2,3),goal_object_quat_w=torch.tensor([[0.,0.,0.,1.]]).repeat(2,1),
        simulator_object_quat_w=torch.tensor([[0.,0.,0.,1.]]).repeat(2,1),
        simulator_object_lin_vel_w=torch.zeros(2,3),simulator_object_ang_vel_w=torch.zeros(2,3))
    urdf=tmp_path/'object.urdf'
    urdf.write_text('<robot name="box"><link name="base"><collision><origin xyz="0 0 0.1"/><geometry><box size="0.2 0.2 0.2"/></geometry></collision></link></robot>')
    e.simulator=SimpleNamespace(robot_config=SimpleNamespace(object=SimpleNamespace(
        object_urdf_paths=[],object_urdf_path=str(urdf))),scene=SimpleNamespace(env_origins=torch.zeros(2,3)))
    e.cmd.hybrid_cfg=SimpleNamespace(size_randomization=SimpleNamespace(enabled=False))
    monkeypatch.setattr(rewards.wbt,'_get_motion_command_and_assert_type',lambda env:env.cmd)
    return e


def term(cls,env,**params):
    return cls(RewardTermCfg(func='unused',params=params),env)


def test_progress_no_repeated_cycle_bonus_and_no_reset_teleport_reward(env):
    p=term(rewards.GoalProgress,env)
    env.cmd.simulator_object_pos_w[:,0]=1.
    assert torch.equal(p(env),torch.zeros(2))
    env.cmd.simulator_object_pos_w[:,0]=.8
    assert torch.allclose(p(env)*env.dt,torch.tensor([.2,0.]))
    env.cmd.simulator_object_pos_w[:,0]=1.
    assert torch.equal(p(env),torch.zeros(2))
    env.cmd.simulator_object_pos_w[:,0]=.8
    assert torch.equal(p(env),torch.zeros(2))
    env.cmd.goal_version+=1
    env.cmd.simulator_object_pos_w[:,0]=.2
    assert torch.equal(p(env),torch.zeros(2))
    p.reset(torch.tensor([0]));env.cmd.simulator_object_pos_w[:,0]=0.
    assert torch.equal(p(env),torch.zeros(2))


def test_progress_includes_orientation(env):
    p=term(rewards.GoalProgress,env)
    env.cmd.simulator_object_quat_w[:]=torch.tensor([0.,0.,math.sin(math.pi/4),math.cos(math.pi/4)])
    p(env)
    env.cmd.simulator_object_quat_w[:]=torch.tensor([0.,0.,0.,1.])
    assert torch.allclose(p(env)*env.dt,torch.tensor([.5,0.]),atol=1e-6)


def test_settled_requires_contiguous_hold_and_only_once(env):
    s=term(rewards.GoalSettledOnce,env,hold_seconds=.3)
    assert not s(env).any()
    env.cmd.simulator_object_lin_vel_w[0,0]=1.
    assert not s(env).any()
    env.cmd.simulator_object_lin_vel_w.zero_()
    assert not s(env).any();assert not s(env).any()
    assert torch.equal(s(env)*env.dt,torch.tensor([3.,0.]))
    assert not s(env).any()
    env.cmd.goal_version+=1
    assert not s(env).any();assert not s(env).any()
    assert torch.equal(s(env)*env.dt,torch.tensor([3.,0.]))


@pytest.mark.parametrize('failure',['position','orientation','angular_speed','termination'])
def test_settled_rejects_bad_pose_speed_or_failure(env,failure):
    s=term(rewards.GoalSettledOnce,env,hold_seconds=.1)
    if failure=='position':env.cmd.simulator_object_pos_w[0,0]=.3
    if failure=='orientation':env.cmd.simulator_object_quat_w[0]=torch.tensor([0.,0.,1.,0.])
    if failure=='angular_speed':env.cmd.simulator_object_ang_vel_w[0,0]=1.
    if failure=='termination':env.reset_buf[0]=True
    assert not s(env).any()


def test_settled_reset_is_per_environment(env):
    env.cmd.gen_env_mask[:]=True
    s=term(rewards.GoalSettledOnce,env,hold_seconds=.2)
    s(env);s.reset(torch.tensor([0]))
    assert torch.equal(s(env)*env.dt,torch.tensor([0.,3.]))
    assert torch.equal(s(env)*env.dt,torch.tensor([3.,0.]))


def test_joint_formula_hdmi_absolute_mean_and_track_mask(env):
    env.simulator=SimpleNamespace(dof_names=['waist_yaw_joint','left_ankle_roll_joint','left_elbow_joint'])
    env.cmd.joint_pos=torch.zeros(2,3);env.cmd.robot_joint_pos=torch.tensor([[.25,100.,.75]]).repeat(2,1)
    actual=rewards.joint_tracking(env,sigma=.25)
    assert torch.allclose(actual,torch.tensor([0.,math.exp(-2)]))
    env.cmd.joint_vel=env.cmd.joint_pos;env.cmd.robot_joint_vel=env.cmd.robot_joint_pos
    assert torch.allclose(rewards.joint_tracking(env,sigma=2.5,velocity=True),torch.tensor([0.,math.exp(-.2)]))


def test_raw_tracking_math_preserved_without_gen_constant(env,monkeypatch):
    monkeypatch.setattr(rewards.wbt,'motion_global_body_lin_vel',lambda env,sigma:torch.tensor([.7,.8]))
    assert torch.equal(rewards.track_only(env,'motion_global_body_lin_vel',1.),torch.tensor([0.,.8]))


def test_actor_dimensions_and_track_only_hand_reference_dependency(env,monkeypatch):
    from holosoma.managers.observation.terms import wbt as obs_wbt
    from holosoma.managers.observation.terms.hdmi_goal import object_to_hands_b, object_ang_vel_b
    monkeypatch.setattr(obs_wbt,'_get_motion_command_and_assert_type',lambda e:e.cmd)
    env.base_quat=torch.tensor([[0.,0.,0.,1.]]).repeat(2,1)
    env.default_dof_pos=torch.zeros(1,29)
    env.action_manager=SimpleNamespace(action=torch.zeros(2,29))
    env.simulator=SimpleNamespace(dof_pos=torch.zeros(2,29),dof_vel=torch.zeros(2,29),
        robot_root_states=torch.zeros(2,13),_body_list=['left_wrist_yaw_link','right_wrist_yaw_link'],
        _rigid_body_pos=torch.tensor([[[1.,0.,0.],[0.,1.,0.]]]).repeat(2,1,1))
    env.cmd.robot_ref_pos_w=torch.zeros(2,3)
    env.cmd.robot_ref_quat_w=env.base_quat.clone()
    env.cmd.motion_cfg=SimpleNamespace(body_names_to_track=env.simulator._body_list)
    env.cmd.robot_body_pos_w=env.simulator._rigid_body_pos.clone()
    env.cmd.body_pos_w=env.cmd.robot_body_pos_w.clone()
    env.cmd.object_pos_w=env.cmd.simulator_object_pos_w.clone()
    env.cmd.object_quat_w=env.cmd.simulator_object_quat_w.clone()
    actor=replace(adapted.observation.groups['actor_obs'],enable_noise=False)
    m=ObservationManager(ObservationManagerCfg(groups={'actor':actor}),env,'cpu')
    assert m.get_obs_dims()=={'actor':571}
    before=m.compute_group('actor',modify_history=False)
    # Joint targets are not observed, but reference hand/object error is now observed.
    env.cmd.joint_pos=torch.full((2,29),99.)
    assert torch.equal(before,m.compute_group('actor',modify_history=False))
    env.cmd.object_pos_w=torch.full((2,3),99.)
    after=m.compute_group('actor',modify_history=False)
    assert torch.equal(before[0],after[0])  # gen reference error is masked out
    assert not torch.equal(before[1],after[1])  # track receives the reference error
    assert bool(torch.isfinite(before).all())
    assert torch.equal(object_to_hands_b(env)[0],torch.tensor([1.,0.,0.,0.,1.,0.]))
    # Object features use the current torso frame, independently of the base frame.
    env.cmd.robot_ref_quat_w[:]=torch.tensor([0.,0.,math.sin(math.pi/4),math.cos(math.pi/4)])
    env.cmd.simulator_object_ang_vel_w[:,0]=1.
    assert torch.allclose(object_to_hands_b(env)[0],torch.tensor([0.,-1.,0.,1.,0.,0.]),atol=1e-6)
    assert torch.allclose(object_ang_vel_b(env)[0],torch.tensor([0.,-1.,0.]),atol=1e-6)


@pytest.mark.parametrize('height,accepted',[(0.,True),(.05,True),(.0501,False),(.20,False)])
def test_settled_bottom_height_limit_independent_of_reference_z(env,height,accepted):
    env.cmd.goal_object_pos_w[:,2]=-0.2  # Retargeted reference below ground.
    env.cmd.simulator_object_pos_w[:,2]=height
    s=term(rewards.GoalSettledOnce,env,hold_seconds=.1)
    assert bool(s(env)[0]>0)==accepted
    s.reset()
    env.cmd.goal_object_pos_w[:,2]=0.3  # Floating reference cannot permit suspension.
    assert bool(s(env)[0]>0)==accepted


def test_ground_clearance_rotates_scales_and_offsets_environment(env):
    env.cmd.hybrid_cfg.size_randomization.enabled=True
    env.simulator.get_actor_scale_factors=lambda name:torch.full((2,3),2.)
    env.simulator.scene.env_origins[:,2]=10.
    env.cmd.simulator_object_pos_w[:,2]=10.25
    env.cmd.simulator_object_quat_w[:]=torch.tensor([math.sin(math.pi/4),0.,0.,math.cos(math.pi/4)])
    clearance=rewards.ObjectGroundClearance(env)
    # Rotated scaled box's bottom is 0.2m below its origin, not z=0 of original box.
    assert torch.allclose(clearance(env,env.cmd),torch.full((2,),.05),atol=1e-6)


def test_collision_vertices_preserve_bounds_and_avoid_aabb_false_bottom(env,tmp_path):
    from holosoma.envs.wbt.hybrid_size_randomization import load_urdf_collision_bounds
    mesh=tmp_path/'triangle.obj'
    mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 0 1\n')
    path=tmp_path/'triangle.urdf'
    path.write_text('<robot name="test"><link name="base"><collision><geometry><mesh filename="triangle.obj"/></geometry></collision></link></robot>')
    env.simulator.robot_config.object.object_urdf_path=str(path)
    assert load_urdf_collision_bounds(str(path))==([0.,0.,0.],[1.,0.,1.])
    env.cmd.simulator_object_quat_w[:]=torch.tensor([0.,math.sin(3*math.pi/8),0.,math.cos(3*math.pi/8)])
    clearance=rewards.ObjectGroundClearance(env)(env,env.cmd)
    # A nonexistent AABB corner (1,0,1) would falsely give -sqrt(2).
    assert torch.allclose(clearance,torch.full((2,),-math.sqrt(.5)),atol=1e-6)


def test_ground_clearance_rejects_unimplemented_variants(env):
    env.simulator.robot_config.object.object_urdf_paths=['one.urdf','two.urdf']
    with pytest.raises(ValueError,match='multiple object variants'):
        rewards.ObjectGroundClearance(env)


def test_settled_can_construct_before_command_manager(env):
    cmd = env.cmd
    del env.cmd
    reward = term(rewards.GoalSettledOnce, env, hold_seconds=.1)
    env.cmd = cmd
    assert torch.equal(reward(env) * env.dt, torch.tensor([3., 0.]))
