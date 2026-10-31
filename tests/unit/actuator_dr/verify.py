"""Focused acceptance for new actuator DR only. CPU; no environment rollouts."""
import ast, copy, dataclasses, json, sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from holosoma.utils.hdmi_actuator_dr import HDMISubstepFilter, grouped_bounds, log_uniform_scales
from holosoma.config_values.wbt.g1.hdmi_actuator_dr import base, g1_29dof_wbt_hybrid_object_hdmi_actuator_dr as new
from holosoma.managers.randomization.terms.hdmi_actuator import HDMIActuatorRandomizerState
from holosoma.managers.action.terms.hdmi_actuator import HDMIJointPositionActionTerm
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm

root = Path(__file__).parent
# Independent oracle: exact legacy methods, not copies of the new implementation.
tree=ast.parse((root/'legacy_actions.py').read_text())
c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JointPosition')
c=copy.deepcopy(c);c.bases=[];c.keywords=[]
c.body=[n for n in c.body if isinstance(n,ast.FunctionDef) and n.name in ('reset','process_action','apply_action')]
ns={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[c],type_ignores=[])),'legacy_actions.py','exec'),ns)
old=ns['JointPosition']();old.num_envs=12;old.action_dim=29;old.device='cpu';old.env=NS(decimation=4)
old.min_delay=1;old.max_delay=3;old.alpha_range=(.8,1.)
old.action_buf=torch.zeros(12,3,29);old.applied_action=torch.zeros(12,29)
old.alpha=torch.ones(12,1);old.delay=torch.zeros(12,1,dtype=torch.long)
old.default_joint_pos=torch.zeros(12,29);old.offset=torch.zeros(12,29);old.joint_ids=torch.arange(29)
old.action_scaling=torch.ones(29);old.asset=NS(set_joint_position_target=lambda value:None)
f=HDMISubstepFilter(12,29,4,'cpu');ids=torch.arange(12)
torch.manual_seed(149);old.reset(ids)
torch.manual_seed(149);f.reset(ids)
assert torch.equal(old.delay,f.delay) and torch.equal(old.alpha,f.alpha)
checks=0
for t in range(80):
 a=torch.randn(12,29)
 old.process_action(a);f.process_action(a)
 for sub in range(4):
  old.apply_action(sub);f.apply_action(sub)
  assert torch.equal(old.applied_action,f.applied_action),(t,sub)
  checks+=1
 if t==30:
  subset=torch.tensor([1,5,9]);untouched=f.applied_action[[0,2,3]].clone()
  torch.manual_seed(345);old.reset(subset)
  torch.manual_seed(345);f.reset(subset)
  assert torch.equal(untouched,f.applied_action[[0,2,3]])
  assert torch.equal(old.delay,f.delay) and torch.equal(old.alpha,f.alpha)
# Explicit timing oracle, including exact delay units and substep LPF.
g=HDMISubstepFilter(3,1,4,'cpu');g.delay[:,0]=torch.tensor([1,2,3]);g.alpha.fill_(.8)
g.process_action(torch.ones(3,1))
steps=torch.stack([g.apply_action(i).clone() for i in range(4)],1).squeeze(-1)
expected=torch.tensor([[0,.8,.96,.992],[0,0,.8,.96],[0,0,0,.8]])
assert torch.allclose(steps,expected,atol=1e-7,rtol=0)
# Literal legacy log-uniform draw, same RNG, exact values.
bounds=grouped_bounds(base.robot.dof_names,new.randomization.setup_terms['actuator_randomizer_state'].params['kp_groups'],device='cpu')
lo,hi=bounds.unbind(-1);lo=lo.unsqueeze(0).expand(1024,-1);hi=hi.unsqueeze(0).expand(1024,-1)
torch.manual_seed(63);expected=(lo.log()+torch.rand(lo.shape)*(hi.log()-lo.log())).exp()
torch.manual_seed(63);actual=log_uniform_scales(bounds,1024)
assert torch.equal(expected,actual)
assert ((actual>=lo)&(actual<=hi)).all()
# All configuration outside the requested actuator changes must remain exact.
a=dataclasses.asdict(base);b=dataclasses.asdict(new)
b['action']=a['action']
b['randomization']['setup_terms']['actuator_randomizer_state']=a['randomization']['setup_terms']['actuator_randomizer_state']
b['randomization']['setup_terms'].pop('hdmi_armature')
b['randomization']['reset_terms']['actuator_randomizer_state']=a['randomization']['reset_terms']['actuator_randomizer_state']
assert a==b
# Integration with existing PD implementation and clipping; neutral filter is exact.
env=NS(num_envs=12,num_dof=29,device='cpu',robot_config=base.robot,
       dof_names=list(base.robot.dof_names),default_dof_pos=torch.zeros(12,29),torque_limits=torch.tensor(base.robot.dof_effort_limit_list),
       log_dict={},_randomize_ctrl_delay=False,_pending_torque_rfi=(False,0.),randomization_manager=None)
env.simulator=NS(sim_dt=.005,simulator_config=NS(sim=NS(control_decimation=4)),
 dof_pos=torch.randn(12,29)*.02,dof_vel=torch.randn(12,29)*.05,apply_torques_at_dof=lambda x:None)
original=JointPositionActionTerm(base.action.terms['joint_control'],env);original.setup()
term=HDMIJointPositionActionTerm(new.action.terms['joint_control'],env);term.setup()
state=HDMIActuatorRandomizerState(new.randomization.setup_terms['actuator_randomizer_state'],env);state.setup();state.reset()
assert not torch.equal(state.kp_scale,torch.ones_like(state.kp_scale))
before=state.kp_scale.clone();state.reset(torch.tensor([1,5]));assert torch.equal(before[[0,2,3]],state.kp_scale[[0,2,3]])
original.attach_actuator_scales(state.kp_scale,state.kd_scale,state.rfi_lim_scale)
term.attach_actuator_scales(state.kp_scale,state.kd_scale,state.rfi_lim_scale)
term.hdmi_filter.delay.zero_();term.hdmi_filter.alpha.fill_(1.)
for i in range(5):
 action=torch.randn(12,29)*10
 original.process_actions(action);term.process_actions(action)
 for s in range(4):
  original.apply_actions();term.apply_actions();assert torch.equal(original.torques,term.torques)
# Non-neutral filter routes its target into the same PD+torque clipping implementation.
term.reset();action=torch.randn(12,29);term.process_actions(action)
for s in range(4):
 term.apply_actions()
 assert torch.equal(term.torques,original._compute_torques(term.hdmi_filter.applied_action))
# Reject unsupported/overlapping groups and accidental stacked delay.
try: grouped_bounds(['unknown_joint'],{'.*wrist.*':(.9,1.1)},device='cpu')
except ValueError: pass
else: raise AssertionError('unmatched group accepted')
env._randomize_ctrl_delay=True
try: HDMIJointPositionActionTerm(new.action.terms['joint_control'],env).setup()
except ValueError: pass
else: raise AssertionError('double delay accepted')
report={'passed':True,'legacy_filter_exact_substep_comparisons':checks,
'partial_reset_exact':True,'delay_response':steps.tolist(),'legacy_log_uniform_exact':True,
'other_config_exact':True,'neutral_filter_PD_exact':True,'filtered_PD_target_exact':True,
'PD_group_counts':{'arms':int((bounds[:,0]==.9).sum()),'waist_legs':int((bounds[:,0]==.8).sum())},
'delay_unit_seconds':.005,'formal_training_started':False}
(root/'cpu_acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
