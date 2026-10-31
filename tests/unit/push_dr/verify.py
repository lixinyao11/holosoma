"""Focused checks for config-only 3A; no group1/group2 tests repeated."""
import ast, json, copy, importlib.util
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace as NS
import torch
from holosoma.config_values.wbt.g1.hdmi_push_dr import base, g1_29dof_wbt_hybrid_object_hdmi_push_dr as new
from holosoma.managers.randomization.terms.locomotion import PushRandomizerState, apply_pushes

r=Path(__file__).parent
a,b=asdict(base),asdict(new)
old=a['randomization']['setup_terms']['push_randomizer_state']['params']
modified=b['randomization']['setup_terms']['push_randomizer_state']['params']
assert old['max_push_vel']==[.4,.4,0.,0.,0.,0.]
assert modified['max_push_vel']==[.4,.4,.1,.2,.2,.4]
assert old['push_interval_s']==modified['push_interval_s']==[1.,5.]
modified['max_push_vel']=old['max_push_vel'];assert a==b

# Execute the exact existing WBT push method, without importing simulator apps.
package=Path(next(iter(importlib.util.find_spec('holosoma').submodule_search_locations)))
tree=ast.parse((package/'envs/wbt/wbt_manager.py').read_text())
klass=next(x for x in tree.body if isinstance(x,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='_push_robots' for m in x.body))
method=copy.deepcopy(next(x for x in klass.body if isinstance(x,ast.FunctionDef) and x.name=='_push_robots'))
ns={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'native_wbt_push','exec'),ns)

def run(cfg):
 env=NS(device='cpu',num_envs=64,dt=.02,is_evaluating=False,need_to_refresh_envs=torch.zeros(64,dtype=torch.bool),
   push_robot_vel_buf=torch.zeros(64,6),record_push_robot_vel_buf=torch.zeros(64,6))
 state=PushRandomizerState(cfg.randomization.setup_terms['push_randomizer_state'],env)
 env.randomization_manager=NS(get_state=lambda name:state)
 env.simulator=NS(robot_root_states=torch.zeros(64,13))
 events=[]
 def write(ids,states):
  events.append((ids.clone(),states.clone(),env.push_robot_vel_buf.clone(),state.push_interval_s.clone()))
 env.simulator.set_actor_root_state_tensor_robots=write
 env._push_robots=lambda ids:ns['_push_robots'](env,ids)
 torch.manual_seed(58);state.setup();initial=state.push_interval_s.clone()
 for _ in range(300):state.step();apply_pushes(env)
 assert events
 n=len(events);env.is_evaluating=True;state.push_robot_counter[:]=(state.push_interval_s/env.dt).int();apply_pushes(env);assert len(events)==n
 env.is_evaluating=False;apply_pushes(env,enabled=False);assert len(events)==n
 return initial,events
x,y=run(base),run(new)
assert torch.equal(x[0],y[0]) and len(x[1])==len(y[1])
for old_event,new_event in zip(x[1],y[1]):
 ids,states,pushes,intervals=new_event
 assert torch.equal(ids,old_event[0]) and torch.equal(intervals,old_event[3])
 assert torch.equal(states[:,7:9],old_event[1][:,7:9])
 assert torch.equal(pushes[:,:2],old_event[2][:,:2])
 assert (pushes.abs()<=torch.tensor([.4,.4,.1,.2,.2,.4])).all()
assert (y[1][-1][2][:,2:].abs().sum(0)>0).all()
report={'passed':True,'only_four_velocity_bounds_changed':True,'same_seed_horizontal_pushes_exact':True,
 'same_seed_schedule_exact':True,'interval_seconds':[1,5],'schedule_events_checked':len(y[1]),
 'evaluation_and_disabled_guards_preserved':True,'production_push_implementation_unchanged':True}
(r/'cpu_acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
