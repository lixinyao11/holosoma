import importlib.util,json,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
import torch
r=Path(sys.argv[1]);r.mkdir(parents=True, exist_ok=True)
module_path=Path(__file__).resolve().parents[3]/'src/holosoma/holosoma/simulator/mjlab/ground_materials.py'
s=importlib.util.spec_from_file_location('ground_materials',module_path);mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod)
xml='''<mujoco><worldbody><geom name="floor" type="plane" size="3 3 .1" friction="1 .005 .0001"/><body name="box" pos="0 0 .049"><freejoint/><geom name="box_geom" type="box" size=".1 .1 .05" mass="1" friction=".2 .005 .0001" solref=".02 .25"/><geom name="visual" type="sphere" size=".01" contype="0" conaffinity="0"/></body></worldbody></mujoco>'''
def contacts(m):
 d=mujoco.MjData(m);mujoco.mj_forward(m,d)
 assert d.ncon>0
 return {'ncon':d.ncon,'friction':d.contact.friction[:d.ncon].tolist(),'solref':d.contact.solref[:d.ncon].tolist()}
old=mujoco.MjModel.from_xml_string(xml);old_c=contacts(old)
spec=mujoco.MjSpec.from_string(xml);records=mod.add_ground_pairs(spec,1.);new=spec.compile();new_c=contacts(new)
assert len(records)==1 and new.npair==1
assert all(abs(c[0]-1.)<1e-12 for c in old_c['friction'])
assert all(abs(c[0]-.2)<1e-12 for c in new_c['friction'])
assert all(abs(c[1]-1.)<1e-12 for c in new_c['solref'])
for k in ['body_mass','body_inertia','body_ipos','jnt_range','dof_armature','geom_contype','geom_conaffinity']:
 assert np.array_equal(getattr(old,k),getattr(new,k)),k
# Per-world mapping at the actual sampled values, including subset updates.
batch=SimpleNamespace(geom_friction=torch.tensor(np.repeat(new.geom_friction[None],3,axis=0)),pair_friction=torch.tensor(np.repeat(new.pair_friction[None],3,axis=0)),pair_solref=torch.tensor(np.repeat(new.pair_solref[None],3,axis=0)))
gid=new.geom('box_geom').id;batch.geom_friction[:,gid,0]=torch.tensor([.1,.3,.6]);sim=SimpleNamespace(device='cpu',_sim=SimpleNamespace(mj_model=new,model=batch))
before=batch.pair_friction.clone();mod.update_ground_pairs(sim,torch.tensor([0,2]));assert torch.equal(batch.pair_friction[1],before[1]);assert torch.allclose(batch.pair_friction[[0,2],0,0],torch.tensor([.1,.6],dtype=torch.float64));mod.update_ground_pairs(sim,torch.tensor([1]));assert torch.allclose(batch.pair_friction[:,0,0],torch.tensor([.1,.3,.6],dtype=torch.float64))
# Actual native collision readback for each world's mapped material.
for world in range(3):
 new.pair_friction[:]=batch.pair_friction[world].numpy();new.pair_solref[:]=batch.pair_solref[world].numpy()
 c=contacts(new);assert all(abs(x[0]-[.1,.3,.6][world])<1e-6 for x in c['friction'])
# Warp model field shape/expansion on CPU only, no GPU contexts or training.
import warp as wp
wp.init()
import mujoco_warp as mjwarp
from mjlab.sim.randomization import expand_model_fields
with wp.ScopedDevice('cpu'):
 wm=mjwarp.put_model(new)
expand_model_fields(wm,3,['pair_friction','pair_solref']);shapes={k:list(wp.to_torch(getattr(wm,k)).shape) for k in ['pair_friction','pair_solref']};assert shapes['pair_friction']==[3,1,5] and shapes['pair_solref']==[3,1,2],shapes
out={'passed':True,'old_native_contact':old_c,'corrected_native_contact':new_c,'pairs':records,'per_world_samples':[.1,.3,.6],'unselected_world_unchanged':True,'visual_geoms_excluded':True,'body_and_joint_fields_unchanged':True,'warp_field_shapes':shapes,'code_sha256':hashlib.sha256((module_path).read_bytes()).hexdigest(),'scope':'CPU contact/material contract only; no policy rollout, no learning-performance claim'}
(r/'cpu_acceptance.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
