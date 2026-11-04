"""CPU oracles reused from dev; no native Isaac launch."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from holosoma.config_values.wbt.g1.hdmi_goal import make_legacy_m15_config
from holosoma.managers.randomization.terms.hdmi_body import randomize_body_startup, sample_body_parameters, readback_error
from holosoma.managers.command.terms.hdmi_body import HDMIBodyMotionCommand
from holosoma.managers.command.terms.wbt import HybridMotionCommand

def test_body_dr_writeback_and_partial_reset():
    new = make_legacy_m15_config(extra_dr=True)
    mass=torch.ones(4096)*3;inertia=torch.arange(1,10).float().expand(4096,-1)*.01
    com=torch.ones(4096,3)*.05
    torch.manual_seed(47)
    m,i,c,s,o=sample_body_parameters(mass,inertia,com,(.9,1.1),(-.03,.03))
    assert ((s>=.9)&(s<=1.1)).all() and s.std()>.04
    assert ((o>=-.03)&(o<=.03)).all() and o.std()>.01
    assert torch.equal(m,mass*s) and torch.equal(i,inertia*s[:,None]) and torch.equal(c,com+o)

    # Tight native representation budget: accept one-ULP roundtrip, reject real changes.
    expected=(torch.ones(4,3), torch.ones(4,3,9)*.01, torch.ones(4,3,7)*.1)
    actual=tuple(torch.nextafter(x,torch.full_like(x,float('inf'))) for x in expected)
    assert all(x['passed'] for x in readback_error(actual,expected,exact=False).values())
    assert not all(x['passed'] for x in readback_error(actual,expected,exact=True).values())
    bad=tuple(x*1.00001 for x in expected)
    assert not any(x['passed'] for x in readback_error(bad,expected,exact=False).values())

    class View:
     def __init__(self):
      self.m=torch.ones(16,5);self.i=torch.ones(16,5,9)*.1;self.c=torch.zeros(16,5,7);self.c[:,:,6]=1
     def get_masses(self):return self.m
     def get_inertias(self):return self.i
     def get_coms(self):return self.c
     def set_masses(self,x,ids):self.m[ids]=x[ids]
     def set_inertias(self,x,ids):self.i[ids]=x[ids]
     def set_coms(self,x,ids):self.c[ids]=x[ids]
    view=View();sim=type('IsaacSim',(),{})();sim._robot=NS(body_names=['other','pelvis','torso_link','leg','arm'],root_physx_view=view)
    env=NS(simulator=sim,num_envs=16)
    before=[x.clone() for x in (view.m,view.i,view.c)]
    randomize_body_startup(env)
    assert torch.equal(view.m[:,[0,1,3,4]],before[0][:,[0,1,3,4]])
    assert torch.equal(view.i[:,[0,1,3,4]],before[1][:,[0,1,3,4]])
    assert torch.equal(view.c[:,[0,2,3,4]],before[2][:,[0,2,3,4]])
    assert torch.equal(view.c[:,:,3:],before[2][:,:,3:])
    try:randomize_body_startup(env)
    except RuntimeError:pass
    else:raise AssertionError('Repeated startup not rejected')

    # Native parent-reset oracle is isolated; only the new delta is under test.
    class Fixture(HDMIBodyMotionCommand):
     @property
     def joint_vel(self):return self.reference
    cmd=object.__new__(Fixture);cmd.device='cpu';cmd.num_envs=16
    cmd.cfg=new.command.setup_terms['motion_command']
    cmd.reference=torch.randn(16,29);cmd._env=NS(simulator=NS(dof_vel=torch.zeros(16,29)))
    def parent_reset(self,ids):self._env.simulator.dof_vel[ids]=self.reference[ids]
    with patch.object(HybridMotionCommand,'reset',parent_reset):
     cmd.reset(torch.arange(16))
     old=cmd._env.simulator.dof_vel.clone();old_noise=cmd.hdmi_velocity_audit['noise'].clone()
     cmd.reset(torch.tensor([2,7]))
     rest=torch.tensor([k for k in range(16) if k not in (2,7)])
     assert torch.equal(old[rest],cmd._env.simulator.dof_vel[rest])
     assert not torch.equal(old_noise[[2,7]],cmd.hdmi_velocity_audit['noise'][[2,7]])
     assert torch.equal(cmd.hdmi_velocity_audit['reset_count'][rest],torch.ones(len(rest),dtype=torch.long))
     rng=torch.get_rng_state();cmd.reset(torch.tensor([],dtype=torch.long));assert torch.equal(rng,torch.get_rng_state())
    v=cmd.hdmi_velocity_audit
    assert (v['noise'].abs()<=.1).all()
    assert torch.equal(v['reference']+v['noise'],v['actual'])


def test_friction_partial_reset_preserves_other_envs_and_viscous_parameter():
    from unittest.mock import Mock
    from holosoma.managers.randomization.terms.hdmi_friction import HDMIJointFrictionRandomizer
    cfg = make_legacy_m15_config(extra_dr=True).randomization.reset_terms['hdmi_joint_friction']
    env = NS(num_envs=8, device='cpu')
    term = HDMIJointFrictionRandomizer(cfg, env)
    term.low, term.high = cfg.params['effort_range_nm']
    term.joint_names = ['j0', 'j1']
    term.joint_ids = torch.tensor([1, 3])
    term.robot = NS(write_joint_friction_coefficient_to_sim=Mock())
    term.sampled_effort = torch.zeros(8, 2)
    term.reset_count = torch.zeros(8, dtype=torch.long)
    term.reset(torch.tensor([2, 6]))
    args, kwargs = term.robot.write_joint_friction_coefficient_to_sim.call_args
    assert torch.equal(args[0], kwargs['joint_dynamic_friction_coeff'])
    assert set(kwargs) == {'joint_dynamic_friction_coeff', 'joint_ids', 'env_ids'}
    assert ((args[0] >= .05) & (args[0] <= .25)).all()
    assert torch.equal(term.sampled_effort[[2, 6]], args[0])
    assert term.sampled_effort[[0, 1, 3, 4, 5, 7]].count_nonzero() == 0
    assert term.reset_count.tolist() == [0, 0, 1, 0, 0, 0, 1, 0]
    state = torch.get_rng_state()
    term.reset(torch.tensor([], dtype=torch.long))
    assert torch.equal(state, torch.get_rng_state())
    assert term.robot.write_joint_friction_coefficient_to_sim.call_count == 1
