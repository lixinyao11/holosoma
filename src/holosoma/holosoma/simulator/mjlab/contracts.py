"""Opt-in state boundary audit; never runs in the production Isaac backend."""
import json
import torch


def audit(sim):
    ids=torch.tensor([0,sim.num_envs-1],device=sim.device).unique()
    qpos=sim._sim.data.qpos.clone();qvel=sim._sim.data.qvel.clone()
    original=sim.all_root_states.clone(); joints=sim.dof_state.clone()
    errors={}
    try:
        for name in ('robot','object'):
            indices=sim.get_actor_indices(name,ids)
            target=sim.get_actor_states_by_index(indices)
            target[:,:3]=torch.tensor([.3,-.2,1.4],device=sim.device)
            q=torch.tensor([.1,.2,.3,.9],device=sim.device);q=q/q.norm()
            target[:,3:7]=q
            target[:,7:]=torch.tensor([.13,-.27,.41,.4,-.3,.2],device=sim.device)
            sim.set_actor_states_by_index(indices,target)
            sim.refresh_sim_tensors()
            err=(sim.get_actor_states_by_index(indices)-target).abs().max().item()
            errors[name+'_pose_com_velocity_roundtrip']=err
            assert err<=1e-6,(name,err)
        target_joints=joints.clone();target_joints[ids,:,0]+=.01;target_joints[ids,:,1]=.1
        sim.set_dof_state_tensor_robots(ids,target_joints)
        sim.refresh_sim_tensors()
        err=(sim.dof_state[ids]-target_joints[ids]).abs().max().item()
        errors['joint_order_roundtrip']=err;assert err<=1e-6,err
        untouched=torch.ones(sim.num_envs,dtype=torch.bool,device=sim.device);untouched[ids]=False
        assert torch.equal(sim._sim.data.qpos[untouched],qpos[untouched])
        assert torch.equal(sim._sim.data.qvel[untouched],qvel[untouched])
        assert sim.contact_forces.shape==(sim.num_envs,sim.num_bodies,3)
        assert torch.isfinite(sim.contact_forces).all()
        errors['other_envs_unchanged']=True
    finally:
        sim._sim.data.qpos.copy_(qpos);sim._sim.data.qvel.copy_(qvel)
        sim._dirty=True;sim.refresh_sim_tensors()
    assert torch.equal(sim._sim.data.qpos,qpos)
    assert torch.equal(sim._sim.data.qvel,qvel)
    (sim.audit_dir/'state_contract.json').write_text(json.dumps({'passed':True,'checks':errors},indent=2))
