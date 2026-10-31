"""Audited native mappings for the unchanged Holosoma randomizer settings."""
import math
import os
import torch
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_matrix


def randomize(sim, kind, p):
    env = p["env"]
    if not p.get("enabled", True):
        return
    ids = p.get("idx")
    if ids is None:
        ids = p.get("env_ids")
        ids = torch.arange(sim.num_envs, device=sim.device) if ids is None else torch.as_tensor(ids, device=sim.device)
    if not ids.numel():
        return
    m = sim._sim.model
    samples = {}
    def draw(bounds, shape, name):
        x = torch.empty(shape, device=sim.device).uniform_(float(bounds[0]),float(bounds[1]))
        samples[name] = x.detach().cpu()
        return x
    def field_mass(body_ids, bounds, operation):
        old = m.body_mass[ids[:,None], body_ids].clone()
        x = draw(bounds, old.shape, "mass_"+operation)
        new = old*x if operation == "scale" else old+x
        if (new <= 0).any():
            raise ValueError("Nonpositive randomized mass")
        m.body_mass[ids[:,None], body_ids] = new
        m.body_inertia[ids[:,None], body_ids] *= (new/old)[...,None]
        samples["actual_mass"] = new.detach().cpu()
    if kind.endswith("rigid_body_material_startup"):
        if os.environ.get("HOLOSOMA_MJLAB_APPROXIMATE_MATERIALS") != "1":
            raise RuntimeError("PhysX materials have no exact MuJoCo mapping; explicit opt-in required")
        entity = sim.entities["object" if "object" in kind else "robot"]
        geoms = entity.indexing.geom_ids
        # Original Isaac material randomizer uses 64 buckets, assigned per collision shape.
        shape = (len(ids), len(geoms))
        buckets = torch.stack([draw(p[k], (64,), k+"_buckets") for k in
            ("static_friction_range", "dynamic_friction_range", "restitution_range")],-1)
        bucket_ids = torch.randint(64, shape, device=sim.device)
        sf, df, e = buckets[bucket_ids].unbind(-1)
        samples["bucket_ids"] = bucket_ids.cpu()
        samples["static_friction_recorded_not_applied"] = sf.cpu()
        samples["dynamic_friction"] = df.cpu()
        samples["restitution"] = e.cpu()
        loge = e.clamp(min=1e-8, max=1-1e-8).log()
        damping = -loge / torch.sqrt(loge.square()+math.pi**2)
        damping = torch.where(e == 0, torch.ones_like(damping), damping)
        m.geom_friction[ids[:,None],geoms,0] = df
        m.geom_solref[ids[:,None],geoms,1] = damping
        samples["actual_damping_ratio"] = damping.detach().cpu()
        samples["mapping"] = "sliding=dynamic; static retained only; damping=-ln(e)/sqrt(pi^2+ln(e)^2); native MuJoCo contact combination"
    elif kind == "randomize_base_com_startup":
        b = sim._robot.indexing.body_ids[sim._robot.body_names.index(env.robot_config.torso_name)]
        bias = torch.stack([draw(p["base_com_range"][axis],(len(ids),),axis) for axis in "xyz"],-1)
        m.body_ipos[ids,b] += bias
        samples["actual_com"] = m.body_ipos[ids,b].detach().cpu()
    elif kind == "randomize_mass_startup":
        if p["enable_link_mass"]:
            body_ids = torch.tensor([int(sim._robot.indexing.body_ids[sim._robot.body_names.index(n)])
                for n in env.robot_config.randomize_link_body_names],device=sim.device)
            field_mass(body_ids,p["link_mass_range"],"scale")
        if p["enable_base_mass"]:
            body_ids = sim._robot.indexing.body_ids[[sim._robot.body_names.index(env.robot_config.torso_name)]]
            field_mass(body_ids,p["added_mass_range"],"add")
    elif kind == "randomize_object_rigid_body_mass_startup":
        field_mass(sim.entities["object"].indexing.body_ids,p["mass_distribution_params"],"add")
    elif kind == "randomize_object_rigid_body_inertia_startup":
        b = sim.entities["object"].indexing.body_ids
        diagonal = m.body_inertia[ids[:,None],b].clone()
        rotation = matrix_from_quat(m.body_iquat[ids[:,None],b])
        inertia = rotation @ torch.diag_embed(diagonal) @ rotation.transpose(-1,-2)
        for key,i,j in (("Ixx",0,0),("Iyy",1,1),("Izz",2,2),("Ixy",0,1),("Iyz",1,2),("Ixz",0,2)):
            x = draw(p["inertia_distribution_params_dict"][key], inertia.shape[:2],key)
            inertia[...,i,j] *= x
            if i != j:
                inertia[...,j,i] = inertia[...,i,j]
        vals, axes = torch.linalg.eigh(inertia)
        if (vals<=0).any():
            raise ValueError("Sampled full inertia is not physically admissible; no silent clipping")
        axes[..., :,0] *= torch.linalg.det(axes)[...,None]
        m.body_inertia[ids[:,None],b] = vals
        m.body_iquat[ids[:,None],b] = quat_from_matrix(axes)
        samples["actual_inertia_body_frame"] = inertia.detach().cpu()
    else:
        raise NotImplementedError(kind)
    sim._sim.recompute_constants(RecomputeLevel.set_const)
    sim.randomization_samples[kind] = samples
    torch.save(sim.randomization_samples, sim.audit_dir/"randomization_actual.pt")
