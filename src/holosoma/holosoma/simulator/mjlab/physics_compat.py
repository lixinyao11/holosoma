"""PhysX-facing force semantics for the isolated Holosoma MJLab adapter.

No material/solver changes. Normal forces sum native solver contact normals on each rigid body;
tangential friction is excluded. The direction is force ON the body.
Body damping is a velocity-decay rate, not MuJoCo's joint damping coefficient.
"""
import torch
import warp as wp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul


@wp.kernel(enable_backward=False)
def _normal_forces(cone: int, frame: wp.array[wp.mat33], friction: wp.array[vec5],
                   dim: wp.array[int], efc_address: wp.array2d[int],
                   efc_force: wp.array2d[float], njmax: int, nacon: wp.array[int],
                   worldids: wp.array[int], geoms: wp.array[wp.vec2i],
                   geom_bodyids: wp.array[int], body_ids: wp.array[int],
                   out: wp.array2d[wp.vec3]):
    cid = wp.tid()
    if cid >= nacon[0] or efc_address[cid, 0] < 0:
        return
    world = worldids[cid]
    force = contact_force_fn(cone, frame, friction, dim, efc_address,
                             efc_force, njmax, nacon, world, cid, False)
    normal = force[0] * frame[cid][0]
    b1 = body_ids[geom_bodyids[geoms[cid][0]]]
    b2 = body_ids[geom_bodyids[geoms[cid][1]]]
    # MuJoCo normal points geom1 -> geom2. Return force ON each body,
    # matching IsaacLab net_forces_w, rather than force exerted by that body.
    if b1 >= 0:
        wp.atomic_add(out, world, b1, -normal)
    if b2 >= 0:
        wp.atomic_add(out, world, b2, normal)


class NormalContacts:
    def __init__(self, sim):
        self.sim = sim
        native = sim._sim
        ids = [-1] * native.mj_model.nbody
        for i, b in enumerate(sim._robot.indexing.body_ids[sim.body_ids].tolist()):
            ids[b] = i
        self.ids = wp.array(ids, dtype=int, device=str(sim.device))
        self.output = wp.zeros((sim.num_envs, sim.num_bodies), dtype=wp.vec3, device=str(sim.device))
        self.tensor = wp.to_torch(self.output)

    def read(self):
        sim = self.sim._sim
        m, d = sim.wp_model, sim.wp_data
        self.output.zero_()
        wp.launch(_normal_forces, dim=d.naconmax, inputs=[
            int(m.opt.cone), d.contact.frame, d.contact.friction, d.contact.dim,
            d.contact.efc_address, d.efc.force, d.njmax, d.nacon,
            d.contact.worldid, d.contact.geom, m.geom_bodyid, self.ids],
            outputs=[self.output], device=str(self.sim.device))
        if self.tensor.is_cuda:
            torch.cuda.current_stream(self.tensor.device).wait_stream(
                torch.cuda.ExternalStream(wp.get_stream(str(self.sim.device)).cuda_stream))
        return self.tensor


def object_damping_wrench(sim, linear=0.01, angular=0.01):
    """Rate-matched rigid-body drag about COM, using randomized mass/inertia.

    PhysX scales unconstrained velocity by max(0, 1-rate*dt). The gravity
    correction matches its gravity-before-damping order in free flight.
    Contact solvers remain native and are not claimed numerically equivalent.
    """
    native, entity = sim._sim, sim.entities['object']
    body = entity.indexing.root_body_id
    q = native.data.qpos[:, entity.indexing.free_joint_q_adr]
    v = native.data.qvel[:, entity.indexing.free_joint_v_adr]
    rotation = q[:, 3:7]
    omega = quat_apply(rotation, v[:, 3:6])
    offset = quat_apply(rotation, native.model.body_ipos[:, body])
    velocity = v[:, :3] + torch.linalg.cross(omega, offset)
    gravity = torch.as_tensor(native.mj_model.opt.gravity, device=str(sim.device), dtype=v.dtype)
    force = -linear * native.model.body_mass[:, body, None] * (velocity + sim.sim_dt * gravity)
    inertia_rotation = quat_mul(rotation, native.model.body_iquat[:, body])
    angular_momentum = quat_apply(inertia_rotation,
        native.model.body_inertia[:, body] * quat_apply_inverse(inertia_rotation, omega))
    torque = -angular * angular_momentum
    # MuJoCo xfrc_applied is force:torque in world frame, applied at body COM.
    return torch.cat((force, torque), dim=-1)
