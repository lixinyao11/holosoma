"""Scoped MuJoCo-Warp 3.11 contact response for Holosoma's PhysX DR adapter.

Material draws and native mixing are unchanged. Positive mixed solref (t,zeta)
was previously treated as fixed relaxation time, making k grow as 1/zeta**2.
Use direct solref to retain the t,zeta=1 baseline stiffness while varying damping.
Then cap the penetration-bias target velocity at PhysX's configured 1 m/s.
This is NOT restitution-equivalent PhysX contact or a cap on physical velocity.
No joint-limit/equality rows, friction coefficients, impulses, qpos or qvel are clipped.
See MuJoCo modeling/solver-parameters; old HDMI uses explicit softness, not a
claim of exact restitution transfer. Scoped models only; native dependencies untouched.
"""
import functools
import warp as wp
from mujoco_warp._src import constraint
from mujoco_warp._src.types import vec5, ConeType, DisableBit, MJ_MINIMP, MJ_MAXIMP


@wp.kernel(enable_backward=False)
def _fixed_stiffness(n: wp.array[int], world: wp.array[int],
                     dt: wp.array[float], disable: int, ref: wp.array[wp.vec2]):
    i = wp.tid()
    if i >= n[0]:
        return
    r = ref[i]
    if r[0] > 0.0 and r[1] >= 0.0:
        t = r[0]
        if not (disable & DisableBit.REFSAFE):
            t = wp.max(t, 2.0 * dt[world[i] % dt.shape[0]])
        # Direct format: stiffness and damping, both represented as negatives.
        ref[i] = wp.vec2(-1.0 / (t*t), -2.0 * r[1] / t)


@wp.kernel(enable_backward=False)
def _cap_penetration_bias(n: wp.array[int], world: wp.array[int],
                         dim: wp.array[int], addresses: wp.array2d[int], cone: int,
                         ref: wp.array[wp.vec2], imp: wp.array[vec5],
                         velocity: wp.array2d[float], pos: wp.array2d[float],
                         margin: wp.array2d[float], vmax: float,
                         aref: wp.array2d[float]):
    i, j = wp.tid()
    if i >= n[0]:
        return
    dims = dim[i]
    if cone == int(ConeType.ELLIPTIC):
        if j != 0:
            return  # Pure friction directions have no penetration residual.
    elif (dims == 1 and j != 0) or (dims > 1 and j >= 2*(dims-1)):
        return
    row = addresses[i,j]
    if row < 0:
        return
    w = world[i]
    if pos[w,row] - margin[w,row] >= 0.0:
        return
    dmax = wp.clamp(imp[i][1], MJ_MINIMP, MJ_MAXIMP)
    # _fixed_stiffness has already put positive references in direct format.
    damping = -ref[i][1] / dmax
    old = aref[w,row]
    bias = old + damping * velocity[w,row]
    # aref = b*(target_separation_velocity - J*qvel). Limit ONLY bias/b.
    cap = damping * vmax
    if bias > cap:
        aref[w,row] = cap - damping * velocity[w,row]


_MODELS = {}
_ORIGINAL = constraint.make_constraint


@functools.wraps(_ORIGINAL)
def _make_constraint(model, data):
    item = _MODELS.get(id(model))
    if item is None or item[0] is not model:
        return _ORIGINAL(model, data)
    vmax = item[1]
    wp.launch(_fixed_stiffness, dim=data.naconmax,
              inputs=[data.nacon, data.contact.worldid, model.opt.timestep,
                      int(model.opt.disableflags), data.contact.solref], device=data.qpos.device)
    _ORIGINAL(model, data)
    wp.launch(_cap_penetration_bias, dim=(data.naconmax, data.contact.efc_address.shape[1]),
              inputs=[data.nacon, data.contact.worldid, data.contact.dim,
                      data.contact.efc_address, int(model.opt.cone), data.contact.solref,
                      data.contact.solimp, data.efc.vel, data.efc.pos, data.efc.margin,
                      vmax, data.efc.aref], device=data.qpos.device)


def install(native):
    """Install only for this model; refresh CUDA graphs to include both kernels."""
    import importlib.metadata
    if importlib.metadata.version('mujoco-warp').split('.')[:2] != ['3','11']:
        raise RuntimeError('Contact response adapter requires audited mujoco-warp 3.11')
    if constraint.make_constraint not in (_ORIGINAL, _make_constraint):
        raise RuntimeError('Conflicting constraint hook; refusing to overwrite it')
    wp.load_module(module=__name__, device=native.wp_device)
    _MODELS[id(native.wp_model)] = (native.wp_model, 1.0)
    constraint.make_constraint = _make_constraint
    native.create_graph()
