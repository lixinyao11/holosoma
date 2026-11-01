"""Match Holosoma's plane material combination, without changing DR samples.

Isaac's plane uses multiply friction/restitution and zero restitution. MuJoCo
normally uses max friction and averaged solref instead. Explicit plane pairs
preserve the sampled dynamic friction and use critical damping for zero bounce.
Static/dynamic friction separation and the native contact solvers remain different.
"""
import numpy as np
import mujoco
import torch

PREFIX = "holosoma_ground_"


def add_ground_pairs(spec, dynamic_friction):
    if dynamic_friction < 0:
        raise ValueError("Negative ground friction")
    planes = [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE
              and (g.contype or g.conaffinity)]
    if len(planes) != 1:
        raise ValueError("Ground material adapter requires one collidable plane")
    floor = planes[0]
    if not floor.name:
        floor.name = PREFIX + "plane"
    floor.friction[0] = dynamic_friction
    records = []
    existing = {frozenset((p.geomname1, p.geomname2)) for p in spec.pairs}
    for index, geom in enumerate(spec.geoms):
        if geom is floor or geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if not ((geom.contype & floor.conaffinity) or (floor.contype & geom.conaffinity)):
            continue
        if not geom.name:
            geom.name = PREFIX + f"geom_{index}"
        if frozenset((geom.name, floor.name)) in existing:
            raise ValueError(f"Existing explicit ground pair for {geom.name}")
        # Current Holosoma assets use equal-priority positive solref. Reject a
        # different regime rather than accidentally replacing its semantics.
        if geom.priority != floor.priority or min(geom.solref[0], floor.solref[0]) <= 0:
            raise ValueError("Unsupported ground contact priority/solref")
        total = geom.solmix + floor.solmix
        weight = geom.solmix / total if total > 0 else 0.5
        ref = weight * np.asarray(geom.solref) + (1 - weight) * np.asarray(floor.solref)
        ref[1] = 1.0  # e_object * e_ground(=0) -> zero restitution approximation
        imp = weight * np.asarray(geom.solimp) + (1 - weight) * np.asarray(floor.solimp)
        mu = float(geom.friction[0]) * float(dynamic_friction)
        torsion = max(geom.friction[1], floor.friction[1])
        rolling = max(geom.friction[2], floor.friction[2])
        name = PREFIX + f"pair_{index}"
        spec.add_pair(name=name, geomname1=floor.name, geomname2=geom.name,
                      condim=max(geom.condim, floor.condim),
                      friction=[mu, mu, torsion, rolling, rolling], solref=ref,
                      solimp=imp, margin=max(geom.margin, floor.margin),
                      gap=max(geom.gap, floor.gap))
        records.append({"pair": name, "ground": floor.name, "geom": geom.name})
    return records


def update_ground_pairs(sim, env_ids):
    """Update only selected worlds after existing per-geom material sampling."""
    native = sim._sim
    model = native.mj_model
    pairs = [i for i in range(model.npair) if model.pair(i).name.startswith(PREFIX)]
    if not pairs:
        return
    device = sim.device
    ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    if ids.numel() == 0:
        return
    pids = torch.tensor(pairs, device=device, dtype=torch.long)
    g1 = torch.as_tensor(model.pair_geom1[pairs], device=device, dtype=torch.long)
    g2 = torch.as_tensor(model.pair_geom2[pairs], device=device, dtype=torch.long)
    m = native.model
    mu = m.geom_friction[ids[:, None], g1, 0] * m.geom_friction[ids[:, None], g2, 0]
    m.pair_friction[ids[:, None], pids, 0] = mu
    m.pair_friction[ids[:, None], pids, 1] = mu
    m.pair_solref[ids[:, None], pids, 1] = 1.0
