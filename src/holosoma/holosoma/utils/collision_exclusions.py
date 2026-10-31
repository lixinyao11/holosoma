"""Explicit body-pair filters shared by HDMI Isaac and MJLab assets.

These rules do not disable a body's collision geometry or its external contacts.
"""

HDMI_G1_SELF_COLLISION_EXCLUSIONS = (
    ('left_elbow_link', 'left_wrist_pitch_link'),
    ('right_elbow_link', 'right_wrist_pitch_link'),
    ('pelvis', 'right_hip_roll_link'),
    ('pelvis', 'left_hip_roll_link'),
)


def apply_mjcf_collision_exclusions(spec, pairs):
    """Add only missing pairs before compilation; preserve all other filters."""
    names = {body.name for body in spec.bodies}
    for a, b in pairs:
        if a == b or a not in names or b not in names:
            raise ValueError(f'Invalid collision exclusion: {(a, b)}')
    existing = {frozenset((ex.bodyname1, ex.bodyname2)) for ex in spec.excludes}
    for a, b in pairs:
        key = frozenset((a, b))
        if key not in existing:
            spec.add_exclude(bodyname1=a, bodyname2=b)
            existing.add(key)


def apply_usd_collision_exclusions(stage, robot_paths, pairs):
    """Apply native USD FilteredPairs to rigid bodies, including each clone."""
    if not pairs:
        return
    from pxr import Usd, UsdPhysics

    resolved = []
    for root_path in robot_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            raise ValueError(f'Missing robot root: {root_path}')
        bodies = {}
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                name = prim.GetName()
                if name in bodies:
                    raise ValueError(f'Ambiguous rigid body: {root_path}/{name}')
                bodies[name] = prim
        for a, b in pairs:
            if a == b or a not in bodies or b not in bodies:
                raise ValueError(f'Invalid collision exclusion under {root_path}: {(a, b)}')
            resolved.append((bodies[a], bodies[b]))
    # Resolve all paths first so an invalid body never creates partial rules.
    for a, b in resolved:
        relation = UsdPhysics.FilteredPairsAPI.Apply(a).CreateFilteredPairsRel()
        if b.GetPath() not in relation.GetTargets():
            relation.AddTarget(b.GetPath())
