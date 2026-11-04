"""Opt-in group 2: startup torso mass/inertia and pelvis COM perturbations."""
import torch


def sample_body_parameters(mass, inertia, com, mass_range, com_range):
    """Legacy HDMI semantics: uniform mass scale, matched inertia, additive COM."""
    scale = torch.empty_like(mass).uniform_(*mass_range)
    offset = torch.empty_like(com).uniform_(*com_range)
    return mass * scale, inertia * scale[..., None], com + offset, scale, offset


def readback_error(actual, expected, *, exact):
    """PhysX tensor setters round-trip inverse masses/inertia frames/quaternions.

    Bound each body's error by eight float32 eps times its parameter magnitude;
    COM positions and rotations have separate bounds. No task/physics tolerance
    is changed. The optional exact check remains available for callers.
    """
    report = {}
    for key, a, e in zip(('mass', 'inertia', 'com'), actual, expected):
        delta = (a - e).abs()
        if exact:
            valid = torch.equal(a, e)
            max_bound = 0.0
        else:
            eps = torch.finfo(e.dtype).eps
            if key == 'mass':
                magnitude = e.abs()
            elif key == 'inertia':
                magnitude = e.abs().amax(-1, keepdim=True).expand_as(e)
            else:
                magnitude = torch.cat((
                    e[..., :3].abs().amax(-1, keepdim=True).expand_as(e[..., :3]),
                    e[..., 3:].abs().amax(-1, keepdim=True).expand_as(e[..., 3:])), -1)
            bound = 8 * eps * magnitude
            valid = bool(torch.isfinite(a).all() and (delta <= bound).all())
            max_bound = float(bound.max())
        report[key] = {'passed': valid, 'max_abs': float(delta.max()),
                       'max_bound': max_bound, 'bitwise_equal': torch.equal(a, e)}
    return report


def randomize_body_startup(env, *, mass_range=(0.9, 1.1), com_range=(-0.03, 0.03)):
    sim = env.simulator
    if hasattr(sim, 'hdmi_body_audit'):
        raise RuntimeError('Body startup DR must not accumulate on reset')
    robot = sim._robot
    torso = robot.body_names.index('torso_link')
    pelvis = robot.body_names.index('pelvis')
    if sim.__class__.__name__ == 'IsaacSim':
        view = robot.root_physx_view
        before_mass = view.get_masses().clone()
        before_inertia = view.get_inertias().clone()
        before_com = view.get_coms().clone()
    else:
        raise NotImplementedError(type(sim).__name__)
    mass, inertia, com, scale, offset = sample_body_parameters(
        before_mass[:, torso], before_inertia[:, torso], before_com[:, pelvis, :3],
        mass_range, com_range)
    expected_mass = before_mass.clone()
    expected_inertia = before_inertia.clone()
    expected_com = before_com.clone()
    expected_mass[:, torso] = mass
    expected_inertia[:, torso] = inertia
    expected_com[:, pelvis, :3] = com
    ids = torch.arange(env.num_envs, device='cpu', dtype=torch.int)
    # Native PhysX setters; preserve all non-selected links and COM rotations.
    view.set_masses(expected_mass, ids)
    view.set_inertias(expected_inertia, ids)
    view.set_coms(expected_com, ids)
    actual = (view.get_masses(), view.get_inertias(), view.get_coms())
    expected = (expected_mass, expected_inertia, expected_com)
    readback = readback_error(actual, expected, exact=False)
    if not all(x['passed'] for x in readback.values()):
        raise RuntimeError(f'Body DR engine readback differs beyond float precision: {readback}')
    sim.hdmi_body_audit = {
        'mass_body': 'torso_link', 'com_body': 'pelvis',
        'baseline_mass': before_mass[:, torso].clone(),
        'baseline_inertia': before_inertia[:, torso].clone(),
        'baseline_com': before_com[:, pelvis, :3].clone(),
        'scale': scale.clone(), 'offset': offset.clone(),
        'actual_mass': actual[0][:, torso].clone(),
        'actual_inertia': actual[1][:, torso].clone(),
        'actual_com': actual[2][:, pelvis, :3].clone(),
        'requested_mass': mass.clone(), 'requested_inertia': inertia.clone(),
        'requested_com': com.clone(), 'readback': readback,
        'readback_passed': True, 'nonselected_fields_preserved_within_float_precision': True,
    }
