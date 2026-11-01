"""Only restore native friction DR; restitution remains nominal zero."""
from dataclasses import replace

MATERIAL_TERMS = {
    'randomize_robot_rigid_body_material_startup',
    'randomize_object_rigid_body_material_startup',
}

def configure(cfg):
    changes = {}; audit = {}; found = set()
    for group in ('setup_terms', 'reset_terms', 'step_terms'):
        terms = {}
        for name, term in getattr(cfg.randomization, group).items():
            before = dict(term.params or {}); params = dict(before)
            for key in before:
                if key == 'enabled' or key.startswith('enable_') or key == 'randomize_dof_pos_bias':
                    assert isinstance(before[key], bool), (name, key)
                    params[key] = False
            if name in MATERIAL_TERMS:
                assert group == 'setup_terms'
                params['enabled'] = True
                params['restitution_range'] = (0.0, 0.0)
                found.add(name)
            terms[name] = replace(term, params=params)
            if before != params:
                audit[group+'/'+name] = {'before': before, 'after': params}
        changes[group] = terms
    assert found == MATERIAL_TERMS, found
    return replace(cfg, randomization=replace(cfg.randomization, **changes)), audit
