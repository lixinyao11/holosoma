"""Explicit actuator-DR variant; the original HDMI teacher preset stays unchanged."""
from dataclasses import replace
from holosoma.config_types.randomization import RandomizationTermCfg
from holosoma.config_values.wbt.g1.hdmi_goal import g1_29dof_wbt_hybrid_object_hdmi as base

module = 'holosoma.managers.randomization.terms.hdmi_actuator'
groups = {'.*(shoulder|elbow|wrist).*': (0.9, 1.1),
          '.*(waist|hip|knee|ankle).*': (0.8, 1.2)}
setup = dict(base.randomization.setup_terms)
reset = dict(base.randomization.reset_terms)
setup['actuator_randomizer_state'] = replace(
    setup['actuator_randomizer_state'], func=f'{module}:HDMIActuatorRandomizerState',
    params={**setup['actuator_randomizer_state'].params, 'enable_pd_gain': True,
            'kp_groups': groups, 'kd_groups': groups})
reset['actuator_randomizer_state'] = replace(
    reset['actuator_randomizer_state'], func=f'{module}:HDMIActuatorRandomizerState')
setup['hdmi_armature'] = RandomizationTermCfg(
    func=f'{module}:randomize_armature_startup', params={'scale_range': (0.75, 1.25)})
actions = dict(base.action.terms)
actions['joint_control'] = replace(
    actions['joint_control'],
    func='holosoma.managers.action.terms.hdmi_actuator:HDMIJointPositionActionTerm',
    params={'min_delay': 1, 'max_delay': 3, 'alpha_range': (0.8, 1.0)})
g1_29dof_wbt_hybrid_object_hdmi_actuator_dr = replace(
    base, action=replace(base.action, terms=actions),
    randomization=replace(base.randomization, setup_terms=setup, reset_terms=reset))
