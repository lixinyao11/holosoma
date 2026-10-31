"""Group 1 + group 2; the original and actuator-only presets stay unchanged."""
from dataclasses import replace
from holosoma.config_types.randomization import RandomizationTermCfg
from holosoma.config_values.wbt.g1.hdmi_actuator_dr import (
    g1_29dof_wbt_hybrid_object_hdmi_actuator_dr as base,
)

setup = dict(base.randomization.setup_terms)
setup['hdmi_body'] = RandomizationTermCfg(
    func='holosoma.managers.randomization.terms.hdmi_body:randomize_body_startup',
    params={'mass_range': (0.9, 1.1), 'com_range': (-0.03, 0.03)})

def command_terms(terms, setup=False):
    terms = dict(terms)
    term = terms['motion_command']
    params = dict(term.params)
    if setup:
        params['joint_velocity_noise_range'] = (-0.1, 0.1)
    terms['motion_command'] = replace(
        term, func='holosoma.managers.command.terms.hdmi_body:HDMIBodyMotionCommand', params=params)
    return terms

g1_29dof_wbt_hybrid_object_hdmi_body_dr = replace(
    base,
    randomization=replace(base.randomization, setup_terms=setup),
    command=replace(base.command,
        setup_terms=command_terms(base.command.setup_terms, True),
        reset_terms=command_terms(base.command.reset_terms),
        step_terms=command_terms(base.command.step_terms)))
