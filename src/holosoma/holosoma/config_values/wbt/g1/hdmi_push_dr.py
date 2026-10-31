"""Groups 1+2+3A. Legacy HDMI push amplitude and interval; unchanged scheduler."""
from dataclasses import replace
from holosoma.config_values.wbt.g1.hdmi_body_dr import (
    g1_29dof_wbt_hybrid_object_hdmi_body_dr as base,
)

setup = dict(base.randomization.setup_terms)
term = setup['push_randomizer_state']
max_vel = list(term.params['max_push_vel'])
max_vel[:2] = [0.2, 0.2]
max_vel[2:] = [0.1, 0.2, 0.2, 0.4]
setup['push_randomizer_state'] = replace(
    term, params={**term.params, 'max_push_vel': max_vel, 'push_interval_s': [4.0, 6.0]})
g1_29dof_wbt_hybrid_object_hdmi_push_dr = replace(
    base, randomization=replace(base.randomization, setup_terms=setup))
