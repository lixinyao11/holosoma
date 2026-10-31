"""Existing groups 1+2+3A and collision exclusions, plus native joint friction."""
from dataclasses import replace
from holosoma.config_types.randomization import RandomizationTermCfg
from holosoma.config_values.wbt.g1.hdmi_push_dr import g1_29dof_wbt_hybrid_object_hdmi_push_dr as base

term = RandomizationTermCfg(
    func='holosoma.managers.randomization.terms.hdmi_friction:HDMIJointFrictionRandomizer',
    params={'effort_range_nm': (0.05, 0.25)})
setup = {**base.randomization.setup_terms, 'hdmi_joint_friction': term}
reset = {**base.randomization.reset_terms, 'hdmi_joint_friction': term}
g1_29dof_wbt_hybrid_object_hdmi_friction_dr = replace(
    base, randomization=replace(base.randomization, setup_terms=setup, reset_terms=reset))
