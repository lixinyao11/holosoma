"""Independent, opt-out M15 hardware options for the legacy HDMI teacher.

Always start from the unenhanced teacher; do not stack this on a modified config.
Use hdmi_goal.make_legacy_m15_config to construct any switch combination.
"""
from dataclasses import replace

from holosoma.config_types.randomization import RandomizationTermCfg
from holosoma.config_values.wbt.g1.robot_mode15 import with_mode15
from holosoma.utils.collision_exclusions import HDMI_G1_SELF_COLLISION_EXCLUSIONS


def with_legacy_m15(cfg, *, collision_exclusions=True, motor_dr=True, delay_filter=True):
    cfg = replace(cfg, robot=with_mode15(replace(cfg.robot, asset=replace(
        cfg.robot.asset,
        collision_exclusions=HDMI_G1_SELF_COLLISION_EXCLUSIONS if collision_exclusions else (),
    ))))
    if motor_dr:
        module = 'holosoma.managers.randomization.terms.hdmi_actuator'
        groups = {'.*(shoulder|elbow|wrist).*': (0.9, 1.1),
                  '.*(waist|hip|knee|ankle).*': (0.8, 1.2)}
        setup = dict(cfg.randomization.setup_terms)
        reset = dict(cfg.randomization.reset_terms)
        setup['actuator_randomizer_state'] = replace(
            setup['actuator_randomizer_state'], func=f'{module}:HDMIActuatorRandomizerState',
            params={**setup['actuator_randomizer_state'].params, 'enable_pd_gain': True,
                    'kp_groups': groups, 'kd_groups': groups})
        reset['actuator_randomizer_state'] = replace(
            reset['actuator_randomizer_state'], func=f'{module}:HDMIActuatorRandomizerState')
        setup['hdmi_armature'] = RandomizationTermCfg(
            func=f'{module}:randomize_armature_startup', params={'scale_range': (0.75, 1.25)})
        cfg = replace(cfg, randomization=replace(cfg.randomization, setup_terms=setup, reset_terms=reset))
    if delay_filter:
        actions = dict(cfg.action.terms)
        actions['joint_control'] = replace(
            actions['joint_control'],
            func='holosoma.managers.action.terms.hdmi_actuator:HDMIJointPositionActionTerm',
            params={'min_delay': 1, 'max_delay': 3, 'alpha_range': (0.8, 1.0)})
        cfg = replace(cfg, action=replace(cfg.action, terms=actions))
    return cfg
