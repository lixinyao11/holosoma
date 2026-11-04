"""Optional B bundle; defaults remain in legacy_m15, without track phase."""
from dataclasses import replace

from holosoma.config_types.randomization import RandomizationTermCfg


def with_extra_dr(cfg):
    """Add the four agreed dev changes without changing the base in place."""
    setup = dict(cfg.randomization.setup_terms)
    reset = dict(cfg.randomization.reset_terms)
    setup["hdmi_body"] = RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.hdmi_body:randomize_body_startup",
        params={"mass_range": (0.9, 1.1), "com_range": (-0.03, 0.03)},
    )
    push = setup["push_randomizer_state"]
    setup["push_randomizer_state"] = replace(
        push, params={**push.params, "max_push_vel": [0.2, 0.2, 0.1, 0.2, 0.2, 0.4],
                      "push_interval_s": [4.0, 6.0]},
    )
    friction = RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.hdmi_friction:HDMIJointFrictionRandomizer",
        params={"effort_range_nm": (0.05, 0.25)},
    )
    setup["hdmi_joint_friction"] = friction
    reset["hdmi_joint_friction"] = friction

    def command_terms(terms, *, is_setup=False):
        terms = dict(terms)
        term = terms["motion_command"]
        params = dict(term.params)
        if is_setup:
            params["joint_velocity_noise_range"] = (-0.1, 0.1)
        terms["motion_command"] = replace(
            term, func="holosoma.managers.command.terms.hdmi_body:HDMIBodyMotionCommand",
            params=params,
        )
        return terms

    return replace(
        cfg,
        randomization=replace(cfg.randomization, setup_terms=setup, reset_terms=reset),
        command=replace(
            cfg.command,
            setup_terms=command_terms(cfg.command.setup_terms, is_setup=True),
            reset_terms=command_terms(cfg.command.reset_terms),
            step_terms=command_terms(cfg.command.step_terms),
        ),
    )
