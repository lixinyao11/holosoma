"""Optional actor-only phase: compose on the selected legacy config, not dev DR."""
from dataclasses import replace
from holosoma.config_types.observation import ObsTermCfg


def with_track_phase(cfg):
    motion = cfg.command.setup_terms["motion_command"].params["motion_config"]
    if motion.enable_default_pose_prepend or motion.enable_default_pose_append:
        raise ValueError("track_phase requires motion clips without default-pose transitions")
    groups = dict(cfg.observation.groups)
    actor = groups["actor_obs"]
    if actor.history_length != 1:
        raise ValueError("track_phase expects one current scalar, not group history")
    groups["actor_obs"] = replace(actor, terms={
        **actor.terms,
        "track_phase": ObsTermCfg(func="holosoma.managers.observation.terms.track_phase:track_phase"),
    })
    return replace(cfg, training=replace(cfg.training, name=cfg.training.name + "_track_phase"),
                   observation=replace(cfg.observation, groups=groups))
