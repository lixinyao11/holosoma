"""Opt-in 1D track phase on top of the accepted mode15 + cumulative DR preset."""
from dataclasses import replace
from holosoma.config_types.observation import ObsTermCfg
from holosoma.config_values.wbt.g1.hdmi_friction_dr import g1_29dof_wbt_hybrid_object_hdmi_friction_dr as base

groups = dict(base.observation.groups)
actor = groups["actor_obs"]
groups["actor_obs"] = replace(actor, terms={
    **actor.terms,
    "track_phase": ObsTermCfg(func="holosoma.managers.observation.terms.track_phase:track_phase"),
})
g1_29dof_wbt_hybrid_object_hdmi_friction_dr_track_phase = replace(
    base, observation=replace(base.observation, groups=groups))
