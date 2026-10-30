"""Default observation manager configurations."""

from holosoma.config_values.loco.g1.observation import g1_29dof_loco_single_wolinvel
from holosoma.config_values.loco.t1.observation import t1_29dof_loco_single_wolinvel
from holosoma.config_values.wbt.g1.observation import (
    g1_29dof_wbt_observation,
    g1_29dof_wbt_observation_hybrid_climb,
    g1_29dof_wbt_observation_hybrid_multi_object,
    g1_29dof_wbt_observation_hybrid_multi_object_future,
    g1_29dof_wbt_observation_hybrid_multi_object_future_hands,
    g1_29dof_wbt_observation_hybrid_object,
    g1_29dof_wbt_observation_w_object,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof_loco_single_wolinvel": t1_29dof_loco_single_wolinvel,
    "g1_29dof_loco_single_wolinvel": g1_29dof_loco_single_wolinvel,
    "g1_29dof_wbt": g1_29dof_wbt_observation,
    "g1_29dof_wbt_hybrid_climb": g1_29dof_wbt_observation_hybrid_climb,
    "g1_29dof_wbt_hybrid_object": g1_29dof_wbt_observation_hybrid_object,
    "g1_29dof_wbt_hybrid_multi_object": g1_29dof_wbt_observation_hybrid_multi_object,
    "g1_29dof_wbt_hybrid_multi_object_future": g1_29dof_wbt_observation_hybrid_multi_object_future,
    "g1_29dof_wbt_hybrid_multi_object_future_hands": g1_29dof_wbt_observation_hybrid_multi_object_future_hands,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_observation_w_object,
}
