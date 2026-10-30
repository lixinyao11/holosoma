"""Default termination manager configurations."""

from holosoma.config_values.loco.g1.termination import g1_29dof_termination
from holosoma.config_values.loco.t1.termination import t1_29dof_termination
from holosoma.config_values.wbt.g1.termination import (
    g1_29dof_wbt_termination,
    g1_29dof_wbt_termination_hybrid_climb,
    g1_29dof_wbt_termination_hybrid_object,
    g1_29dof_wbt_termination_hybrid_object_persistent,
    g1_29dof_wbt_termination_hybrid_object_persistent_anchor,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_termination,
    "g1_29dof": g1_29dof_termination,
    "g1_29dof_wbt": g1_29dof_wbt_termination,
    "g1_29dof_wbt_hybrid_climb": g1_29dof_wbt_termination_hybrid_climb,
    "g1_29dof_wbt_hybrid_object": g1_29dof_wbt_termination_hybrid_object,
    "g1_29dof_wbt_hybrid_object_persistent": g1_29dof_wbt_termination_hybrid_object_persistent,
    "g1_29dof_wbt_hybrid_object_persistent_anchor": g1_29dof_wbt_termination_hybrid_object_persistent_anchor,
}
