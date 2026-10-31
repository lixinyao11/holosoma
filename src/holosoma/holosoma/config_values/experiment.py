import tyro
from typing_extensions import Annotated

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.loco.g1.experiment import g1_29dof, g1_29dof_fast_sac
from holosoma.config_values.loco.t1.experiment import t1_29dof, t1_29dof_fast_sac
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt,
    g1_29dof_wbt_climb,
    g1_29dof_wbt_fast_sac,
    g1_29dof_wbt_fast_sac_w_object,
    g1_29dof_wbt_hybrid_climb,
    g1_29dof_wbt_hybrid_multi_object,
    g1_29dof_wbt_hybrid_object,
    g1_29dof_wbt_w_object,
)

from holosoma.config_values.wbt.g1.hdmi_goal import g1_29dof_wbt_hybrid_object_hdmi

from holosoma.config_values.wbt.g1.hdmi_actuator_dr import g1_29dof_wbt_hybrid_object_hdmi_actuator_dr
from holosoma.config_values.wbt.g1.hdmi_body_dr import g1_29dof_wbt_hybrid_object_hdmi_body_dr
from holosoma.config_values.wbt.g1.hdmi_push_dr import g1_29dof_wbt_hybrid_object_hdmi_push_dr

from holosoma.config_values.wbt.g1.hdmi_friction_dr import g1_29dof_wbt_hybrid_object_hdmi_friction_dr

from holosoma.config_values.wbt.g1.hdmi_track_phase import g1_29dof_wbt_hybrid_object_hdmi_friction_dr_track_phase

DEFAULTS = {
    "g1_29dof_wbt_hybrid_object_hdmi_friction_dr_track_phase": g1_29dof_wbt_hybrid_object_hdmi_friction_dr_track_phase,
    "g1_29dof_wbt_hybrid_object_hdmi_friction_dr": g1_29dof_wbt_hybrid_object_hdmi_friction_dr,
    "g1_29dof_wbt_hybrid_object_hdmi_actuator_dr": g1_29dof_wbt_hybrid_object_hdmi_actuator_dr,
    "g1_29dof_wbt_hybrid_object_hdmi_body_dr": g1_29dof_wbt_hybrid_object_hdmi_body_dr,
    "g1_29dof_wbt_hybrid_object_hdmi_push_dr": g1_29dof_wbt_hybrid_object_hdmi_push_dr,
    "g1_29dof_wbt_hybrid_object_hdmi": g1_29dof_wbt_hybrid_object_hdmi,
    "g1_29dof": g1_29dof,
    "g1_29dof_fast_sac": g1_29dof_fast_sac,
    "t1_29dof": t1_29dof,
    "t1_29dof_fast_sac": t1_29dof_fast_sac,
    "g1_29dof_wbt": g1_29dof_wbt,
    "g1_29dof_wbt_climb": g1_29dof_wbt_climb,
    "g1_29dof_wbt_hybrid_climb": g1_29dof_wbt_hybrid_climb,
    "g1_29dof_wbt_hybrid_object": g1_29dof_wbt_hybrid_object,
    "g1_29dof_wbt_hybrid_multi_object": g1_29dof_wbt_hybrid_multi_object,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_w_object,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac,
    "g1_29dof_wbt_fast_sac_w_object": g1_29dof_wbt_fast_sac_w_object,
}

AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{k.replace('_', '-')}": v for k, v in DEFAULTS.items()}
        )
    ),
]
