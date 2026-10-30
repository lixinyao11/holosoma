"""Whole Body Tracking termination presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

g1_29dof_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "body_proximity": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BodyGroupProximity",
            params={
                "min_distance": 0.05,
                "body_groups": [["left_foot_contact_point", "right_foot_contact_point"]],
            },
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            params={
                # robot tracking
                "bad_ref_pos_threshold": 0.4,
                "bad_ref_ori_threshold": 0.4,
                "bad_motion_body_pos_threshold": 0.25,
                # NOTE: body_names_to_track is shared with command_manager
                "body_names_to_track": [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                # object tracking
                # only triggered when has_object=True
                "bad_object_pos_threshold": 0.3,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

g1_29dof_wbt_termination_hybrid_climb = TerminationManagerCfg(
    terms={
        "timeout": g1_29dof_wbt_termination.terms["timeout"],
        "body_proximity": g1_29dof_wbt_termination.terms["body_proximity"],
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:HybridBadTracking",
            params={
                **g1_29dof_wbt_termination.terms["bad_tracking"].params,
                "min_root_height": 0.45,
                "gen_bad_ref_pos_z_threshold": 0.4,
                "gen_bad_ref_pos_xyz_threshold": 100.0,
            },
        ),
    }
)

g1_29dof_wbt_termination_hybrid_object = TerminationManagerCfg(
    terms={
        "timeout": g1_29dof_wbt_termination_hybrid_climb.terms["timeout"],
        "body_proximity": g1_29dof_wbt_termination_hybrid_climb.terms["body_proximity"],
        "bad_tracking": TerminationTermCfg(
            func=g1_29dof_wbt_termination_hybrid_climb.terms["bad_tracking"].func,
            params={
                **g1_29dof_wbt_termination_hybrid_climb.terms["bad_tracking"].params,
                "gen_bad_object_pos_z_threshold": 0.6,
                "gen_bad_object_ori_threshold": 1.0,
            },
        ),
    }
)

# Objects checked as HDMI does: an object check (track: 0.6 m / 1.2 rad; gen: as above) ends the episode only after 5
# consecutive bad steps.
g1_29dof_wbt_termination_hybrid_object_persistent = TerminationManagerCfg(
    terms={
        **g1_29dof_wbt_termination_hybrid_object.terms,
        "bad_tracking": TerminationTermCfg(
            func=g1_29dof_wbt_termination_hybrid_object.terms["bad_tracking"].func,
            params={
                **g1_29dof_wbt_termination_hybrid_object.terms["bad_tracking"].params,
                "bad_object_pos_threshold": 0.6,
                "bad_object_ori_threshold": 1.2,
                "object_persist_steps": 5,
            },
        ),
    }
)

# Same, with the track envs' anchor ending the episode at 0.3 m off instead of 0.4 m.
g1_29dof_wbt_termination_hybrid_object_persistent_anchor = TerminationManagerCfg(
    terms={
        **g1_29dof_wbt_termination_hybrid_object_persistent.terms,
        "bad_tracking": replace(
            g1_29dof_wbt_termination_hybrid_object_persistent.terms["bad_tracking"],
            params={**g1_29dof_wbt_termination_hybrid_object_persistent.terms["bad_tracking"].params, "bad_ref_pos_threshold": 0.3},
        ),
    }
)

__all__ = [
    "g1_29dof_wbt_termination",
    "g1_29dof_wbt_termination_hybrid_climb",
    "g1_29dof_wbt_termination_hybrid_object",
    "g1_29dof_wbt_termination_hybrid_object_persistent",
    "g1_29dof_wbt_termination_hybrid_object_persistent_anchor",
]
