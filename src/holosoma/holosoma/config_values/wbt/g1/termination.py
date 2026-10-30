"""Whole Body Tracking termination presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

g1_29dof_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            params={
                # robot tracking
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
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
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

# The object checked against the reference object re-anchored to the robot, like the tracked bodies.
g1_29dof_wbt_termination_rel_object = TerminationManagerCfg(
    terms={
        **g1_29dof_wbt_termination.terms,
        "bad_tracking": replace(
            g1_29dof_wbt_termination.terms["bad_tracking"],
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnlyRelObject",
        ),
    }
)

# Relative object, checked as HDMI does: the object ends the episode once it has been more than 0.6 m / 1.2 rad off
# for 5 consecutive steps (instead of 0.25 m / 0.8 rad at once).
g1_29dof_wbt_termination_rel_object_persistent = TerminationManagerCfg(
    terms={
        **g1_29dof_wbt_termination.terms,
        "bad_tracking": replace(
            g1_29dof_wbt_termination.terms["bad_tracking"],
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnlyRelObjectPersistent",
            params={
                **g1_29dof_wbt_termination.terms["bad_tracking"].params,
                "bad_object_pos_threshold": 0.6,
                "bad_object_ori_threshold": 1.2,
                "object_persist_steps": 5,
            },
        ),
    }
)

__all__ = [
    "g1_29dof_wbt_termination",
    "g1_29dof_wbt_termination_rel_object",
    "g1_29dof_wbt_termination_rel_object_persistent",
]
