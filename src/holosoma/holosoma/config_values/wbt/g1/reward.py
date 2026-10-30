"""Whole Body Tracking reward presets for the G1 robot."""

from dataclasses import replace
from math import pi

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

g1_29dof_wbt_reward = RewardManagerCfg(
    terms={
        # Motion tracking rewards - global reference frame
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=0.5,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        # Motion tracking rewards - body velocities
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14},
            weight=1.0,
        ),
        # Regularization rewards
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.1,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
                    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
                ),
            },
            weight=-0.1,
        ),
    }
)

g1_29dof_wbt_fast_sac_reward = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-1.0,
        ),
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=2.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        # Motion tracking rewards - global reference frame
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

hybrid_goal_ref_reached_once_term = RewardTermCfg(
    func="holosoma.managers.reward.terms.wbt:HybridGoalRefReachedOnce",
    params={"bonus": 2.0, "pos_threshold": 0.2, "ori_threshold": pi / 6.0},
    weight=1.0,
)

hybrid_goal_object_reached_once_term = RewardTermCfg(
    func="holosoma.managers.reward.terms.wbt:HybridGoalObjectReachedOnce",
    params={"bonus": 3.0, "pos_threshold": 0.2, "ori_threshold": pi / 6.0},
    weight=1.0,
)

g1_29dof_wbt_reward_hybrid_climb = RewardManagerCfg(
    terms={
        "motion_global_ref_position_error_exp": replace(
            g1_29dof_wbt_reward.terms["motion_global_ref_position_error_exp"],
            func="holosoma.managers.reward.terms.wbt:hybrid_motion_global_ref_position_error_exp",
        ),
        "motion_global_ref_orientation_error_exp": replace(
            g1_29dof_wbt_reward.terms["motion_global_ref_orientation_error_exp"],
            func="holosoma.managers.reward.terms.wbt:hybrid_motion_global_ref_orientation_error_exp",
        ),
        "motion_relative_body_position_error_exp": replace(
            g1_29dof_wbt_reward.terms["motion_relative_body_position_error_exp"],
            func="holosoma.managers.reward.terms.wbt:hybrid_motion_relative_body_position_error_exp",
        ),
        "motion_relative_body_orientation_error_exp": replace(
            g1_29dof_wbt_reward.terms["motion_relative_body_orientation_error_exp"],
            func="holosoma.managers.reward.terms.wbt:hybrid_motion_relative_body_orientation_error_exp",
        ),
        "motion_global_body_lin_vel": replace(
            g1_29dof_wbt_reward.terms["motion_global_body_lin_vel"],
            func="holosoma.managers.reward.terms.wbt:hybrid_motion_global_body_lin_vel",
        ),
        "motion_global_body_ang_vel": replace(
            g1_29dof_wbt_reward.terms["motion_global_body_ang_vel"],
            func="holosoma.managers.reward.terms.wbt:hybrid_motion_global_body_ang_vel",
        ),
        "goal_ref_reached_once": hybrid_goal_ref_reached_once_term,
        "action_rate_l2": g1_29dof_wbt_reward.terms["action_rate_l2"],
        "limits_dof_pos": g1_29dof_wbt_reward.terms["limits_dof_pos"],
        "undesired_contacts": g1_29dof_wbt_reward.terms["undesired_contacts"],
    }
)

g1_29dof_wbt_reward_hybrid_object = RewardManagerCfg(
    terms={
        "motion_global_ref_position_error_exp": g1_29dof_wbt_reward_hybrid_climb.terms[
            "motion_global_ref_position_error_exp"
        ],
        "motion_global_ref_orientation_error_exp": g1_29dof_wbt_reward_hybrid_climb.terms[
            "motion_global_ref_orientation_error_exp"
        ],
        "motion_relative_body_position_error_exp": g1_29dof_wbt_reward_hybrid_climb.terms[
            "motion_relative_body_position_error_exp"
        ],
        "motion_relative_body_orientation_error_exp": g1_29dof_wbt_reward_hybrid_climb.terms[
            "motion_relative_body_orientation_error_exp"
        ],
        "motion_global_body_lin_vel": g1_29dof_wbt_reward_hybrid_climb.terms["motion_global_body_lin_vel"],
        "motion_global_body_ang_vel": g1_29dof_wbt_reward_hybrid_climb.terms["motion_global_body_ang_vel"],
        "object_global_ref_position_error_exp": replace(
            g1_29dof_wbt_reward_w_object.terms["object_global_ref_position_error_exp"],
            func="holosoma.managers.reward.terms.wbt:hybrid_object_global_ref_position_error_exp",
        ),
        "object_global_ref_orientation_error_exp": replace(
            g1_29dof_wbt_reward_w_object.terms["object_global_ref_orientation_error_exp"],
            func="holosoma.managers.reward.terms.wbt:hybrid_object_global_ref_orientation_error_exp",
        ),
        "goal_object_reached_once": hybrid_goal_object_reached_once_term,
        "action_rate_l2": g1_29dof_wbt_reward.terms["action_rate_l2"],
        "limits_dof_pos": g1_29dof_wbt_reward.terms["limits_dof_pos"],
        "undesired_contacts": g1_29dof_wbt_reward.terms["undesired_contacts"],
    }
)

# Same, with (track envs) the hands rewarded for being where the reference puts them on the object (in the object's
# frame, sigma 0.1 m, while within 0.5 m of it in the reference) and HDMI's joint position term: with only the
# relative body terms a hand 10 cm off still gets ~90 % of the reward, and never touches the object.
g1_29dof_wbt_reward_hybrid_object_hands = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_hybrid_object.terms,
        "hand_object_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_hand_object_position_error_exp",
            params={
                "sigma": 0.1,
                "body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
                "max_ref_distance": 0.5,
            },
            weight=1.0,
        ),
        "motion_joint_pos_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_motion_joint_pos_error_exp",
            params={"sigma": 0.25},
            weight=0.5,
        ),
    }
)

__all__ = [
    "g1_29dof_wbt_fast_sac_reward",
    "g1_29dof_wbt_reward",
    "g1_29dof_wbt_reward_hybrid_climb",
    "g1_29dof_wbt_reward_hybrid_object",
    "g1_29dof_wbt_reward_hybrid_object_hands",
    "g1_29dof_wbt_reward_w_object",
    "hybrid_goal_object_reached_once_term",
    "hybrid_goal_ref_reached_once_term",
]
