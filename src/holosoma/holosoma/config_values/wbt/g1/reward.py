"""Whole Body Tracking reward presets for the G1 robot."""

from dataclasses import replace

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

# The object tracked relative to the robot (the reference object re-anchored to the robot, like the relative body
# terms) instead of in the world frame.
g1_29dof_wbt_reward_w_object_rel = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "object_relative_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_relative_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "object_relative_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_relative_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

# Same, with the anchor's world position weighted as much as the relative body terms (1.0 instead of 0.5), as HDMI's
# root-aware-global tracking: the robot has to get to the object, not only move like the reference.
g1_29dof_wbt_reward_w_object_rel_global_root = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object_rel.terms,
        "motion_global_ref_position_error_exp": replace(
            g1_29dof_wbt_reward_w_object_rel.terms["motion_global_ref_position_error_exp"], weight=1.0
        ),
    }
)

# Same, with the hands rewarded for being where the reference puts them on the object (in the object's frame, sigma
# 0.1 m, while within 0.5 m of it in the reference) and HDMI's joint position term: with only the relative body terms
# (sigma 0.3 m over 14 bodies) a hand 10 cm off still gets ~90 % of the reward, and never touches the object.
g1_29dof_wbt_reward_w_object_rel_global_root_hands = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object_rel_global_root.terms,
        "hand_object_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hand_object_position_error_exp",
            params={
                "sigma": 0.1,
                "body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
                "max_ref_distance": 0.5,
            },
            weight=1.0,
        ),
        "motion_joint_pos_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_pos_error_exp",
            params={"sigma": 0.25},
            weight=0.5,
        ),
    }
)

__all__ = [
    "g1_29dof_wbt_fast_sac_reward",
    "g1_29dof_wbt_reward",
    "g1_29dof_wbt_reward_w_object",
    "g1_29dof_wbt_reward_w_object_rel",
    "g1_29dof_wbt_reward_w_object_rel_global_root",
    "g1_29dof_wbt_reward_w_object_rel_global_root_hands",
]
