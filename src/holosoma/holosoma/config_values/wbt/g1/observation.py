"""Whole Body Tracking observation presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsHistoryAugCfg, ObsTermCfg

WBT_OBJECT_HISTORY_1S_SOURCE_TERMS = ("base_ang_vel", "dof_pos", "dof_vel", "actions", "obj_pos_b", "obj_ori_b")

wbt_object_history_1s = ObsHistoryAugCfg(
    source_terms=WBT_OBJECT_HISTORY_1S_SOURCE_TERMS,
    sample_offsets=(),
    duration_s=1.0,
    stride_steps=10,
)

actor_obs_shared_terms = {
    "motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command",
        scale=1.0,
        noise=0.0,
    ),
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
        scale=1.0,
        noise=0.05,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
}

actor_obs_shared = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms=actor_obs_shared_terms,
)

actor_obs_w_object_terms = actor_obs_shared_terms.copy()
actor_obs_w_object_terms.update(
    {
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_lin_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ref_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ref_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ref_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ref_ori_b",
            scale=1.0,
            noise=0.0,
        ),
    }
)

actor_obs_w_object = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms=actor_obs_w_object_terms,
)

actor_obs_hybrid_state_terms = {
    "motion_command": replace(actor_obs_shared_terms["motion_command"], noise=0.0),
    "motion_ref_ori_b": replace(actor_obs_shared_terms["motion_ref_ori_b"], noise=0.0),
    "base_ang_vel": replace(actor_obs_shared_terms["base_ang_vel"], noise=0.10),
    "dof_pos": replace(actor_obs_shared_terms["dof_pos"], noise=0.005),
    "dof_vel": replace(actor_obs_shared_terms["dof_vel"], noise=0.25),
    "actions": replace(actor_obs_shared_terms["actions"], noise=0.0),
}

actor_obs_hybrid_object_state_terms = actor_obs_hybrid_state_terms.copy()
actor_obs_hybrid_object_state_terms.update(
    {
        "obj_pos_b": replace(actor_obs_w_object_terms["obj_pos_b"], noise=0.0),
        "obj_ori_b": replace(actor_obs_w_object_terms["obj_ori_b"], noise=0.0),
        "obj_lin_vel_b": replace(actor_obs_w_object_terms["obj_lin_vel_b"], noise=0.0),
        "obj_ref_pos_b": replace(actor_obs_w_object_terms["obj_ref_pos_b"], noise=0.0),
        "obj_ref_ori_b": replace(actor_obs_w_object_terms["obj_ref_ori_b"], noise=0.0),
    }
)

motion_goal_pos_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:motion_goal_pos_b",
    scale=1.0,
    noise=0.0,
)

motion_goal_ori_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:motion_goal_ori_b",
    scale=1.0,
    noise=0.0,
)

obj_goal_pos_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:obj_goal_pos_b",
    scale=1.0,
    noise=0.0,
)

obj_goal_ori_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:obj_goal_ori_b",
    scale=1.0,
    noise=0.0,
)

actor_obs_hybrid_climb_terms = {
    "motion_goal_pos_b": motion_goal_pos_term,
    "motion_goal_ori_b": motion_goal_ori_term,
    "base_ang_vel": replace(actor_obs_hybrid_state_terms["base_ang_vel"]),
    "dof_pos": replace(actor_obs_hybrid_state_terms["dof_pos"]),
    "dof_vel": replace(actor_obs_hybrid_state_terms["dof_vel"]),
    "actions": replace(actor_obs_hybrid_state_terms["actions"]),
}

actor_obs_hybrid_climb = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms=actor_obs_hybrid_climb_terms,
)

actor_obs_hybrid_object_terms = actor_obs_hybrid_climb_terms.copy()
actor_obs_hybrid_object_terms.pop("motion_goal_pos_b")
actor_obs_hybrid_object_terms.pop("motion_goal_ori_b")
actor_obs_hybrid_object_terms.update(
    {
        "obj_pos_b": replace(actor_obs_hybrid_object_state_terms["obj_pos_b"]),
        "obj_ori_b": replace(actor_obs_hybrid_object_state_terms["obj_ori_b"]),
        "obj_lin_vel_b": replace(actor_obs_hybrid_object_state_terms["obj_lin_vel_b"]),
        "obj_goal_pos_b": obj_goal_pos_term,
        "obj_goal_ori_b": obj_goal_ori_term,
    }
)

actor_obs_hybrid_object = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    history_augmentation=wbt_object_history_1s,
    terms=actor_obs_hybrid_object_terms,
)

critic_obs_shared_terms = {
    "motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command",
        scale=1.0,
        noise=0.0,
    ),
    "motion_ref_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
        scale=1.0,
        noise=0.25,
    ),
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
        scale=1.0,
        noise=0.05,
    ),
    "robot_body_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_pos_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_ori_b",
        scale=1.0,
        noise=0.0,
    ),
    "base_lin_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_lin_vel",
        scale=1.0,
        noise=0.0,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
}

critic_obs_w_object_terms = critic_obs_shared_terms.copy()
critic_obs_w_object_terms.update(
    {
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_lin_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ref_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ref_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ref_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ref_ori_b",
            scale=1.0,
            noise=0.0,
        ),
    }
)

critic_obs_hybrid_climb_terms = critic_obs_shared_terms.copy()
critic_obs_hybrid_climb_terms.update(
    {
        "motion_goal_pos_b": motion_goal_pos_term,
        "motion_goal_ori_b": motion_goal_ori_term,
    }
)

critic_obs_hybrid_object_terms = critic_obs_w_object_terms.copy()
critic_obs_hybrid_object_terms.update(
    {
        "obj_goal_pos_b": obj_goal_pos_term,
        "obj_goal_ori_b": obj_goal_ori_term,
    }
)

critic_obs_hybrid_climb_terms.update(
    {
        "motion_command": replace(
            critic_obs_hybrid_climb_terms["motion_command"],
            func="holosoma.managers.observation.terms.wbt:hybrid_motion_command",
        ),
        "motion_ref_pos_b": replace(
            critic_obs_hybrid_climb_terms["motion_ref_pos_b"],
            func="holosoma.managers.observation.terms.wbt:hybrid_motion_ref_pos_b",
        ),
        "motion_ref_ori_b": replace(
            critic_obs_hybrid_climb_terms["motion_ref_ori_b"],
            func="holosoma.managers.observation.terms.wbt:hybrid_motion_ref_ori_b",
        ),
    }
)

critic_obs_hybrid_object_terms.update(
    {
        "motion_command": replace(
            critic_obs_hybrid_object_terms["motion_command"],
            func="holosoma.managers.observation.terms.wbt:hybrid_motion_command",
        ),
        "motion_ref_pos_b": replace(
            critic_obs_hybrid_object_terms["motion_ref_pos_b"],
            func="holosoma.managers.observation.terms.wbt:hybrid_motion_ref_pos_b",
        ),
        "motion_ref_ori_b": replace(
            critic_obs_hybrid_object_terms["motion_ref_ori_b"],
            func="holosoma.managers.observation.terms.wbt:hybrid_motion_ref_ori_b",
        ),
        "obj_ref_pos_b": replace(
            critic_obs_hybrid_object_terms["obj_ref_pos_b"],
            func="holosoma.managers.observation.terms.wbt:hybrid_obj_ref_pos_b",
        ),
        "obj_ref_ori_b": replace(
            critic_obs_hybrid_object_terms["obj_ref_ori_b"],
            func="holosoma.managers.observation.terms.wbt:hybrid_obj_ref_ori_b",
        ),
    }
)

g1_29dof_wbt_observation = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_w_object,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
    },
)

g1_29dof_wbt_observation_hybrid_climb = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_hybrid_climb,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_hybrid_climb_terms,
        ),
    },
)

g1_29dof_wbt_observation_hybrid_object = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_hybrid_object,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            history_augmentation=wbt_object_history_1s,
            terms=critic_obs_hybrid_object_terms,
        ),
    },
)

__all__ = [
    "g1_29dof_wbt_observation",
    "g1_29dof_wbt_observation_hybrid_climb",
    "g1_29dof_wbt_observation_hybrid_object",
    "g1_29dof_wbt_observation_w_object",
]
