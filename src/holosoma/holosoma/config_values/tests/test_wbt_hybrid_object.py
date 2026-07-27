from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values import command as command_defaults
from holosoma.config_values import experiment as experiment_defaults
from holosoma.config_values import observation as observation_defaults
from holosoma.config_values import reward as reward_defaults
from holosoma.config_values import termination as termination_defaults
from holosoma.config_values.wbt.g1.command import (
    g1_29dof_wbt_command_hybrid_object,
    hybrid_climb_motion_config,
    hybrid_object_motion_config,
)
from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_curriculum_hybrid_object
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt_hybrid_object,
    g1_29dof_wbt_w_object,
)
from holosoma.config_values.wbt.g1.observation import g1_29dof_wbt_observation_hybrid_object
from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization_hybrid_object
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward_hybrid_object
from holosoma.config_values.wbt.g1.termination import g1_29dof_wbt_termination_hybrid_object


def _hybrid_keys(defaults: dict) -> set[str]:
    return {key for key in defaults if "_wbt_hybrid" in key}


def test_public_release_exposes_exactly_two_hybrid_tasks():
    expected = {"g1_29dof_wbt_hybrid_climb", "g1_29dof_wbt_hybrid_object"}
    for defaults in (
        experiment_defaults.DEFAULTS,
        command_defaults.DEFAULTS,
        observation_defaults.DEFAULTS,
        reward_defaults.DEFAULTS,
        termination_defaults.DEFAULTS,
    ):
        assert _hybrid_keys(defaults) == expected

    aliases = {f"exp:{key.replace('_', '-')}": value for key, value in experiment_defaults.DEFAULTS.items()}
    assert aliases["exp:g1-29dof-wbt-hybrid-object"] is g1_29dof_wbt_hybrid_object


def test_hybrid_object_stack_is_explicit_and_renamed():
    assert isinstance(g1_29dof_wbt_hybrid_object, ExperimentConfig)
    assert g1_29dof_wbt_hybrid_object.env_class == g1_29dof_wbt_w_object.env_class
    assert g1_29dof_wbt_hybrid_object.action == g1_29dof_wbt_w_object.action
    assert g1_29dof_wbt_hybrid_object.randomization is g1_29dof_wbt_randomization_hybrid_object
    assert g1_29dof_wbt_hybrid_object.command is g1_29dof_wbt_command_hybrid_object
    assert g1_29dof_wbt_hybrid_object.observation is g1_29dof_wbt_observation_hybrid_object
    assert g1_29dof_wbt_hybrid_object.curriculum is g1_29dof_wbt_curriculum_hybrid_object
    assert g1_29dof_wbt_hybrid_object.reward is g1_29dof_wbt_reward_hybrid_object
    assert g1_29dof_wbt_hybrid_object.termination is g1_29dof_wbt_termination_hybrid_object
    assert g1_29dof_wbt_hybrid_object.training.name == "g1_29dof_wbt_hybrid_object"
    assert g1_29dof_wbt_hybrid_object.nightly is None


def test_hybrid_object_noise_and_size_defaults():
    cfg = hybrid_object_motion_config.hybrid
    assert cfg is not None
    assert hybrid_object_motion_config.start_at_timestep_zero_prob == 0.05
    assert cfg.track_ratio == 0.5
    assert cfg.env_partition_seed == 0
    assert cfg.gen_start_at_timestep_zero_prob == 0.2
    assert cfg.gen_step_zero_root_noise is not None
    assert cfg.gen_step_zero_root_noise.pos_std_xyz == [0.01, 0.01, 0.003]
    assert cfg.gen_step_zero_root_noise.pos_clip_xyz == [0.05, 0.05, 0.01]
    assert cfg.gen_step_zero_root_noise.rpy_std == [0.01, 0.01, 0.03]
    assert cfg.gen_step_zero_root_noise.rpy_clip == [0.05, 0.05, 0.1]
    assert cfg.robot_goal_noise.pos_std_xyz == [0.05, 0.05, 0.001]
    assert cfg.robot_goal_noise.pos_clip_xyz == [0.10, 0.10, 0.001]
    assert cfg.object_goal_noise.pos_std_xyz == [0.2, 0.2, 0.001]
    assert cfg.object_goal_noise.pos_clip_xyz == [0.3, 0.3, 0.001]
    assert cfg.object_goal_noise.rpy_std == [0.001, 0.001, 0.2]
    assert cfg.object_goal_noise.rpy_clip == [0.001, 0.001, 0.4]
    assert cfg.size_randomization.enabled is False
    assert cfg.size_randomization.mode == "uniform_xyz"
    assert cfg.size_randomization.track_scale_range == [0.95, 1.05]
    assert cfg.size_randomization.gen_scale_range == [0.85, 1.15]


def test_hybrid_climb_adds_small_yaw_only_robot_goal_noise():
    cfg = hybrid_climb_motion_config.hybrid
    assert cfg is not None
    assert cfg.robot_goal_noise.pos_std_xyz == [0.05, 0.05, 0.001]
    assert cfg.robot_goal_noise.pos_clip_xyz == [0.10, 0.10, 0.001]
    assert cfg.robot_goal_noise.rpy_std == [0.001, 0.001, 0.03]
    assert cfg.robot_goal_noise.rpy_clip == [0.001, 0.001, 0.10]

    object_cfg = hybrid_object_motion_config.hybrid
    assert object_cfg is not None
    assert object_cfg.robot_goal_noise.rpy_std == [0.001, 0.001, 0.001]
    assert object_cfg.robot_goal_noise.rpy_clip == [0.001, 0.001, 0.001]


def test_hybrid_object_keeps_only_the_object_goal_curriculum():
    assert set(g1_29dof_wbt_curriculum_hybrid_object.setup_terms) == {
        "average_episode_tracker",
        "hybrid_gen_noise_curriculum",
    }
    term = g1_29dof_wbt_curriculum_hybrid_object.setup_terms["hybrid_gen_noise_curriculum"]
    assert term.func.endswith(":HybridGenNoiseCurriculum")
    assert term.params == {
        "enabled": True,
        "initial_scale": 0.3,
        "min_scale": 0.3,
        "max_scale": 1.0,
        "scale_step": 0.01,
        "success_threshold_up": 0.90,
        "success_threshold_down": 0.80,
        "update_interval": 20,
        "ema_alpha": 0.5,
    }


def test_hybrid_object_observation_and_task_terms_are_clean():
    actor_terms = g1_29dof_wbt_observation_hybrid_object.groups["actor_obs"].terms
    critic_terms = g1_29dof_wbt_observation_hybrid_object.groups["critic_obs"].terms
    assert {"assist_beta", "assist_force_b", "assist_torque_b"}.isdisjoint(critic_terms)
    assert {"obj_goal_pos_b", "obj_goal_ori_b"}.issubset(actor_terms)
    assert critic_terms["motion_command"].func.endswith(":hybrid_motion_command")
    assert critic_terms["motion_ref_pos_b"].func.endswith(":hybrid_motion_ref_pos_b")
    assert critic_terms["obj_ref_pos_b"].func.endswith(":hybrid_obj_ref_pos_b")
    assert g1_29dof_wbt_reward_hybrid_object.terms["goal_object_reached_once"].func.endswith(
        ":HybridGoalObjectReachedOnce"
    )
    assert g1_29dof_wbt_termination_hybrid_object.terms["bad_tracking"].func.endswith(":HybridBadTracking")
