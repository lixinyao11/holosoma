from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values import command as command_defaults_module
from holosoma.config_values import experiment as experiment_defaults_module
from holosoma.config_values import robot as robot_defaults_module
from holosoma.config_values.wbt.g1.command import (
    g1_29dof_wbt_command_hybrid_climb,
    hybrid_climb_motion_config,
    motion_config_climb,
)
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt, g1_29dof_wbt_hybrid_climb
from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization_hybrid


def test_wbt_hybrid_climb_defaults_are_registered():
    assert command_defaults_module.DEFAULTS["g1_29dof_wbt_hybrid_climb"] is g1_29dof_wbt_command_hybrid_climb
    assert experiment_defaults_module.DEFAULTS["g1_29dof_wbt_hybrid_climb"] is g1_29dof_wbt_hybrid_climb

    aliases = {f"exp:{key.replace('_', '-')}": value for key, value in experiment_defaults_module.DEFAULTS.items()}
    assert aliases["exp:g1-29dof-wbt-hybrid-climb"] is g1_29dof_wbt_hybrid_climb


def test_wbt_hybrid_climb_size_randomization_defaults_are_disabled():
    size_cfg = hybrid_climb_motion_config.hybrid.size_randomization

    assert size_cfg.enabled is False
    assert size_cfg.mode == "height_only"
    assert size_cfg.track_scale_range == [0.95, 1.05]
    assert size_cfg.gen_scale_range == [0.85, 1.15]
    assert size_cfg.sampling_seed == 0
    assert size_cfg.ground_clearance == 0.001
    assert size_cfg.ground_contact_tolerance == 0.02


def test_wbt_hybrid_climb_experiment_uses_an_explicit_stack():
    assert isinstance(g1_29dof_wbt_hybrid_climb, ExperimentConfig)

    assert g1_29dof_wbt_hybrid_climb.env_class == g1_29dof_wbt.env_class
    assert g1_29dof_wbt_hybrid_climb.algo == g1_29dof_wbt.algo
    assert g1_29dof_wbt_hybrid_climb.simulator == g1_29dof_wbt.simulator
    assert g1_29dof_wbt_hybrid_climb.robot.asset.urdf_file == robot_defaults_module.g1_29dof_w_object.asset.urdf_file
    assert g1_29dof_wbt_hybrid_climb.robot.object.object_urdf_path is None
    assert g1_29dof_wbt_hybrid_climb.terrain == g1_29dof_wbt.terrain
    assert g1_29dof_wbt_hybrid_climb.action == g1_29dof_wbt.action
    assert g1_29dof_wbt_hybrid_climb.randomization is g1_29dof_wbt_randomization_hybrid
    assert set(g1_29dof_wbt_hybrid_climb.curriculum.setup_terms) == {"average_episode_tracker"}
    assert g1_29dof_wbt_hybrid_climb.command is g1_29dof_wbt_command_hybrid_climb
    assert g1_29dof_wbt_hybrid_climb.training.name == "g1_29dof_wbt_hybrid_climb"
    assert g1_29dof_wbt_hybrid_climb.nightly is None


def test_wbt_hybrid_climb_motion_config_reuses_climb_clip_and_hybrid_command():
    hybrid_cfg = hybrid_climb_motion_config.hybrid

    assert (
        hybrid_climb_motion_config.motion_file
        == "holosoma/data/motions/g1_29dof/whole_body_tracking/climbox1/walk_climb1_reframed.npz"
    )
    assert hybrid_climb_motion_config.motion_dir == ""
    assert hybrid_climb_motion_config.load_fixed_scene_from_motion is True
    assert hybrid_climb_motion_config.goal_orientation.robot_yaw_only is False
    assert hybrid_climb_motion_config.goal_orientation.object_yaw_only is False
    assert hybrid_climb_motion_config.start_at_timestep_zero_prob == 0.05
    assert hybrid_cfg is not None
    assert hybrid_cfg.track_ratio == 0.5
    assert hybrid_cfg.env_partition_seed == 0
    assert hybrid_cfg.gen_start_at_timestep_zero_prob == 0.2
    assert hybrid_cfg.size_randomization.enabled is False
    assert hybrid_cfg.size_randomization.mode == "height_only"
    assert hybrid_cfg.size_randomization.track_scale_range == [0.95, 1.05]
    assert hybrid_cfg.size_randomization.gen_scale_range == [0.85, 1.15]
    assert hybrid_climb_motion_config.noise_to_initial_pose.step_zero_root_noise is not None
    assert hybrid_climb_motion_config.noise_to_initial_pose.step_zero_root_noise.pos_std_xyz == [0.05, 0.05, 0.01]
    assert hybrid_climb_motion_config.noise_to_initial_pose.step_zero_root_noise.pos_clip_xyz == [0.2, 0.2, 0.02]
    assert hybrid_climb_motion_config.noise_to_initial_pose.step_zero_root_noise.rpy_std == [0.05, 0.05, 0.1]
    assert hybrid_climb_motion_config.noise_to_initial_pose.step_zero_root_noise.rpy_clip == [0.1, 0.1, 0.3]
    assert hybrid_cfg.gen_step_zero_root_noise is not None
    assert hybrid_cfg.gen_step_zero_root_noise.pos_std_xyz == [0.2, 0.2, 0.01]
    assert hybrid_cfg.gen_step_zero_root_noise.pos_clip_xyz == [1.0, 1.0, 0.02]
    assert hybrid_cfg.gen_step_zero_root_noise.rpy_std == [0.05, 0.05, 0.2]
    assert hybrid_cfg.gen_step_zero_root_noise.rpy_clip == [0.15, 0.15, 0.6]
    assert g1_29dof_wbt_command_hybrid_climb.setup_terms["motion_command"].params["motion_config"].hybrid == hybrid_cfg
    assert hybrid_climb_motion_config.body_names_to_track == motion_config_climb.body_names_to_track
    assert hybrid_climb_motion_config.body_name_ref == motion_config_climb.body_name_ref
    assert g1_29dof_wbt_command_hybrid_climb.setup_terms["motion_command"].func.endswith(":HybridMotionCommand")
    assert g1_29dof_wbt_hybrid_climb.termination.terms["bad_tracking"].func.endswith(":HybridBadTracking")
    assert g1_29dof_wbt_hybrid_climb.termination.terms["bad_tracking"].params[
        "gen_bad_ref_pos_xyz_threshold"
    ] == 100.0
