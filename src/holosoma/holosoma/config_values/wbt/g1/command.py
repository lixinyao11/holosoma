"""Whole Body Tracking command presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.command import (
    CommandManagerCfg,
    CommandTermCfg,
    GoalPoseNoiseConfig,
    HybridMotionConfig,
    HybridSizeRandomizationConfig,
    MotionConfig,
    NoiseToInitialPoseConfig,
    StepZeroRootNoiseConfig,
)

# init_pose_config = NoiseToInitialPoseConfig(
#     overall_noise_scale=1.0,
#     dof_pos=0.1,
#     root_pos=[0.05, 0.05, 0.01],
#     root_rot=[0.1, 0.1, 0.2],
#     root_lin_vel=[0.5, 0.5, 0.2],
#     root_ang_vel=[0.52, 0.52, 0.78],
#     object_pos=[0.05, 0.05, 0.0],
# )

init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.025, 0.025, 0.01],
    root_rot=[0.05, 0.05, 0.1],
    root_lin_vel=[0.25, 0.25, 0.1],
    root_ang_vel=[0.26, 0.26, 0.39],
    object_pos=[0.025, 0.025, 0.0],
)

motion_config = MotionConfig(
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz",
    body_names_to_track=[
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
    body_name_ref=["torso_link"],
    use_adaptive_timesteps_sampler=True,
    # adaptive_timestep_sampler inherits MotionConfig defaults unless an experiment overrides it explicitly
    start_at_timestep_zero_prob=0.1,
    enable_terminal_pose_hold=True,
    terminal_pose_hold_duration_s=1.0,
    noise_to_initial_pose=init_pose_config,
)

motion_config_w_object = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz",
    load_fixed_scene_from_motion=True,
)

motion_config_climb = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/climbox1/walk_climb1_reframed.npz",
    load_fixed_scene_from_motion=True,
)

_hybrid_motion_cfg = HybridMotionConfig(
    track_ratio=0.5,
    env_partition_seed=0,
    gen_start_at_timestep_zero_prob=0.2,
    robot_goal_noise=GoalPoseNoiseConfig(
        pos_std_xyz=[0.05, 0.05, 0.001],
        pos_clip_xyz=[0.10, 0.10, 0.001],
        rpy_std=[0.001, 0.001, 0.001],
        rpy_clip=[0.001, 0.001, 0.001],
    ),
    object_goal_noise=GoalPoseNoiseConfig(
        pos_std_xyz=[0.2, 0.2, 0.001],
        pos_clip_xyz=[0.3, 0.3, 0.001],
        rpy_std=[0.001, 0.001, 0.2],
        rpy_clip=[0.001, 0.001, 0.4],
    ),
)

init_pose_config_hybrid_climb = replace(
    init_pose_config,
    step_zero_root_noise=StepZeroRootNoiseConfig(
        pos_std_xyz=[0.05, 0.05, 0.01],
        pos_clip_xyz=[0.2, 0.2, 0.02],
        rpy_std=[0.05, 0.05, 0.1],
        rpy_clip=[0.1, 0.1, 0.3],
    ),
)

hybrid_climb_cfg = replace(
    _hybrid_motion_cfg,
    robot_goal_noise=GoalPoseNoiseConfig(
        pos_std_xyz=[0.05, 0.05, 0.001],
        pos_clip_xyz=[0.10, 0.10, 0.001],
        rpy_std=[0.001, 0.001, 0.03],
        rpy_clip=[0.001, 0.001, 0.10],
    ),
    size_randomization=HybridSizeRandomizationConfig(
        enabled=False,
        track_scale_range=[0.95, 1.05],
        gen_scale_range=[0.85, 1.15],
        mode="height_only",
        sampling_seed=0,
        ground_clearance=0.001,
        ground_contact_tolerance=0.02,
    ),
    gen_step_zero_root_noise=StepZeroRootNoiseConfig(
        pos_std_xyz=[0.2, 0.2, 0.01],
        pos_clip_xyz=[1.0, 1.0, 0.02],
        rpy_std=[0.05, 0.05, 0.2],
        rpy_clip=[0.15, 0.15, 0.6],
    ),
)

hybrid_object_cfg = replace(
    _hybrid_motion_cfg,
    size_randomization=HybridSizeRandomizationConfig(
        enabled=False,
        track_scale_range=[0.95, 1.05],
        gen_scale_range=[0.85, 1.15],
        mode="uniform_xyz",
        sampling_seed=0,
        ground_clearance=0.001,
        ground_contact_tolerance=0.02,
    ),
    gen_step_zero_root_noise=StepZeroRootNoiseConfig(
        pos_std_xyz=[0.01, 0.01, 0.003],
        pos_clip_xyz=[0.05, 0.05, 0.01],
        rpy_std=[0.01, 0.01, 0.03],
        rpy_clip=[0.05, 0.05, 0.1],
    ),
)

hybrid_climb_motion_config = replace(
    motion_config_climb,
    start_at_timestep_zero_prob=0.05,
    hybrid=hybrid_climb_cfg,
    noise_to_initial_pose=init_pose_config_hybrid_climb,
)

hybrid_object_motion_config = replace(
    motion_config_w_object,
    start_at_timestep_zero_prob=0.05,
    hybrid=hybrid_object_cfg,
)

g1_29dof_wbt_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config,
            },
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
)

g1_29dof_wbt_command_w_object = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object,
            },
        )
    },
)

g1_29dof_wbt_command_climb = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_climb,
            },
        )
    },
)

def _hybrid_command(motion_config: MotionConfig) -> CommandManagerCfg:
    return replace(
        g1_29dof_wbt_command,
        setup_terms={
            "motion_command": CommandTermCfg(
                func="holosoma.managers.command.terms.wbt:HybridMotionCommand",
                params={"motion_config": motion_config},
            ),
        },
        reset_terms={
            "motion_command": CommandTermCfg(
                func="holosoma.managers.command.terms.wbt:HybridMotionCommand",
            )
        },
        step_terms={
            "motion_command": CommandTermCfg(
                func="holosoma.managers.command.terms.wbt:HybridMotionCommand",
            )
        },
    )


g1_29dof_wbt_command_hybrid_climb = _hybrid_command(hybrid_climb_motion_config)

g1_29dof_wbt_command_hybrid_object = _hybrid_command(hybrid_object_motion_config)

# Several objects: entry i of the comma-separated motion_dir holds the clips of robot.object.object_urdf_paths[i].
hybrid_multi_object_motion_config = replace(
    hybrid_object_motion_config,
    motion_file="",
    motion_dir_object_groups=True,
)
g1_29dof_wbt_command_hybrid_multi_object = _hybrid_command(hybrid_multi_object_motion_config)

# Same, with the object reset exactly to its reference pose (no position noise).
g1_29dof_wbt_command_hybrid_multi_object_exact_object = _hybrid_command(
    replace(
        hybrid_multi_object_motion_config,
        noise_to_initial_pose=replace(hybrid_multi_object_motion_config.noise_to_initial_pose, object_pos=[0.0, 0.0, 0.0]),
    )
)

# Same, with HDMI's rewind: after a failure, 80 % of track envs restart on the same clip 25-125 steps (0.5-2.5 s) before
# it, so that reaching for and lifting the object is practised rather than starting with it already in the hands.
g1_29dof_wbt_command_hybrid_multi_object_exact_object_rewind = _hybrid_command(
    replace(
        hybrid_multi_object_motion_config,
        noise_to_initial_pose=replace(hybrid_multi_object_motion_config.noise_to_initial_pose, object_pos=[0.0, 0.0, 0.0]),
        rewind_prob=0.8,
        rewind_steps_range=(25, 125),
    )
)

__all__ = [
    "g1_29dof_wbt_command",
    "g1_29dof_wbt_command_climb",
    "g1_29dof_wbt_command_hybrid_climb",
    "g1_29dof_wbt_command_hybrid_multi_object",
    "g1_29dof_wbt_command_hybrid_multi_object_exact_object",
    "g1_29dof_wbt_command_hybrid_multi_object_exact_object_rewind",
    "g1_29dof_wbt_command_hybrid_object",
    "g1_29dof_wbt_command_w_object",
    "hybrid_climb_cfg",
    "hybrid_climb_motion_config",
    "hybrid_multi_object_motion_config",
    "hybrid_object_cfg",
    "hybrid_object_motion_config",
    "motion_config_climb",
]
