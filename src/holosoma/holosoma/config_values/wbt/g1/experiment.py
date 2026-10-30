from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    scene,
    simulator,
    termination,
    terrain,
)

g1_29dof_wbt = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=30000,
            num_learning_epochs=5,
            save_interval=4000,
            entropy_coef=0.005,
            init_noise_std=1.0,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            init_at_random_ep_len=True,
            empirical_normalization=True,
            use_symmetry=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(
            robot.g1_29dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_reward,
    nightly=NightlyConfig(
        iterations=8000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.10, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.4, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [0.85, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.60, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.60, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.37, "inf"],
        },
    ),
)

g1_29dof_wbt_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_fast_sac_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=400000,
            v_max=20.0,
            v_min=-20.0,
            gamma=0.99,  # For motion tracking, high gamma + high num_steps is better
            num_steps=1,
            num_updates=4,
            num_atoms=501,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(
            robot.g1_29dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.18, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.09, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.29, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.09, "inf"],
        },
    ),
)

g1_29dof_wbt_w_object = replace(
    g1_29dof_wbt,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(
            robot.g1_29dof_w_object.asset,
            enable_self_collisions=True,
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    scene=scene.g1_29dof_wbt_object_scene,
)

# Several objects at once: one asset and one directory of clips per object, environment e holds object e % N
# and tracks only that object's clips. Set the clips and URDFs on the CLI, in the same order:
#   --command.setup-terms.motion-command.params.motion-config.motion-dir=<dir_0>,<dir_1>,...
#   --scene.rigid-objects.object.urdf-files <urdf_0> <urdf_1> ...
g1_29dof_wbt_w_multi_object = replace(
    g1_29dof_wbt_w_object,
    command=command.g1_29dof_wbt_command_w_multi_object,
    observation=observation.g1_29dof_wbt_observation_w_multi_object,
    randomization=randomization.g1_29dof_wbt_randomization_w_multi_object,
    scene=scene.g1_29dof_wbt_multi_object_scene,
)

# Same, with an actor that sees the object, the reference object and the reference root.
g1_29dof_wbt_w_multi_object_state = replace(
    g1_29dof_wbt_w_multi_object,
    observation=observation.g1_29dof_wbt_observation_w_multi_object_state,
)

# Same as g1_29dof_wbt_w_multi_object, with the G1's rubber hands (robot g1_29dof_w_object_rubberhand).
g1_29dof_wbt_w_multi_object_rubberhand = replace(
    g1_29dof_wbt_w_multi_object,
    robot=replace(
        g1_29dof_wbt_w_multi_object.robot,
        asset=replace(g1_29dof_wbt_w_multi_object.robot.asset, urdf_file="g1/main_mesh_collision_rubberhand.urdf"),
    ),
)

# Rubber hands, and the object tracked and checked relative to the robot (as the bodies are) and reset exactly to its
# reference pose: the robot drifting while it holds the object no longer ends the episode.
g1_29dof_wbt_w_multi_object_rubberhand_relobj = replace(
    g1_29dof_wbt_w_multi_object_rubberhand,
    command=command.g1_29dof_wbt_command_w_multi_object_exact_object,
    reward=reward.g1_29dof_wbt_reward_w_object_rel,
    termination=termination.g1_29dof_wbt_termination_rel_object,
)

g1_29dof_wbt_fast_sac_w_object = replace(
    g1_29dof_wbt_fast_sac,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    scene=scene.g1_29dof_wbt_object_scene,
)

__all__ = [
    "g1_29dof_wbt",
    "g1_29dof_wbt_fast_sac",
    "g1_29dof_wbt_fast_sac_w_object",
    "g1_29dof_wbt_w_multi_object",
    "g1_29dof_wbt_w_multi_object_rubberhand",
    "g1_29dof_wbt_w_multi_object_rubberhand_relobj",
    "g1_29dof_wbt_w_multi_object_state",
    "g1_29dof_wbt_w_object",
]

"""
Example 1: Robot only:
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt

Example 2: Robot+Object:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object

Example 3: Robot+Terrain:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path="holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj" \
  --command.setup_terms.motion_command.params.motion_config.motion_file\
="holosoma/data/motions/g1_29dof/whole_body_tracking/motion_crawl_slope.npz" \
  --scene.env_spacing=0.0
"""
