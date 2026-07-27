from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt,
    g1_29dof_wbt_hybrid_climb,
    g1_29dof_wbt_hybrid_object,
    g1_29dof_wbt_w_object,
)
from holosoma.envs.wbt.config_utils import apply_wbt_robot_defaults
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager, _inject_fixed_scene_objects_from_motion


def test_apply_wbt_robot_defaults_reapplies_shared_robot_settings():
    updated_cfg = apply_wbt_robot_defaults(g1_29dof_wbt_w_object)

    assert updated_cfg.robot.asset.enable_self_collisions is True
    assert updated_cfg.robot.control.action_scales_by_effort_limit_over_p_gain is True
    assert updated_cfg.robot.init_state.pos == [0.0, 0.0, 0.76]
    assert updated_cfg.command == g1_29dof_wbt_w_object.command
    assert updated_cfg.reward == g1_29dof_wbt_w_object.reward
    assert updated_cfg.termination == g1_29dof_wbt_w_object.termination


def test_video_overlay_text_includes_hybrid_mode():
    motion_command = SimpleNamespace(
        motion_ids=torch.tensor([2], dtype=torch.long),
        time_steps=torch.tensor([5], dtype=torch.long),
        motion=SimpleNamespace(
            motion_start_idx=torch.tensor([0, 0, 3], dtype=torch.long),
            motion_end_idx=torch.tensor([2, 2, 8], dtype=torch.long),
        ),
        ref_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        robot_ref_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        track_env_mask=torch.tensor([False]),
        gen_env_mask=torch.tensor([True]),
    )
    env = SimpleNamespace(
        num_envs=1,
        command_manager=SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None),
        curriculum_manager=SimpleNamespace(get_term=lambda name: None),
    )

    text = WholeBodyTrackingManager.get_video_overlay_text(env, env_id=0)

    assert "mode=gen" in text


def test_fill_extras_adds_only_useful_hybrid_episode_metrics():
    env = object.__new__(WholeBodyTrackingManager)
    env.device = "cpu"
    env.time_out_buf = torch.tensor([False, True, False], dtype=torch.bool)
    env.extras = {}
    env._pending_termination_episode_metrics = {
        "termination/bad_tracking": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        "hybrid/termination/track/bad_tracking": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "hybrid/termination/gen/low_root_height": torch.tensor([1.0], dtype=torch.float32),
        "hybrid/termination/gen/ref_pos_z": torch.tensor([1.0], dtype=torch.float32),
        "hybrid/termination/gen/object_pos_z": torch.tensor([0.0], dtype=torch.float32),
    }
    env.reward_manager = SimpleNamespace(
        reset=lambda env_ids: {
            "episode": {
                "rew_motion_global_ref_position_error_exp": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
                "rew_goal_ref_reached_once": torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32),
                "rew_action_rate_l2": torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32),
            },
            "episode_all": {},
            "raw_episode": {
                "raw_rew_motion_global_ref_position_error_exp": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
                "raw_rew_goal_ref_reached_once": torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32),
            },
            "raw_episode_all": {},
        }
    )
    motion_command = SimpleNamespace(
        track_env_mask=torch.tensor([True, False, True], dtype=torch.bool),
        gen_env_mask=torch.tensor([False, True, False], dtype=torch.bool),
    )
    env.command_manager = SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None)
    env.episode_length_buf = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    env.max_episode_length_s = 20.0

    WholeBodyTrackingManager._fill_extras(env, torch.tensor([0, 1, 2], dtype=torch.long))

    assert torch.equal(
        env.extras["episode"]["hybrid/reward/track/rew_motion_global_ref_position_error_exp"],
        torch.tensor([0.1, 0.3], dtype=torch.float32),
    )
    assert torch.equal(
        env.extras["episode"]["hybrid/reward/gen/rew_goal_ref_reached_once"],
        torch.tensor([2.0], dtype=torch.float32),
    )
    assert "hybrid/reward/track/rew_goal_ref_reached_once" not in env.extras["episode"]
    assert "hybrid/reward/gen/rew_motion_global_ref_position_error_exp" not in env.extras["episode"]
    assert "hybrid/reward/track/rew_action_rate_l2" not in env.extras["episode"]
    assert not any(key.startswith("hybrid/") for key in env.extras["raw_episode"])
    assert torch.equal(
        env.extras["episode"]["hybrid/termination/track/bad_tracking"],
        torch.tensor([1.0, 1.0], dtype=torch.float32),
    )
    assert torch.equal(
        env.extras["episode"]["hybrid/termination/gen/low_root_height"],
        torch.tensor([1.0], dtype=torch.float32),
    )
    assert torch.equal(
        env.extras["episode"]["hybrid/termination/gen/ref_pos_z"],
        torch.tensor([1.0], dtype=torch.float32),
    )
    assert torch.equal(
        env.extras["episode"]["hybrid/termination/gen/object_pos_z"],
        torch.tensor([0.0], dtype=torch.float32),
    )
    assert "hybrid/termination/gen/object_unmoved" not in env.extras["episode"]
    assert torch.equal(
        env.extras["episode"]["hybrid/mean_reward/track"],
        torch.tensor([82.0, 126.0], dtype=torch.float32),
    )
    assert torch.equal(
        env.extras["episode"]["hybrid/mean_reward/gen"],
        torch.tensor([144.0], dtype=torch.float32),
    )
    assert torch.equal(
        env.extras["episode"]["hybrid/success/gen"],
        torch.tensor([1.0], dtype=torch.float32),
    )


def test_fill_extras_skips_irrelevant_or_empty_hybrid_reward_logs():
    env = object.__new__(WholeBodyTrackingManager)
    env.device = "cpu"
    env.time_out_buf = torch.tensor([False, False], dtype=torch.bool)
    env.extras = {}
    env._pending_termination_episode_metrics = {}
    env.reward_manager = SimpleNamespace(
        reset=lambda env_ids: {
            "episode": {
                "rew_action_rate_l2": torch.tensor([0.4, 0.6], dtype=torch.float32),
                "rew_goal_ref_reached_once": torch.tensor([1.0, 2.0], dtype=torch.float32),
            },
            "episode_all": {},
            "raw_episode": {},
            "raw_episode_all": {},
        }
    )
    motion_command = SimpleNamespace(
        track_env_mask=torch.tensor([True, True], dtype=torch.bool),
        gen_env_mask=torch.tensor([False, False], dtype=torch.bool),
    )
    env.command_manager = SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None)
    env.max_episode_length_s = 20.0

    WholeBodyTrackingManager._fill_extras(env, torch.tensor([0, 1], dtype=torch.long))

    assert not any(key.startswith("hybrid/reward/") for key in env.extras["episode"])


def test_record_termination_metrics_adds_hybrid_mode_specific_fractions():
    env = object.__new__(WholeBodyTrackingManager)
    env.device = "cpu"
    env.log_dict = {}
    env._pending_termination_episode_metrics = {}
    env.episode_length_buf = torch.tensor([12.0, 8.0, 3.0], dtype=torch.float32)
    motion_command = SimpleNamespace(
        track_env_mask=torch.tensor([True, False, True], dtype=torch.bool),
        gen_env_mask=torch.tensor([False, True, False], dtype=torch.bool),
    )
    env.command_manager = SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None)
    env.termination_manager = SimpleNamespace(
        last_term_results={
            "bad_tracking": torch.tensor([True, True, False], dtype=torch.bool),
        },
        last_term_breakdowns={
            "bad_tracking": {
                "track_ref_pos_z": torch.tensor([True, False, False], dtype=torch.bool),
                "gen_low_root_height": torch.tensor([False, True, False], dtype=torch.bool),
                "gen_ref_pos_z": torch.tensor([False, True, False], dtype=torch.bool),
                "gen_object_pos_z": torch.tensor([False, True, False], dtype=torch.bool),
            }
        },
    )

    WholeBodyTrackingManager._record_termination_metrics(
        env,
        reset_flags=torch.tensor([True, True, False], dtype=torch.bool),
        timeout_flags=torch.tensor([False, False, False], dtype=torch.bool),
    )

    assert env.log_dict["hybrid/termination/track/reset_frac"].item() == 0.5
    assert env.log_dict["hybrid/termination/gen/reset_frac"].item() == 1.0
    assert env.log_dict["hybrid/termination/track/bad_tracking_frac"].item() == 0.5
    assert env.log_dict["hybrid/termination/gen/low_root_height_frac"].item() == 1.0
    assert env.log_dict["hybrid/termination/gen/ref_pos_z_frac"].item() == 1.0
    assert env.log_dict["hybrid/termination/gen/object_pos_z_frac"].item() == 1.0
    assert "hybrid/termination/gen/object_unmoved_frac" not in env.log_dict
    assert "hybrid/termination/gen/bad_tracking_frac" not in env.log_dict
    assert "hybrid/termination/track/bad_tracking/ref_pos_z_frac" not in env.log_dict
    assert torch.equal(
        env._pending_termination_episode_metrics["hybrid/termination/track/bad_tracking"],
        torch.tensor([1.0], dtype=torch.float32),
    )
    assert torch.equal(
        env._pending_termination_episode_metrics["hybrid/termination/gen/low_root_height"],
        torch.tensor([1.0], dtype=torch.float32),
    )
    assert torch.equal(
        env._pending_termination_episode_metrics["hybrid/termination/gen/ref_pos_z"],
        torch.tensor([1.0], dtype=torch.float32),
    )
    assert torch.equal(
        env._pending_termination_episode_metrics["hybrid/termination/gen/object_pos_z"],
        torch.tensor([1.0], dtype=torch.float32),
    )
    assert "hybrid/termination/gen/object_unmoved" not in env._pending_termination_episode_metrics
    assert torch.equal(
        env._pending_termination_episode_metrics["hybrid/episode_length/track"],
        torch.tensor([12.0], dtype=torch.float32),
    )
    assert torch.equal(
        env._pending_termination_episode_metrics["hybrid/episode_length/gen"],
        torch.tensor([8.0], dtype=torch.float32),
    )


def test_inject_fixed_scene_objects_from_motion_adds_climb_box(tmp_path):
    motion_path = tmp_path / "motion_with_box.npz"
    np.savez(
        motion_path,
        fixed_scene_object_names=np.array(["climb_box"], dtype="<U32"),
        fixed_scene_object_geom_types=np.array(["box"], dtype="<U16"),
        fixed_scene_object_pos_w=np.array([[2.7, 0.0, 0.2]], dtype=np.float32),
        fixed_scene_object_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        fixed_scene_object_size=np.array([[0.8, 2.0, 0.4]], dtype=np.float32),
    )

    base_motion_cfg = g1_29dof_wbt.command.setup_terms["motion_command"].params["motion_config"]
    motion_cfg = replace(base_motion_cfg, motion_file=str(motion_path), load_fixed_scene_from_motion=True)
    motion_term_cfg = replace(
        g1_29dof_wbt.command.setup_terms["motion_command"],
        params={"motion_config": motion_cfg},
    )
    command_cfg = replace(g1_29dof_wbt.command, setup_terms={"motion_command": motion_term_cfg})
    experiment_cfg = replace(g1_29dof_wbt, command=command_cfg)

    updated_cfg = _inject_fixed_scene_objects_from_motion(experiment_cfg)

    assert experiment_cfg.simulator.config.scene.rigid_objects is None
    assert updated_cfg.simulator.config.scene.rigid_objects is not None
    assert len(updated_cfg.simulator.config.scene.rigid_objects) == 1

    climb_box = updated_cfg.simulator.config.scene.rigid_objects[0]
    assert climb_box.name == "climb_box"
    assert climb_box.geom_type == "box"
    assert np.allclose(climb_box.size, [0.8, 2.0, 0.4])
    assert np.allclose(climb_box.position, [2.7, 0.0, 0.2])
    assert np.allclose(climb_box.orientation, [1.0, 0.0, 0.0, 0.0])
    assert climb_box.physics is not None
    assert climb_box.physics.kinematic_enabled is True
    assert climb_box.physics.isaacsim is not None
    assert climb_box.physics.isaacsim.static_friction == experiment_cfg.terrain.terrain_term.static_friction
    assert climb_box.physics.isaacsim.dynamic_friction == experiment_cfg.terrain.terrain_term.dynamic_friction
    assert climb_box.physics.isaacsim.restitution == experiment_cfg.terrain.terrain_term.restitution


def test_inject_fixed_scene_objects_from_motion_adds_tracked_object_usd_and_clears_legacy_urdf(tmp_path):
    motion_path = tmp_path / "motion_with_usd_object.npz"
    object_usd_path = tmp_path / "alcohol.usd"
    np.savez(
        motion_path,
        object_pos_w=np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.3]], dtype=np.float32),
        object_quat_w=np.array([[1.0, 0.0, 0.0, 0.0], [0.99, 0.01, 0.0, 0.0]], dtype=np.float32),
        object_lin_vel_w=np.zeros((2, 3), dtype=np.float32),
        object_ang_vel_w=np.zeros((2, 3), dtype=np.float32),
        object_geom_type=np.array("mesh"),
        object_usd_path=np.array(str(object_usd_path)),
    )

    base_motion_cfg = g1_29dof_wbt_w_object.command.setup_terms["motion_command"].params["motion_config"]
    motion_cfg = replace(base_motion_cfg, motion_file=str(motion_path), motion_dir="")
    motion_term_cfg = replace(
        g1_29dof_wbt_w_object.command.setup_terms["motion_command"],
        params={"motion_config": motion_cfg},
    )
    command_cfg = replace(g1_29dof_wbt_w_object.command, setup_terms={"motion_command": motion_term_cfg})
    experiment_cfg = replace(g1_29dof_wbt_w_object, command=command_cfg)

    updated_cfg = _inject_fixed_scene_objects_from_motion(experiment_cfg)

    assert updated_cfg.robot.object.object_urdf_path is None
    assert updated_cfg.simulator.config.scene.rigid_objects is not None
    assert len(updated_cfg.simulator.config.scene.rigid_objects) == 1
    tracked_object = updated_cfg.simulator.config.scene.rigid_objects[0]
    assert tracked_object.name == "object"
    assert tracked_object.usd_path == str(object_usd_path)
    assert tracked_object.urdf_path is None
    assert tracked_object.geom_type is None
    assert np.allclose(tracked_object.position, [0.1, 0.2, 0.3])
    assert np.allclose(tracked_object.orientation, [1.0, 0.0, 0.0, 0.0])
    assert tracked_object.physics is not None
    assert tracked_object.physics.kinematic_enabled is False


def test_inject_fixed_scene_objects_from_hybrid_climb_motion_adds_climb_box():
    updated_cfg = _inject_fixed_scene_objects_from_motion(g1_29dof_wbt_hybrid_climb)

    assert g1_29dof_wbt_hybrid_climb.simulator.config.scene.rigid_objects is None
    assert updated_cfg.simulator.config.scene.rigid_objects is not None
    assert len(updated_cfg.simulator.config.scene.rigid_objects) == 1

    climb_box = updated_cfg.simulator.config.scene.rigid_objects[0]
    assert climb_box.name == "climb_box"
    assert climb_box.geom_type == "box"
    assert np.allclose(climb_box.size, [0.8, 2.0, 0.5])
    assert np.allclose(climb_box.position, [2.7, 0.0, 0.25])
    assert np.allclose(climb_box.orientation, [1.0, 0.0, 0.0, 0.0])


def test_inject_fixed_scene_objects_from_motion_dir_adds_shared_climb_box(tmp_path):
    motion_dir = tmp_path / "climb_dir"
    motion_dir.mkdir()
    fixed_scene_kwargs = dict(
        fixed_scene_object_names=np.array(["climb_box"], dtype="<U32"),
        fixed_scene_object_geom_types=np.array(["box"], dtype="<U16"),
        fixed_scene_object_pos_w=np.array([[2.7, 0.0, 0.25]], dtype=np.float32),
        fixed_scene_object_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        fixed_scene_object_size=np.array([[0.8, 2.0, 0.5]], dtype=np.float32),
    )
    np.savez(motion_dir / "clip_a.npz", **fixed_scene_kwargs)
    np.savez(motion_dir / "clip_b.npz", **fixed_scene_kwargs)

    base_motion_cfg = g1_29dof_wbt_hybrid_climb.command.setup_terms["motion_command"].params["motion_config"]
    motion_cfg = replace(base_motion_cfg, motion_file="", motion_dir=str(motion_dir), load_fixed_scene_from_motion=True)
    motion_term_cfg = replace(
        g1_29dof_wbt_hybrid_climb.command.setup_terms["motion_command"],
        params={"motion_config": motion_cfg},
    )
    command_cfg = replace(g1_29dof_wbt_hybrid_climb.command, setup_terms={"motion_command": motion_term_cfg})
    experiment_cfg = replace(g1_29dof_wbt_hybrid_climb, command=command_cfg)

    updated_cfg = _inject_fixed_scene_objects_from_motion(experiment_cfg)

    assert updated_cfg.simulator.config.scene.rigid_objects is not None
    assert len(updated_cfg.simulator.config.scene.rigid_objects) == 1
    climb_box = updated_cfg.simulator.config.scene.rigid_objects[0]
    assert climb_box.name == "climb_box"
    assert np.allclose(climb_box.position, [2.7, 0.0, 0.25])
    assert np.allclose(climb_box.size, [0.8, 2.0, 0.5])
    assert climb_box.physics is not None
    assert climb_box.physics.kinematic_enabled is True
    assert climb_box.physics.isaacsim is not None


def test_inject_fixed_scene_objects_from_serialized_climb_checkpoint_config(tmp_path):
    motion_dir = tmp_path / "climb_dir"
    motion_dir.mkdir()
    np.savez(
        motion_dir / "clip.npz",
        fixed_scene_object_names=np.array(["climb_box"], dtype="<U32"),
        fixed_scene_object_geom_types=np.array(["box"], dtype="<U16"),
        fixed_scene_object_pos_w=np.array([[2.7, 0.0, 0.25]], dtype=np.float32),
        fixed_scene_object_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        fixed_scene_object_size=np.array([[0.8, 2.0, 0.5]], dtype=np.float32),
    )

    checkpoint_cfg = ExperimentConfig(**g1_29dof_wbt_hybrid_climb.to_serializable_dict())
    motion_cfg = checkpoint_cfg.command.setup_terms["motion_command"].params["motion_config"]
    assert isinstance(motion_cfg, dict)
    motion_cfg["motion_file"] = ""
    motion_cfg["motion_dir"] = str(motion_dir)

    updated_cfg = _inject_fixed_scene_objects_from_motion(checkpoint_cfg)

    assert updated_cfg.simulator.config.scene.rigid_objects is not None
    assert len(updated_cfg.simulator.config.scene.rigid_objects) == 1
    climb_box = updated_cfg.simulator.config.scene.rigid_objects[0]
    assert climb_box.name == "climb_box"
    assert np.allclose(climb_box.position, [2.7, 0.0, 0.25])
    assert np.allclose(climb_box.size, [0.8, 2.0, 0.5])
    assert climb_box.physics is not None
    assert climb_box.physics.kinematic_enabled is True
    assert climb_box.physics.isaacsim is not None


def test_serialized_object_checkpoint_keeps_legacy_dynamic_object_urdf(tmp_path):
    motion_dir = tmp_path / "movebox_dir"
    motion_dir.mkdir()
    np.savez(
        motion_dir / "clip.npz",
        object_pos_w=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        object_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        object_lin_vel_w=np.zeros((1, 3), dtype=np.float32),
        object_ang_vel_w=np.zeros((1, 3), dtype=np.float32),
    )

    checkpoint_cfg = ExperimentConfig(**g1_29dof_wbt_hybrid_object.to_serializable_dict())
    motion_cfg = checkpoint_cfg.command.setup_terms["motion_command"].params["motion_config"]
    assert isinstance(motion_cfg, dict)
    motion_cfg["motion_file"] = ""
    motion_cfg["motion_dir"] = str(motion_dir)

    updated_cfg = _inject_fixed_scene_objects_from_motion(checkpoint_cfg)

    # Legacy move-box motions carry trajectories but no asset metadata. Isaac Sim
    # must therefore keep spawning the dynamic object from robot.object.
    assert updated_cfg.simulator.config.scene.rigid_objects is None
    assert (
        updated_cfg.robot.object.object_urdf_path
        == "holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
    )
