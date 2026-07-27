from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import torch
from loguru import logger

from holosoma.config_types.command import HybridSizeRandomizationConfig
from holosoma.config_types.simulator import IsaacSimPhysicsConfig, PhysicsConfig, RigidObjectConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.envs.wbt.config_utils import apply_wbt_robot_defaults
from holosoma.envs.wbt.hybrid_size_randomization import (
    build_fixed_hybrid_track_mask,
    expand_size_scalars,
    load_urdf_collision_bounds,
    sample_fixed_hybrid_size_scales,
)
from holosoma.utils.fixed_scene_motion import FixedSceneObjectSpec, load_fixed_scene_objects_from_motion_source
from holosoma.utils.rotations import quat_error_magnitude

# from holosoma.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from holosoma.utils.simulator_config import SimulatorType

HYBRID_TRACK_REWARD_TERMS = {
    "motion_global_ref_position_error_exp",
    "motion_global_ref_orientation_error_exp",
    "motion_relative_body_position_error_exp",
    "motion_relative_body_orientation_error_exp",
    "motion_global_body_lin_vel",
    "motion_global_body_ang_vel",
    "object_global_ref_position_error_exp",
    "object_global_ref_orientation_error_exp",
}

HYBRID_GEN_REWARD_TERMS = {
    "goal_ref_reached_once",
    "goal_object_reached_once",
}


def _extract_fixed_scene_loading_request(tyro_config) -> dict[str, str] | None:
    command_cfg = getattr(tyro_config, "command", None)
    if command_cfg is None:
        return None

    setup_term = command_cfg.setup_terms.get("motion_command")
    if setup_term is None:
        return None

    motion_cfg = setup_term.params.get("motion_config")
    if motion_cfg is None:
        return None
    if isinstance(motion_cfg, Mapping):
        if not bool(motion_cfg.get("load_fixed_scene_from_motion", False)):
            return None
        return {
            "motion_file": str(motion_cfg.get("motion_file", "") or ""),
            "motion_dir": str(motion_cfg.get("motion_dir", "") or ""),
        }
    if not bool(getattr(motion_cfg, "load_fixed_scene_from_motion", False)):
        return None
    return {
        "motion_file": str(motion_cfg.motion_file or ""),
        "motion_dir": str(motion_cfg.motion_dir or ""),
    }


def _inject_fixed_scene_objects_from_motion(tyro_config):
    loading_request = _extract_fixed_scene_loading_request(tyro_config)
    if loading_request is None:
        return tyro_config

    motion_file = loading_request["motion_file"]
    motion_dir = loading_request["motion_dir"]
    if not motion_dir and not motion_file:
        raise ValueError("load_fixed_scene_from_motion requires motion_file or motion_dir to be set.")

    fixed_scene_specs = load_fixed_scene_objects_from_motion_source(
        motion_file=motion_file,
        motion_dir=motion_dir,
        recursive=False,
    )
    if not fixed_scene_specs:
        return tyro_config

    terrain_term = tyro_config.terrain.terrain_term

    def object_physics(spec: FixedSceneObjectSpec) -> PhysicsConfig:
        damping = 0.01 if spec.is_motion_object else 0.0
        return PhysicsConfig(
            kinematic_enabled=not spec.is_motion_object,
            linear_damping=damping,
            angular_damping=damping,
            isaacsim=IsaacSimPhysicsConfig(
                static_friction=terrain_term.static_friction,
                dynamic_friction=terrain_term.dynamic_friction,
                restitution=terrain_term.restitution,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
        )

    scene_cfg = tyro_config.simulator.config.scene
    existing_objects = list(scene_cfg.rigid_objects or [])
    existing_names = {obj.name for obj in existing_objects}
    injected_objects: list[RigidObjectConfig] = []
    for spec in fixed_scene_specs:
        if spec.name in existing_names:
            raise ValueError(f"Fixed scene object '{spec.name}' collides with an existing scene rigid object.")
        common_kwargs = {
            "name": spec.name,
            "position": spec.position_w.tolist(),
            "orientation": spec.quat_wxyz.tolist(),
            "physics": object_physics(spec),
        }
        if spec.geom_type == "box":
            if spec.size is None:
                raise ValueError(f"Motion object '{spec.name}' is missing box size metadata.")
            injected_objects.append(
                RigidObjectConfig(
                    **common_kwargs,
                    geom_type="box",
                    size=spec.size.tolist(),
                )
            )
        elif spec.geom_type == "usd":
            if not spec.usd_path:
                raise ValueError(f"Motion object '{spec.name}' is missing usd_path metadata.")
            injected_objects.append(
                RigidObjectConfig(
                    **common_kwargs,
                    usd_path=spec.usd_path,
                )
            )
        elif spec.geom_type == "urdf":
            if not spec.urdf_path:
                raise ValueError(f"Motion object '{spec.name}' is missing urdf_path metadata.")
            injected_objects.append(
                RigidObjectConfig(
                    **common_kwargs,
                    urdf_path=spec.urdf_path,
                )
            )
        else:
            raise ValueError(f"Unsupported motion scene geom_type '{spec.geom_type}'.")

    updated_scene = replace(scene_cfg, rigid_objects=existing_objects + injected_objects)
    updated_simulator_init = replace(tyro_config.simulator.config, scene=updated_scene)
    updated_simulator = replace(tyro_config.simulator, config=updated_simulator_init)
    updated_config = replace(tyro_config, simulator=updated_simulator)

    # A motion-provided tracked object named "object" supersedes the legacy robot.object URDF path.
    if any(spec.is_motion_object and spec.name == "object" for spec in fixed_scene_specs):
        robot_object_cfg = getattr(updated_config.robot, "object", None)
        if robot_object_cfg is not None and getattr(robot_object_cfg, "object_urdf_path", None):
            updated_robot = replace(
                updated_config.robot,
                object=replace(robot_object_cfg, object_urdf_path=None),
            )
            updated_config = replace(updated_config, robot=updated_robot)

    return updated_config


def _extract_hybrid_size_randomization(tyro_config) -> tuple[object, HybridSizeRandomizationConfig] | None:
    command_cfg = getattr(tyro_config, "command", None)
    if command_cfg is None:
        return None
    setup_term = command_cfg.setup_terms.get("motion_command")
    if setup_term is None:
        return None
    raw_motion_cfg = setup_term.params.get("motion_config")
    if raw_motion_cfg is None:
        return None

    if hasattr(raw_motion_cfg, "hybrid"):
        raw_hybrid_cfg = raw_motion_cfg.hybrid
    else:
        raw_hybrid_cfg = dict(raw_motion_cfg).get("hybrid")
    if raw_hybrid_cfg is None:
        return None

    if hasattr(raw_hybrid_cfg, "size_randomization"):
        raw_size_cfg = raw_hybrid_cfg.size_randomization
    else:
        raw_size_cfg = dict(raw_hybrid_cfg).get("size_randomization")
    if raw_size_cfg is None:
        size_cfg = HybridSizeRandomizationConfig()
    elif isinstance(raw_size_cfg, HybridSizeRandomizationConfig):
        size_cfg = raw_size_cfg
    else:
        size_cfg = HybridSizeRandomizationConfig(**dict(raw_size_cfg))
    return raw_hybrid_cfg, size_cfg


def _prepare_hybrid_size_randomization(tyro_config):
    """Author fixed per-environment scaling data before the simulator is created."""
    extracted = _extract_hybrid_size_randomization(tyro_config)
    if extracted is None:
        return tyro_config
    raw_hybrid_cfg, size_cfg = extracted
    if not size_cfg.enabled:
        return tyro_config

    track_ratio = float(
        raw_hybrid_cfg.track_ratio
        if hasattr(raw_hybrid_cfg, "track_ratio")
        else dict(raw_hybrid_cfg).get("track_ratio", 0.5)
    )
    partition_seed = int(
        raw_hybrid_cfg.env_partition_seed
        if hasattr(raw_hybrid_cfg, "env_partition_seed")
        else dict(raw_hybrid_cfg).get("env_partition_seed", 0)
    )
    num_envs = int(tyro_config.training.num_envs)
    track_mask = build_fixed_hybrid_track_mask(num_envs, track_ratio, partition_seed)
    size_scalars = sample_fixed_hybrid_size_scales(track_mask, size_cfg)
    scales_xyz = expand_size_scalars(size_scalars, size_cfg.mode)

    scene_cfg = tyro_config.simulator.config.scene
    existing_objects = list(scene_cfg.rigid_objects or [])
    target_name = "object" if size_cfg.mode == "uniform_xyz" else "climb_box"
    target_indices = [index for index, obj in enumerate(existing_objects) if obj.name == target_name]
    if len(target_indices) > 1:
        raise ValueError(f"Hybrid size randomization found multiple scene actors named '{target_name}'.")
    target_index = target_indices[0] if target_indices else None
    updated_robot = tyro_config.robot

    if target_index is None and target_name == "object":
        robot_object_cfg = getattr(tyro_config.robot, "object", None)
        object_urdf_path = getattr(robot_object_cfg, "object_urdf_path", None)
        if not object_urdf_path:
            raise ValueError(
                "Hybrid object size randomization requires robot.object.object_urdf_path or a scene rigid object "
                "named 'object'."
            )
        collision_min, collision_max = load_urdf_collision_bounds(str(object_urdf_path))
        existing_objects.append(
            RigidObjectConfig(
                name="object",
                urdf_path=str(object_urdf_path),
                position=[0.0, 0.0, 0.5],
                orientation=[1.0, 0.0, 0.0, 0.0],
                physics=PhysicsConfig(
                    kinematic_enabled=False,
                    linear_damping=0.01,
                    angular_damping=0.01,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                ),
                per_env_scales=scales_xyz.tolist(),
                per_env_positions=[[0.0, 0.0, 0.5] for _ in range(num_envs)],
                local_collision_bounds=[collision_min, collision_max],
            )
        )
        target_index = len(existing_objects) - 1
        updated_robot = replace(
            tyro_config.robot,
            object=replace(robot_object_cfg, object_urdf_path=None),
        )

    if target_index is None:
        raise ValueError(f"Hybrid size randomization requires a scene rigid object named '{target_name}'.")

    target_cfg = existing_objects[target_index]
    if target_name == "object":
        if target_cfg.urdf_path is None:
            raise ValueError("Hybrid object size randomization currently requires an URDF-backed 'object'.")
        collision_min, collision_max = load_urdf_collision_bounds(target_cfg.urdf_path)
        per_env_positions = [[float(value) for value in target_cfg.position] for _ in range(num_envs)]
        updated_target = replace(
            target_cfg,
            per_env_scales=scales_xyz.tolist(),
            per_env_positions=per_env_positions,
            local_collision_bounds=[collision_min, collision_max],
        )
    else:
        if target_cfg.geom_type != "box" or target_cfg.size is None or len(target_cfg.size) != 3:
            raise ValueError("Hybrid climb size randomization requires 'climb_box' to be a primitive box.")
        nominal_height = float(target_cfg.size[2])
        per_env_positions = []
        for scalar in size_scalars.tolist():
            per_env_positions.append(
                [
                    float(target_cfg.position[0]),
                    float(target_cfg.position[1]),
                    float(size_cfg.ground_clearance) + 0.5 * nominal_height * float(scalar),
                ]
            )
        half_size = [0.5 * float(value) for value in target_cfg.size]
        updated_target = replace(
            target_cfg,
            per_env_scales=scales_xyz.tolist(),
            per_env_positions=per_env_positions,
            local_collision_bounds=[[-value for value in half_size], half_size],
        )
    existing_objects[target_index] = updated_target

    updated_scene = replace(scene_cfg, replicate_physics=False, rigid_objects=existing_objects)
    updated_simulator_init = replace(tyro_config.simulator.config, scene=updated_scene)
    updated_simulator = replace(tyro_config.simulator, config=updated_simulator_init)
    logger.info(
        "Enabled fixed hybrid size randomization for '{}' across {} environments; scale seed={}, "
        "track bounds={}, gen bounds={}, replicate_physics=False.",
        target_name,
        num_envs,
        int(size_cfg.sampling_seed),
        size_cfg.track_scale_range,
        size_cfg.gen_scale_range,
    )
    return replace(tyro_config, simulator=updated_simulator, robot=updated_robot)


class WholeBodyTrackingManager(BaseTask):
    def __init__(self, tyro_config, *, device):
        tyro_config = apply_wbt_robot_defaults(tyro_config)
        tyro_config = _inject_fixed_scene_objects_from_motion(tyro_config)
        tyro_config = _prepare_hybrid_size_randomization(tyro_config)
        super().__init__(tyro_config, device=device)
        assert not hasattr(self.simulator, "gym"), "WBT requires IsaacSim — IsaacGym is not supported."

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        super()._init_buffers()

        # -------------------------------- terms same with locomotion_manager.py [start]--------------------------------
        self.base_quat = self.simulator.base_quat
        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self._configure_default_dof_pos()
        self._init_domain_rand_buffers()

    def _configure_default_dof_pos(self):
        self.default_dof_pos_base = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if name not in self.robot_config.init_state.default_joint_angles:
                raise ValueError(f"Missing default joint angle for DOF '{name}' in robot configuration.")
            angle = self.robot_config.init_state.default_joint_angles[name]
            self.default_dof_pos_base[i] = angle

        self.default_dof_pos_base = self.default_dof_pos_base.unsqueeze(0)  # (1, num_dof)
        self.default_dof_pos = self.default_dof_pos_base.repeat(self.num_envs, 1).clone()  # (num_envs, num_dof)

    def _pre_compute_observations_callback(self):
        self.base_quat[:] = self.simulator.base_quat[:]

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        self.need_to_refresh_envs[env_ids] = True
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # pending_episode_update_mask is only used in curriculum_term::AverageEpisodeLengthTracker.
        self._pending_episode_update_mask[env_ids] = True

    def _get_envs_to_refresh(self):
        return self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()

    def _refresh_envs_after_reset(self, env_ids):
        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)
        self.simulator.clear_contact_forces_history(env_ids)
        self.need_to_refresh_envs[env_ids] = False
        self.simulator.refresh_sim_tensors()
        self._pre_compute_observations_callback()

    def _get_average_episode_tracker(self):
        tracker = self.curriculum_manager.get_term("average_episode_tracker")
        if tracker is None:
            raise RuntimeError("AverageEpisodeLengthTracker is not registered with the curriculum manager.")
        return tracker

    # -------------------------------- terms same with locomotion_manager.py [end]--------------------------------

    def _update_log_dict(self):
        # _update_log_dict happens before reset_envs_idx
        # -------------------------------- terms same with locomotion_manager.py [start]--------------------------------
        avg = self._get_average_episode_tracker().get_average()
        self.log_dict["average_episode_length"] = avg.detach().cpu()
        # -------------------------------- terms same with locomotion_manager.py [end]--------------------------------
        # Add tracking metrics to log_dict
        motion_command = self.command_manager.get_state("motion_command")
        motion_command.update_metrics()
        self.log_dict.update(motion_command.metrics)
        hybrid_gen_noise_curriculum = self.curriculum_manager.get_term("hybrid_gen_noise_curriculum")
        gen_noise_curriculum = hybrid_gen_noise_curriculum
        if gen_noise_curriculum is not None and getattr(gen_noise_curriculum, "enabled", False):
            if getattr(motion_command, "hybrid_cfg", None) is not None:
                scale = motion_command.object_goal_noise_curriculum_scale.detach().cpu()
                pos_scale = motion_command.object_goal_noise_pos_axis_scale
                rpy_scale = motion_command.object_goal_noise_rpy_axis_scale
                object_goal_noise_cfg = motion_command.hybrid_cfg.object_goal_noise
                pos_std = torch.tensor(object_goal_noise_cfg.pos_std_xyz, dtype=torch.float32, device=self.device)
                pos_clip = torch.tensor(object_goal_noise_cfg.pos_clip_xyz, dtype=torch.float32, device=self.device)
                rpy_std = torch.tensor(object_goal_noise_cfg.rpy_std, dtype=torch.float32, device=self.device)
                rpy_clip = torch.tensor(object_goal_noise_cfg.rpy_clip, dtype=torch.float32, device=self.device)

                self.log_dict["curriculum/object_goal_noise_success_ema"] = gen_noise_curriculum.success_ema.detach().cpu()
                self.log_dict["curriculum/object_goal_noise_collect_rollout"] = torch.tensor(
                    float(gen_noise_curriculum.collect_rollout),
                    dtype=torch.float32,
                )
                self.log_dict["curriculum/object_goal_noise_collected_gen_episodes"] = (
                    gen_noise_curriculum.accumulated_success_count.detach().cpu().to(dtype=torch.float32)
                )
                if getattr(gen_noise_curriculum, "has_success_ema", False):
                    self.log_dict["curriculum/object_goal_noise_latest_batch_success"] = (
                        gen_noise_curriculum.latest_batch_success.detach().cpu()
                    )
                self.log_dict["curriculum/object_goal_noise_scale"] = scale
                self.log_dict["curriculum/object_goal_noise_pos_std_xy_mean"] = (
                    (pos_std[:2] * pos_scale[:2]).mean().detach().cpu()
                )
                self.log_dict["curriculum/object_goal_noise_pos_clip_xy_mean"] = (
                    (pos_clip[:2] * pos_scale[:2]).mean().detach().cpu()
                )
                self.log_dict["curriculum/object_goal_noise_yaw_std"] = (
                    (rpy_std[2] * rpy_scale[2]).detach().cpu()
                )
                self.log_dict["curriculum/object_goal_noise_yaw_clip"] = (
                    (rpy_clip[2] * rpy_scale[2]).detach().cpu()
                )


    def _get_hybrid_mode_masks(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor] | None:
        motion_command = self.command_manager.get_state("motion_command")
        if motion_command is None:
            return None
        if not hasattr(motion_command, "track_env_mask") or not hasattr(motion_command, "gen_env_mask"):
            return None

        track_env_mask = motion_command.track_env_mask
        gen_env_mask = motion_command.gen_env_mask
        if env_ids is not None:
            env_ids = env_ids.to(device=track_env_mask.device, dtype=torch.long)
            track_env_mask = track_env_mask[env_ids]
            gen_env_mask = gen_env_mask[env_ids]
        return track_env_mask, gen_env_mask

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=values.device, dtype=torch.bool)
        if not bool(mask.any().item()):
            return torch.zeros((), dtype=torch.float32, device=values.device)
        return values[mask].to(dtype=torch.float32).mean()

    def _select_by_mask(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=values.device, dtype=torch.bool)
        return values[mask].detach().clone()

    def _termination_cause_mask(
        self,
        *,
        base_mask: torch.Tensor,
        cause_name: str,
        object_env_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if object_env_mask is not None and cause_name in {"object_pos", "object_ori", "object_pos_z"}:
            return base_mask & object_env_mask
        return base_mask

    def _hybrid_reward_term_name(self, metric_name: str) -> str | None:
        for prefix in ("rew_", "raw_rew_"):
            if metric_name.startswith(prefix):
                return metric_name[len(prefix) :]
        return None

    def _append_hybrid_reward_episode_metrics(
        self,
        destination: dict[str, torch.Tensor],
        reward_metrics: dict[str, torch.Tensor],
        track_mask: torch.Tensor,
        gen_mask: torch.Tensor,
    ) -> None:
        for metric_name, values in list(reward_metrics.items()):
            term_name = self._hybrid_reward_term_name(metric_name)
            if term_name in HYBRID_TRACK_REWARD_TERMS:
                track_values = self._select_by_mask(values, track_mask)
                if track_values.numel() > 0:
                    destination[f"hybrid/reward/track/{metric_name}"] = track_values
            elif term_name in HYBRID_GEN_REWARD_TERMS:
                gen_values = self._select_by_mask(values, gen_mask)
                if gen_values.numel() > 0:
                    destination[f"hybrid/reward/gen/{metric_name}"] = gen_values

    def _append_hybrid_episode_return_metrics(
        self,
        destination: dict[str, torch.Tensor],
        reward_metrics: dict[str, torch.Tensor],
        track_mask: torch.Tensor,
        gen_mask: torch.Tensor,
    ) -> None:
        if not reward_metrics:
            return
        values = [metric for metric in reward_metrics.values() if isinstance(metric, torch.Tensor)]
        if not values:
            return
        total_return = torch.zeros_like(values[0])
        for metric in values:
            total_return = total_return + metric
        total_return = total_return * float(self.max_episode_length_s)
        track_values = self._select_by_mask(total_return, track_mask)
        if track_values.numel() > 0:
            destination["hybrid/mean_reward/track"] = track_values
        gen_values = self._select_by_mask(total_return, gen_mask)
        if gen_values.numel() > 0:
            destination["hybrid/mean_reward/gen"] = gen_values

    def _append_hybrid_success_metrics(
        self,
        destination: dict[str, torch.Tensor],
        reward_metrics: dict[str, torch.Tensor],
        gen_mask: torch.Tensor,
    ) -> None:
        has_object_success = "rew_goal_object_reached_once" in reward_metrics
        has_robot_success = "rew_goal_ref_reached_once" in reward_metrics
        if not has_object_success and not has_robot_success:
            return

        success_key = "rew_goal_object_reached_once" if has_object_success else "rew_goal_ref_reached_once"
        success_values = reward_metrics[success_key] > 0.0
        success_gen = self._select_by_mask(success_values.to(dtype=torch.float32), gen_mask)
        if success_gen.numel() > 0:
            destination["hybrid/success/gen"] = success_gen

    def _get_hybrid_termination_term_name(self) -> str | None:
        for term_name, breakdown in self.termination_manager.last_term_breakdowns.items():
            if any(
                key in breakdown
                for key in (
                    "gen_low_root_height",
                    "gen_ref_pos_z",
                    "gen_object_pos_z",
                )
            ):
                return term_name
        if "bad_tracking" in self.termination_manager.last_term_results:
            return "bad_tracking"
        return None

    def reset_all(self):
        # If reset_all is called several times, clear buffer in motion_command
        motion_command = self.command_manager.get_state("motion_command")
        motion_command.init_buffers()
        return super().reset_all()

    def _fill_extras(self, env_ids):
        if self.reward_manager is None:
            return

        reward_extras = self.reward_manager.reset(env_ids)
        reward_episode_metrics = dict(reward_extras.get("episode", {}))
        raw_reward_episode_metrics = dict(reward_extras.get("raw_episode", {}))
        termination_metrics = dict(self._pending_termination_episode_metrics)

        self.extras["episode"] = dict(reward_episode_metrics)
        self.extras["episode_all"] = reward_extras.get("episode_all", {})
        self.extras["raw_episode"] = dict(raw_reward_episode_metrics)
        self.extras["raw_episode_all"] = reward_extras.get("raw_episode_all", {})
        self.extras["episode"].update(termination_metrics)

        hybrid_mode_masks = self._get_hybrid_mode_masks(env_ids)
        if hybrid_mode_masks is not None:
            track_reset_mask, gen_reset_mask = hybrid_mode_masks
            self._append_hybrid_reward_episode_metrics(
                self.extras["episode"], reward_episode_metrics, track_reset_mask, gen_reset_mask
            )
            self._append_hybrid_episode_return_metrics(
                self.extras["episode"], reward_episode_metrics, track_reset_mask, gen_reset_mask
            )
            self._append_hybrid_success_metrics(
                self.extras["episode"],
                reward_episode_metrics,
                gen_reset_mask,
            )
        self._pending_termination_episode_metrics = {}
        self.extras["time_outs"] = self.time_out_buf

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        # TODO(jchen): Now,reset robot/object states is implemented in command/terms/wbt.MotionCommand.reset
        # discuss whether to move to here in the future.
        pass

    def _record_termination_metrics(self, reset_flags: torch.Tensor, timeout_flags: torch.Tensor) -> None:
        super()._record_termination_metrics(reset_flags=reset_flags, timeout_flags=timeout_flags)

        hybrid_mode_masks = self._get_hybrid_mode_masks()
        if hybrid_mode_masks is not None:
            track_env_mask, gen_env_mask = hybrid_mode_masks
            reset_mask = (reset_flags | timeout_flags).to(dtype=torch.bool)

            self.log_dict["hybrid/termination/track/reset_frac"] = self._masked_mean(
                reset_mask, track_env_mask
            ).detach().cpu()
            self.log_dict["hybrid/termination/gen/reset_frac"] = self._masked_mean(
                reset_mask, gen_env_mask
            ).detach().cpu()
            hybrid_term_name = self._get_hybrid_termination_term_name()
            if hybrid_term_name is not None:
                hybrid_term_mask = self.termination_manager.last_term_results[hybrid_term_name]
                self.log_dict["hybrid/termination/track/bad_tracking_frac"] = self._masked_mean(
                    hybrid_term_mask, track_env_mask
                ).detach().cpu()

                low_root_height_mask = self.termination_manager.last_term_breakdowns.get(hybrid_term_name, {}).get(
                    "gen_low_root_height"
                )
                if low_root_height_mask is None:
                    low_root_height_mask = torch.zeros_like(gen_env_mask)
                self.log_dict["hybrid/termination/gen/low_root_height_frac"] = self._masked_mean(
                    low_root_height_mask, gen_env_mask
                ).detach().cpu()
                gen_ref_pos_z_mask = self.termination_manager.last_term_breakdowns.get(hybrid_term_name, {}).get(
                    "gen_ref_pos_z"
                )
                if gen_ref_pos_z_mask is not None:
                    self.log_dict["hybrid/termination/gen/ref_pos_z_frac"] = self._masked_mean(
                        gen_ref_pos_z_mask, gen_env_mask
                    ).detach().cpu()
                gen_object_pos_z_mask = self.termination_manager.last_term_breakdowns.get(hybrid_term_name, {}).get(
                    "gen_object_pos_z"
                )
                if gen_object_pos_z_mask is not None:
                    self.log_dict["hybrid/termination/gen/object_pos_z_frac"] = self._masked_mean(
                        gen_object_pos_z_mask, gen_env_mask
                    ).detach().cpu()

                reset_env_ids = reset_mask.nonzero(as_tuple=False).flatten()
                if reset_env_ids.numel() > 0:
                    track_episode_lengths = self._select_by_mask(
                        self.episode_length_buf[reset_env_ids].to(dtype=torch.float32),
                        track_env_mask[reset_env_ids],
                    )
                    if track_episode_lengths.numel() > 0:
                        self._pending_termination_episode_metrics["hybrid/episode_length/track"] = (
                            track_episode_lengths.detach().cpu().clone()
                        )
                    gen_episode_lengths = self._select_by_mask(
                        self.episode_length_buf[reset_env_ids].to(dtype=torch.float32),
                        gen_env_mask[reset_env_ids],
                    )
                    if gen_episode_lengths.numel() > 0:
                        self._pending_termination_episode_metrics["hybrid/episode_length/gen"] = (
                            gen_episode_lengths.detach().cpu().clone()
                        )
                    hybrid_track_episodes = self._select_by_mask(
                        hybrid_term_mask.to(dtype=torch.float32)[reset_env_ids], track_env_mask[reset_env_ids]
                    )
                    if hybrid_track_episodes.numel() > 0:
                        self._pending_termination_episode_metrics["hybrid/termination/track/bad_tracking"] = (
                            hybrid_track_episodes.detach().cpu().clone()
                        )

                    hybrid_gen_episodes = self._select_by_mask(
                        low_root_height_mask.to(dtype=torch.float32)[reset_env_ids], gen_env_mask[reset_env_ids]
                    )
                    if hybrid_gen_episodes.numel() > 0:
                        self._pending_termination_episode_metrics["hybrid/termination/gen/low_root_height"] = (
                            hybrid_gen_episodes.detach().cpu().clone()
                        )
                    if gen_ref_pos_z_mask is not None:
                        gen_ref_pos_z_episodes = self._select_by_mask(
                            gen_ref_pos_z_mask.to(dtype=torch.float32)[reset_env_ids],
                            gen_env_mask[reset_env_ids],
                        )
                        if gen_ref_pos_z_episodes.numel() > 0:
                            self._pending_termination_episode_metrics["hybrid/termination/gen/ref_pos_z"] = (
                                gen_ref_pos_z_episodes.detach().cpu().clone()
                            )
                    if gen_object_pos_z_mask is not None:
                        gen_object_pos_z_episodes = self._select_by_mask(
                            gen_object_pos_z_mask.to(dtype=torch.float32)[reset_env_ids],
                            gen_env_mask[reset_env_ids],
                        )
                        if gen_object_pos_z_episodes.numel() > 0:
                            self._pending_termination_episode_metrics["hybrid/termination/gen/object_pos_z"] = (
                                gen_object_pos_z_episodes.detach().cpu().clone()
                            )


    def _apply_force_in_physics_step(self):
        if self.action_manager is not None:
            self.action_manager.apply_actions()

    def get_video_overlay_text(self, env_id: int = 0) -> str:
        motion_command = self.command_manager.get_state("motion_command")
        env_id = max(0, min(int(env_id), self.num_envs - 1))

        motion_id = int(motion_command.motion_ids[env_id].item())
        time_step = int(motion_command.time_steps[env_id].item())
        motion_start = int(motion_command.motion.motion_start_idx[motion_id].item())
        motion_end = int(motion_command.motion.motion_end_idx[motion_id].item())
        phase = float(time_step - motion_start) / max(motion_end - motion_start - 1, 1)

        ref_pos_error = float(torch.norm(motion_command.ref_pos_w[env_id] - motion_command.robot_ref_pos_w[env_id]).item())
        ref_rot_error = float(
            quat_error_magnitude(
                motion_command.ref_quat_w[env_id].unsqueeze(0),
                motion_command.robot_ref_quat_w[env_id].unsqueeze(0),
            ).item()
        )

        labels = [
            f"WBT motion={motion_id}",
            f"t={time_step}",
            f"phase={phase:.2f}",
            f"e_pos={ref_pos_error:.2f}",
            f"e_rot={ref_rot_error:.2f}",
        ]
        if hasattr(motion_command, "track_env_mask") and hasattr(motion_command, "gen_env_mask"):
            mode = "track" if bool(motion_command.track_env_mask[env_id].item()) else "gen"
            labels.append(f"mode={mode}")
        return ", ".join(labels)

    def _visualize_world_frame_marker(
        self,
        motion_command,
        marker_name: str,
        pos_w: torch.Tensor,
        quat_xyzw: torch.Tensor,
    ) -> None:
        if marker_name not in motion_command.visualization_markers:
            return
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
        motion_command.visualization_markers[marker_name].visualize(pos_w, quat_wxyz)

    ########################################################### Push robots #########################################
    # TODO: This should be moved to the randomization manager.
    def _init_domain_rand_buffers(self):
        ######################################### DR related tensors #########################################
        # Action delay buffers are now initialized by randomization manager's setup_action_delay_buffers term

        self.push_robot_vel_buf = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.record_push_robot_vel_buf = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._randomize_push_robots = False
        self._max_push_vel = torch.zeros(6, dtype=torch.float32, device=self.device)

    def _push_robots(self, env_ids):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        max_vel_tensor = self._max_push_vel
        if self.randomization_manager is not None:
            state = self.randomization_manager.get_state("push_randomizer_state")
            if state is not None:
                max_vel_tensor = state.max_push_vel.clone().to(self.device)

        if not isinstance(max_vel_tensor, torch.Tensor) or max_vel_tensor.numel() != 6:
            raise ValueError("WholeBodyTracking push velocity vector must have exactly 6 components.")

        rand = torch.rand(len(env_ids), 6, device=self.device) * 2 - 1
        self.push_robot_vel_buf[env_ids] = rand * max_vel_tensor.unsqueeze(0)
        self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone()
        # Additive push to match BeyondMimic/IsaacLab's push_by_setting_velocity.
        self.simulator.robot_root_states[env_ids, 7:13] += self.push_robot_vel_buf[env_ids]
        # Push impulses only take effect in the simulator once we write the mutated root state tensor back.
        self.simulator.set_actor_root_state_tensor_robots(env_ids, self.simulator.robot_root_states)
        self._max_push_vel = max_vel_tensor.clone()

    #########################################################################################################
    ## Debug visualization
    #########################################################################################################

    def _draw_debug_vis_isaacsim(self):
        motion_command = self.command_manager.get_state("motion_command")
        if not hasattr(motion_command, "visualization_markers"):
            return
        track_env_mask = getattr(motion_command, "track_env_mask", None)
        if track_env_mask is None:
            track_env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        hidden_translation = torch.full((self.num_envs, 3), -1000.0, device=self.device, dtype=torch.float32)

        for body_idx, body_names in enumerate(motion_command.motion_cfg.body_names_to_track):
            motion_robot_body_pos_xyz = motion_command.body_pos_w[:, body_idx].clone()
            motion_robot_body_pos_xyz = torch.where(
                track_env_mask.unsqueeze(-1),
                motion_robot_body_pos_xyz,
                hidden_translation,
            )
            motion_command.visualization_markers[f"motion_{body_names}"].visualize(motion_robot_body_pos_xyz)

        self._visualize_world_frame_marker(
            motion_command,
            "goal_robot",
            motion_command.goal_ref_pos_w.clone(),
            motion_command.goal_ref_quat_w.clone(),
        )

        # object
        if motion_command.motion.has_object:
            self._visualize_world_frame_marker(
                motion_command,
                "real_object",
                motion_command.simulator_object_pos_w.clone(),
                motion_command.simulator_object_quat_w.clone(),
            )
            self._visualize_world_frame_marker(
                motion_command,
                "motion_object",
                torch.where(
                    track_env_mask.unsqueeze(-1),
                    motion_command.object_pos_w.clone(),
                    hidden_translation,
                ),
                motion_command.object_quat_w.clone(),
            )
            self._visualize_world_frame_marker(
                motion_command,
                "goal_object",
                motion_command.goal_object_pos_w.clone(),
                motion_command.goal_object_quat_w.clone(),
            )

    def _draw_debug_vis_isaacgym(self):
        self.simulator.clear_lines()
        n_bodies = len(self.motion_command.motion_cfg.body_names_to_track)
        track_env_mask = getattr(self.motion_command, "track_env_mask", None)
        if track_env_mask is None:
            track_env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for env_id in range(self.num_envs):
            for body_idx in range(n_bodies):
                if bool(track_env_mask[env_id].item()):
                    color = (0.0, 1.0, 0.0)
                    self.simulator.draw_sphere(
                        self.motion_command.body_pos_relative_w[env_id, body_idx], 0.03, color, env_id, body_idx
                    )

                color = (0.0, 0.0, 1.0)
                self.simulator.draw_sphere(
                    self.motion_command.robot_body_pos_w[env_id, body_idx], 0.03, color, env_id, n_bodies + body_idx
                )

            if bool(track_env_mask[env_id].item()):
                color = (0.0, 1.0, 0.0)
                self.simulator.draw_sphere(
                    self.motion_command.ref_pos_w[env_id], 0.05, color, env_id, n_bodies * 2 + 0
                )
            color = (0.0, 0.0, 1.0)
            self.simulator.draw_sphere(
                self.motion_command.robot_ref_pos_w[env_id], 0.05, color, env_id, n_bodies * 2 + 1
            )

    def _draw_debug_vis(self):
        if self.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._draw_debug_vis_isaacsim()
        elif self.simulator.get_simulator_type() == SimulatorType.ISAACGYM:
            self._draw_debug_vis_isaacgym()

    def step_visualize_motion(self, actions):
        motion_command = self.command_manager.get_state("motion_command")
        dt = 1.0 / float(motion_command.motion.fps)
        motion_command.step()
        print("time_steps: ", motion_command.time_steps[0].item())
        self._draw_debug_vis()

        # set root_states_from_motion_command
        root_pos = motion_command.root_pos_w.clone()
        root_ori = motion_command.root_quat_w.clone()  # wxyz
        root_lin_vel = motion_command.body_lin_vel_w[:, 0].clone()
        root_ang_vel = motion_command.body_ang_vel_w[:, 0].clone()

        joint_pos = motion_command.joint_pos.clone()
        joint_vel = motion_command.joint_vel.clone()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.simulator.dof_pos[env_ids] = joint_pos
        self.simulator.dof_vel[env_ids] = joint_vel

        self.simulator.robot_root_states[env_ids, :3] = root_pos
        self.simulator.robot_root_states[env_ids, 3:7] = root_ori
        self.simulator.robot_root_states[env_ids, 7:10] = root_lin_vel
        self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel

        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)

        if motion_command.motion.has_object:
            # set object root_states from motion command
            object_pos = motion_command.object_pos_w.clone()
            object_ori = motion_command.object_quat_w.clone()
            object_lin_vel = motion_command.object_lin_vel_w.clone()

            object_states = torch.zeros(len(env_ids), 13, device=self.device)
            object_states[:, :3] = object_pos[:]
            object_states[:, 3:7] = object_ori[:]
            object_states[:, 7:10] = object_lin_vel[:]
            object_states[:, 10:13] = torch.zeros_like(object_lin_vel[:])
            self.simulator.set_actor_states(["object"], env_ids, object_states)

        self.simulator.scene.write_data_to_sim()
        self.simulator.sim.forward()
        self.simulator.sim.render()
        self.simulator.refresh_sim_tensors()

        time.sleep(dt)

        return motion_command.time_steps[0].item() >= motion_command.motion.time_step_total - 2
