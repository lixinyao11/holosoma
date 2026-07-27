"""Replay reference motion through the full sim pipeline in viser.

Use this to debug whether motion is correctly handled by the simulator.
For lightweight replay without IsaacSim, use replay_viser.py instead.

Based on the working replay.py + env.step_visualize_motion() logic, but
skips _draw_debug_vis (requires GUI) and adds viser rendering instead.

Usage:
    source scripts/source_isaacsim_setup.sh
    python src/holosoma/holosoma/replay_viser_sim.py \
        exp:g1-29dof-wbt \
        --training.headless=True \
        --training.num-envs 1
"""

from __future__ import annotations

import os
import sys
import time

import tyro

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.eval_utils import init_sim_imports
from holosoma.utils.helpers import get_class
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG


def replay_viser_sim(tyro_config: ExperimentConfig, port: int = 8012):
    from loguru import logger

    # Keep standalone replay output readable unless the user explicitly opts
    # into verbose debugging via LOGURU_LEVEL=DEBUG.
    logger.remove()
    logger.add(sys.stdout, level=os.environ.get("LOGURU_LEVEL", "INFO").upper(), colorize=True)

    simulation_app = init_sim_imports(tyro_config)

    import numpy as np
    import torch
    from pathlib import Path

    from holosoma.utils.common import seeding
    from holosoma.utils.fixed_scene_motion import load_fixed_scene_objects_from_motion_config
    from holosoma.utils.module_utils import get_holosoma_root
    from holosoma.utils.viser_helper import ViserHelper
    from holosoma.utils.viser_robot import RobotAssetViser
    from holosoma.utils.viser_scene import StaticBoxViser, StaticUrdfViser, add_ground_grid, setup_scene_lighting

    target_color = (0.18, 0.92, 0.38)
    target_torso_radius = 0.09
    target_torso_arrow_length = 0.24
    target_torso_arrow_head_length = 0.08
    target_torso_arrow_head_width = 0.06
    target_torso_arrow_line_width = 3.0

    seeding(42, torch_deterministic=False)

    env_target = tyro_config.env_class
    tyro_env_config = get_tyro_env_config(tyro_config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(env_target)(tyro_env_config, device=device)

    sim = env.simulator
    motion_command = env.command_manager.get_state("motion_command")
    motion_fps = float(motion_command.motion.fps)
    dt = 1.0 / motion_fps

    # --- Setup viser ---
    viser = ViserHelper(port=port)
    setup_scene_lighting(viser)
    add_ground_grid(viser)

    # Resolve URDF
    asset_cfg = env.robot_config.asset
    asset_root = asset_cfg.asset_root
    if asset_root.startswith("@holosoma/"):
        asset_root = asset_root.replace("@holosoma", get_holosoma_root())
    urdf_path = str(Path(asset_root) / asset_cfg.urdf_file)
    body_names = list(sim.body_names) if hasattr(sim, "body_names") else None

    # Create per-env visualizers
    num_envs = sim.num_envs if hasattr(sim, "num_envs") else 1
    env_vizs = []
    env_offsets = []
    try:
        origins = sim.scene.env_origins.detach().cpu().numpy()
        origin_0 = origins[0].astype(np.float32)
    except (AttributeError, TypeError):
        origins = None
        origin_0 = np.zeros(3, dtype=np.float32)

    for eid in range(num_envs):
        viz = RobotAssetViser(viser, urdf_path, body_names, name_prefix=f"/env_{eid}/robot")
        env_vizs.append(viz)
        if origins is not None and eid < origins.shape[0]:
            env_offsets.append((origins[eid] - origin_0).astype(np.float32))
        else:
            env_offsets.append(np.zeros(3, dtype=np.float32))

    object_vizs = []
    object_ref_vizs = []
    object_goal_vizs = []
    target_torso_names = [f"/env_{eid}/target_torso" for eid in range(num_envs)]
    target_torso_heading_names = [f"/env_{eid}/target_torso_heading" for eid in range(num_envs)]
    is_hybrid_motion = hasattr(motion_command, "track_env_mask") and hasattr(motion_command, "gen_env_mask")

    def _motion_has_object_goal() -> bool:
        return bool(getattr(motion_command.motion, "has_object", False))

    def _should_show_torso_goal(env_id: int) -> bool:
        if not is_hybrid_motion:
            return True
        return not _motion_has_object_goal()

    def _should_show_object_goal(env_id: int) -> bool:
        if not is_hybrid_motion:
            return bool(getattr(motion_command.motion, "has_object", False))
        return _motion_has_object_goal()

    fixed_scene_box_specs = []
    motion_cfg = getattr(motion_command, "motion_cfg", None)
    if motion_cfg is not None:
        try:
            for spec in load_fixed_scene_objects_from_motion_config(motion_cfg):
                if spec.geom_type != "box":
                    logger.warning(
                        f"Replay viser (sim): skipping unsupported fixed scene geom_type '{spec.geom_type}' "
                        f"for '{spec.name}'."
                    )
                    continue
                fixed_scene_box_specs.append(spec)
        except Exception as exc:
            logger.warning(f"Replay viser (sim): failed to load fixed scene metadata: {exc}")

    fixed_scene_object_names = [spec.name for spec in fixed_scene_box_specs]
    fixed_scene_vizs: list[list[StaticBoxViser]] = []
    if fixed_scene_box_specs:
        for eid in range(num_envs):
            env_fixed_scene_vizs: list[StaticBoxViser] = []
            for spec in fixed_scene_box_specs:
                env_fixed_scene_vizs.append(
                    StaticBoxViser.from_fixed_scene(
                        viser,
                        object_name=spec.name,
                        size_xyz=spec.size,
                        prefix=f"/env_{eid}/fixed_scene/{spec.name}",
                    )
                )
            fixed_scene_vizs.append(env_fixed_scene_vizs)
        logger.info(f"Replay viser (sim): loaded {len(fixed_scene_box_specs)} fixed scene box(es)")

    object_urdf_path = env.robot_config.object.object_urdf_path
    if motion_command.motion.has_object and object_urdf_path is not None:
        resolved_object_urdf = resolve_data_file_path(object_urdf_path)
        for eid in range(num_envs):
            object_vizs.append(StaticUrdfViser(viser, resolved_object_urdf, prefix=f"/env_{eid}/object"))
            object_ref_vizs.append(
                StaticUrdfViser(
                    viser,
                    resolved_object_urdf,
                    prefix=f"/env_{eid}/object_ref",
                    color_override=(0.95, 0.55, 0.18),
                )
            )
            object_goal_vizs.append(
                StaticUrdfViser(
                    viser,
                    resolved_object_urdf,
                    prefix=f"/env_{eid}/object_target",
                    color_override=target_color,
                )
            )
        logger.info(f"Replay viser (sim): loaded object URDF: {resolved_object_urdf}")
        logger.info("Replay viser (sim): showing simulator object pose, reference object pose, and target object pose")
    elif motion_command.motion.has_object:
        logger.warning("Motion contains an object, but the experiment has no object_urdf_path configured.")

    logger.info(f"Replay viser (sim): {num_envs} env(s), motion_fps={motion_fps:.1f}, port={port}")
    logger.info(f"Open http://localhost:{port} in your browser")

    # --- FPS widget ---
    fps_widget = None
    try:
        fps_widget = viser.server.gui.add_text(
            "Sim FPS", f"target {motion_fps:.1f} fps", disabled=True, order=-1.0
        )
    except Exception:
        pass

    def _to_np(t) -> np.ndarray:
        return t.detach().cpu().numpy().copy()

    def _get_fixed_scene_root_states():
        states = []
        for object_name in fixed_scene_object_names:
            rigid_object = sim.scene.rigid_objects[object_name]
            raw_states = rigid_object.data.root_state_w[:num_envs]
            converted_states = raw_states.clone()
            converted_states[:, 3:7] = raw_states[:, [4, 5, 6, 3]]
            states.append(converted_states)
        return states

    last_time = None
    fps_ema = None
    env_ids = torch.arange(num_envs, device=device)
    fixed_scene_state_warning_emitted = False

    # --- Replay loop ---
    # Replicates step_visualize_motion logic but skips _draw_debug_vis (needs GUI)
    while True:
        # Advance motion timestep
        motion_command.step()

        # Write motion poses into sim state (same as step_visualize_motion)
        root_pos = motion_command.root_pos_w.clone()
        root_ori = motion_command.root_quat_w.clone()
        root_lin_vel = motion_command.body_lin_vel_w[:, 0].clone()
        root_ang_vel = motion_command.body_ang_vel_w[:, 0].clone()
        joint_pos = motion_command.joint_pos.clone()
        joint_vel = motion_command.joint_vel.clone()

        sim.dof_pos[env_ids] = joint_pos
        sim.dof_vel[env_ids] = joint_vel
        sim.robot_root_states[env_ids, :3] = root_pos
        sim.robot_root_states[env_ids, 3:7] = root_ori
        sim.robot_root_states[env_ids, 7:10] = root_lin_vel
        sim.robot_root_states[env_ids, 10:13] = root_ang_vel

        sim.set_actor_root_state_tensor(env_ids, sim.all_root_states)
        sim.set_dof_state_tensor(env_ids, sim.dof_state)

        # Handle object if present
        if motion_command.motion.has_object:
            object_pos = motion_command.object_pos_w.clone()
            object_ori = motion_command.object_quat_w.clone()
            object_lin_vel = motion_command.object_lin_vel_w.clone()
            object_states = torch.zeros(len(env_ids), 13, device=device)
            object_states[:, :3] = object_pos[:]
            object_states[:, 3:7] = object_ori[:]
            object_states[:, 7:10] = object_lin_vel[:]
            sim.set_actor_states(["object"], env_ids, object_states)

        # Step physics
        sim.scene.write_data_to_sim()
        sim.sim.step()
        sim.sim.render()
        sim.refresh_sim_tensors()

        fixed_scene_states = None
        if fixed_scene_object_names:
            try:
                fixed_scene_states = _get_fixed_scene_root_states()
            except Exception as exc:
                fixed_scene_states = None
                if not fixed_scene_state_warning_emitted:
                    logger.warning(f"Replay viser (sim): failed to read fixed scene states from simulator: {exc}")
                    fixed_scene_state_warning_emitted = True

        # Read back body poses and update viser
        for eid in range(num_envs):
            body_pos = _to_np(sim._rigid_body_pos[eid] - sim.scene.env_origins[eid])
            body_quat = _to_np(sim._rigid_body_rot[eid])
            env_vizs[eid].update(body_pos, body_quat, world_offset=env_offsets[eid])
            if fixed_scene_states is not None:
                for object_idx, box_viz in enumerate(fixed_scene_vizs[eid]):
                    box_state = fixed_scene_states[object_idx][eid]
                    box_pos = _to_np(box_state[:3] - sim.scene.env_origins[eid])
                    box_quat = _to_np(box_state[3:7])
                    box_viz.update(box_pos, box_quat, world_offset=env_offsets[eid])
            else:
                for box_viz in fixed_scene_vizs[eid]:
                    box_viz.hide()
            if _should_show_torso_goal(eid):
                target_torso_pos = _to_np(motion_command.goal_ref_pos_w[eid] - sim.scene.env_origins[eid]) + env_offsets[eid]
                target_torso_quat = _to_np(motion_command.goal_ref_quat_w[eid])
                viser.add_sphere(
                    target_torso_names[eid],
                    radius=target_torso_radius,
                    color=target_color,
                    position=target_torso_pos,
                )
                viser.update_heading_arrow(
                    target_torso_heading_names[eid],
                    position=target_torso_pos,
                    quat_xyzw=target_torso_quat,
                    color=target_color,
                    shaft_length=target_torso_arrow_length,
                    head_length=target_torso_arrow_head_length,
                    head_width=target_torso_arrow_head_width,
                    vertical_offset=0.0,
                    line_width=target_torso_arrow_line_width,
                )
            else:
                viser.remove(target_torso_names[eid])
                viser.remove(target_torso_heading_names[eid])
            if eid < len(object_vizs):
                object_pos = _to_np(motion_command.simulator_object_pos_w[eid] - sim.scene.env_origins[eid])
                object_quat = _to_np(motion_command.simulator_object_quat_w[eid])
                object_vizs[eid].update(object_pos, object_quat, world_offset=env_offsets[eid])
            if eid < len(object_ref_vizs):
                object_ref_pos = _to_np(motion_command.object_pos_w[eid] - sim.scene.env_origins[eid])
                object_ref_quat = _to_np(motion_command.object_quat_w[eid])
                object_ref_vizs[eid].update(object_ref_pos, object_ref_quat, world_offset=env_offsets[eid])
            if eid < len(object_goal_vizs):
                if _should_show_object_goal(eid):
                    object_goal_pos = _to_np(motion_command.goal_object_pos_w[eid] - sim.scene.env_origins[eid])
                    object_goal_quat = _to_np(motion_command.goal_object_quat_w[eid])
                    object_goal_vizs[eid].update(object_goal_pos, object_goal_quat, world_offset=env_offsets[eid])
                else:
                    object_goal_vizs[eid].hide()

        # FPS tracking
        now = time.perf_counter()
        if last_time is not None and fps_widget is not None:
            frame_dt = now - last_time
            if frame_dt > 1e-6:
                inst_fps = 1.0 / frame_dt
                fps_ema = inst_fps if fps_ema is None else 0.8 * fps_ema + 0.2 * inst_fps
                try:
                    fps_widget.value = f"run {fps_ema:4.1f} | target {motion_fps:.1f} fps"
                except Exception:
                    pass
        last_time = now

        # Real-time pacing
        elapsed = time.perf_counter() - now
        sleep_dur = dt - elapsed
        if sleep_dur > 0:
            time.sleep(sleep_dur)

        # Loop when motion ends — reset timesteps to 0
        if motion_command.time_steps[0].item() >= motion_command.motion.time_step_total - 2:
            motion_command.time_steps[:] = 0

    close_simulation_app(simulation_app)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    replay_viser_sim(tyro_cfg)


if __name__ == "__main__":
    main()
