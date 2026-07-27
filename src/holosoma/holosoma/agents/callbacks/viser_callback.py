"""Eval callback that provides real-time web visualization via viser.

Renders the policy robot (and optionally a reference motion robot and objects)
in a browser at ``http://localhost:{port}``.
Supports multi-env visualization with a grid layout.
Includes FPS display and optional real-time pacing.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from moviepy import VideoFileClip  # type: ignore[import-untyped]

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.eval_callback import ViserConfig
from holosoma.utils.fixed_scene_motion import load_fixed_scene_objects_from_motion_config
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path


class EvalViserCallback(RLEvalCallback):
    """Real-time web visualization of policy rollouts during evaluation."""

    _TARGET_TORSO_COLOR = (0.18, 0.92, 0.38)
    _TARGET_OBJECT_COLOR = (0.18, 0.92, 0.38)
    _TARGET_TORSO_RADIUS = 0.09
    _TARGET_TORSO_ARROW_LENGTH = 0.24
    _TARGET_TORSO_ARROW_HEAD_LENGTH = 0.08
    _TARGET_TORSO_ARROW_HEAD_WIDTH = 0.06
    _TARGET_TORSO_ARROW_LINE_WIDTH = 3.0

    def __init__(
        self,
        config: ViserConfig,
        training_loop: Any = None,
    ):
        super().__init__(config, training_loop)
        self.cfg = config

        self._viser: Any = None
        self._policy_vizs: list = []  # one per visualized env
        self._ref_vizs: list = []  # one per visualized env (or empty)
        self._object_vizs: list = []  # one per visualized env (or empty)
        self._object_ref_vizs: list = []  # one per visualized env (or empty)
        self._object_goal_vizs: list = []  # one per visualized env (or empty)
        self._fixed_scene_vizs: list[list] = []
        self._fixed_scene_ref_vizs: list[list] = []
        self._fixed_scene_box_specs: list = []
        self._fixed_scene_object_names: list[str] = []
        self._fixed_scene_state_warning_emitted = False
        self._target_torso_names: list[str] = []
        self._target_torso_heading_names: list[str] = []
        self._motion_command = None
        self._env_ids: list[int] = []
        self._env_offsets: list[np.ndarray] = []  # world offset per env slot
        self._first_step = True

        # FPS tracking
        self._fps_widget = None
        self._last_update_time: float | None = None
        self._fps_ema: float | None = None
        self._sim_dt: float = 0.0  # sim dt (seconds per control step)

        # Browser viewport video recording
        self._recording_lock = threading.Lock()
        self._recording = False
        self._recording_saving = False
        self._recording_session = 0
        self._recording_index = 0
        self._recording_client: Any = None
        self._recording_resolution: tuple[int, int] | None = None
        self._recording_path: Path | None = None
        self._recording_temp_path: Path | None = None
        self._recording_writer: Any = None
        self._recording_frame_count = 0
        self._camera_reset_button: Any = None
        self._recording_start_button: Any = None
        self._recording_stop_button: Any = None
        self._recording_status_widget: Any = None

        # Real-time pacing
        self._last_tick_time: float | None = None

    @staticmethod
    def _is_hybrid_motion_command(motion_cmd: Any) -> bool:
        return hasattr(motion_cmd, "track_env_mask") and hasattr(motion_cmd, "gen_env_mask")

    @staticmethod
    def _motion_has_object_goal(motion_cmd: Any) -> bool:
        return bool(getattr(getattr(motion_cmd, "motion", None), "has_object", False))

    @classmethod
    def _should_show_torso_goal(cls, motion_cmd: Any, env_id: int) -> bool:
        if motion_cmd is None or not cls._is_hybrid_motion_command(motion_cmd):
            return False
        return not cls._motion_has_object_goal(motion_cmd)

    @classmethod
    def _should_show_object_goal(cls, motion_cmd: Any, env_id: int) -> bool:
        if motion_cmd is None or not cls._is_hybrid_motion_command(motion_cmd):
            return False
        return cls._motion_has_object_goal(motion_cmd)

    @staticmethod
    def _load_fixed_scene_box_specs(motion_cmd: Any) -> list:
        motion_cfg = getattr(motion_cmd, "motion_cfg", None)
        return [
            spec
            for spec in load_fixed_scene_objects_from_motion_config(motion_cfg)
            if spec.geom_type == "box"
        ]

    @staticmethod
    def _load_motion_object_specs(motion_cmd: Any) -> list:
        motion_cfg = getattr(motion_cmd, "motion_cfg", None)
        return [
            spec
            for spec in load_fixed_scene_objects_from_motion_config(motion_cfg)
            if getattr(spec, "is_motion_object", False)
        ]

    @staticmethod
    def _get_fixed_scene_root_states(sim: Any, object_names: list[str], env_ids: list[int]):
        states = []
        for object_name in object_names:
            rigid_object = sim.scene.rigid_objects[object_name]
            raw_states = rigid_object.data.root_state_w[env_ids]
            converted_states = raw_states.clone()
            converted_states[:, 3:7] = raw_states[:, [4, 5, 6, 3]]
            states.append(converted_states)
        return states

    @staticmethod
    def _create_motion_object_visualizer(
        viser: Any,
        spec: Any,
        *,
        prefix: str,
        color_override: tuple[float, float, float] | None = None,
    ) -> Any:
        from holosoma.utils.viser_scene import RigidMeshViser, StaticBoxViser, StaticUrdfViser

        color = color_override if color_override is not None else RigidMeshViser._DEFAULT_COLOR
        mesh_vertices = getattr(spec, "mesh_vertices", None)
        mesh_faces = getattr(spec, "mesh_faces", None)
        if mesh_vertices is not None and mesh_faces is not None:
            return RigidMeshViser(
                viser,
                np.asarray(mesh_vertices, dtype=np.float32),
                np.asarray(mesh_faces, dtype=np.int32),
                prefix=prefix,
                color=color,
            )

        urdf_path = getattr(spec, "urdf_path", None)
        if urdf_path:
            return StaticUrdfViser(
                viser,
                resolve_data_file_path(urdf_path),
                prefix=prefix,
                color_override=color_override,
            )

        mesh_path = getattr(spec, "mesh_path", None)
        if mesh_path:
            try:
                return RigidMeshViser.from_mesh_path(
                    viser,
                    resolve_data_file_path(mesh_path),
                    prefix=prefix,
                    color=color,
                )
            except Exception as exc:
                logger.warning(f"EvalViserCallback: failed to load object mesh '{mesh_path}': {exc}")

        size = getattr(spec, "size", None)
        if size is not None:
            return StaticBoxViser(
                viser,
                np.asarray(size, dtype=np.float32),
                prefix=prefix,
                color=color,
            )

        raise ValueError(f"No visualizable mesh/URDF/box metadata for motion object '{getattr(spec, 'name', 'object')}'.")

    def _get_env(self):
        return self.training_loop._unwrap_env()

    def _resolve_asset_root(self, asset_root: str) -> str:
        if asset_root.startswith("@holosoma/"):
            return asset_root.replace("@holosoma", get_holosoma_root())
        return asset_root

    def _compute_grid_offsets(self, num_vis: int, spacing: float) -> list[np.ndarray]:
        """Compute grid layout offsets for multi-env visualization."""
        cols = max(1, int(math.ceil(math.sqrt(num_vis))))
        offsets = []
        for slot in range(num_vis):
            col = slot % cols
            row = slot // cols
            offsets.append(np.array([col * spacing, row * spacing, 0.0], dtype=np.float32))
        return offsets

    def on_pre_evaluate_policy(self) -> None:
        from holosoma.utils.viser_helper import ViserHelper
        from holosoma.utils.viser_robot import RobotAssetViser
        from holosoma.utils.viser_scene import StaticBoxViser, StaticUrdfViser, add_ground_grid, setup_scene_lighting

        env = self._get_env()
        sim = env.simulator

        # Determine which env IDs to visualize
        total_envs = sim.num_envs if hasattr(sim, "num_envs") else 1
        if self.cfg.num_envs < 0:
            num_vis = total_envs
        else:
            num_vis = min(self.cfg.num_envs, total_envs)
        self._env_ids = list(range(num_vis))

        # Compute per-env offsets from the actual sim env_origins (ground truth)
        self._env_offsets = []
        origin_0 = np.zeros(3, dtype=np.float32)
        try:
            origins = sim.scene.env_origins.detach().cpu().numpy()  # [num_envs, 3]
            origin_0 = origins[0].astype(np.float32)
        except (AttributeError, TypeError):
            origins = None

        for slot, eid in enumerate(self._env_ids):
            if origins is not None and eid < origins.shape[0]:
                self._env_offsets.append((origins[eid] - origin_0).astype(np.float32))
            else:
                self._env_offsets.append(np.zeros(3, dtype=np.float32))

        # Get sim dt for FPS display and real-time pacing
        self._sim_dt = float(env.dt) if hasattr(env, "dt") else 0.0
        sim_fps = 1.0 / self._sim_dt if self._sim_dt > 0 else 0.0

        logger.info(f"EvalViserCallback: total_envs={total_envs}, visualizing={num_vis}, sim_fps={sim_fps:.1f}")

        # --- Server + scene ---
        self._viser = ViserHelper(port=self.cfg.port)
        if not self._viser.ok():
            logger.warning("EvalViserCallback: failed to start viser server")
            return
        self._set_default_camera(np.zeros(3, dtype=np.float32), apply_to_clients=False)
        setup_scene_lighting(self._viser)
        add_ground_grid(
            self._viser,
            width=self.cfg.ground_width,
            depth=self.cfg.ground_depth,
            spacing=self.cfg.ground_spacing,
        )

        # --- FPS widget ---
        try:
            gui = self._viser.server.gui
            self._fps_widget = gui.add_text(
                "Sim FPS",
                self._format_fps_text(0.0),
                disabled=True,
                hint="run = measured update FPS | sim = physics FPS | pace = target FPS",
                order=-1.0,
            )
        except Exception:
            self._fps_widget = None

        self._setup_camera_controls()
        self._setup_recording_controls()

        # --- Resolve robot URDF ---
        asset_cfg = env.robot_config.asset
        asset_root = self._resolve_asset_root(asset_cfg.asset_root)
        urdf_path = str(Path(asset_root) / asset_cfg.urdf_file)
        body_names = list(sim.body_names) if hasattr(sim, "body_names") else None

        logger.info(f"EvalViserCallback: loading robot URDF: {urdf_path}")

        # --- Create per-env policy visualizers ---
        self._policy_vizs = []
        for slot, eid in enumerate(self._env_ids):
            viz = RobotAssetViser(
                self._viser, urdf_path, body_names, name_prefix=f"/env_{slot}/robot_policy"
            )
            self._policy_vizs.append(viz)

        # --- Reference motion (WBT only) ---
        self._ref_vizs = []
        self._motion_command = None
        if hasattr(env, "command_manager") and env.command_manager is not None:
            motion_cmd = env.command_manager.get_state("motion_command")
            if motion_cmd is not None:
                self._motion_command = motion_cmd
        if self.cfg.show_reference and self._motion_command is not None:
            motion_cmd = self._motion_command
            if motion_cmd is not None:
                for slot, eid in enumerate(self._env_ids):
                    ref_viz = RobotAssetViser(
                        self._viser,
                        urdf_path,
                        body_names,
                        name_prefix=f"/env_{slot}/robot_reference",
                        color_override=self.cfg.reference_color,
                    )
                    self._ref_vizs.append(ref_viz)
                logger.info("EvalViserCallback: reference motion visualization enabled")

        # --- Object visualization ---
        self._object_vizs = []
        self._object_ref_vizs = []
        self._object_goal_vizs = []
        self._target_torso_names = []
        self._target_torso_heading_names = []
        object_cfg = env.robot_config.object
        is_hybrid_motion = self._is_hybrid_motion_command(self._motion_command)
        if is_hybrid_motion:
            for slot, _eid in enumerate(self._env_ids):
                self._target_torso_names.append(f"/env_{slot}/target_torso")
                self._target_torso_heading_names.append(f"/env_{slot}/target_torso_heading")
        if object_cfg.object_urdf_path is not None and self._motion_command is not None:
            try:
                object_urdf = resolve_data_file_path(object_cfg.object_urdf_path)
                for slot, _eid in enumerate(self._env_ids):
                    self._object_vizs.append(
                        StaticUrdfViser(
                            self._viser,
                            object_urdf,
                            prefix=f"/env_{slot}/object_policy",
                        )
                    )
                    if self.cfg.show_reference:
                        self._object_ref_vizs.append(
                            StaticUrdfViser(
                                self._viser,
                                object_urdf,
                                prefix=f"/env_{slot}/object_reference",
                                color_override=self.cfg.reference_color,
                            )
                        )
                    if is_hybrid_motion and bool(self._motion_command.motion.has_object):
                        self._object_goal_vizs.append(
                            StaticUrdfViser(
                                self._viser,
                                object_urdf,
                                prefix=f"/env_{slot}/object_target",
                                color_override=self._TARGET_OBJECT_COLOR,
                            )
                        )
                logger.info(f"EvalViserCallback: loaded dynamic object URDF: {object_urdf}")
            except Exception as exc:
                logger.warning(f"EvalViserCallback: failed to load object URDF: {exc}")
        elif object_cfg.object_urdf_path is not None:
            logger.warning("EvalViserCallback: object URDF is configured, but no motion_command was found.")
        elif self._motion_command is not None and bool(getattr(self._motion_command.motion, "has_object", False)):
            try:
                motion_object_specs = self._load_motion_object_specs(self._motion_command)
                motion_object_spec = next((spec for spec in motion_object_specs if spec.name == "object"), None)
                if motion_object_spec is None and motion_object_specs:
                    motion_object_spec = motion_object_specs[0]
                if motion_object_spec is not None:
                    for slot, _eid in enumerate(self._env_ids):
                        self._object_vizs.append(
                            self._create_motion_object_visualizer(
                                self._viser,
                                motion_object_spec,
                                prefix=f"/env_{slot}/object_policy",
                            )
                        )
                        if self.cfg.show_reference:
                            self._object_ref_vizs.append(
                                self._create_motion_object_visualizer(
                                    self._viser,
                                    motion_object_spec,
                                    prefix=f"/env_{slot}/object_reference",
                                    color_override=self.cfg.reference_color,
                                )
                            )
                        if is_hybrid_motion and bool(self._motion_command.motion.has_object):
                            self._object_goal_vizs.append(
                                self._create_motion_object_visualizer(
                                    self._viser,
                                    motion_object_spec,
                                    prefix=f"/env_{slot}/object_target",
                                    color_override=self._TARGET_OBJECT_COLOR,
                                )
                            )
                    logger.info(
                        "EvalViserCallback: loaded dynamic object visualization from motion metadata "
                        f"({motion_object_spec.geom_type})"
                    )
            except Exception as exc:
                logger.warning(f"EvalViserCallback: failed to load motion object visualization: {exc}")

        # --- Fixed-scene box visualization (e.g. climb obstacles) ---
        self._fixed_scene_vizs = []
        self._fixed_scene_ref_vizs = []
        self._fixed_scene_box_specs = []
        self._fixed_scene_object_names = []
        if self._motion_command is not None:
            try:
                self._fixed_scene_box_specs = self._load_fixed_scene_box_specs(self._motion_command)
            except Exception as exc:
                logger.warning(f"EvalViserCallback: failed to load fixed scene metadata: {exc}")

        if self._fixed_scene_box_specs:
            self._fixed_scene_object_names = [spec.name for spec in self._fixed_scene_box_specs]
            for slot, _eid in enumerate(self._env_ids):
                env_fixed_scene_vizs = []
                env_fixed_scene_ref_vizs = []
                for spec in self._fixed_scene_box_specs:
                    env_fixed_scene_vizs.append(
                        StaticBoxViser.from_fixed_scene(
                            self._viser,
                            object_name=spec.name,
                            size_xyz=spec.size,
                            prefix=f"/env_{slot}/fixed_scene_policy/{spec.name}",
                        )
                    )
                    if self.cfg.show_reference:
                        env_fixed_scene_ref_vizs.append(
                            StaticBoxViser.from_fixed_scene(
                                self._viser,
                                object_name=spec.name,
                                size_xyz=spec.size,
                                prefix=f"/env_{slot}/fixed_scene_reference/{spec.name}",
                                color_override=self.cfg.reference_color,
                            )
                        )
                self._fixed_scene_vizs.append(env_fixed_scene_vizs)
                if self.cfg.show_reference:
                    self._fixed_scene_ref_vizs.append(env_fixed_scene_ref_vizs)
            logger.info(
                f"EvalViserCallback: loaded {len(self._fixed_scene_box_specs)} fixed scene box(es) from motion metadata"
            )

        self._first_step = True
        self._last_update_time = None
        self._fps_ema = None
        self._last_tick_time = None
        logger.info(
            f"EvalViserCallback: visualizing {num_vis} env(s), serving at http://localhost:{self.cfg.port}"
        )

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        if self._viser is None or not self._viser.ok():
            return actor_state

        env = self._get_env()
        sim = env.simulator

        def _to_np(t) -> np.ndarray:
            return t.detach().cpu().numpy().copy()

        fixed_scene_states = None
        if self._fixed_scene_object_names:
            try:
                fixed_scene_states = self._get_fixed_scene_root_states(sim, self._fixed_scene_object_names, self._env_ids)
            except Exception as exc:
                fixed_scene_states = None
                if not self._fixed_scene_state_warning_emitted:
                    logger.warning(f"EvalViserCallback: failed to read fixed scene states from simulator: {exc}")
                    self._fixed_scene_state_warning_emitted = True

        reference_offset = np.array(self.cfg.reference_offset, dtype=np.float32)

        for slot, eid in enumerate(self._env_ids):
            offset = self._env_offsets[slot]

            # --- Update policy robot ---
            if slot < len(self._policy_vizs):
                body_pos = _to_np(sim._rigid_body_pos[eid] - sim.scene.env_origins[eid])  # [num_bodies, 3]
                body_quat = _to_np(sim._rigid_body_rot[eid])  # [num_bodies, 4] xyzw
                self._policy_vizs[slot].update(body_pos, body_quat, world_offset=offset)

            # --- Update reference robot (WBT) ---
            if slot < len(self._ref_vizs) and self._motion_command is not None:
                try:
                    mc = self._motion_command
                    time_step = mc.time_steps[eid].item()

                    ref_body_pos = _to_np(mc.motion.body_pos_w[time_step])
                    ref_body_quat = _to_np(mc.motion.body_quat_w[time_step])

                    ref_offset = offset + np.array(self.cfg.reference_offset, dtype=np.float32)
                    self._ref_vizs[slot].update(ref_body_pos, ref_body_quat, world_offset=ref_offset)
                except Exception:
                    pass

            # --- Update policy object (WBT-with-object) ---
            if slot < len(self._object_vizs) and self._motion_command is not None:
                try:
                    mc = self._motion_command
                    object_pos = _to_np(mc.simulator_object_pos_w[eid] - sim.scene.env_origins[eid])
                    object_quat = _to_np(mc.simulator_object_quat_w[eid])
                    self._object_vizs[slot].update(object_pos, object_quat, world_offset=offset)
                except Exception:
                    pass

            # --- Update fixed scene boxes from simulator actor states ---
            if fixed_scene_states is not None and slot < len(self._fixed_scene_vizs):
                try:
                    for object_idx, box_viz in enumerate(self._fixed_scene_vizs[slot]):
                        box_state = fixed_scene_states[object_idx][slot]
                        box_pos = _to_np(box_state[:3] - sim.scene.env_origins[eid])
                        box_quat = _to_np(box_state[3:7])
                        box_viz.update(box_pos, box_quat, world_offset=offset)
                except Exception:
                    pass
            elif slot < len(self._fixed_scene_vizs):
                for box_viz in self._fixed_scene_vizs[slot]:
                    box_viz.hide()

            # --- Update target torso goal ---
            if slot < len(self._target_torso_names) and self._motion_command is not None:
                try:
                    mc = self._motion_command
                    if self._should_show_torso_goal(mc, eid):
                        target_torso_pos = _to_np(mc.goal_ref_pos_w[eid] - sim.scene.env_origins[eid]) + offset
                        target_torso_quat = _to_np(mc.goal_ref_quat_w[eid])
                        self._viser.add_sphere(
                            self._target_torso_names[slot],
                            radius=self._TARGET_TORSO_RADIUS,
                            color=self._TARGET_TORSO_COLOR,
                            position=target_torso_pos,
                        )
                        if slot < len(self._target_torso_heading_names):
                            self._viser.update_heading_arrow(
                                self._target_torso_heading_names[slot],
                                position=target_torso_pos,
                                quat_xyzw=target_torso_quat,
                                color=self._TARGET_TORSO_COLOR,
                                shaft_length=self._TARGET_TORSO_ARROW_LENGTH,
                                head_length=self._TARGET_TORSO_ARROW_HEAD_LENGTH,
                                head_width=self._TARGET_TORSO_ARROW_HEAD_WIDTH,
                                vertical_offset=0.0,
                                line_width=self._TARGET_TORSO_ARROW_LINE_WIDTH,
                            )
                    else:
                        self._viser.remove(self._target_torso_names[slot])
                        if slot < len(self._target_torso_heading_names):
                            self._viser.remove(self._target_torso_heading_names[slot])
                except Exception:
                    pass

            # --- Update reference object (WBT-with-object) ---
            if slot < len(self._object_ref_vizs) and self._motion_command is not None:
                try:
                    mc = self._motion_command
                    object_ref_pos = _to_np(mc.object_pos_w[eid] - sim.scene.env_origins[eid])
                    object_ref_quat = _to_np(mc.object_quat_w[eid])
                    ref_offset = offset + reference_offset
                    self._object_ref_vizs[slot].update(object_ref_pos, object_ref_quat, world_offset=ref_offset)
                except Exception:
                    pass

            # --- Update reference fixed scene boxes from motion metadata ---
            if slot < len(self._fixed_scene_ref_vizs):
                try:
                    ref_offset = offset + reference_offset
                    for object_idx, box_viz in enumerate(self._fixed_scene_ref_vizs[slot]):
                        spec = self._fixed_scene_box_specs[object_idx]
                        box_viz.update(spec.position_w, spec.quat_wxyz[[1, 2, 3, 0]], world_offset=ref_offset)
                except Exception:
                    pass

            # --- Update target object (goal-conditioned WBT/hybrid) ---
            if slot < len(self._object_goal_vizs) and self._motion_command is not None:
                try:
                    mc = self._motion_command
                    if self._should_show_object_goal(mc, eid):
                        object_goal_pos = _to_np(mc.goal_object_pos_w[eid] - sim.scene.env_origins[eid])
                        object_goal_quat = _to_np(mc.goal_object_quat_w[eid])
                        self._object_goal_vizs[slot].update(object_goal_pos, object_goal_quat, world_offset=offset)
                    else:
                        self._object_goal_vizs[slot].hide()
                except Exception:
                    pass

        # --- FPS tracking ---
        self._update_fps()

        # --- Real-time pacing ---
        if self.cfg.realtime:
            self._sleep_to_realtime()

        # --- Set camera on first step ---
        if self._first_step:
            self._first_step = False
            try:
                env_id = self._env_ids[0]
                root_pos = _to_np(sim.robot_root_states[env_id, :3] - sim.scene.env_origins[env_id])
                self._set_default_camera(root_pos, apply_to_clients=True)
            except Exception:
                logger.warning("EvalViserCallback: failed to set the default camera pose", exc_info=True)

        self._capture_recording_frame()

        return actor_state

    # ------------------------------------------------------------------
    # Default camera
    # ------------------------------------------------------------------

    def _setup_camera_controls(self) -> None:
        if self._viser is None:
            return
        try:
            reset_button = self._viser.server.gui.add_button(
                "Reset camera view",
                hint="Return this browser to the default evaluation camera.",
                order=-5.0,
            )
            self._camera_reset_button = reset_button

            @reset_button.on_click
            def _on_reset_camera(event) -> None:
                if event.client is not None:
                    self._reset_client_camera(event.client)

        except Exception:
            self._camera_reset_button = None
            logger.warning("EvalViserCallback: failed to create the reset-camera control", exc_info=True)

    def _reset_client_camera(self, client: Any) -> None:
        if self._viser is None or self._viser.server is None:
            return

        initial_camera = self._viser.server.initial_camera
        client.camera.position = initial_camera.position
        client.camera.look_at = initial_camera.look_at
        if initial_camera.up is not None:
            client.camera.up = initial_camera.up
        client.camera.fov = initial_camera.fov

    def _set_default_camera(self, root_pos: np.ndarray, *, apply_to_clients: bool) -> None:
        if self._viser is None or self._viser.server is None:
            return

        position = np.asarray(root_pos, dtype=np.float64) + np.asarray(
            self.cfg.camera_position_offset,
            dtype=np.float64,
        )
        look_at = np.asarray(root_pos, dtype=np.float64) + np.asarray(
            self.cfg.camera_look_at_offset,
            dtype=np.float64,
        )
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        fov = math.radians(float(self.cfg.camera_fov_degrees))

        initial_camera = self._viser.server.initial_camera
        initial_camera.position = position
        initial_camera.look_at = look_at
        initial_camera.up = up
        initial_camera.fov = fov

        if apply_to_clients:
            for client in self._viser.server.get_clients().values():
                self._reset_client_camera(client)

        logger.info(
            f"EvalViserCallback: default camera position={position.tolist()}, "
            f"look_at={look_at.tolist()}, fov={self.cfg.camera_fov_degrees:.1f} degrees"
        )

    # ------------------------------------------------------------------
    # Browser viewport video recording
    # ------------------------------------------------------------------

    def _setup_recording_controls(self) -> None:
        if self._viser is None:
            return
        try:
            gui = self._viser.server.gui
            start_button = gui.add_button(
                "Start recording",
                hint="Record this browser's viser viewport to the evaluation log directory.",
                order=-4.0,
            )
            stop_button = gui.add_button(
                "Stop recording",
                disabled=True,
                hint="Stop and save the current viser recording.",
                order=-3.0,
            )
            self._recording_status_widget = gui.add_text(
                "Recording",
                "Idle",
                disabled=True,
                order=-2.0,
            )
            self._recording_start_button = start_button
            self._recording_stop_button = stop_button

            @start_button.on_click
            def _on_start_recording(event) -> None:
                self._start_recording(event.client)

            @stop_button.on_click
            def _on_stop_recording(_event) -> None:
                self._stop_recording()

        except Exception:
            self._recording_start_button = None
            self._recording_stop_button = None
            self._recording_status_widget = None
            logger.warning("EvalViserCallback: failed to create video recording controls", exc_info=True)

    def _recording_save_dir(self) -> Path:
        log_dir = getattr(self.training_loop, "log_dir", None)
        if log_dir is None:
            raise RuntimeError("The evaluation training loop does not have a log directory.")
        save_dir = Path(log_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _next_recording_path(self) -> Path:
        save_dir = self._recording_save_dir()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        while True:
            self._recording_index += 1
            path = save_dir / f"viser_{timestamp}_{self._recording_index:03d}.mp4"
            if not path.exists() and not path.with_suffix(".raw.mp4").exists():
                return path

    def _get_recording_resolution(self, client: Any) -> tuple[int, int]:
        target_width = max(2, int(self.cfg.recording_width))
        target_height = max(2, int(self.cfg.recording_height))
        camera = getattr(client, "camera", None)
        viewport_width = int(getattr(camera, "image_width", 0) or target_width)
        viewport_height = int(getattr(camera, "image_height", 0) or target_height)
        scale = min(1.0, viewport_width / target_width, viewport_height / target_height)
        width = max(2, int(target_width * scale) // 2 * 2)
        height = max(2, int(target_height * scale) // 2 * 2)
        return width, height

    def _set_recording_ui(self, *, recording: bool, status: str) -> None:
        try:
            if self._recording_start_button is not None:
                self._recording_start_button.disabled = recording or self._recording_saving
            if self._recording_stop_button is not None:
                self._recording_stop_button.disabled = not recording
            if self._recording_status_widget is not None:
                self._recording_status_widget.value = status
        except Exception:
            pass

    def _start_recording(self, client: Any) -> bool:
        if client is None:
            logger.warning("EvalViserCallback: cannot start recording without a connected viser browser")
            self._set_recording_ui(recording=False, status="No browser client connected")
            return False

        try:
            recording_path = self._next_recording_path()
            resolution = self._get_recording_resolution(client)
        except Exception as exc:
            logger.warning(f"EvalViserCallback: cannot start video recording: {exc}")
            self._set_recording_ui(recording=False, status=f"Could not start: {exc}")
            return False

        with self._recording_lock:
            if self._recording or self._recording_saving:
                logger.warning("EvalViserCallback: a viser video recording is already active or being saved")
                return False
            self._recording_session += 1
            self._recording = True
            self._recording_client = client
            self._recording_resolution = resolution
            self._recording_path = recording_path
            self._recording_temp_path = recording_path.with_suffix(".raw.mp4")
            self._recording_writer = None
            self._recording_frame_count = 0

        width, height = resolution
        fps = 1.0 / self._sim_dt if self._sim_dt > 0 else 30.0
        logger.info(
            f"EvalViserCallback: started viser video recording "
            f"({width}x{height} at {fps:.1f} fps); output={recording_path.resolve()}"
        )
        self._set_recording_ui(recording=True, status=f"Recording 0 frames ({width}x{height})")
        return True

    def _capture_recording_frame(self) -> None:
        with self._recording_lock:
            if not self._recording:
                return
            recording_session = self._recording_session
            client = self._recording_client
            resolution = self._recording_resolution

        if client is None or resolution is None or self._viser is None:
            return

        try:
            self._viser.server.flush()
            width, height = resolution
            frame = client.get_render(
                height=height,
                width=width,
                transport_format="jpeg",
            )
        except Exception as exc:
            logger.warning(f"EvalViserCallback: failed to capture viser video frame: {exc}")
            self._stop_recording()
            return

        with self._recording_lock:
            if not self._recording or self._recording_session != recording_session:
                return

            if self._recording_writer is None:
                assert self._recording_path is not None
                assert self._recording_temp_path is not None
                fps = 1.0 / self._sim_dt if self._sim_dt > 0 else 30.0
                writer = cv2.VideoWriter(
                    str(self._recording_temp_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
                    fps,
                    resolution,
                )
                if not writer.isOpened():
                    writer.release()
                    self._recording = False
                    self._recording_client = None
                    logger.warning(f"EvalViserCallback: failed to open video writer for {self._recording_temp_path}")
                    self._set_recording_ui(recording=False, status="Could not open video writer")
                    return
                self._recording_writer = writer

            if frame.ndim != 3 or frame.shape[2] not in (3, 4):
                logger.warning(f"EvalViserCallback: ignoring unexpected render shape {frame.shape}")
                return
            color_conversion = cv2.COLOR_RGBA2BGR if frame.shape[2] == 4 else cv2.COLOR_RGB2BGR
            self._recording_writer.write(cv2.cvtColor(frame, color_conversion))
            self._recording_frame_count += 1
            frame_count = self._recording_frame_count

        self._set_recording_ui(recording=True, status=f"Recording {frame_count} frames")

    def _stop_recording(self) -> Path | None:
        with self._recording_lock:
            if not self._recording:
                if not self._recording_saving:
                    self._set_recording_ui(recording=False, status="Idle")
                return None

            self._recording = False
            self._recording_saving = True
            writer = self._recording_writer
            recording_path = self._recording_path
            recording_temp_path = self._recording_temp_path
            frame_count = self._recording_frame_count
            self._recording_client = None
            self._recording_resolution = None
            self._recording_writer = None
            self._recording_path = None
            self._recording_temp_path = None
            self._recording_frame_count = 0

        self._set_recording_ui(recording=False, status="Encoding H.264...")
        if writer is not None:
            writer.release()

        if writer is None or recording_path is None or recording_temp_path is None or frame_count == 0:
            with self._recording_lock:
                self._recording_saving = False
            logger.warning("EvalViserCallback: stopped viser recording before any frames were captured; no video saved")
            self._set_recording_ui(recording=False, status="No frames captured")
            return None

        try:
            self._encode_recording(recording_temp_path, recording_path)
        except Exception as exc:
            if recording_path.exists():
                recording_path.unlink()
            with self._recording_lock:
                self._recording_saving = False
            logger.warning(
                f"EvalViserCallback: failed to encode viser video with MoviePy: {exc}. "
                f"Raw capture preserved at {recording_temp_path.resolve()}",
                exc_info=True,
            )
            self._set_recording_ui(recording=False, status=f"Encoding failed; kept {recording_temp_path.name}")
            return None

        with self._recording_lock:
            self._recording_saving = False

        resolved_path = recording_path.resolve()
        logger.info(f"EvalViserCallback: saved H.264 viser video ({frame_count} frames) to {resolved_path}")
        self._set_recording_ui(recording=False, status=f"Saved {resolved_path.name}")
        return resolved_path

    @staticmethod
    def _encode_recording(raw_path: Path, output_path: Path) -> None:
        clip = VideoFileClip(str(raw_path), audio=False)
        try:
            clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio=False,
                preset="medium",
                pixel_format="yuv420p",
                ffmpeg_params=["-movflags", "+faststart"],
                logger=None,
            )
        finally:
            clip.close()
        raw_path.unlink()

    # ------------------------------------------------------------------
    # FPS display (matches MimicKit's _update_viser_fps / _format_viser_fps_text)
    # ------------------------------------------------------------------

    def _update_fps(self) -> None:
        if self._fps_widget is None:
            return
        now = time.perf_counter()
        if self._last_update_time is not None:
            dt = now - self._last_update_time
            if dt > 1e-6:
                inst_fps = 1.0 / dt
                if self._fps_ema is None:
                    self._fps_ema = inst_fps
                else:
                    self._fps_ema = 0.8 * self._fps_ema + 0.2 * inst_fps
                try:
                    self._fps_widget.value = self._format_fps_text(self._fps_ema)
                except Exception:
                    pass
        self._last_update_time = now

    def _format_fps_text(self, run_fps: float) -> str:
        sim_fps = 1.0 / self._sim_dt if self._sim_dt > 0 else 0.0
        pace_fps = sim_fps if self.cfg.realtime and sim_fps > 0 else 0.0
        sim_text = f"{sim_fps:4.1f}" if sim_fps > 0 else "n/a"
        pace_text = f"{pace_fps:4.1f}" if pace_fps > 0 else "n/a"
        return f"run {run_fps:4.1f} | sim {sim_text} | pace {pace_text} fps"

    # ------------------------------------------------------------------
    # Real-time pacing (matches MimicKit's _sleep_to_realtime)
    # ------------------------------------------------------------------

    def _sleep_to_realtime(self) -> None:
        if self._sim_dt <= 0:
            return
        if self._last_tick_time is None:
            self._last_tick_time = time.perf_counter()
            return
        deadline = self._last_tick_time + self._sim_dt
        now = time.perf_counter()
        sleep_dur = deadline - now
        if sleep_dur > 0:
            time.sleep(sleep_dur)
            now = time.perf_counter()
        self._last_tick_time = max(deadline, now)

    def on_post_evaluate_policy(self) -> None:
        self._stop_recording()
        if self._viser is not None:
            self._viser.stop()
            self._viser = None
        self._policy_vizs = []
        self._ref_vizs = []
        self._object_vizs = []
        self._object_ref_vizs = []
        self._object_goal_vizs = []
        self._fixed_scene_vizs = []
        self._fixed_scene_ref_vizs = []
        self._fixed_scene_box_specs = []
        self._fixed_scene_object_names = []
        self._target_torso_names = []
        self._target_torso_heading_names = []
        self._motion_command = None
        self._fps_widget = None
        self._camera_reset_button = None
        self._recording_start_button = None
        self._recording_stop_button = None
        self._recording_status_widget = None
        self._fixed_scene_state_warning_emitted = False
        logger.info("EvalViserCallback: stopped viser server")
