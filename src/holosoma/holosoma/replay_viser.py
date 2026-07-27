"""Replay reference motion in viser (no IsaacSim required).

Usage:
    python src/holosoma/holosoma/replay_viser.py \
        --motion holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz \
        --urdf holosoma/data/robots/g1/g1_29dof.urdf
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

try:
    from loguru import logger
except ImportError:  # pragma: no cover - env-specific fallback
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logger = logging.getLogger(__name__)

src_root = Path(__file__).resolve().parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.fixed_scene_motion import load_fixed_scene_objects_from_npz
from holosoma.utils.viser_helper import ViserHelper
from holosoma.utils.viser_robot import RobotAssetViser
from holosoma.utils.viser_scene import (
    RigidMeshViser,
    StaticBoxViser,
    StaticUrdfViser,
    add_ground_grid,
    setup_scene_lighting,
)


def _load_optional_string(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def load_motion_npz(motion_path: str, device: str = "cpu") -> dict:
    """Load motion NPZ and return body poses + metadata."""
    resolved = resolve_data_file_path(motion_path)
    logger.info(f"Loading motion: {resolved}")

    with open(resolved, "rb") as f:
        data = np.load(f, allow_pickle=False)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        body_names = data["body_names"].tolist()
        body_pos_w = data["body_pos_w"].astype(np.float32)  # [T, num_bodies, 3]
        body_quat_w_wxyz = data["body_quat_w"].astype(np.float32)  # [T, num_bodies, 4] wxyz
        # Convert wxyz -> xyzw (holosoma convention)
        body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]

        result = {
            "fps": fps,
            "body_names": body_names,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "total_frames": body_pos_w.shape[0],
        }

        fixed_scene_objects = load_fixed_scene_objects_from_npz(data)
        if fixed_scene_objects:
            result["fixed_scene_objects"] = fixed_scene_objects

        # Load object data if present
        if "object_pos_w" in data:
            result["object_pos_w"] = data["object_pos_w"].astype(np.float32)
            obj_quat_wxyz = data["object_quat_w"].astype(np.float32)
            result["object_quat_w"] = obj_quat_wxyz[:, [1, 2, 3, 0]]  # wxyz -> xyzw
        if "object_box_size" in data:
            result["object_box_size"] = data["object_box_size"].astype(np.float32)
        if "object_mesh_vertices" in data and "object_mesh_faces" in data:
            result["object_mesh_vertices"] = data["object_mesh_vertices"].astype(np.float32)
            result["object_mesh_faces"] = data["object_mesh_faces"].astype(np.int32)
            if "object_mesh_color" in data:
                result["object_mesh_color"] = tuple(float(v) for v in data["object_mesh_color"].reshape(-1)[:3])
        object_geom_type = _load_optional_string(data, "object_geom_type")
        if object_geom_type is not None:
            result["object_geom_type"] = object_geom_type
        elif "object_box_size" in result:
            result["object_geom_type"] = "box"

    return result


def resolve_urdf_path(urdf_path: str) -> str:
    """Resolve URDF path, handling holosoma/data/ prefix."""
    return resolve_data_file_path(urdf_path)


def replay_viser(
    motion_path: str,
    urdf_path: str,
    object_urdf_path: str | None = None,
    object_mesh_path: str | None = None,
    port: int = 8012,
):
    motion = load_motion_npz(motion_path)
    fps = motion["fps"]
    body_names = motion["body_names"]
    body_pos_w = motion["body_pos_w"]
    body_quat_w = motion["body_quat_w"]
    total_frames = motion["total_frames"]

    resolved_urdf = resolve_urdf_path(urdf_path)
    logger.info(f"URDF: {resolved_urdf}")
    logger.info(f"Motion: {total_frames} frames, {fps:.1f} fps, {body_pos_w.shape[1]} bodies")

    # --- Setup viser ---
    viser = ViserHelper(port=port)
    setup_scene_lighting(viser)
    add_ground_grid(viser)

    viz = RobotAssetViser(viser, resolved_urdf, body_names, name_prefix="/robot")
    object_ref_viz = None
    fixed_scene_viz: list[StaticBoxViser] = []
    has_object_motion = "object_pos_w" in motion and "object_quat_w" in motion
    has_box_metadata = motion.get("object_geom_type") == "box" and "object_box_size" in motion
    has_mesh_metadata = "object_mesh_vertices" in motion and "object_mesh_faces" in motion
    for spec in motion.get("fixed_scene_objects", []):
        if spec.geom_type != "box":
            logger.warning(f"Skipping unsupported fixed scene geom_type '{spec.geom_type}' for '{spec.name}'.")
            continue
        box_viz = StaticBoxViser.from_fixed_scene(
            viser,
            object_name=spec.name,
            size_xyz=spec.size,
            prefix=f"/fixed_scene/{spec.name}",
        )
        box_viz.update(spec.position_w, spec.quat_wxyz[[1, 2, 3, 0]])
        fixed_scene_viz.append(box_viz)
    if fixed_scene_viz:
        logger.info(f"Loaded {len(fixed_scene_viz)} fixed scene boxes from motion metadata")

    if object_urdf_path:
        resolved_object_urdf = resolve_data_file_path(object_urdf_path)
        object_ref_viz = StaticUrdfViser(
            viser,
            resolved_object_urdf,
            prefix="/object_ref",
            color_override=(0.95, 0.55, 0.18),
        )
        logger.info(f"Object URDF (reference track): {resolved_object_urdf}")
    elif object_mesh_path:
        resolved_object_mesh = resolve_data_file_path(object_mesh_path)
        try:
            object_ref_viz = RigidMeshViser.from_mesh_path(
                viser,
                resolved_object_mesh,
                prefix="/object_ref",
                color=(0.92, 0.90, 0.82),
            )
            logger.info(f"Object mesh (reference track): {resolved_object_mesh}")
        except Exception as exc:
            logger.warning(f"Could not load object mesh '{resolved_object_mesh}': {exc}. Only the robot will be shown.")
    elif has_object_motion and has_mesh_metadata:
        object_ref_viz = RigidMeshViser(
            viser,
            motion["object_mesh_vertices"],
            motion["object_mesh_faces"],
            prefix="/object_ref",
            color=motion.get("object_mesh_color", (0.92, 0.90, 0.82)),
        )
        logger.info(
            "Object mesh (reference track): "
            f"vertices={motion['object_mesh_vertices'].shape[0]} faces={motion['object_mesh_faces'].shape[0]}"
        )
    elif has_object_motion and has_box_metadata:
        object_ref_viz = StaticBoxViser(
            viser,
            motion["object_box_size"],
            prefix="/object_ref",
            color=(0.95, 0.55, 0.18),
        )
        logger.info(f"Object box (reference track): size={motion['object_box_size'].tolist()}")
    elif has_object_motion:
        logger.warning(
            "Motion contains object tracks, but no --object-urdf or supported inline object geometry was provided. "
            "Only the robot will be shown."
        )

    logger.info(f"Open http://localhost:{port} in your browser")

    # --- Playback controls ---
    fps_widget = None
    frame_slider = None
    play_fps_widget = None
    playback_state = {
        "frame_idx": 0,
        "playing": True,
        "play_fps": max(float(fps), 1e-3),
        "reset_clock": False,
    }
    playback_lock = threading.Lock()
    slider_callback_suppressed = False

    def _clamp_frame(value: int | float) -> int:
        return max(0, min(total_frames - 1, int(round(value))))

    def _sync_frame_slider(value: int) -> None:
        nonlocal slider_callback_suppressed
        if frame_slider is None:
            return
        try:
            if int(frame_slider.value) == value:
                return
            slider_callback_suppressed = True
            frame_slider.value = value
        finally:
            slider_callback_suppressed = False

    try:
        import viser as viser_module

        gui = viser.server.gui
        frame_slider = gui.add_slider(
            "Frame",
            min=0,
            max=max(total_frames - 1, 0),
            step=1,
            initial_value=0,
            hint="Select the exact motion frame to display.",
            order=-4.0,
        )
        play_fps_widget = gui.add_number(
            "Play FPS",
            initial_value=float(fps),
            min=0.1,
            max=max(float(fps) * 4.0, 1.0),
            step=0.1,
            hint="Playback speed in motion frames per second.",
            order=-3.0,
        )
        play_button = gui.add_button(
            "Play",
            icon=viser_module.Icon.PLAYER_PLAY,
            hint="Resume motion playback.",
            order=-2.0,
        )
        pause_button = gui.add_button(
            "Pause",
            icon=viser_module.Icon.PLAYER_PAUSE,
            hint="Pause motion playback for frame inspection.",
            order=-1.9,
        )
        fps_widget = gui.add_text(
            "Playback",
            f"frame 0 / {total_frames - 1} | run --.- fps | play {fps:.1f} fps",
            disabled=True,
            order=-1.0,
        )

        @frame_slider.on_update
        def _on_frame_slider_update(_event) -> None:
            if slider_callback_suppressed:
                return
            with playback_lock:
                playback_state["frame_idx"] = _clamp_frame(frame_slider.value)
                playback_state["reset_clock"] = True

        @play_fps_widget.on_update
        def _on_play_fps_update(_event) -> None:
            with playback_lock:
                playback_state["play_fps"] = max(float(play_fps_widget.value), 0.1)
                playback_state["reset_clock"] = True

        @play_button.on_click
        def _on_play_clicked(_event) -> None:
            with playback_lock:
                playback_state["playing"] = True
                playback_state["reset_clock"] = True

        @pause_button.on_click
        def _on_pause_clicked(_event) -> None:
            with playback_lock:
                playback_state["playing"] = False

    except Exception:
        logger.warning("Could not create viser playback controls; falling back to native-FPS autoplay.", exc_info=True)

    # --- Replay loop ---
    rendered_frame_idx = None
    last_time = None
    fps_ema = None
    next_frame_time = time.perf_counter() + 1.0 / max(float(fps), 0.1)

    while True:
        now = time.perf_counter()
        with playback_lock:
            frame_idx = int(playback_state["frame_idx"])
            playing = bool(playback_state["playing"])
            play_fps = max(float(playback_state["play_fps"]), 0.1)
            reset_clock = bool(playback_state["reset_clock"])
            if reset_clock:
                playback_state["reset_clock"] = False

        if reset_clock:
            next_frame_time = now + 1.0 / play_fps
            last_time = None

        if playing and now >= next_frame_time:
            with playback_lock:
                frame_idx = (int(playback_state["frame_idx"]) + 1) % total_frames
                playback_state["frame_idx"] = frame_idx
                play_fps = max(float(playback_state["play_fps"]), 0.1)
            _sync_frame_slider(frame_idx)
            next_frame_time = now + 1.0 / play_fps

            if last_time is not None:
                frame_dt = now - last_time
                if frame_dt > 1e-6:
                    inst_fps = 1.0 / frame_dt
                    fps_ema = inst_fps if fps_ema is None else 0.8 * fps_ema + 0.2 * inst_fps
            last_time = now

        if rendered_frame_idx != frame_idx:
            viz.update(body_pos_w[frame_idx], body_quat_w[frame_idx])
            if object_ref_viz is not None and has_object_motion:
                object_ref_viz.update(motion["object_pos_w"][frame_idx], motion["object_quat_w"][frame_idx])
            rendered_frame_idx = frame_idx

        if fps_widget is not None:
            try:
                run_fps = "--.-" if fps_ema is None else f"{fps_ema:4.1f}"
                state_label = "play" if playing else "pause"
                fps_widget.value = (
                    f"frame {frame_idx} / {total_frames - 1} | {state_label} | "
                    f"run {run_fps} fps | play {play_fps:.1f} fps"
                )
            except Exception:
                pass

        sleep_dur = 0.03
        if playing:
            sleep_dur = max(0.0, min(0.03, next_frame_time - time.perf_counter()))
        time.sleep(sleep_dur)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay motion in viser (no IsaacSim)")
    parser.add_argument(
        "--motion",
        type=str,
        default="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz",
        help="Motion NPZ path (holosoma/data/... or absolute path)",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="holosoma/data/robots/g1/g1_29dof.urdf",
        help="URDF path (holosoma/data/... or absolute path)",
    )
    parser.add_argument(
        "--object-urdf",
        type=str,
        default=None,
        help="Optional rigid object URDF to render alongside the motion",
    )
    parser.add_argument(
        "--object-mesh",
        type=str,
        default=None,
        help="Optional rigid object mesh to render alongside the motion",
    )
    parser.add_argument("--port", type=int, default=8012, help="Viser port")
    args = parser.parse_args()

    replay_viser(
        args.motion,
        args.urdf,
        object_urdf_path=args.object_urdf,
        object_mesh_path=args.object_mesh,
        port=args.port,
    )


if __name__ == "__main__":
    main()
