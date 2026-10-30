from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import torch
from loguru import logger

from holosoma.config_types.command import (
    AdaptiveTimestepSamplerConfig,
    EvalFixedResetConfig,
    GoalPoseNoiseConfig,
    GoalOrientationConfig,
    HybridMotionConfig,
    HybridSizeRandomizationConfig,
    MotionConfig,
    NoiseToInitialPoseConfig,
    StepZeroRootNoiseConfig,
)
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.envs.wbt.hybrid_size_randomization import (
    build_fixed_hybrid_track_mask,
    collision_bounds_corners,
    correct_object_positions_for_ground,
    expand_size_scalars,
    sample_fixed_hybrid_size_scales,
)
from holosoma.managers.command.base import CommandTermBase
from holosoma.utils.file_cache import cached_open
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import (
    get_euler_xyz,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inverse,
    quat_mul,
    quat_normalize,
    quat_rotate_inverse,
    quat_to_angle_axis,
    slerp,
    yaw_quat,
)
from holosoma.utils.simulator_config import SimulatorType


#########################################################################################################
## MotionLoader and AdaptiveTimestepsSampler
#########################################################################################################
def derive_angular_velocity_from_quat_sequence(quat_sequence_xyzw: torch.Tensor, *, fps: float) -> torch.Tensor:
    """Estimate angular velocity from a quaternion sequence using clip-safe central differences."""
    if quat_sequence_xyzw.numel() == 0:
        return quat_sequence_xyzw.new_zeros((0, 3))
    if quat_sequence_xyzw.ndim != 2 or quat_sequence_xyzw.shape[-1] != 4:
        raise ValueError(
            "Expected quaternion sequence with shape (T, 4) in xyzw format, "
            f"got {tuple(quat_sequence_xyzw.shape)}"
        )

    num_steps = quat_sequence_xyzw.shape[0]
    if num_steps == 1:
        return quat_sequence_xyzw.new_zeros((1, 3))

    idx = torch.arange(num_steps, device=quat_sequence_xyzw.device, dtype=torch.long)
    prev_idx = torch.clamp(idx - 1, min=0)
    next_idx = torch.clamp(idx + 1, max=num_steps - 1)
    denom = ((next_idx - prev_idx).to(dtype=torch.float32) / float(fps)).clamp(min=1.0 / float(fps))

    quat_prev = quat_sequence_xyzw.index_select(0, prev_idx)
    quat_next = quat_sequence_xyzw.index_select(0, next_idx)
    quat_delta = quat_mul(quat_next, quat_inverse(quat_prev, w_last=True), w_last=True)
    _, axis_angle = quat_to_angle_axis(quat_delta)
    return axis_angle / denom.unsqueeze(-1)


def _build_terminal_hold_frames(frame: torch.Tensor, num_steps: int, *, zero_values: bool) -> torch.Tensor:
    """Create repeated or zeroed hold frames from a terminal state slice."""
    frame_shape = (num_steps,) + tuple(frame.shape)
    if zero_values:
        return frame.new_zeros(frame_shape)
    return frame.unsqueeze(0).repeat((num_steps,) + (1,) * frame.ndim)


# Object variants an environment does not track wait here, relative to their pose at the start of their group's
# first clip: x + OBJECT_PARK_OFFSET[0], y + OBJECT_PARK_OFFSET[1] + k * OBJECT_PARK_SPACING for variant k.
OBJECT_PARK_OFFSET = (0.0, 30.0)
OBJECT_PARK_SPACING = 3.0


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
    ):
        # Resolve the motion file path using importlib.resources
        motion_file = resolve_data_file_path(motion_file)

        logger.info(f"Loading motion file: {motion_file}")
        body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_npz(motion_file, device)
        body_indexes = self._get_index_of_a_in_b(robot_body_names, body_names_in_motion_data, device)
        joint_indexes = self._get_index_of_a_in_b(robot_joint_names, joint_names_in_motion_data, device)

        self._joint_indexes = joint_indexes
        self._body_indexes = body_indexes
        self.time_step_total = self._joint_pos.shape[0]

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)




    # Expected holosoma NPZ keys
    _REQUIRED_KEYS = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "body_names",
        "joint_names",
    }

    def _load_data_from_motion_npz(self, motion_file: str, device: str) -> tuple[list[str], list[str]]:
        with cached_open(motion_file, "rb") as f, np.load(f) as data:
            # Sanity check: warn if not in expected holosoma format
            keys = set(data.files)
            missing = self._REQUIRED_KEYS - keys
            if missing:
                logger.warning(
                    f"Motion NPZ '{motion_file}' is missing expected holosoma keys: {missing}. "
                    f"All motion data should be in holosoma format (with body_names, joint_names, "
                    f"and root DOFs in joint_pos). Convert from TML/BeyondMimic first."
                )
                raise ValueError(
                    f"Unsupported motion format in '{motion_file}': missing keys {missing}. "
                    f"Please convert to holosoma format."
                )

            self.fps = float(data["fps"].item())

            body_names = data["body_names"].tolist()
            joint_names = data["joint_names"].tolist()

            joint_pos_raw = data["joint_pos"]
            joint_vel_raw = data["joint_vel"]
            body_pos_w_raw = data["body_pos_w"]
            body_quat_w_raw = data["body_quat_w"]
            body_lin_vel_w_raw = data["body_lin_vel_w"]
            body_ang_vel_w_raw = data["body_ang_vel_w"]

            # Holosoma format: joint_pos includes root DOFs [xyz, wxyz] as first 7 values
            # joint_vel includes root velocity [vel_xyz, vel_wxyz] as first 6 values
            num_joint_cols = joint_pos_raw.shape[1]
            num_vel_cols = joint_vel_raw.shape[1]
            num_bodies = body_pos_w_raw.shape[1]

            if num_joint_cols != len(joint_names) + 7:
                logger.warning(
                    f"Unexpected joint_pos columns: got {num_joint_cols}, expected {len(joint_names) + 7} "
                    f"(= {len(joint_names)} joints + 7 root DOFs). File: {motion_file}"
                )
            if num_vel_cols != len(joint_names) + 6:
                logger.warning(
                    f"Unexpected joint_vel columns: got {num_vel_cols}, expected {len(joint_names) + 6} "
                    f"(= {len(joint_names)} joints + 6 root DOFs). File: {motion_file}"
                )
            if num_bodies != len(body_names):
                logger.warning(
                    f"Body count mismatch: body_pos_w has {num_bodies} bodies but body_names has "
                    f"{len(body_names)}. File: {motion_file}"
                )

            # Strip root DOFs
            self._joint_pos = torch.tensor(joint_pos_raw[:, 7:], dtype=torch.float32, device=device)
            self._joint_vel = torch.tensor(joint_vel_raw[:, 6:], dtype=torch.float32, device=device)

            assert len(joint_names) == self._joint_pos.shape[1], (
                f"Joint names ({len(joint_names)}) != joint_pos columns ({self._joint_pos.shape[1]}) in {motion_file}"
            )
            assert len(body_names) == body_pos_w_raw.shape[1], (
                f"Body names ({len(body_names)}) != body_pos_w bodies ({body_pos_w_raw.shape[1]}) in {motion_file}"
            )

            self._body_pos_w = torch.tensor(body_pos_w_raw, dtype=torch.float32, device=device)

            # NOTE: wxyz after loading from npz
            body_quat_w_wxyz = torch.tensor(body_quat_w_raw, dtype=torch.float32, device=device)  # This is wxyz
            self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]  # Change to xyzw

            self._body_lin_vel_w = torch.tensor(body_lin_vel_w_raw, dtype=torch.float32, device=device)
            self._body_ang_vel_w = torch.tensor(body_ang_vel_w_raw, dtype=torch.float32, device=device)

            # add object pos and quat
            object_keys = {"object_pos_w", "object_quat_w", "object_lin_vel_w"}
            present_object_keys = object_keys & keys
            if present_object_keys and present_object_keys != object_keys:
                missing_object_keys = object_keys - present_object_keys
                raise ValueError(
                    f"Motion NPZ '{motion_file}' has incomplete object data; missing {missing_object_keys}."
                )

            self.has_object = object_keys.issubset(keys)
            if self.has_object:
                self._object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
                # NOTE: wxyz after loading from npz
                object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
                self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]  # Change to xyzw
                self._object_lin_vel_w = torch.tensor(data["object_lin_vel_w"], dtype=torch.float32, device=device)
                if "object_ang_vel_w" in data:
                    self._object_ang_vel_w = torch.tensor(data["object_ang_vel_w"], dtype=torch.float32, device=device)
                else:
                    logger.warning(
                        f"Motion NPZ '{motion_file}' is missing object_ang_vel_w; deriving object angular velocity "
                        "from object quaternions."
                    )
                    self._object_ang_vel_w = derive_angular_velocity_from_quat_sequence(
                        self._object_quat_w,
                        fps=float(self.fps),
                    )
            else:
                self._object_pos_w = torch.zeros(0, 3, device=device)
                self._object_quat_w = torch.zeros(0, 4, device=device)
                self._object_lin_vel_w = torch.zeros(0, 3, device=device)
                self._object_ang_vel_w = torch.zeros(0, 3, device=device)
        return body_names, joint_names

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel[:, self._joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        return self._object_ang_vel_w[:]

    @property
    def num_motions(self) -> int:
        return 1

    @property
    def motion_start_idx(self) -> torch.Tensor:
        return torch.tensor([0], dtype=torch.long, device=self._joint_pos.device)

    @property
    def motion_end_idx(self) -> torch.Tensor:
        return torch.tensor([self.time_step_total], dtype=torch.long, device=self._joint_pos.device)

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MotionLoader:
        """Merge interpolated segments with motion data, mutating this MotionLoader."""
        concat_targets = [
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                    ("object_ang_vel", "_object_ang_vel_w"),
                ]
            )

        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            tensors = (segments[seg_key], existing) if prepend else (existing, segments[seg_key])
            setattr(self, attr_name, torch.cat(tensors, dim=0))

        self.time_step_total = self._joint_pos.shape[0]
        return self

    def append_terminal_pose_hold(self, num_steps: int) -> MotionLoader:
        """Append fixed terminal hold frames that keep pose targets and zero velocities."""
        if num_steps <= 0:
            return self

        append_targets = [
            ("_joint_pos", False),
            ("_joint_vel", True),
            ("_body_pos_w", False),
            ("_body_quat_w", False),
            ("_body_lin_vel_w", True),
            ("_body_ang_vel_w", True),
        ]
        if self.has_object:
            append_targets.extend(
                [
                    ("_object_pos_w", False),
                    ("_object_quat_w", False),
                    ("_object_lin_vel_w", True),
                    ("_object_ang_vel_w", True),
                ]
            )

        for attr_name, zero_values in append_targets:
            existing = getattr(self, attr_name)
            terminal_frame = existing[-1]
            hold_frames = _build_terminal_hold_frames(terminal_frame, num_steps, zero_values=zero_values)
            setattr(self, attr_name, torch.cat((existing, hold_frames), dim=0))

        self.time_step_total = self._joint_pos.shape[0]
        return self


class MultiMotionLoader:
    """Loads multiple NPZ motion files from a directory and concatenates them at runtime.

    Tracks per-motion boundaries so environments can sample within individual clips.
    Compatible with the same interface as MotionLoader.
    """

    def __init__(
        self,
        motion_dir: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
        object_groups: bool = False,
    ):
        # Support comma-separated directories for combining multiple datasets
        dirs = [d.strip() for d in motion_dir.split(",")]
        motion_files = []
        file_groups = []  # index of the directory each file came from (its object group)
        for group, d in enumerate(dirs):
            expanded = resolve_data_file_path(os.path.expanduser(d))
            path = Path(expanded)
            files = sorted(str(p) for p in path.glob("*.npz"))
            logger.info(f"MultiMotionLoader: found {len(files)} .npz files in {expanded}")
            motion_files.extend(files)
            file_groups.extend([group] * len(files))
        assert len(motion_files) > 0, f"No .npz files found in {motion_dir}"
        logger.info(f"MultiMotionLoader: loading {len(motion_files)} total motion files")

        loaders = []
        groups = []
        skipped = 0
        for mf, group in zip(motion_files, file_groups):
            try:
                loader = MotionLoader(mf, robot_body_names, robot_joint_names, device=device)
                loaders.append(loader)
                groups.append(group)
            except (KeyError, AssertionError, ValueError) as e:  # noqa: PERF203
                # Skip files with incompatible format (e.g., missing body_names, wrong body count)
                skipped += 1
                if skipped <= 3:
                    logger.warning(f"MultiMotionLoader: skipping {mf}: {e}")
        if skipped > 3:
            logger.warning(f"MultiMotionLoader: skipped {skipped} files total due to format issues")
        assert len(loaders) > 0, f"No compatible motion files found (skipped {skipped})"

        # Track per-motion boundaries
        lengths = [loader.time_step_total for loader in loaders]
        cumulative = torch.tensor(lengths, dtype=torch.long, device=device).cumsum(dim=0)
        self._motion_start_idx = torch.cat([torch.tensor([0], dtype=torch.long, device=device), cumulative[:-1]])
        self._motion_end_idx = cumulative
        self._num_motions = len(loaders)
        # Object group of each motion (all 0 unless object_groups); -1 marks added transition segments.
        self.num_groups = len(dirs) if object_groups else 1
        self._motion_group = torch.tensor(groups if object_groups else [0] * len(loaders), dtype=torch.long,
                                          device=device)

        # Concatenate all motion data. Body tensors are normalized to the simulator robot body subset first:
        # object-reference clips may include an extra object body such as largebox_link in body_* arrays.
        self._joint_pos = torch.cat([ld._joint_pos for ld in loaders], dim=0)
        self._joint_vel = torch.cat([ld._joint_vel for ld in loaders], dim=0)
        self._body_pos_w = torch.cat([ld.body_pos_w for ld in loaders], dim=0)
        self._body_quat_w = torch.cat([ld.body_quat_w for ld in loaders], dim=0)
        self._body_lin_vel_w = torch.cat([ld.body_lin_vel_w for ld in loaders], dim=0)
        self._body_ang_vel_w = torch.cat([ld.body_ang_vel_w for ld in loaders], dim=0)

        # Use indexes from first loader (all loaders share the same robot)
        self._joint_indexes = loaders[0]._joint_indexes
        self._body_indexes = torch.arange(len(robot_body_names), dtype=torch.long, device=device)
        fps_values = [float(loader.fps) for loader in loaders]
        if any(not np.isclose(fps, fps_values[0]) for fps in fps_values[1:]):
            raise ValueError(
                "MultiMotionLoader requires all motion clips to share the same fps. "
                f"Found fps values {fps_values}."
            )
        self.fps = fps_values[0]
        self.time_step_total = self._joint_pos.shape[0]

        object_flags = [loader.has_object for loader in loaders]
        if any(flag != object_flags[0] for flag in object_flags[1:]):
            raise ValueError(
                "MultiMotionLoader requires all motion clips to either include or omit object trajectory data."
            )
        self.has_object = object_flags[0]
        if self.has_object:
            self._object_pos_w = torch.cat([loader._object_pos_w for loader in loaders], dim=0)
            self._object_quat_w = torch.cat([loader._object_quat_w for loader in loaders], dim=0)
            self._object_lin_vel_w = torch.cat([loader._object_lin_vel_w for loader in loaders], dim=0)
            self._object_ang_vel_w = torch.cat([loader._object_ang_vel_w for loader in loaders], dim=0)
        else:
            self._object_pos_w = torch.zeros(0, 3, device=device)
            self._object_quat_w = torch.zeros(0, 4, device=device)
            self._object_lin_vel_w = torch.zeros(0, 3, device=device)
            self._object_ang_vel_w = torch.zeros(0, 3, device=device)

        logger.info(
            "MultiMotionLoader: {} motions, {} total frames, has_object={}",
            self._num_motions,
            self.time_step_total,
            self.has_object,
        )

    @property
    def num_motions(self) -> int:
        return self._num_motions

    @property
    def motion_start_idx(self) -> torch.Tensor:
        return self._motion_start_idx

    @property
    def motion_end_idx(self) -> torch.Tensor:
        return self._motion_end_idx

    @property
    def motion_group(self) -> torch.Tensor:
        return self._motion_group

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel[:, self._joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        return self._object_ang_vel_w[:]

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MultiMotionLoader:
        """Merge interpolated segments with motion data, mutating this MultiMotionLoader."""
        concat_targets = [
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                    ("object_ang_vel", "_object_ang_vel_w"),
                ]
            )

        added_frames = 0
        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            tensors = (segments[seg_key], existing) if prepend else (existing, segments[seg_key])
            setattr(self, attr_name, torch.cat(tensors, dim=0))
            if added_frames == 0:
                added_frames = segments[seg_key].shape[0]

        # Update boundaries — shift all motion boundaries if prepending
        if prepend:
            self._motion_start_idx = self._motion_start_idx + added_frames
            self._motion_end_idx = self._motion_end_idx + added_frames
            dev = self._motion_start_idx.device
            self._motion_start_idx = torch.cat(
                [torch.tensor([0], dtype=torch.long, device=dev), self._motion_start_idx]
            )
            self._motion_end_idx = torch.cat(
                [torch.tensor([added_frames], dtype=torch.long, device=dev), self._motion_end_idx]
            )
            self._motion_group = torch.cat([torch.tensor([-1], dtype=torch.long, device=dev), self._motion_group])
        else:
            old_total = self.time_step_total
            dev = self._motion_start_idx.device
            self._motion_start_idx = torch.cat(
                [self._motion_start_idx, torch.tensor([old_total], dtype=torch.long, device=dev)]
            )
            self._motion_end_idx = torch.cat(
                [self._motion_end_idx, torch.tensor([old_total + added_frames], dtype=torch.long, device=dev)]
            )
            self._motion_group = torch.cat([self._motion_group, torch.tensor([-1], dtype=torch.long, device=dev)])
        self.time_step_total = self._joint_pos.shape[0]
        self._num_motions = len(self._motion_start_idx)
        return self

    def append_terminal_pose_hold(self, num_steps: int) -> MultiMotionLoader:
        """Append fixed terminal hold frames to each motion clip without changing clip count."""
        if num_steps <= 0:
            return self

        motion_ranges = list(zip(self._motion_start_idx.cpu().tolist(), self._motion_end_idx.cpu().tolist()))
        append_targets = [
            ("_joint_pos", False),
            ("_joint_vel", True),
            ("_body_pos_w", False),
            ("_body_quat_w", False),
            ("_body_lin_vel_w", True),
            ("_body_ang_vel_w", True),
        ]
        if self.has_object:
            append_targets.extend(
                [
                    ("_object_pos_w", False),
                    ("_object_quat_w", False),
                    ("_object_lin_vel_w", True),
                    ("_object_ang_vel_w", True),
                ]
            )

        for attr_name, zero_values in append_targets:
            existing = getattr(self, attr_name)
            segments = []
            for start, end in motion_ranges:
                clip = existing[start:end]
                segments.append(clip)
                terminal_frame = clip[-1]
                hold_frames = _build_terminal_hold_frames(terminal_frame, num_steps, zero_values=zero_values)
                segments.append(hold_frames)
            setattr(self, attr_name, torch.cat(segments, dim=0))

        num_motions = len(motion_ranges)
        device = self._motion_start_idx.device
        hold_offsets = torch.arange(num_motions, device=device, dtype=torch.long) * num_steps
        self._motion_start_idx = self._motion_start_idx + hold_offsets
        self._motion_end_idx = self._motion_end_idx + hold_offsets + num_steps
        self.time_step_total = self._joint_pos.shape[0]
        self._num_motions = num_motions
        return self




class AdaptiveTimestepsSampler:
    """Dexhand-style adaptive sampler with per-group, per-motion, per-bin difficulty tables."""

    def __init__(
        self,
        motion_start_idx: torch.Tensor,
        motion_end_idx: torch.Tensor,
        device: str,
        env_dt: float,
        group_names: list[str],
        *,
        bin_width_s: float = 1.0,
        ema_alpha: float = 0.01,
        bin_uniform_mix: float = 0.1,
        motion_uniform_mix: float = 0.1,
        score_power: float = 1.0,
        min_segment_steps: int = 10,
    ):
        self.device = device
        self.motion_start_idx = motion_start_idx.to(device=self.device, dtype=torch.long).clone()
        self.motion_end_idx = motion_end_idx.to(device=self.device, dtype=torch.long).clone()
        self.motion_lengths = self.motion_end_idx - self.motion_start_idx
        if torch.any(self.motion_lengths <= 0):
            raise ValueError("AdaptiveTimestepsSampler requires every motion clip to have at least one frame.")

        if env_dt <= 0.0:
            raise ValueError(f"AdaptiveTimestepsSampler requires env_dt > 0, got {env_dt}")
        if bin_width_s <= 0.0:
            raise ValueError(f"AdaptiveTimestepSamplerConfig.bin_width_s must be > 0, got {bin_width_s}")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"AdaptiveTimestepSamplerConfig.ema_alpha must be in (0, 1], got {ema_alpha}")
        if not 0.0 <= bin_uniform_mix <= 1.0:
            raise ValueError(
                f"AdaptiveTimestepSamplerConfig.bin_uniform_mix must be in [0, 1], got {bin_uniform_mix}"
            )
        if not 0.0 <= motion_uniform_mix <= 1.0:
            raise ValueError(
                f"AdaptiveTimestepSamplerConfig.motion_uniform_mix must be in [0, 1], got {motion_uniform_mix}"
            )
        if score_power < 0.0:
            raise ValueError(f"AdaptiveTimestepSamplerConfig.score_power must be >= 0, got {score_power}")
        if min_segment_steps < 0:
            raise ValueError(
                f"AdaptiveTimestepSamplerConfig.min_segment_steps must be >= 0, got {min_segment_steps}"
            )
        if len(group_names) == 0:
            raise ValueError("AdaptiveTimestepsSampler requires at least one sampler group.")

        self.group_names = list(group_names)
        self.group_name_to_index = {name: idx for idx, name in enumerate(self.group_names)}
        self.num_groups = len(self.group_names)
        self.num_motions = int(self.motion_start_idx.numel())
        self.bin_width_s = float(bin_width_s)
        self.bin_width_steps = max(1, int(round(bin_width_s / env_dt)))
        self.ema_alpha = float(ema_alpha)
        self.bin_uniform_mix = float(bin_uniform_mix)
        self.motion_uniform_mix = float(motion_uniform_mix)
        self.score_power = float(score_power)
        self.min_segment_steps = int(min_segment_steps)

        self.motion_num_bins = self._num_bins_for_motion_lengths(self.motion_lengths, self.bin_width_steps)
        self.max_num_bins = int(self.motion_num_bins.max().item())
        bin_ids = torch.arange(self.max_num_bins, device=self.device, dtype=torch.long)
        self.valid_bin_mask = bin_ids.unsqueeze(0) < self.motion_num_bins.unsqueeze(1)
        self.bin_local_start_steps = bin_ids.unsqueeze(0).expand(self.num_motions, -1) * self.bin_width_steps
        bin_local_end_steps = torch.minimum(
            self.bin_local_start_steps + self.bin_width_steps,
            self.motion_lengths.unsqueeze(1),
        )
        last_bin_mask = bin_ids.unsqueeze(0) == (self.motion_num_bins.unsqueeze(1) - 1)
        self.bin_local_end_steps = torch.where(
            self.valid_bin_mask & last_bin_mask,
            self.motion_lengths.unsqueeze(1),
            bin_local_end_steps,
        )

        self.init_buffers()
        self.metrics: dict[str, torch.Tensor] = {}

    @staticmethod
    def _num_bins_for_motion_lengths(motion_lengths: torch.Tensor, bin_width_steps: int) -> torch.Tensor:
        motion_num_bins = torch.div(
            motion_lengths + (bin_width_steps - 1),
            bin_width_steps,
            rounding_mode="floor",
        )
        last_bin_lengths = motion_lengths - (motion_num_bins - 1) * bin_width_steps
        merge_short_final_bins = (motion_num_bins > 1) & (last_bin_lengths * 2 < bin_width_steps)
        return motion_num_bins - merge_short_final_bins.to(dtype=motion_num_bins.dtype)

    @staticmethod
    def num_bins_for_motion_length(motion_length: int, bin_width_steps: int) -> int:
        if motion_length <= 0:
            raise ValueError(f"motion_length must be > 0, got {motion_length}")
        if bin_width_steps <= 0:
            raise ValueError(f"bin_width_steps must be > 0, got {bin_width_steps}")

        num_bins = (motion_length + bin_width_steps - 1) // bin_width_steps
        last_bin_length = motion_length - (num_bins - 1) * bin_width_steps
        if num_bins > 1 and last_bin_length * 2 < bin_width_steps:
            num_bins -= 1
        return num_bins

    def init_buffers(self):
        shape = (self.num_groups, self.num_motions, self.max_num_bins)
        valid_bins = self.valid_bin_mask.unsqueeze(0).expand(shape)
        self.bin_fail_ema = torch.where(
            valid_bins,
            torch.ones(shape, dtype=torch.float32, device=self.device),
            torch.zeros(shape, dtype=torch.float32, device=self.device),
        )
        self.pending_visits = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.pending_failures = torch.zeros(shape, dtype=torch.float32, device=self.device)

    def _segment_end_counts(
        self,
        *,
        group_ids: torch.Tensor,
        motion_ids: torch.Tensor,
        start_bins: torch.Tensor,
        segment_step_counts: torch.Tensor,
        terminal_time_steps: torch.Tensor,
        failure_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visits = torch.zeros_like(self.bin_fail_ema)
        failures = torch.zeros_like(self.bin_fail_ema)
        if group_ids.numel() == 0:
            return visits, failures

        group_ids = group_ids.to(device=self.device, dtype=torch.long)
        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
        start_bins = start_bins.to(device=self.device, dtype=torch.long)
        segment_step_counts = segment_step_counts.to(device=self.device, dtype=torch.long)
        terminal_time_steps = terminal_time_steps.to(device=self.device, dtype=torch.long)
        failure_mask = failure_mask.to(device=self.device, dtype=torch.bool)
        terminal_bins = self.time_step_to_bin(motion_ids, terminal_time_steps)

        for idx in range(group_ids.numel()):
            if int(segment_step_counts[idx].item()) < self.min_segment_steps:
                continue

            group_id = int(group_ids[idx].item())
            motion_id = int(motion_ids[idx].item())
            num_bins = int(self.motion_num_bins[motion_id].item())
            if num_bins <= 0:
                continue

            start_bin = int(torch.clamp(start_bins[idx], min=0, max=num_bins - 1).item())
            terminal_bin = int(torch.clamp(terminal_bins[idx], min=0, max=num_bins - 1).item())
            end_bin = min(max(terminal_bin, start_bin), num_bins - 1)

            visits[group_id, motion_id, start_bin : end_bin + 1] += 1.0
            if bool(failure_mask[idx].item()) and 0 <= terminal_bin < num_bins:
                failures[group_id, motion_id, terminal_bin] += 1.0

        return visits, failures

    def _apply_counts(self, visits: torch.Tensor, failures: torch.Tensor) -> None:
        update_mask = visits > 0.0
        if not update_mask.any():
            return

        failure_rate = torch.zeros_like(self.bin_fail_ema)
        failure_rate[update_mask] = failures[update_mask] / visits[update_mask]
        self.bin_fail_ema[update_mask] = (
            (1.0 - self.ema_alpha) * self.bin_fail_ema[update_mask] + self.ema_alpha * failure_rate[update_mask]
        )

    def time_step_to_bin(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
        time_steps = time_steps.to(device=self.device, dtype=torch.long)
        local_steps = time_steps - self.motion_start_idx[motion_ids]
        local_steps = torch.clamp(local_steps, min=0)
        max_local = self.motion_lengths[motion_ids] - 1
        local_steps = torch.minimum(local_steps, max_local)
        bin_ids = torch.div(local_steps, self.bin_width_steps, rounding_mode="floor")
        return torch.minimum(bin_ids, self.motion_num_bins[motion_ids] - 1)

    def _group_first_frame_probs(self, group_first_frame_probs: torch.Tensor | None) -> torch.Tensor:
        if group_first_frame_probs is None:
            return torch.zeros(self.num_groups, dtype=torch.float32, device=self.device)
        probs = group_first_frame_probs.to(device=self.device, dtype=torch.float32)
        if probs.numel() != self.num_groups:
            raise ValueError(
                f"AdaptiveTimestepsSampler expected {self.num_groups} group first-frame probabilities, "
                f"got {probs.numel()}."
            )
        if torch.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError("AdaptiveTimestepsSampler group first-frame probabilities must be in [0, 1].")
        return probs

    def _mix_with_uniform(self, scores: torch.Tensor, uniform_mix: float) -> torch.Tensor:
        if scores.numel() == 0:
            return scores
        scores = torch.clamp(scores, min=0.0)
        if self.score_power != 1.0:
            scores = scores.pow(self.score_power)
        uniform = torch.full_like(scores, 1.0 / float(scores.numel()))
        total = scores.sum()
        if total <= 0.0:
            return uniform
        probs = scores / total
        return (1.0 - uniform_mix) * probs + uniform_mix * uniform

    def _mix_with_uniform_masked(self, scores: torch.Tensor, uniform_mix: float, valid_mask: torch.Tensor) -> torch.Tensor:
        if scores.numel() == 0:
            return scores
        scores = torch.clamp(scores, min=0.0)
        valid_mask = valid_mask.to(device=self.device, dtype=torch.bool)
        scores = scores * valid_mask.to(dtype=scores.dtype)
        if self.score_power != 1.0:
            scores = scores.pow(self.score_power)

        valid_counts = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        uniform = valid_mask.to(dtype=scores.dtype) / valid_counts.to(dtype=scores.dtype)
        total = scores.sum(dim=-1, keepdim=True)
        probs = torch.where(total > 0.0, scores / total.clamp_min(torch.finfo(scores.dtype).eps), uniform)
        return (1.0 - uniform_mix) * probs + uniform_mix * uniform

    def motion_probabilities(self, group_id: int, candidate_motion_ids: torch.Tensor) -> torch.Tensor:
        candidate_motion_ids = candidate_motion_ids.to(device=self.device, dtype=torch.long)
        if candidate_motion_ids.numel() == 0:
            raise ValueError("AdaptiveTimestepsSampler.motion_probabilities requires a non-empty motion pool.")

        fail_ema = self.bin_fail_ema[group_id, candidate_motion_ids].clamp(min=0.0)
        valid_mask = self.valid_bin_mask[candidate_motion_ids]
        motion_scores = (fail_ema * valid_mask.float()).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1).float()
        return self._mix_with_uniform(motion_scores, self.motion_uniform_mix)

    def _bin_probabilities_for_motion_ids(self, group_id: int, motion_ids: torch.Tensor) -> torch.Tensor:
        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
        fail_ema = self.bin_fail_ema[group_id, motion_ids].clamp(min=0.0)
        valid_mask = self.valid_bin_mask[motion_ids]
        return self._mix_with_uniform_masked(fail_ema, self.bin_uniform_mix, valid_mask)

    def bin_probabilities(self, group_id: int, motion_id: int) -> torch.Tensor:
        all_probs = self._bin_probabilities_for_motion_ids(
            group_id,
            torch.tensor([motion_id], dtype=torch.long, device=self.device),
        ).squeeze(0)
        return all_probs[self.valid_bin_mask[motion_id]]

    def sample_from_motion_pool(
        self,
        group_id: int,
        candidate_motion_ids: torch.Tensor,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_motion_ids = candidate_motion_ids.to(device=self.device, dtype=torch.long)
        if num_samples <= 0:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty, empty

        motion_probs = self.motion_probabilities(group_id, candidate_motion_ids)
        sampled_motion_local = torch.multinomial(motion_probs, num_samples, replacement=True)
        sampled_motion_ids = candidate_motion_ids[sampled_motion_local]
        sampled_bin_probs = self._bin_probabilities_for_motion_ids(group_id, sampled_motion_ids)
        sampled_bins = torch.multinomial(sampled_bin_probs, 1, replacement=True).squeeze(-1)

        local_start = self.bin_local_start_steps[sampled_motion_ids, sampled_bins]
        local_end = self.bin_local_end_steps[sampled_motion_ids, sampled_bins]
        local_width = (local_end - local_start).clamp_min(1)
        local_offset = torch.floor(torch.rand(num_samples, device=self.device) * local_width.float()).to(
            dtype=torch.long
        )
        sampled_time_steps = self.motion_start_idx[sampled_motion_ids] + local_start + local_offset

        return sampled_motion_ids, sampled_bins, sampled_time_steps

    def record_segment_end(
        self,
        *,
        group_ids: torch.Tensor,
        motion_ids: torch.Tensor,
        start_bins: torch.Tensor,
        segment_step_counts: torch.Tensor,
        terminal_time_steps: torch.Tensor,
        failure_mask: torch.Tensor,
    ) -> None:
        visits, failures = self._segment_end_counts(
            group_ids=group_ids,
            motion_ids=motion_ids,
            start_bins=start_bins,
            segment_step_counts=segment_step_counts,
            terminal_time_steps=terminal_time_steps,
            failure_mask=failure_mask,
        )
        self._apply_counts(visits, failures)

    def accumulate_segment_end(
        self,
        *,
        group_ids: torch.Tensor,
        motion_ids: torch.Tensor,
        start_bins: torch.Tensor,
        segment_step_counts: torch.Tensor,
        terminal_time_steps: torch.Tensor,
        failure_mask: torch.Tensor,
    ) -> None:
        visits, failures = self._segment_end_counts(
            group_ids=group_ids,
            motion_ids=motion_ids,
            start_bins=start_bins,
            segment_step_counts=segment_step_counts,
            terminal_time_steps=terminal_time_steps,
            failure_mask=failure_mask,
        )
        self.pending_visits += visits
        self.pending_failures += failures

    def has_pending_updates(self) -> bool:
        return bool(self.pending_visits.any().item())

    def apply_pending_updates(self) -> None:
        if not self.has_pending_updates():
            return
        self._apply_counts(self.pending_visits, self.pending_failures)
        self.pending_visits.zero_()
        self.pending_failures.zero_()

    def _normalized_group_priors(self, group_priors: torch.Tensor | None) -> torch.Tensor:
        if group_priors is None:
            return torch.full((self.num_groups,), 1.0 / float(self.num_groups), dtype=torch.float32, device=self.device)
        priors = group_priors.to(device=self.device, dtype=torch.float32).clamp_min(0.0)
        if priors.numel() != self.num_groups:
            raise ValueError(
                f"AdaptiveTimestepsSampler expected {self.num_groups} group priors, got {priors.numel()}."
            )
        total = priors.sum()
        if total <= 0.0:
            return torch.zeros(self.num_groups, dtype=torch.float32, device=self.device)
        return priors / total

    def flattened_sampling_distribution(
        self,
        *,
        group_priors: torch.Tensor | None,
        candidate_motion_ids_by_group: dict[int, torch.Tensor],
        group_first_frame_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, int]]]:
        priors = self._normalized_group_priors(group_priors)
        first_frame_probs = self._group_first_frame_probs(group_first_frame_probs)
        states: list[tuple[int, int, int]] = []
        probs = []

        for group_id in range(self.num_groups):
            group_prior = float(priors[group_id].item())
            group_non_first_frame_mass = group_prior * (1.0 - float(first_frame_probs[group_id].item()))
            if group_non_first_frame_mass <= 0.0:
                continue
            candidate_motion_ids = candidate_motion_ids_by_group.get(group_id)
            if candidate_motion_ids is None or candidate_motion_ids.numel() == 0:
                continue
            candidate_motion_ids = candidate_motion_ids.to(device=self.device, dtype=torch.long)
            motion_probs = self.motion_probabilities(group_id, candidate_motion_ids)
            for motion_prob, motion_id_tensor in zip(motion_probs, candidate_motion_ids):
                motion_id = int(motion_id_tensor.item())
                bin_probs = self.bin_probabilities(group_id, motion_id)
                for bin_id, bin_prob in enumerate(bin_probs.tolist()):
                    states.append((group_id, motion_id, bin_id))
                    probs.append(group_non_first_frame_mass * float(motion_prob.item()) * float(bin_prob))

        if not probs:
            empty = torch.zeros(0, dtype=torch.float32, device=self.device)
            return empty, empty, states

        raw_distribution = torch.tensor(probs, dtype=torch.float32, device=self.device)
        distribution = raw_distribution.clone()
        total = distribution.sum()
        if total > 0.0:
            distribution = distribution / total
        return raw_distribution, distribution, states

    def valid_fail_ema(
        self,
        *,
        group_priors: torch.Tensor | None,
        candidate_motion_ids_by_group: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        priors = self._normalized_group_priors(group_priors)
        valid_values: list[torch.Tensor] = []

        for group_id in range(self.num_groups):
            if float(priors[group_id].item()) <= 0.0:
                continue
            candidate_motion_ids = candidate_motion_ids_by_group.get(group_id)
            if candidate_motion_ids is None or candidate_motion_ids.numel() == 0:
                continue
            candidate_motion_ids = candidate_motion_ids.to(device=self.device, dtype=torch.long)
            fail_ema = self.bin_fail_ema[group_id, candidate_motion_ids]
            valid_mask = self.valid_bin_mask[candidate_motion_ids]
            group_values = fail_ema[valid_mask]
            if group_values.numel() > 0:
                valid_values.append(group_values)

        if not valid_values:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        return torch.cat(valid_values)

    def get_stats(
        self,
        *,
        group_priors: torch.Tensor | None,
        candidate_motion_ids_by_group: dict[int, torch.Tensor],
        group_first_frame_probs: torch.Tensor | None = None,
    ) -> None:
        raw_distribution, distribution, states = self.flattened_sampling_distribution(
            group_priors=group_priors,
            candidate_motion_ids_by_group=candidate_motion_ids_by_group,
            group_first_frame_probs=group_first_frame_probs,
        )
        valid_fail_ema = self.valid_fail_ema(
            group_priors=group_priors,
            candidate_motion_ids_by_group=candidate_motion_ids_by_group,
        )

        if valid_fail_ema.numel() == 0:
            zero = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            self.metrics["sampling_entropy"] = zero
            self.metrics["sampling_effective_bins"] = zero
            self.metrics["sampling_top1_prob"] = zero
            self.metrics["sampling_top1_bin"] = zero
            self.metrics["sampling_expected_progress"] = zero
            self.metrics["sampling_motion_entropy"] = zero
            self.metrics["sampling_motion_top1_prob"] = zero
            self.metrics["sampling_motion_top1_id"] = zero
            self.metrics["fail_ema_mean"] = zero
            self.metrics["fail_ema_min"] = zero
            self.metrics["fail_ema_max"] = zero
            self.metrics["fail_ema_std"] = zero
            self.metrics["overall_bin_prob_max"] = zero
            self.metrics["overall_bin_prob_min"] = zero
            self.metrics["overall_bin_prob_effective_bins"] = zero
            self.metrics["overall_bin_prob_entropy_norm"] = zero
            self.metrics["overall_bin_prob_expected_progress"] = zero
            return

        self.metrics["fail_ema_mean"] = valid_fail_ema.mean()
        self.metrics["fail_ema_min"] = valid_fail_ema.min()
        self.metrics["fail_ema_max"] = valid_fail_ema.max()
        self.metrics["fail_ema_std"] = valid_fail_ema.std(unbiased=False)

        if distribution.numel() == 0:
            zero = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            self.metrics["sampling_entropy"] = zero
            self.metrics["sampling_effective_bins"] = zero
            self.metrics["sampling_top1_prob"] = zero
            self.metrics["sampling_top1_bin"] = zero
            self.metrics["sampling_expected_progress"] = zero
            self.metrics["sampling_motion_entropy"] = zero
            self.metrics["sampling_motion_top1_prob"] = zero
            self.metrics["sampling_motion_top1_id"] = zero
            self.metrics["overall_bin_prob_max"] = zero
            self.metrics["overall_bin_prob_min"] = zero
            self.metrics["overall_bin_prob_effective_bins"] = zero
            self.metrics["overall_bin_prob_entropy_norm"] = zero
            self.metrics["overall_bin_prob_expected_progress"] = zero
            return

        H = -(distribution * (distribution + 1.0e-12).log()).sum()
        H_norm = H / np.log(distribution.numel()) if distribution.numel() > 1 else distribution.new_tensor(0.0)
        effective_bins = torch.exp(H)
        overall_prob_max, imax = raw_distribution.max(dim=0)
        overall_prob_min = raw_distribution.min()
        _, motion_id, bin_id = states[int(imax.item())]
        top1_bin = self.bin_local_start_steps[motion_id, bin_id].float() / self.motion_lengths[motion_id].float()

        normalized_progress = []
        for _, motion_idx, cur_bin_id in states:
            start_step = float(self.bin_local_start_steps[motion_idx, cur_bin_id].item())
            end_step = float(self.bin_local_end_steps[motion_idx, cur_bin_id].item())
            center_step = 0.5 * (start_step + max(end_step - 1.0, start_step))
            motion_len = max(float(self.motion_lengths[motion_idx].item()), 1.0)
            normalized_progress.append(min(center_step / motion_len, 1.0))
        normalized_progress_tensor = torch.tensor(normalized_progress, dtype=torch.float32, device=self.device)
        expected_progress = (distribution * normalized_progress_tensor).sum()

        motion_distribution = torch.zeros(self.num_motions, dtype=torch.float32, device=self.device)
        for prob, (_, motion_idx, _) in zip(distribution.tolist(), states):
            motion_distribution[motion_idx] += float(prob)
        active_motion_mask = motion_distribution > 0.0
        active_motion_count = int(active_motion_mask.sum().item())
        motion_entropy = distribution.new_tensor(0.0)
        if active_motion_count > 1:
            motion_probs = motion_distribution[active_motion_mask]
            motion_entropy = -(motion_probs * (motion_probs + 1.0e-12).log()).sum() / np.log(active_motion_count)
        motion_top1_prob, motion_top1_id = motion_distribution.max(dim=0)

        self.metrics["sampling_top1_bin"] = top1_bin
        self.metrics["sampling_motion_entropy"] = motion_entropy
        self.metrics["sampling_motion_top1_prob"] = motion_top1_prob
        self.metrics["sampling_motion_top1_id"] = motion_top1_id.float()
        self.metrics["overall_bin_prob_max"] = overall_prob_max
        self.metrics["overall_bin_prob_min"] = overall_prob_min
        self.metrics["overall_bin_prob_effective_bins"] = effective_bins
        self.metrics["overall_bin_prob_entropy_norm"] = H_norm
        self.metrics["overall_bin_prob_expected_progress"] = expected_progress

        # Backward-compatible aliases for existing dashboards.
        self.metrics["sampling_entropy"] = H_norm
        self.metrics["sampling_effective_bins"] = effective_bins
        self.metrics["sampling_top1_prob"] = overall_prob_max
        self.metrics["sampling_top1_bin"] = top1_bin
        self.metrics["sampling_expected_progress"] = expected_progress


#########################################################################################################
## Helper functions
#########################################################################################################
FAKE_BODY_NAME_ALIASES: dict[str, str] = {
    # Fake foot contact bodies are authored in the URDF purely for height computation.
    # They do not exist in the motion-capture dataset, so we alias them back to the
    # closest real body when indexing into motion data. These are not actually used in training.
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}


def get_filtered_body_names(body_list: List[str], pattern: str) -> List[str]:
    return [body_name for body_name in body_list if re.match(pattern, body_name)]


def _parse_step_zero_root_noise_cfg(raw_cfg: Any) -> StepZeroRootNoiseConfig | None:
    if raw_cfg is None or isinstance(raw_cfg, StepZeroRootNoiseConfig):
        return raw_cfg
    return StepZeroRootNoiseConfig(**dict(raw_cfg))


def _parse_goal_pose_noise_cfg(raw_cfg: Any) -> GoalPoseNoiseConfig:
    if isinstance(raw_cfg, GoalPoseNoiseConfig):
        return raw_cfg
    return GoalPoseNoiseConfig(**dict(raw_cfg))


def _parse_hybrid_motion_cfg(raw_cfg: Any) -> HybridMotionConfig:
    if isinstance(raw_cfg, HybridMotionConfig):
        return raw_cfg
    hybrid_cfg_dict = dict(raw_cfg)
    gen_step_zero_root_noise = hybrid_cfg_dict.get("gen_step_zero_root_noise")
    if gen_step_zero_root_noise is not None:
        hybrid_cfg_dict["gen_step_zero_root_noise"] = _parse_step_zero_root_noise_cfg(gen_step_zero_root_noise)

    robot_goal_noise = hybrid_cfg_dict.get("robot_goal_noise")
    if robot_goal_noise is not None:
        hybrid_cfg_dict["robot_goal_noise"] = _parse_goal_pose_noise_cfg(robot_goal_noise)

    object_goal_noise = hybrid_cfg_dict.get("object_goal_noise")
    if object_goal_noise is not None:
        hybrid_cfg_dict["object_goal_noise"] = _parse_goal_pose_noise_cfg(object_goal_noise)

    size_randomization = hybrid_cfg_dict.get("size_randomization")
    if size_randomization is not None and not isinstance(size_randomization, HybridSizeRandomizationConfig):
        hybrid_cfg_dict["size_randomization"] = HybridSizeRandomizationConfig(**dict(size_randomization))

    return HybridMotionConfig(**hybrid_cfg_dict)


class MotionCommand(CommandTermBase):
    def __init__(self, cfg: Any, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self._env = env
        # self.motion_cfg: MotionConfig = cfg.params["motion_config"]
        # TODO(jchen):temporary fix for motion_config being a dict after tyro.cli
        if isinstance(cfg.params["motion_config"], MotionConfig):
            self.motion_cfg = cfg.params["motion_config"]
        else:
            motion_cfg_dict = dict(cfg.params["motion_config"])
            noise_cfg = motion_cfg_dict.get("noise_to_initial_pose")
            if isinstance(noise_cfg, dict):
                noise_cfg_dict = dict(noise_cfg)
                step_zero_root_noise = noise_cfg_dict.get("step_zero_root_noise")
                if isinstance(step_zero_root_noise, dict):
                    noise_cfg_dict["step_zero_root_noise"] = StepZeroRootNoiseConfig(**step_zero_root_noise)
                motion_cfg_dict["noise_to_initial_pose"] = NoiseToInitialPoseConfig(**noise_cfg_dict)

            goal_orientation_cfg = motion_cfg_dict.get("goal_orientation")
            if isinstance(goal_orientation_cfg, dict):
                motion_cfg_dict["goal_orientation"] = GoalOrientationConfig(**goal_orientation_cfg)

            adaptive_sampler_cfg = motion_cfg_dict.get("adaptive_timestep_sampler")
            if isinstance(adaptive_sampler_cfg, dict):
                motion_cfg_dict["adaptive_timestep_sampler"] = AdaptiveTimestepSamplerConfig(**adaptive_sampler_cfg)

            eval_fixed_reset_cfg = motion_cfg_dict.get("eval_fixed_reset")
            if isinstance(eval_fixed_reset_cfg, dict):
                motion_cfg_dict["eval_fixed_reset"] = EvalFixedResetConfig(**eval_fixed_reset_cfg)

            hybrid_cfg = motion_cfg_dict.get("hybrid")
            if hybrid_cfg is not None:
                motion_cfg_dict["hybrid"] = _parse_hybrid_motion_cfg(hybrid_cfg)

            self.motion_cfg = MotionConfig(**motion_cfg_dict)
        self.init_pose_cfg: NoiseToInitialPoseConfig = self.motion_cfg.noise_to_initial_pose

    def setup(self) -> None:
        self.num_envs = self._env.num_envs
        self.device = self._env.device

        robot_body_names = self._env.simulator._body_list  # type: ignore[attr-defined]
        robot_body_names_alias = [FAKE_BODY_NAME_ALIASES.get(bn, bn) for bn in robot_body_names]

        robot_joint_names = self._env.simulator.dof_names  # type: ignore[attr-defined]

        # 1. load motion data
        self.motion = self._load_motion_data(robot_body_names_alias, robot_joint_names)

        # Store body and joint indexes for interpolation
        self._body_indexes_in_motion = self.motion._body_indexes
        self._joint_indexes_in_motion = self.motion._joint_indexes

        # Maybe prepend interpolated transition from default pose
        self._maybe_add_default_pose_transition(prepend=True)

        # Maybe append a terminal hold before any return-to-default transition.
        self._maybe_add_terminal_pose_hold()

        # Maybe append interpolated transition back to default pose
        self._maybe_add_default_pose_transition(prepend=False)

        # 2. get the indexes of the root link and the tracked links
        self.ref_body_index = robot_body_names.index(self.motion_cfg.body_name_ref[0])  # int
        self.tracked_body_indexes = self._get_index_of_a_in_b(
            self.motion_cfg.body_names_to_track, robot_body_names, self.device
        )

        # 3. get the name of the object, or indices of the object
        if self.motion.has_object:
            # cache the object_index_in_simulator (object groups: replaced by each environment's variant below)
            self.object_name = "object_00" if self.motion_cfg.motion_dir_object_groups else "object"
            self.object_indices_in_simulator = self._env.simulator.get_actor_indices(self.object_name, env_ids=None)

            assert self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM, (
                "Object is only supported in IsaacSim"
            )

        # 3.1 object groups: environment e holds object asset e % num_groups and only samples that object's clips
        self._setup_object_groups()

        # 4. get the adaptive timesteps sampler
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            adaptive_cfg = self.motion_cfg.adaptive_timestep_sampler
            self.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
                self.motion.motion_start_idx,
                self.motion.motion_end_idx,
                self.device,
                float(self._env.dt),
                list(self._adaptive_sampler_group_names()),
                bin_width_s=adaptive_cfg.bin_width_s,
                ema_alpha=adaptive_cfg.ema_alpha,
                bin_uniform_mix=adaptive_cfg.bin_uniform_mix,
                motion_uniform_mix=adaptive_cfg.motion_uniform_mix,
                score_power=adaptive_cfg.score_power,
                min_segment_steps=adaptive_cfg.min_segment_steps,
            )
            self._adaptive_sampler_last_dump_step = -1
            self._adaptive_sampler_last_metrics_refresh_step = -1
            self._adaptive_sampler_collect_rollout = False

        # 5. metrics
        self.metrics: dict[str, torch.Tensor] = {}

        self.init_buffers()
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self._refresh_adaptive_sampler_runtime_cache()
            self._refresh_adaptive_sampler_metrics()

        # 6. visualization markers for isaacsim
        video_recorder = getattr(self._env.simulator, "video_recorder", None)
        should_setup_markers = self._env.viewer or (video_recorder is not None and video_recorder.enabled)
        if should_setup_markers and self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._setup_visualization_markers_for_isaacsim()

    def _load_motion_data(self, robot_body_names_alias: list[str], robot_joint_names: list[str]) -> MotionLoader | MultiMotionLoader:
        assert self.motion_cfg.motion_file or self.motion_cfg.motion_dir, (
            "Either motion_file or motion_dir must be set in MotionConfig"
        )
        if self.motion_cfg.motion_dir:
            return MultiMotionLoader(
                self.motion_cfg.motion_dir,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
                object_groups=self.motion_cfg.motion_dir_object_groups,
            )
        return MotionLoader(
            self.motion_cfg.motion_file,
            robot_body_names_alias,
            robot_joint_names,
            device=self.device,
        )

    def reset(self, env_ids: torch.Tensor | None) -> None:
        """called per reset_idx, reset timesteps and robot/object poses."""
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        adaptive_group_ids = None
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self._record_adaptive_sampler_episode_resets(env_ids)
            adaptive_group_ids = self._adaptive_sampler_group_ids(env_ids)

        # Environments whose episode failed may restart on the same clip shortly before the failure (rewind_prob).
        rewind = None
        if self.motion_cfg.rewind_prob > 0.0 and not self._env.is_evaluating:
            n = env_ids.numel()
            prev_motion_ids = self.motion_ids[env_ids].clone()
            prev_time_steps = self.time_steps[env_ids].clone()
            rewind = self._env.termination_manager.terminated[env_ids] & (
                torch.rand(n, device=self.device) < self.motion_cfg.rewind_prob
            )
            rewind &= prev_time_steps < self.motion.motion_end_idx[prev_motion_ids] - 50
            track_env_mask = getattr(self, "track_env_mask", None)
            if track_env_mask is not None:
                rewind &= track_env_mask[env_ids]

        # 0. Sample the time steps
        eval_fixed_reset_enabled = False
        if self._env.is_evaluating:
            eval_fixed_reset = getattr(self.motion_cfg, "eval_fixed_reset", None)
            eval_fixed_reset_enabled = bool(getattr(eval_fixed_reset, "enabled", False))
            if eval_fixed_reset_enabled:
                sampled_motion_ids, sampled_time_steps = self._sample_eval_fixed_reset_states(env_ids)
                self.motion_ids[env_ids] = sampled_motion_ids
                self.time_steps[env_ids] = sampled_time_steps
            elif getattr(self.motion_cfg, "eval_sample_random_timestep", False):
                sampled_motion_ids, sampled_time_steps = self._sample_uniform_reset_states(env_ids)
                self.motion_ids[env_ids] = sampled_motion_ids
                self.time_steps[env_ids] = sampled_time_steps
            else:
                self._sample_motion_ids_for_reset(env_ids)
                start_idx = self.motion.motion_start_idx[self.motion_ids[env_ids]]
                self.time_steps[env_ids] = start_idx
        elif self.motion_cfg.use_adaptive_timesteps_sampler:
            assert adaptive_group_ids is not None
            sampled_motion_ids, sampled_time_steps = self._sample_reset_states_with_adaptive_fallback(
                env_ids,
                adaptive_group_ids,
            )
            self.motion_ids[env_ids] = sampled_motion_ids
            self.time_steps[env_ids] = sampled_time_steps
        else:
            sampled_motion_ids, sampled_time_steps = self._sample_uniform_reset_states(env_ids)
            self.motion_ids[env_ids] = sampled_motion_ids
            self.time_steps[env_ids] = sampled_time_steps

        start_idx = self.motion.motion_start_idx[self.motion_ids[env_ids]]
        end_idx = self.motion.motion_end_idx[self.motion_ids[env_ids]]

        # Optionally force a reset to the first frame of the assigned motion clip.
        start_at_zero_probs = self._get_start_at_timestep_zero_probs(env_ids)
        if not eval_fixed_reset_enabled and torch.any(start_at_zero_probs > 0.0):
            subset = self.time_steps[env_ids]
            rand_vals = torch.rand(env_ids.numel(), device=self.device, dtype=torch.float32)
            subset = torch.where(rand_vals < start_at_zero_probs, start_idx, subset)
            self.time_steps[env_ids] = subset

        if rewind is not None and rewind.any():
            back = torch.randint(*self.motion_cfg.rewind_steps_range, (env_ids.numel(),), device=self.device)
            rewound = (prev_time_steps - back).clamp(min=self.motion.motion_start_idx[prev_motion_ids])
            self.motion_ids[env_ids] = torch.where(rewind, prev_motion_ids, self.motion_ids[env_ids])
            self.time_steps[env_ids] = torch.where(rewind, rewound, self.time_steps[env_ids])
            start_idx = self.motion.motion_start_idx[self.motion_ids[env_ids]]
            end_idx = self.motion.motion_end_idx[self.motion_ids[env_ids]]

        # If the motion is at the last timestep, set it to the second last timestep;
        # Otherwise, update_tasks_callback will advance the timestep to the next timestep -> out of bounds error.
        already_last_timestep_mask = self.time_steps[env_ids] >= end_idx - 1
        self.time_steps[env_ids] = torch.where(already_last_timestep_mask, end_idx - 2, self.time_steps[env_ids])
        self._assign_goal_time_steps(env_ids)

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            assert adaptive_group_ids is not None
            adaptive_enabled_mask = self._adaptive_sampler_enabled_env_mask(adaptive_group_ids)
            self.adaptive_sampler_segment_active_mask[env_ids] = adaptive_enabled_mask
            active_env_ids = env_ids[adaptive_enabled_mask]
            if active_env_ids.numel() > 0:
                self.adaptive_sampler_start_group_ids[active_env_ids] = adaptive_group_ids[adaptive_enabled_mask]
                self.adaptive_sampler_start_motion_ids[active_env_ids] = self.motion_ids[active_env_ids]
                self.adaptive_sampler_start_bins[active_env_ids] = self.adaptive_timesteps_sampler.time_step_to_bin(
                    self.motion_ids[active_env_ids],
                    self.time_steps[active_env_ids],
                )
                self.adaptive_sampler_segment_start_env_step[active_env_ids] = int(
                    self._adaptive_sampler_global_env_step
                )

        # 1. Get the root/body poses from the motion data
        root_pos = self.root_pos_w[env_ids].clone()
        root_rot = self.root_quat_w[env_ids].clone()
        root_lin_vel = self.root_lin_vel_w[env_ids].clone()
        root_ang_vel = self.root_ang_vel_w[env_ids].clone()

        dof_pos = self.joint_pos[env_ids].clone()
        dof_vel = self.joint_vel[env_ids].clone()

        # 2. Adding noise
        # 2.1 prepare the noise scale
        initial_pose_noise_scale = self._initial_pose_noise_scale_multipliers(env_ids)
        robot_initial_pose_noise_scale = self._robot_initial_pose_noise_scale_multipliers(env_ids)
        root_pose_initial_noise_scale = self._root_pose_initial_noise_scale_multipliers(env_ids, start_idx)
        dof_pos_noise = (
            self.init_pose_cfg.dof_pos
            * self.init_pose_cfg.overall_noise_scale
            * initial_pose_noise_scale[:, None]
            * robot_initial_pose_noise_scale[:, None]
        )
        root_pos_noise = (
            self._noise_vector_3d(
                self.init_pose_cfg.root_pos,
                field_name="motion_config.noise_to_initial_pose.root_pos",
            )
            * self.init_pose_cfg.overall_noise_scale
            * initial_pose_noise_scale[:, None]
            * robot_initial_pose_noise_scale[:, None]
            * root_pose_initial_noise_scale[:, None]
        )  # (num_envs, 3)
        root_rot_noise_rpy = (
            self._noise_vector_3d(
                self.init_pose_cfg.root_rot,
                field_name="motion_config.noise_to_initial_pose.root_rot",
            )
            * self.init_pose_cfg.overall_noise_scale
            * initial_pose_noise_scale[:, None]
            * robot_initial_pose_noise_scale[:, None]
            * root_pose_initial_noise_scale[:, None]
        )  # (num_envs, 3)
        root_vel_noise = (
            torch.tensor(
                self.init_pose_cfg.root_lin_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
            * initial_pose_noise_scale[:, None]
            * robot_initial_pose_noise_scale[:, None]
        )  # (num_envs, 3)
        root_ang_vel_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_ang_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
            * initial_pose_noise_scale[:, None]
            * robot_initial_pose_noise_scale[:, None]
        )  # (num_envs, 3)

        # 2.2 Adding noise to dof_pos, root_pos, root_vel, root_ang_vel, root_rot
        # 1.2.1 dof_pos
        target_dof_pos = (
            dof_pos + (torch.rand(dof_pos.shape, device=self.device) - 0.5) * 2 * dof_pos_noise
        )  # (num_envs, num_dofs)
        soft_joint_pos_limits = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]  # (num_dofs, 2)
        target_dof_pos = torch.clip(target_dof_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1])

        # 1.2.2 dof_vel no noise
        target_dof_vel = dof_vel

        # 1.2.3 root_pos
        target_root_pos = root_pos + (
            torch.rand(root_pos.shape, device=self.device) - 0.5
        ) * 2 * root_pos_noise  # (num_envs, 3)

        # 1.2.4 root_rot
        rand_sample_rpy = (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 2 * root_rot_noise_rpy
        orientations_delta = quat_from_euler_xyz(
            rand_sample_rpy[:, 0], rand_sample_rpy[:, 1], rand_sample_rpy[:, 2]
        )  # (num_envs, 4), xyzw
        target_root_rot = quat_mul(orientations_delta, root_rot, w_last=True)  # (num_envs, 4), xyzw
        self._apply_step_zero_root_pose_overrides(
            env_ids,
            start_idx,
            root_pos,
            root_rot,
            target_root_pos,
            target_root_rot,
        )

        # 1.2.5 root_lin_vel
        target_root_lin_vel = root_lin_vel + (
            torch.rand(root_lin_vel.shape, device=self.device) - 0.5
        ) * 2 * root_vel_noise  # (num_envs, 3)

        # 1.2.6 root_ang_vel
        target_root_ang_vel = root_ang_vel + (
            torch.rand(root_ang_vel.shape, device=self.device) - 0.5
        ) * 2 * root_ang_vel_noise_rpy  # (num_envs, 3)

        # 3. Set the robot states in simulator
        self._env.simulator.dof_pos[env_ids] = target_dof_pos
        self._env.simulator.dof_vel[env_ids] = target_dof_vel

        self._env.simulator.robot_root_states[env_ids, :3] = target_root_pos
        self._env.simulator.robot_root_states[env_ids, 3:7] = target_root_rot
        self._env.simulator.robot_root_states[env_ids, 7:10] = target_root_lin_vel
        self._env.simulator.robot_root_states[env_ids, 10:13] = target_root_ang_vel

        # 4. Set the object states in simulator
        if self.motion.has_object:
            if self.object_groups:
                self._set_object_variant_states(env_ids, self._object_reset_states(env_ids))
            else:
                self._env.simulator.set_actor_states([self.object_name], env_ids, self._object_reset_states(env_ids))

    def _set_object_variant_states(self, env_ids: torch.Tensor, object_states: torch.Tensor) -> None:
        """Object groups: ``object_states`` [len(env_ids), 13] for each environment's own variant; every other variant
        of these environments is parked (at rest, away from the robot)."""
        env_groups = self.env_object_group[env_ids]
        origins = self._env.simulator.scene.env_origins[env_ids]
        for k, name in enumerate(self.object_names):
            states = torch.zeros(len(env_ids), 13, device=self.device)
            states[:, :3] = self._object_park_pos[k] + origins
            states[:, 3:7] = self._object_park_quat[k]
            own = env_groups == k
            states[own] = object_states[own]
            self._env.simulator.set_actor_states([name], env_ids, states)

    def step(self) -> None:
        """called in _update_tasks_callback of the environment. (after compute_reward, before compute_observations)"""
        self.last_clip_end_env_ids = torch.zeros(0, dtype=torch.long, device=self.device)
        # 0. update time steps, all motion joint/body poses are updated automatically with the time steps.
        advance_mask = torch.ones_like(self.time_steps, dtype=torch.bool)

        # Handle freeze_at_timestep_zero_prob: for envs at their motion's start, randomly decide whether to advance
        freeze_prob = self.motion_cfg.freeze_at_timestep_zero_prob
        if freeze_prob > 0.0:
            zero_mask = self.time_steps == self.motion.motion_start_idx[self.motion_ids]
            if zero_mask.any():
                rand_vals = torch.rand(self.num_envs, device=self.device)
                freeze_mask = (rand_vals < freeze_prob) & zero_mask
                advance_mask = advance_mask & ~freeze_mask

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self._increment_adaptive_sampler_global_env_step()
        self.time_steps += advance_mask.long()

        # BeyondMimic-style behavior: when the clip ends, resample motion and
        # reset robot/object state without terminating the whole episode.
        per_motion_end = self.motion.motion_end_idx[self.motion_ids]
        ended_env_ids = torch.where(self.time_steps >= per_motion_end)[0]
        if ended_env_ids.numel() > 0:
            self.last_clip_end_env_ids = ended_env_ids.clone()
            if self.motion_cfg.use_adaptive_timesteps_sampler:
                self._record_adaptive_sampler_clip_ends(ended_env_ids, per_motion_end[ended_env_ids] - 1)
            self.reset(ended_env_ids)
            # Flush the mutated root/dof state into the simulator so that
            # rigid-body positions are up-to-date for downstream consumers
            # (termination checks, observations, rewards).
            sim = self._env.simulator
            sim.set_actor_root_state_tensor_robots(ended_env_ids, sim.robot_root_states)
            sim.set_dof_state_tensor_robots(ended_env_ids, sim.dof_state)  # type: ignore[attr-defined]
            sim.refresh_sim_tensors()

        # 1. update body_pos_relative_w and body_quat_relative_w
        # definition of body_pos/quat_relative_w:
        # If I take this motion data and adapt it to where my robot currently is
        # (accounting for position(x, y) offset and yaw difference of a reference body),
        # what should each body part's target pose be?

        ## 1.0 get the reference body poses

        # Issue (This is a isaacgym only issue.):
        # ------------------------------------------------------------
        # In isaacgym, immediately after reset (self._env.episode_length_buf == 0), calling
        # simulator.set_actor_root_state_tensor and simulator.set_dof_state_tensor will reset
        # the robot_root_pos_w and robot_root_quat_w successfully.
        # However, the robot_body_pos_w and robot_body_quat_w are not updated successfully,
        # (since kinematic forward has not been applied yet).
        # Therefore, using robot_ref_pos_w and robot_ref_quat_w as reference body poses is not resetted correctly.

        # Solution:
        # ------------------------------------------------------------
        # if episode_length_buf == 0, use robot_root_pos_w and robot_root_quat_w as reference body.
        # else, use configured reference body as reference body.
        use_root = (self._env.episode_length_buf == 0).unsqueeze(1).float()

        ref_pos_w = self.root_pos_w * use_root + self.ref_pos_w * (1 - use_root)
        ref_quat_w = self.root_quat_w * use_root + self.ref_quat_w * (1 - use_root)
        robot_ref_pos_w = self.robot_root_pos_w * use_root + self.robot_ref_pos_w * (1 - use_root)
        robot_ref_quat_w = self.robot_root_quat_w * use_root + self.robot_ref_quat_w * (1 - use_root)

        ## 1.1 repeat to match the number of body parts
        ref_pos_w_repeat = ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        ref_quat_w_repeat = ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]

        ## 1.2 compute the relative body poses
        delta_quat_w = yaw_quat(
            quat_mul(robot_ref_quat_w_repeat, quat_inverse(ref_quat_w_repeat, w_last=True), w_last=True), w_last=True
        )
        ### 1.2.1 body_quat_relative_w
        self.body_quat_relative_w = quat_mul(delta_quat_w, self.body_quat_w, w_last=True)
        ### 1.2.2 body_pos_relative_w
        delta_pos_w_height = ref_pos_w_repeat - robot_ref_pos_w_repeat
        delta_pos_w_height[..., :2] = 0.0  # adjusting for height differences
        self.body_pos_relative_w = (
            robot_ref_pos_w_repeat
            + delta_pos_w_height
            + quat_apply(delta_quat_w, self.body_pos_w - ref_pos_w_repeat, w_last=True)
        )

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    #########################################################################################
    ## Robot from motion data
    #########################################################################################
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.time_steps][:, self.tracked_body_indexes]
            + self._env.simulator.scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps][:, self.tracked_body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps][:, self.tracked_body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps][:, self.tracked_body_indexes]

    @property
    def ref_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.ref_body_index] + self._env.simulator.scene.env_origins

    @property
    def ref_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.ref_body_index]

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.ref_body_index]

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.ref_body_index]

    @property
    def goal_time_steps(self) -> torch.Tensor:
        return self._goal_time_steps

    @property
    def goal_version(self) -> torch.Tensor:
        return self._goal_version

    @property
    def goal_ref_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.goal_time_steps, self.ref_body_index] + self._env.simulator.scene.env_origins

    @property
    def goal_ref_quat_w(self) -> torch.Tensor:
        goal_quat_w = self.motion.body_quat_w[self.goal_time_steps, self.ref_body_index]
        if self._robot_goal_yaw_only_enabled():
            return yaw_quat(goal_quat_w, w_last=True)
        return goal_quat_w

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, 0] + self._env.simulator.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, 0]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, 0]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, 0]

    #########################################################################################
    ## Robot from simulator
    #########################################################################################
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self._env.simulator.dof_pos  # (num_envs, num_dofs)

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self._env.simulator.dof_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.tracked_body_indexes, :]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.tracked_body_indexes, :]  # xyzw

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, :3]  # type: ignore[attr-defined]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 3:7]  # type: ignore[attr-defined]

    @property
    def robot_root_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 7:10]  # type: ignore[attr-defined]

    @property
    def robot_root_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 10:13]  # type: ignore[attr-defined]

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.ref_body_index, :]

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.ref_body_index, :]  # xyzw

    @property
    def goal_conditioning_ref_quat_w(self) -> torch.Tensor:
        if self._robot_goal_yaw_only_enabled():
            return yaw_quat(self.robot_ref_quat_w, w_last=True)
        return self.robot_ref_quat_w

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.ref_body_index, :]

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.ref_body_index, :]

    #########################################################################################
    ## Object from motion data
    #########################################################################################
    @property
    def object_pos_w(self) -> torch.Tensor:
        # Applies env origins, but ideally we should rely on the simulator
        return self.motion.object_pos_w[self.time_steps] + self._env.simulator.scene.env_origins

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self.motion.object_quat_w[self.time_steps]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self.motion.object_lin_vel_w[self.time_steps]

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        return self.motion.object_ang_vel_w[self.time_steps]

    @property
    def goal_object_pos_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            raise RuntimeError("Goal object pose requested for a motion without object data.")
        return self.motion.object_pos_w[self.goal_time_steps] + self._env.simulator.scene.env_origins

    @property
    def goal_object_quat_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            raise RuntimeError("Goal object pose requested for a motion without object data.")
        goal_quat_w = self.motion.object_quat_w[self.goal_time_steps]
        if self._object_goal_yaw_only_enabled():
            return yaw_quat(goal_quat_w, w_last=True)
        return goal_quat_w

    #########################################################################################
    ## Object from simulator
    #########################################################################################
    def _simulator_object_states(self) -> torch.Tensor:
        """Object states [num_envs, 13]. With object groups, each environment's own variant: every variant is read
        in environment order and the tracked one picked per environment (the root-state proxy concatenates mixed
        indices object by object, not in index order)."""
        if getattr(self, "object_groups", False):
            states = torch.stack(
                [self._env.simulator.all_root_states[idx] for idx in self._object_variant_indices]
            )  # [N, num_envs, 13]
            return states[self.env_object_group, torch.arange(self.num_envs, device=self.device)]
        return self._env.simulator.all_root_states[self.object_indices_in_simulator]

    @property
    def simulator_object_pos_w(self) -> torch.Tensor:
        return self._simulator_object_states()[:, :3]

    @property
    def simulator_object_quat_w(self) -> torch.Tensor:
        return self._simulator_object_states()[:, 3:7]

    @property
    def goal_conditioning_object_quat_w(self) -> torch.Tensor:
        if self._object_goal_yaw_only_enabled():
            return yaw_quat(self.simulator_object_quat_w, w_last=True)
        return self.simulator_object_quat_w

    @property
    def simulator_object_lin_vel_w(self) -> torch.Tensor:
        return self._simulator_object_states()[:, 7:10]

    @property
    def simulator_object_ang_vel_w(self) -> torch.Tensor:
        return self._simulator_object_states()[:, 10:13]

    #########################################################################################
    ## Methods that does not fit into setup/step/reset pattern
    #########################################################################################

    def _goal_orientation_cfg(self):
        return getattr(getattr(self, "motion_cfg", None), "goal_orientation", None)

    def _robot_goal_yaw_only_enabled(self) -> bool:
        return bool(getattr(self._goal_orientation_cfg(), "robot_yaw_only", False))

    def _object_goal_yaw_only_enabled(self) -> bool:
        return bool(getattr(self._goal_orientation_cfg(), "object_yaw_only", False))

    def _assign_goal_time_steps(self, env_ids: torch.Tensor) -> None:
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        self._goal_time_steps[env_ids] = self._compute_goal_time_steps(env_ids)
        self._goal_version[env_ids] += 1
        self._on_goal_time_steps_updated(env_ids)

    def _compute_goal_time_steps(self, env_ids: torch.Tensor) -> torch.Tensor:
        env_ids = self._ensure_index_tensor(env_ids)
        motion_end_idx = self.motion.motion_end_idx[self.motion_ids[env_ids]]
        return motion_end_idx - 1

    def _on_goal_time_steps_updated(self, env_ids: torch.Tensor) -> None:
        """Hook for subclasses to refresh goal-dependent buffers when sampled goals change."""
        return

    def init_buffers(self):
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._goal_time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._goal_version = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_clip_end_env_ids = torch.zeros(0, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 3, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 4, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w[:, :, 0] = 1.0

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_sampler_start_group_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.adaptive_sampler_start_motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.adaptive_sampler_start_bins = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.adaptive_sampler_segment_start_env_step = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.adaptive_sampler_segment_active_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._adaptive_sampler_global_env_step = 0
            self.adaptive_timesteps_sampler.init_buffers()

    def update_metrics(self):
        """Update the metrics. After action, before step() is called."""
        self.metrics["motion/error_ref_pos"] = torch.norm(self.ref_pos_w - self.robot_ref_pos_w, dim=-1)
        self.metrics["motion/error_ref_rot"] = quat_error_magnitude(self.ref_quat_w, self.robot_ref_quat_w)
        self.metrics["motion/error_ref_lin_vel"] = torch.norm(self.ref_lin_vel_w - self.robot_ref_lin_vel_w, dim=-1)
        self.metrics["motion/error_ref_ang_vel"] = torch.norm(self.ref_ang_vel_w - self.robot_ref_ang_vel_w, dim=-1)

        ref_body_lin_vel_local = quat_rotate_inverse(self.robot_ref_quat_w, self.robot_ref_lin_vel_w, w_last=True)
        self.metrics["motion/ref_body_speed_world"] = torch.norm(self.robot_ref_lin_vel_w, dim=-1)
        self.metrics["motion/ref_body_vel_local_x"] = torch.abs(ref_body_lin_vel_local[:, 0])
        self.metrics["motion/ref_body_vel_local_y"] = torch.abs(ref_body_lin_vel_local[:, 1])

        self.metrics["motion/error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["motion/error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["motion/error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["motion/error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.metrics["motion/adaptive_timesteps_sampler_entropy"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_entropy"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_effective_bins"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_effective_bins"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_prob"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_prob"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_bin"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_bin"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_expected_progress"] = (
                self.adaptive_timesteps_sampler.metrics["sampling_expected_progress"]
            )
            self.metrics["motion/adaptive_timesteps_sampler_motion_entropy"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_motion_entropy"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_motion_top1_prob"] = (
                self.adaptive_timesteps_sampler.metrics["sampling_motion_top1_prob"]
            )
            self.metrics["motion/adaptive_timesteps_sampler_motion_top1_id"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_motion_top1_id"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_fail_ema_mean"] = self.adaptive_timesteps_sampler.metrics[
                "fail_ema_mean"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_fail_ema_min"] = self.adaptive_timesteps_sampler.metrics[
                "fail_ema_min"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_fail_ema_max"] = self.adaptive_timesteps_sampler.metrics[
                "fail_ema_max"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_fail_ema_std"] = self.adaptive_timesteps_sampler.metrics[
                "fail_ema_std"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_overall_bin_prob_max"] = (
                self.adaptive_timesteps_sampler.metrics["overall_bin_prob_max"]
            )
            self.metrics["motion/adaptive_timesteps_sampler_overall_bin_prob_min"] = (
                self.adaptive_timesteps_sampler.metrics["overall_bin_prob_min"]
            )
            self.metrics["motion/adaptive_timesteps_sampler_overall_bin_prob_effective_bins"] = (
                self.adaptive_timesteps_sampler.metrics["overall_bin_prob_effective_bins"]
            )
            self.metrics["motion/adaptive_timesteps_sampler_overall_bin_prob_entropy_norm"] = (
                self.adaptive_timesteps_sampler.metrics["overall_bin_prob_entropy_norm"]
            )
            self.metrics["motion/adaptive_timesteps_sampler_overall_bin_prob_expected_progress"] = (
                self.adaptive_timesteps_sampler.metrics["overall_bin_prob_expected_progress"]
            )

    def _refresh_adaptive_sampler_metrics(self) -> None:
        if not self.motion_cfg.use_adaptive_timesteps_sampler:
            return
        self.adaptive_timesteps_sampler.get_stats(
            group_priors=self._adaptive_sampler_effective_group_priors(),
            candidate_motion_ids_by_group=self._get_cached_adaptive_sampler_candidate_motion_ids_by_group(),
            group_first_frame_probs=self._get_cached_adaptive_sampler_group_first_frame_probs(),
        )

    def _refresh_adaptive_sampler_runtime_cache(self) -> None:
        if not self.motion_cfg.use_adaptive_timesteps_sampler:
            return

        enabled_group_mask = self._adaptive_sampler_enabled_group_mask().to(device=self.device, dtype=torch.bool)
        group_priors = self._adaptive_sampler_group_priors().to(device=self.device, dtype=torch.float32)
        if group_priors.numel() != enabled_group_mask.numel():
            raise ValueError(
                "Adaptive sampler group priors and enabled-group mask must have the same length, "
                f"got {group_priors.numel()} priors and {enabled_group_mask.numel()} mask entries."
            )

        self._adaptive_sampler_cached_enabled_group_mask = enabled_group_mask.clone()
        self._adaptive_sampler_cached_effective_group_priors = (
            group_priors * enabled_group_mask.to(dtype=group_priors.dtype)
        )
        self._adaptive_sampler_cached_group_first_frame_probs = self._adaptive_sampler_group_first_frame_probs().to(
            device=self.device,
            dtype=torch.float32,
        )
        candidate_motion_ids_by_group = self._adaptive_sampler_candidate_motion_ids_by_group()
        self._adaptive_sampler_cached_candidate_motion_ids_by_group = {
            int(group_id): motion_ids.to(device=self.device, dtype=torch.long).clone()
            for group_id, motion_ids in candidate_motion_ids_by_group.items()
        }
        all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        unique_pool_ids = torch.unique(self._adaptive_sampler_motion_pool_ids(all_env_ids))
        self._adaptive_sampler_cached_motion_pools_by_id = {
            int(pool_id.item()): self._adaptive_sampler_motion_pool(int(pool_id.item())).to(
                device=self.device,
                dtype=torch.long,
            ).clone()
            for pool_id in unique_pool_ids
        }

    def _get_cached_adaptive_sampler_group_first_frame_probs(self) -> torch.Tensor:
        cached = getattr(self, "_adaptive_sampler_cached_group_first_frame_probs", None)
        if cached is not None:
            return cached
        return self._adaptive_sampler_group_first_frame_probs()

    def _get_cached_adaptive_sampler_candidate_motion_ids_by_group(self) -> dict[int, torch.Tensor]:
        cached = getattr(self, "_adaptive_sampler_cached_candidate_motion_ids_by_group", None)
        if cached is not None:
            return cached
        return {
            int(group_id): motion_ids.to(device=self.device, dtype=torch.long)
            for group_id, motion_ids in self._adaptive_sampler_candidate_motion_ids_by_group().items()
        }

    def _get_cached_adaptive_sampler_motion_pools_by_id(self) -> dict[int, torch.Tensor]:
        cached = getattr(self, "_adaptive_sampler_cached_motion_pools_by_id", None)
        if cached is not None:
            return cached
        all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        unique_pool_ids = torch.unique(self._adaptive_sampler_motion_pool_ids(all_env_ids))
        return {
            int(pool_id.item()): self._adaptive_sampler_motion_pool(int(pool_id.item())).to(
                device=self.device,
                dtype=torch.long,
            )
            for pool_id in unique_pool_ids
        }

    def _adaptive_sampler_update_interval(self) -> int:
        return max(1, int(self.motion_cfg.adaptive_timestep_sampler.update_interval))

    def _adaptive_sampler_uses_batched_updates(self) -> bool:
        return self._adaptive_sampler_update_interval() > 1

    def _should_collect_adaptive_sampler_rollout(self, training_step: int) -> bool:
        if not self._adaptive_sampler_uses_batched_updates():
            return False
        return (int(training_step) + 1) % self._adaptive_sampler_update_interval() == 0

    def _maybe_refresh_adaptive_sampler_metrics(self, *, training_step: int) -> bool:
        if not self.motion_cfg.use_adaptive_timesteps_sampler:
            return False

        training_step = int(training_step)
        if training_step <= 0:
            return False
        if getattr(self, "_adaptive_sampler_last_metrics_refresh_step", -1) == training_step:
            return False

        self._refresh_adaptive_sampler_metrics()
        self._adaptive_sampler_last_metrics_refresh_step = training_step
        return True

    def _increment_adaptive_sampler_global_env_step(self) -> None:
        if not self.motion_cfg.use_adaptive_timesteps_sampler:
            return
        self._adaptive_sampler_global_env_step += 1

    def _adaptive_sampler_segment_lengths(self, env_ids: torch.Tensor) -> torch.Tensor:
        current_env_step = int(getattr(self, "_adaptive_sampler_global_env_step", 0))
        return torch.full(
            (env_ids.numel(),),
            current_env_step,
            dtype=torch.long,
            device=self.device,
        ) - self.adaptive_sampler_segment_start_env_step[env_ids]

    def _record_adaptive_sampler_segment_ends(
        self,
        *,
        env_ids: torch.Tensor,
        terminal_time_steps: torch.Tensor,
        failure_mask: torch.Tensor,
    ) -> None:
        if env_ids.numel() == 0:
            return
        segment_step_counts = self._adaptive_sampler_segment_lengths(env_ids)
        if self._adaptive_sampler_uses_batched_updates():
            if not getattr(self, "_adaptive_sampler_collect_rollout", False):
                return
            self.adaptive_timesteps_sampler.accumulate_segment_end(
                group_ids=self.adaptive_sampler_start_group_ids[env_ids],
                motion_ids=self.adaptive_sampler_start_motion_ids[env_ids],
                start_bins=self.adaptive_sampler_start_bins[env_ids],
                segment_step_counts=segment_step_counts,
                terminal_time_steps=terminal_time_steps,
                failure_mask=failure_mask,
            )
            return
        self.adaptive_timesteps_sampler.record_segment_end(
            group_ids=self.adaptive_sampler_start_group_ids[env_ids],
            motion_ids=self.adaptive_sampler_start_motion_ids[env_ids],
            start_bins=self.adaptive_sampler_start_bins[env_ids],
            segment_step_counts=segment_step_counts,
            terminal_time_steps=terminal_time_steps,
            failure_mask=failure_mask,
        )

    #########################################################################################
    ## Internal helpers
    #########################################################################################
    def _adaptive_sampler_group_names(self) -> tuple[str, ...]:
        return ("default",)

    def _adaptive_sampler_group_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(env_ids.numel(), dtype=torch.long, device=self.device)

    def _adaptive_sampler_group_priors(self) -> torch.Tensor:
        return torch.ones(len(self._adaptive_sampler_group_names()), dtype=torch.float32, device=self.device)

    def _adaptive_sampler_enabled_group_mask(self) -> torch.Tensor:
        return torch.ones(len(self._adaptive_sampler_group_names()), dtype=torch.bool, device=self.device)

    def _adaptive_sampler_effective_group_priors(self) -> torch.Tensor:
        cached = getattr(self, "_adaptive_sampler_cached_effective_group_priors", None)
        if cached is not None:
            return cached
        group_priors = self._adaptive_sampler_group_priors()
        enabled_group_mask = self._adaptive_sampler_enabled_group_mask()
        if group_priors.numel() != enabled_group_mask.numel():
            raise ValueError(
                "Adaptive sampler group priors and enabled-group mask must have the same length, "
                f"got {group_priors.numel()} priors and {enabled_group_mask.numel()} mask entries."
            )
        return group_priors * enabled_group_mask.to(dtype=group_priors.dtype)

    def _adaptive_sampler_enabled_env_mask(self, group_ids: torch.Tensor) -> torch.Tensor:
        enabled_group_mask = getattr(self, "_adaptive_sampler_cached_enabled_group_mask", None)
        if enabled_group_mask is None:
            enabled_group_mask = self._adaptive_sampler_enabled_group_mask()
        group_ids = group_ids.to(device=self.device, dtype=torch.long)
        return enabled_group_mask[group_ids]

    def _adaptive_sampler_group_first_frame_probs(self) -> torch.Tensor:
        prob = self._validate_probability(
            self.motion_cfg.start_at_timestep_zero_prob,
            field_name="motion_config.start_at_timestep_zero_prob",
        )
        assert prob is not None
        return torch.full(
            (len(self._adaptive_sampler_group_names()),),
            prob,
            dtype=torch.float32,
            device=self.device,
        )

    def _adaptive_sampler_motion_pool_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self.object_groups:  # one pool per object group
            return self.env_object_group[env_ids]
        return torch.zeros(env_ids.numel(), dtype=torch.long, device=self.device)

    def _adaptive_sampler_motion_pool(self, pool_id: int) -> torch.Tensor:
        if self.object_groups:
            return self._group_motion_ids[pool_id]
        return torch.arange(self.motion.num_motions, dtype=torch.long, device=self.device)

    def _setup_object_groups(self) -> None:
        """Object groups (``motion_dir_object_groups``): environment e holds object asset e % N (the scene spawns
        ``object_urdf_paths[e % N]``) and samples only the clips of motion_dir entry e % N."""
        self.object_groups = bool(getattr(self.motion_cfg, "motion_dir_object_groups", False))
        self.num_object_groups = self.motion.num_groups if self.object_groups else 1
        self.env_object_group = torch.arange(self.num_envs, device=self.device) % self.num_object_groups
        if not self.object_groups:
            return
        if not self.motion.has_object:
            raise ValueError("motion_dir_object_groups needs motions with object data")
        groups = self.motion.motion_group
        self._group_motion_ids = [torch.where(groups == g)[0] for g in range(self.num_object_groups)]
        if any(ids.numel() == 0 for ids in self._group_motion_ids):
            raise ValueError("motion_dir_object_groups: every motion_dir entry needs at least one clip")
        logger.info(
            f"MotionCommand: {self.num_object_groups} object groups, motions per group "
            f"{[ids.numel() for ids in self._group_motion_ids]}"
        )
        # Object variants (robot.object.object_urdf_paths): every environment holds object_00, object_01, ... in group
        # order; environment e tracks variant e % N (its rows of all_root_states), the others are parked at reset
        # (their pose at the start of the group's first clip, moved away from the robot).
        self.object_names = [f"object_{k:02d}" for k in range(self.num_object_groups)]
        missing = [n for n in self.object_names if n not in self._env.simulator.scene.rigid_objects]
        if missing:
            raise ValueError(
                f"motion_dir_object_groups: {self.num_object_groups} motion_dir entries but no objects {missing} "
                "(set robot.object.object_urdf_paths in the same order)"
            )
        self._object_variant_indices = [
            self._env.simulator.get_actor_indices(n, env_ids=None) for n in self.object_names
        ]  # each in environment order; read through _simulator_object_states
        first = torch.stack([self.motion.motion_start_idx[ids[0]] for ids in self._group_motion_ids])
        self._object_park_pos = self.motion.object_pos_w[first].clone()
        self._object_park_pos[:, 0] += OBJECT_PARK_OFFSET[0]
        self._object_park_pos[:, 1] += OBJECT_PARK_OFFSET[1] + OBJECT_PARK_SPACING * torch.arange(
            self.num_object_groups, device=self.device
        )
        self._object_park_quat = self.motion.object_quat_w[first].clone()
        if self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            # Each variant should be its own asset, told apart by its centre of mass (masses are randomized at
            # startup, the centre of mass is not).
            coms = [
                tuple(round(float(c), 3) for c in
                      self._env.simulator.scene.rigid_objects[n].root_physx_view.get_coms()[0, :3])
                for n in self.object_names
            ]
            logger.info(f"MotionCommand: object variants {self.object_names}, centres of mass {coms}")

    def _adaptive_sampler_candidate_motion_ids_by_group(self) -> dict[int, torch.Tensor]:
        all_motion_ids = self._adaptive_sampler_motion_pool(0)
        return {
            group_id: all_motion_ids
            for group_id in range(len(self._adaptive_sampler_group_names()))
        }

    def _sample_adaptive_reset_states(
        self,
        env_ids: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_ids = torch.empty(env_ids.numel(), dtype=torch.long, device=self.device)
        start_bins = torch.empty(env_ids.numel(), dtype=torch.long, device=self.device)
        time_steps = torch.empty(env_ids.numel(), dtype=torch.long, device=self.device)
        pool_ids = self._adaptive_sampler_motion_pool_ids(env_ids)
        pair_ids = torch.stack([group_ids.to(dtype=torch.long), pool_ids.to(dtype=torch.long)], dim=1)
        unique_pairs, inverse = torch.unique(pair_ids, dim=0, return_inverse=True)
        motion_pools_by_id = self._get_cached_adaptive_sampler_motion_pools_by_id()

        for pair_idx in range(unique_pairs.shape[0]):
            subset_mask = inverse == pair_idx
            num_samples = int(subset_mask.sum().item())
            if num_samples == 0:
                continue
            group_id = int(unique_pairs[pair_idx, 0].item())
            pool_id = int(unique_pairs[pair_idx, 1].item())
            sampled_motion_ids, sampled_bins, sampled_time_steps = self.adaptive_timesteps_sampler.sample_from_motion_pool(
                group_id,
                motion_pools_by_id[pool_id],
                num_samples,
            )
            motion_ids[subset_mask] = sampled_motion_ids
            start_bins[subset_mask] = sampled_bins
            time_steps[subset_mask] = sampled_time_steps

        return motion_ids, start_bins, time_steps

    def _sample_eval_fixed_reset_states(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty

        reset_cfg = getattr(self.motion_cfg, "eval_fixed_reset", None)
        motion_id = int(getattr(reset_cfg, "motion_id", 0))
        if motion_id < 0 or motion_id >= int(self.motion.num_motions):
            raise ValueError(
                f"motion_config.eval_fixed_reset.motion_id must be in [0, {int(self.motion.num_motions) - 1}], "
                f"got {motion_id}."
            )

        motion_start = int(self.motion.motion_start_idx[motion_id].item())
        motion_end = int(self.motion.motion_end_idx[motion_id].item())
        motion_len = motion_end - motion_start
        if motion_len <= 0:
            raise ValueError(f"Motion {motion_id} has invalid length {motion_len}.")

        bin_id = getattr(reset_cfg, "bin_id", None)
        local_frame = getattr(reset_cfg, "local_frame", None)
        if bin_id is not None and local_frame is not None:
            raise ValueError("motion_config.eval_fixed_reset.bin_id and local_frame are mutually exclusive.")

        if local_frame is not None:
            local_frame = int(local_frame)
            if local_frame < 0 or local_frame >= motion_len:
                raise ValueError(
                    f"motion_config.eval_fixed_reset.local_frame must be in [0, {motion_len - 1}], "
                    f"got {local_frame} for motion {motion_id}."
                )
            local_steps = torch.full((env_ids.numel(),), local_frame, dtype=torch.long, device=self.device)
        elif bin_id is not None:
            bin_id = int(bin_id)
            bin_width_steps = self._eval_fixed_reset_bin_width_steps()
            num_bins = AdaptiveTimestepsSampler.num_bins_for_motion_length(motion_len, bin_width_steps)
            if bin_id < 0 or bin_id >= num_bins:
                raise ValueError(
                    f"motion_config.eval_fixed_reset.bin_id must be in [0, {num_bins - 1}], "
                    f"got {bin_id} for motion {motion_id}."
                )
            local_start = bin_id * bin_width_steps
            local_end = motion_len if bin_id == num_bins - 1 else min(local_start + bin_width_steps, motion_len)
            local_width = max(1, local_end - local_start)
            if bool(getattr(reset_cfg, "randomize_within_bin", True)):
                local_offset = torch.floor(torch.rand(env_ids.numel(), device=self.device) * float(local_width)).to(
                    dtype=torch.long
                )
                local_steps = local_start + local_offset
            else:
                local_steps = torch.full((env_ids.numel(),), local_start, dtype=torch.long, device=self.device)
        else:
            local_steps = torch.zeros(env_ids.numel(), dtype=torch.long, device=self.device)

        motion_ids = torch.full((env_ids.numel(),), motion_id, dtype=torch.long, device=self.device)
        time_steps = torch.full((env_ids.numel(),), motion_start, dtype=torch.long, device=self.device) + local_steps
        return motion_ids, time_steps

    def _eval_fixed_reset_bin_width_steps(self) -> int:
        adaptive_sampler = getattr(self, "adaptive_timesteps_sampler", None)
        if adaptive_sampler is not None and hasattr(adaptive_sampler, "bin_width_steps"):
            return max(1, int(adaptive_sampler.bin_width_steps))

        adaptive_cfg = getattr(self.motion_cfg, "adaptive_timestep_sampler", None)
        bin_width_s = float(getattr(adaptive_cfg, "bin_width_s", 1.0))
        env_dt = float(getattr(self._env, "dt", 0.0))
        if env_dt <= 0.0:
            raise ValueError(f"Eval fixed bin reset requires env.dt > 0, got {env_dt}.")
        return max(1, int(round(bin_width_s / env_dt)))

    def _sample_uniform_reset_states(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty

        phase = torch.rand(env_ids.numel(), device=self.device)
        self._sample_motion_ids_for_reset(env_ids)
        sampled_motion_ids = self.motion_ids[env_ids].clone()
        start_idx = self.motion.motion_start_idx[sampled_motion_ids]
        end_idx = self.motion.motion_end_idx[sampled_motion_ids]
        motion_len = end_idx - start_idx
        sampled_time_steps = start_idx + (phase * (motion_len - 1).float()).long()
        return sampled_motion_ids, sampled_time_steps

    def _sample_reset_states_with_adaptive_fallback(
        self,
        env_ids: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = self._ensure_index_tensor(env_ids)
        group_ids = group_ids.to(device=self.device, dtype=torch.long)
        motion_ids = torch.empty(env_ids.numel(), dtype=torch.long, device=self.device)
        time_steps = torch.empty(env_ids.numel(), dtype=torch.long, device=self.device)
        adaptive_env_mask = self._adaptive_sampler_enabled_env_mask(group_ids)

        if adaptive_env_mask.any():
            sampled_motion_ids, _, sampled_time_steps = self._sample_adaptive_reset_states(
                env_ids[adaptive_env_mask],
                group_ids[adaptive_env_mask],
            )
            motion_ids[adaptive_env_mask] = sampled_motion_ids
            time_steps[adaptive_env_mask] = sampled_time_steps

        uniform_env_mask = ~adaptive_env_mask
        if uniform_env_mask.any():
            sampled_motion_ids, sampled_time_steps = self._sample_uniform_reset_states(env_ids[uniform_env_mask])
            motion_ids[uniform_env_mask] = sampled_motion_ids
            time_steps[uniform_env_mask] = sampled_time_steps

        return motion_ids, time_steps

    def _record_adaptive_sampler_episode_resets(self, env_ids: torch.Tensor) -> None:
        reset_failure_mask = self._env.termination_manager.terminated[env_ids]
        timeout_mask = self._env.termination_manager.time_outs[env_ids]
        ended_mask = reset_failure_mask | timeout_mask
        if not ended_mask.any():
            return

        ended_env_ids = env_ids[ended_mask]
        active_mask = self.adaptive_sampler_segment_active_mask[ended_env_ids]
        if not active_mask.any():
            return
        ended_env_ids = ended_env_ids[active_mask]
        reset_failure_mask = reset_failure_mask[ended_mask][active_mask]
        self._record_adaptive_sampler_segment_ends(
            env_ids=ended_env_ids,
            terminal_time_steps=self.time_steps[ended_env_ids],
            failure_mask=reset_failure_mask,
        )

    def _record_adaptive_sampler_clip_ends(self, env_ids: torch.Tensor, terminal_time_steps: torch.Tensor) -> None:
        active_mask = self.adaptive_sampler_segment_active_mask[env_ids]
        if not active_mask.any():
            return
        env_ids = env_ids[active_mask]
        terminal_time_steps = terminal_time_steps[active_mask]
        self._record_adaptive_sampler_segment_ends(
            env_ids=env_ids,
            terminal_time_steps=terminal_time_steps,
            failure_mask=torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device),
        )

    def on_training_step(self, training_step: int, *, log_dir: str | None = None, is_main_process: bool = True) -> None:
        if not self.motion_cfg.use_adaptive_timesteps_sampler:
            return
        self._adaptive_sampler_collect_rollout = self._should_collect_adaptive_sampler_rollout(int(training_step))
        self._maybe_refresh_adaptive_sampler_metrics(training_step=int(training_step))
        if not is_main_process or log_dir is None:
            return
        self._maybe_dump_adaptive_sampler_state(training_step=training_step, log_dir=log_dir)

    def on_rollout_end(self, training_step: int, *, log_dir: str | None = None, is_main_process: bool = True) -> None:
        del training_step, log_dir, is_main_process
        if not self.motion_cfg.use_adaptive_timesteps_sampler:
            return
        if not self._adaptive_sampler_uses_batched_updates():
            return
        self.adaptive_timesteps_sampler.apply_pending_updates()
        self._adaptive_sampler_collect_rollout = False

    def _maybe_dump_adaptive_sampler_state(self, *, training_step: int, log_dir: str) -> None:
        dump_interval = max(0, int(self.motion_cfg.adaptive_timestep_sampler.dump_interval))
        if dump_interval <= 0:
            return

        training_step = int(training_step)
        if training_step <= 0 or training_step % dump_interval != 0:
            return
        if getattr(self, "_adaptive_sampler_last_dump_step", -1) == training_step:
            return
        self._adaptive_sampler_last_dump_step = training_step

        os.makedirs(log_dir, exist_ok=True)
        dump_path = os.path.join(log_dir, str(self.motion_cfg.adaptive_timestep_sampler.dump_file))
        dump_dir = os.path.dirname(dump_path)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        self._write_adaptive_sampler_dump(training_step=training_step, dump_path=dump_path)

    def _write_adaptive_sampler_dump(self, *, training_step: int, dump_path: str) -> None:
        assert self.motion_cfg.use_adaptive_timesteps_sampler

        group_names = self._adaptive_sampler_group_names()
        enabled_group_mask = getattr(self, "_adaptive_sampler_cached_enabled_group_mask", None)
        if enabled_group_mask is None:
            enabled_group_mask = self._adaptive_sampler_enabled_group_mask().to(device=self.device, dtype=torch.bool)
        group_priors = self.adaptive_timesteps_sampler._normalized_group_priors(
            self._adaptive_sampler_effective_group_priors()
        )
        group_first_frame_probs = self.adaptive_timesteps_sampler._group_first_frame_probs(
            self._get_cached_adaptive_sampler_group_first_frame_probs()
        )
        candidate_motion_ids_by_group = self._get_cached_adaptive_sampler_candidate_motion_ids_by_group()

        with open(dump_path, "a", encoding="utf-8") as f:
            f.write(f"step {int(training_step)}\n")
            f.write(f"bin_width_steps {self.adaptive_timesteps_sampler.bin_width_steps}\n")
            f.write(f"bin_width_sec {self.adaptive_timesteps_sampler.bin_width_s:.6f}\n")
            f.write(
                f"motion_uniform_mix {self.adaptive_timesteps_sampler.motion_uniform_mix:.8f}\n"
            )
            f.write(f"bin_uniform_mix {self.adaptive_timesteps_sampler.bin_uniform_mix:.8f}\n")
            f.write(f"score_power {self.adaptive_timesteps_sampler.score_power:.8f}\n")
            f.write(
                "columns group motion_id bin start_frame end_frame fail_ema "
                "conditional_bin_prob overall_bin_prob\n"
            )

            for group_id, group_name in enumerate(group_names):
                if not bool(enabled_group_mask[group_id].item()):
                    continue
                group_prob = float(group_priors[group_id].item())
                first_frame_prob = float(group_first_frame_probs[group_id].item())
                candidate_motion_ids = candidate_motion_ids_by_group.get(group_id)
                if candidate_motion_ids is None or candidate_motion_ids.numel() == 0:
                    f.write(
                        f"group {group_name} group_prob {group_prob:.8f} "
                        f"first_frame_prob {first_frame_prob:.8f} num_motions 0\n"
                    )
                    continue

                candidate_motion_ids = candidate_motion_ids.to(device=self.device, dtype=torch.long)
                motion_probs = self.adaptive_timesteps_sampler.motion_probabilities(group_id, candidate_motion_ids)
                f.write(
                    f"group {group_name} group_prob {group_prob:.8f} "
                    f"first_frame_prob {first_frame_prob:.8f} num_motions {int(candidate_motion_ids.numel())}\n"
                )
                for motion_prob, motion_id_tensor in zip(motion_probs, candidate_motion_ids):
                    motion_id = int(motion_id_tensor.item())
                    num_bins = int(self.adaptive_timesteps_sampler.motion_num_bins[motion_id].item())
                    motion_start = int(self.motion.motion_start_idx[motion_id].item())
                    motion_end = int(self.motion.motion_end_idx[motion_id].item())
                    motion_prob_value = float(motion_prob.item())
                    f.write(
                        f"motion {motion_id} global_start {motion_start} global_end {motion_end} "
                        f"seq_len {motion_end - motion_start} num_bins {num_bins} motion_prob {motion_prob_value:.8f}\n"
                    )
                    conditional_bin_probs = self.adaptive_timesteps_sampler.bin_probabilities(group_id, motion_id)
                    for bin_id in range(num_bins):
                        start_frame = int(self.adaptive_timesteps_sampler.bin_local_start_steps[motion_id, bin_id].item())
                        end_frame = int(self.adaptive_timesteps_sampler.bin_local_end_steps[motion_id, bin_id].item()) - 1
                        fail_ema = float(self.adaptive_timesteps_sampler.bin_fail_ema[group_id, motion_id, bin_id].item())
                        conditional_prob = float(conditional_bin_probs[bin_id].item())
                        overall_prob = group_prob * (1.0 - first_frame_prob) * motion_prob_value * conditional_prob
                        f.write(
                            f"{group_name} {motion_id} {bin_id} {start_frame} {end_frame} "
                            f"{fail_ema:.8f} {conditional_prob:.8f} {overall_prob:.8f}\n"
                        )
            f.write("\n")

    @staticmethod
    def _validate_probability(prob: float | None, *, field_name: str) -> float | None:
        if prob is None:
            return None
        prob = float(prob)
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"{field_name} must be in [0, 1], got {prob}")
        return prob

    def _initial_pose_noise_scale_multipliers(self, env_ids: torch.Tensor) -> torch.Tensor:
        return torch.ones(env_ids.numel(), dtype=torch.float32, device=self.device)

    def _robot_initial_pose_noise_scale_multipliers(self, env_ids: torch.Tensor) -> torch.Tensor:
        return torch.ones(env_ids.numel(), dtype=torch.float32, device=self.device)

    def _root_pose_initial_noise_scale_multipliers(
        self,
        env_ids: torch.Tensor,
        start_idx: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ones(env_ids.numel(), dtype=torch.float32, device=self.device)

    def _get_start_at_timestep_zero_probs(self, env_ids: torch.Tensor) -> torch.Tensor:
        prob = self._validate_probability(
            self.motion_cfg.start_at_timestep_zero_prob,
            field_name="motion_config.start_at_timestep_zero_prob",
        )
        assert prob is not None
        return torch.full((env_ids.numel(),), prob, dtype=torch.float32, device=self.device)

    def _sample_motion_ids_for_reset(self, env_ids: torch.Tensor) -> None:
        if self.object_groups:
            env_groups = self.env_object_group[env_ids]
            motion_ids = torch.empty(env_ids.numel(), dtype=torch.long, device=self.device)
            for g, ids in enumerate(self._group_motion_ids):
                sel = env_groups == g
                motion_ids[sel] = ids[torch.randint(0, ids.numel(), (int(sel.sum()),), device=self.device)]
            self.motion_ids[env_ids] = motion_ids
            return
        self.motion_ids[env_ids] = torch.randint(0, self.motion.num_motions, (env_ids.numel(),), device=self.device)

    def _noise_vector_3d(self, values: list[float], *, field_name: str) -> torch.Tensor:
        if len(values) != 3:
            raise ValueError(f"{field_name} must have exactly 3 entries, got {len(values)}")
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _sample_step_zero_root_noise(
        self,
        num_envs: int,
        noise_cfg: StepZeroRootNoiseConfig,
        *,
        pos_axis_scale_xyz: torch.Tensor | None = None,
        rpy_axis_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_std = self._noise_vector_3d(
            noise_cfg.pos_std_xyz,
            field_name="step_zero_root_noise.pos_std_xyz",
        ).unsqueeze(0)
        pos_clip = self._noise_vector_3d(
            noise_cfg.pos_clip_xyz,
            field_name="step_zero_root_noise.pos_clip_xyz",
        ).unsqueeze(0)
        if pos_axis_scale_xyz is not None:
            pos_scale = pos_axis_scale_xyz.to(device=self.device, dtype=torch.float32).reshape(1, 3)
            pos_std = pos_std * pos_scale
            pos_clip = pos_clip * pos_scale
        pos_noise = torch.randn(num_envs, 3, device=self.device, dtype=torch.float32) * pos_std
        pos_noise = torch.clamp(pos_noise, min=-pos_clip, max=pos_clip)

        rpy_std = self._noise_vector_3d(
            noise_cfg.rpy_std,
            field_name="step_zero_root_noise.rpy_std",
        ).unsqueeze(0)
        rpy_clip = self._noise_vector_3d(
            noise_cfg.rpy_clip,
            field_name="step_zero_root_noise.rpy_clip",
        ).unsqueeze(0)
        if rpy_axis_scale is not None:
            rpy_scale = rpy_axis_scale.to(device=self.device, dtype=torch.float32).reshape(1, 3)
            rpy_std = rpy_std * rpy_scale
            rpy_clip = rpy_clip * rpy_scale
        rpy_noise = torch.randn(num_envs, 3, device=self.device, dtype=torch.float32) * rpy_std
        rpy_noise = torch.clamp(rpy_noise, min=-rpy_clip, max=rpy_clip)
        quat_noise = quat_from_euler_xyz(rpy_noise[:, 0], rpy_noise[:, 1], rpy_noise[:, 2])
        return pos_noise, quat_normalize(quat_noise)

    def _apply_step_zero_root_noise_cfg(
        self,
        *,
        mask: torch.Tensor,
        noise_cfg: StepZeroRootNoiseConfig | None,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        target_root_pos: torch.Tensor,
        target_root_rot: torch.Tensor,
        pos_axis_scale_xyz: torch.Tensor | None = None,
        rpy_axis_scale: torch.Tensor | None = None,
    ) -> None:
        if noise_cfg is None or not mask.any():
            return

        pos_noise, quat_noise = self._sample_step_zero_root_noise(
            int(mask.sum().item()),
            noise_cfg,
            pos_axis_scale_xyz=pos_axis_scale_xyz,
            rpy_axis_scale=rpy_axis_scale,
        )
        target_root_pos[mask] = root_pos[mask] + pos_noise
        target_root_rot[mask] = quat_normalize(quat_mul(quat_noise, root_rot[mask], w_last=True))

    def _apply_step_zero_root_noise_overrides(
        self,
        env_ids: torch.Tensor,
        step_zero_mask: torch.Tensor,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        target_root_pos: torch.Tensor,
        target_root_rot: torch.Tensor,
    ) -> None:
        self._apply_step_zero_root_noise_cfg(
            mask=step_zero_mask,
            noise_cfg=self.init_pose_cfg.step_zero_root_noise,
            root_pos=root_pos,
            root_rot=root_rot,
            target_root_pos=target_root_pos,
            target_root_rot=target_root_rot,
        )

    def _apply_step_zero_root_pose_overrides(
        self,
        env_ids: torch.Tensor,
        start_idx: torch.Tensor,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        target_root_pos: torch.Tensor,
        target_root_rot: torch.Tensor,
    ) -> None:
        step_zero_mask = self.time_steps[env_ids] == start_idx
        if not step_zero_mask.any():
            return
        self._apply_step_zero_root_noise_overrides(
            env_ids,
            step_zero_mask,
            root_pos,
            root_rot,
            target_root_pos,
            target_root_rot,
        )

    def _object_reset_states(self, env_ids: torch.Tensor) -> torch.Tensor:
        obj_pos = self.object_pos_w[env_ids]
        obj_ori = self.object_quat_w[env_ids]
        obj_lin_vel = self.object_lin_vel_w[env_ids]
        obj_ang_vel = self.object_ang_vel_w[env_ids]

        obj_pos_noise = torch.tensor(
            [self.init_pose_cfg.object_pos],
            device=self.device,
        )
        obj_pos_noise = obj_pos_noise * self.init_pose_cfg.overall_noise_scale
        obj_pos_noise = obj_pos_noise * self._initial_pose_noise_scale_multipliers(env_ids)[:, None]
        target_obj_pos = obj_pos + (torch.rand(obj_pos.shape, device=self.device) - 0.5) * 2 * obj_pos_noise

        return torch.cat(
            [target_obj_pos, obj_ori, obj_lin_vel, obj_ang_vel],
            dim=-1,
        )

    def _maybe_add_terminal_pose_hold(self) -> None:
        """Optionally append fixed terminal hold frames to the end of each motion clip."""
        if not self.motion_cfg.enable_terminal_pose_hold:
            return

        duration = self.motion_cfg.terminal_pose_hold_duration_s
        if duration <= 0.0:
            return

        num_steps = round(duration / self._env.dt)
        if num_steps <= 0:
            logger.warning(
                "Terminal pose hold duration {}s is too short for dt {}; skipping augmentation.",
                duration,
                self._env.dt,
            )
            return

        self.motion = self.motion.append_terminal_pose_hold(num_steps)
        logger.info("append {} terminal hold frames ({}s)", num_steps, duration)

    def _maybe_add_default_pose_transition(self, *, prepend: bool) -> None:
        """Shared path for optionally inserting default-pose interpolation before/after the clip."""
        enabled = self.motion_cfg.enable_default_pose_prepend if prepend else self.motion_cfg.enable_default_pose_append
        if not enabled:
            return

        duration = (
            self.motion_cfg.default_pose_prepend_duration_s
            if prepend
            else self.motion_cfg.default_pose_append_duration_s
        )
        if duration <= 0.0:
            return

        num_steps = round(duration / self._env.dt)
        if num_steps <= 1:
            logger.warning(
                "Default pose {} duration {}s is too short for dt {}; skipping augmentation.",
                "prepend" if prepend else "append",
                duration,
                self._env.dt,
            )
            return

        default_state = self._build_default_pose_state(use_motion_end=not prepend)

        action = "prepend" if prepend else "append"
        log_str = f"{action} {num_steps} interpolated frames ({duration}s) from default pose to motion"
        try:
            self._add_transition_to_motion(default_state, num_steps, prepend=prepend)
            logger.info(log_str)
        except Exception as exc:
            logger.error(f"Failed to {action} default pose transition: {exc}")
            raise RuntimeError(
                f"Critical error during motion interpolation setup: {exc}\n"
                "This indicates a mismatch in tensor dimensions during interpolation. "
                "Please check that the motion file and robot configuration are compatible."
            ) from exc

    def _build_default_pose_state(self, use_motion_end: bool = False) -> dict[str, torch.Tensor]:
        """Build the state dict representing the robot's default standing pose.

        By default, anchor root pos/yaw to the motion start; when use_motion_end is True, anchor to motion end.
        """
        init_state = self._env.robot_config.init_state
        joint_pos = self._env.default_dof_pos_base.squeeze(0).to(self.device)
        joint_vel = torch.zeros_like(joint_pos)

        init_root_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
        init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)

        motion_idx = -1 if use_motion_end else 0

        # Assume the pelvis is the first in robot_body_names
        motion_root_pos = self.motion.body_pos_w[motion_idx, 0].to(self.device)
        motion_root_quat = self.motion.body_quat_w[motion_idx, 0].to(self.device).unsqueeze(0)
        _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

        # Keep z from init config but adopt the clip's x,y at the chosen anchor frame.
        default_root_pos = torch.tensor(
            [motion_root_pos[0], motion_root_pos[1], init_state.pos[2]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        # Keep roll/pitch from init config but adopt the clip's yaw at the chosen anchor frame.
        default_root_quat = quat_from_euler_xyz(
            init_roll.squeeze(0),
            init_pitch.squeeze(0),
            motion_yaw.squeeze(0),
        )
        default_root_lin_vel = torch.tensor(init_state.lin_vel, dtype=torch.float32, device=self.device)
        default_root_ang_vel = torch.tensor(init_state.ang_vel, dtype=torch.float32, device=self.device)

        body_states = self._capture_body_states(
            joint_pos,
            joint_vel,
            default_root_pos,
            default_root_quat,
            default_root_lin_vel,
            default_root_ang_vel,
        )

        default_body_pos = self._map_robot_bodies_to_motion_order(body_states["pos"])
        default_body_quat = self._map_robot_bodies_to_motion_order(body_states["quat"])
        default_body_lin_vel = self._map_robot_bodies_to_motion_order(body_states["lin_vel"])
        default_body_ang_vel = self._map_robot_bodies_to_motion_order(body_states["ang_vel"])

        if self.motion.has_object:
            object_pos = self.motion._object_pos_w[motion_idx].to(self.device)
            object_quat = self.motion._object_quat_w[motion_idx].to(self.device)
            # The default-pose target keeps the object fixed at the chosen motion pose.
            object_lin_vel = torch.zeros(3, device=self.device, dtype=torch.float32)
            object_ang_vel = torch.zeros(3, device=self.device, dtype=torch.float32)
        else:
            object_pos = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            object_quat = torch.zeros(0, 4, device=self.device, dtype=torch.float32)
            object_lin_vel = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            object_ang_vel = torch.zeros(0, 3, device=self.device, dtype=torch.float32)

        return {
            "joint_pos": joint_pos.clone(),
            "joint_vel": joint_vel,
            "root_pos": default_root_pos,
            "root_quat": default_root_quat,
            "root_lin_vel": default_root_lin_vel,
            "root_ang_vel": default_root_ang_vel,
            "body_pos": default_body_pos,
            "body_quat": default_body_quat,
            "body_lin_vel": default_body_lin_vel,
            "body_ang_vel": default_body_ang_vel,
            "object_pos": object_pos,
            "object_quat": object_quat,
            "object_lin_vel": object_lin_vel,
            "object_ang_vel": object_ang_vel,
        }

    def _add_transition_to_motion(self, default_state: dict[str, torch.Tensor], num_steps: int, prepend: bool) -> None:
        """Add interpolated frames either before or after the motion data."""
        assert self._body_indexes_in_motion is not None
        assert self._joint_indexes_in_motion is not None

        if num_steps <= 0:
            return

        device = self.device
        dtype = self.motion._joint_pos.dtype

        default_motion_state = self._default_motion_state(default_state, dtype=dtype, device=device)
        motion_state = self._motion_state(0 if prepend else -1, dtype=dtype, device=device)

        start_state = default_motion_state if prepend else motion_state
        target_state = motion_state if prepend else default_motion_state
        drop_first, drop_last = (False, True) if prepend else (True, False)

        self._build_and_apply_transition(
            start_state=start_state,
            target_state=target_state,
            num_steps=num_steps,
            prepend=prepend,
            drop_first=drop_first,
            drop_last=drop_last,
            dtype=dtype,
            device=device,
        )

    def _slerp_quat_sequence(self, start: torch.Tensor, end: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        """Spherically interpolate quaternions across multiple time steps."""
        if alphas.numel() == 0:
            return start.new_zeros((0,) + start.shape)

        num_steps = alphas.shape[0]
        start_expand = start.unsqueeze(0).expand(num_steps, -1, -1)
        end_expand = end.unsqueeze(0).expand(num_steps, -1, -1)
        alpha_flat = alphas.repeat_interleave(start.shape[0]).unsqueeze(-1)
        blended = slerp(
            start_expand.reshape(-1, 4),
            end_expand.reshape(-1, 4),
            alpha_flat,
        )
        return blended.view(num_steps, start.shape[0], 4)

    def _capture_body_states(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Capture body states by temporarily setting the robot state in the simulator."""
        simulator = self._env.simulator
        assert simulator.get_simulator_type() == SimulatorType.ISAACSIM, (
            "Default-pose interpolation only supports IsaacSim; IsaacGym write_state_updates does not run FK."
        )
        env_id = 0
        env_origin = simulator.scene.env_origins[env_id].to(self.device)

        root_backup = simulator.robot_root_states[env_id].clone()
        dof_pos_backup = simulator.dof_pos[env_id].clone()
        dof_vel_backup = simulator.dof_vel[env_id].clone()

        try:
            simulator.robot_root_states[env_id, :3] = root_pos + env_origin
            simulator.robot_root_states[env_id, 3:7] = root_quat
            simulator.robot_root_states[env_id, 7:10] = root_lin_vel
            simulator.robot_root_states[env_id, 10:13] = root_ang_vel
            simulator.dof_pos[env_id] = joint_pos
            simulator.dof_vel[env_id] = joint_vel

            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

            body_pos = (simulator._rigid_body_pos[env_id] - env_origin).clone()
            body_quat = simulator._rigid_body_rot[env_id].clone()
            body_lin_vel = simulator._rigid_body_vel[env_id].clone()
            body_ang_vel = simulator._rigid_body_ang_vel[env_id].clone()
        finally:
            simulator.robot_root_states[env_id] = root_backup
            simulator.dof_pos[env_id] = dof_pos_backup
            simulator.dof_vel[env_id] = dof_vel_backup
            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

        return {
            "pos": body_pos,
            "quat": body_quat,
            "lin_vel": body_lin_vel,
            "ang_vel": body_ang_vel,
        }

    def _map_robot_bodies_to_motion_order(self, robot_tensor: torch.Tensor) -> torch.Tensor:
        """Map robot body tensor to motion data order using body indexes."""
        assert self._body_indexes_in_motion is not None
        num_motion_bodies = self.motion._body_pos_w.shape[1]
        motion_shape = (num_motion_bodies,) + robot_tensor.shape[1:]
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[self._body_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _map_robot_joints_to_motion_order(
        self, robot_tensor: torch.Tensor, num_motion_joints: int | None = None
    ) -> torch.Tensor:
        """Map robot joint tensor to motion data order using joint indexes."""
        assert self._joint_indexes_in_motion is not None
        if num_motion_joints is None:
            num_motion_joints = self.motion._joint_pos.shape[1]
        motion_shape = robot_tensor.shape[:-1] + (num_motion_joints,)
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[..., self._joint_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _motion_state(self, idx: int, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
        """Slice motion tensors at a given index into a state dict."""
        state = {
            "joint_pos": self.motion._joint_pos[idx].to(device=device, dtype=dtype),
            "joint_vel": self.motion._joint_vel[idx].to(device=device, dtype=dtype),
            "body_pos": self.motion._body_pos_w[idx].to(device=device, dtype=dtype),
            "body_quat": self.motion._body_quat_w[idx].to(device=device, dtype=dtype),
            "body_lin_vel": self.motion._body_lin_vel_w[idx].to(device=device, dtype=dtype),
            "body_ang_vel": self.motion._body_ang_vel_w[idx].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = self.motion._object_pos_w[idx].to(device=device, dtype=dtype)
            state["object_quat"] = self.motion._object_quat_w[idx].to(device=device, dtype=dtype)
            state["object_lin_vel"] = self.motion._object_lin_vel_w[idx].to(device=device, dtype=dtype)
            state["object_ang_vel"] = self.motion._object_ang_vel_w[idx].to(device=device, dtype=dtype)
        return state

    def _default_motion_state(
        self, default_state: dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Map default robot-state tensors into motion order for interpolation."""
        state = {
            "joint_pos": self._map_robot_joints_to_motion_order(
                default_state["joint_pos"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_pos.shape[1],
            ),
            "joint_vel": self._map_robot_joints_to_motion_order(
                default_state["joint_vel"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_vel.shape[1],
            ),
            "body_pos": default_state["body_pos"].to(device=device, dtype=dtype),
            "body_quat": default_state["body_quat"].to(device=device, dtype=dtype),
            "body_lin_vel": default_state["body_lin_vel"].to(device=device, dtype=dtype),
            "body_ang_vel": default_state["body_ang_vel"].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = default_state["object_pos"].to(device=device, dtype=dtype)
            state["object_quat"] = default_state["object_quat"].to(device=device, dtype=dtype)
            state["object_lin_vel"] = default_state["object_lin_vel"].to(device=device, dtype=dtype)
            state["object_ang_vel"] = default_state["object_ang_vel"].to(device=device, dtype=dtype)
        return state

    def _build_transition_segments(
        self,
        start: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        alphas: torch.Tensor,
        alphas_joint: torch.Tensor,
        alphas_body: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Linearly/spherically interpolate between start and target states."""

        def _lerp(a: torch.Tensor, b: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
            return a.unsqueeze(0) + view * (b - a).unsqueeze(0)

        segments = {
            "joint_pos": _lerp(start["joint_pos"], target["joint_pos"], alphas_joint),
            "joint_vel": _lerp(start["joint_vel"], target["joint_vel"], alphas_joint),
            "body_pos": _lerp(start["body_pos"], target["body_pos"], alphas_body),
            "body_lin_vel": _lerp(start["body_lin_vel"], target["body_lin_vel"], alphas_body),
            "body_ang_vel": _lerp(start["body_ang_vel"], target["body_ang_vel"], alphas_body),
            "body_quat": self._slerp_quat_sequence(start["body_quat"], target["body_quat"], alphas),
        }

        if self.motion.has_object:
            segments["object_pos"] = _lerp(start["object_pos"], target["object_pos"], alphas_joint)
            segments["object_lin_vel"] = _lerp(start["object_lin_vel"], target["object_lin_vel"], alphas_joint)
            segments["object_ang_vel"] = _lerp(start["object_ang_vel"], target["object_ang_vel"], alphas_joint)
            segments["object_quat"] = self._slerp_quat_sequence(
                start["object_quat"].unsqueeze(0), target["object_quat"].unsqueeze(0), alphas
            ).squeeze(1)

        return segments

    def _apply_transition_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> None:
        """Splice interpolated segments into motion data, either prepending or appending."""
        self.motion = self.motion.extend_with_segments(segments, prepend=prepend)

    def _build_and_apply_transition(
        self,
        start_state: dict[str, torch.Tensor],
        target_state: dict[str, torch.Tensor],
        num_steps: int,
        prepend: bool,
        drop_first: bool,
        drop_last: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Shared interpolation path for prepend/append transitions."""
        if num_steps <= 0:
            return

        alphas = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=dtype)
        if drop_first:
            alphas = alphas[1:]
        if drop_last:
            alphas = alphas[:-1]
        if alphas.numel() == 0:
            return

        alphas_joint = alphas.view(num_steps, 1)
        alphas_body = alphas.view(num_steps, 1, 1)

        segments = self._build_transition_segments(start_state, target_state, alphas, alphas_joint, alphas_body)
        self._apply_transition_segments(segments, prepend=prepend)

    def _setup_visualization_markers_for_isaacsim(self):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG

        self.visualization_markers = {}

        for body_names in self.motion_cfg.body_names_to_track:
            visualization_markers_cfg = RAY_CASTER_MARKER_CFG.replace(
                prim_path=f"/Visuals/Command/motion_robot_body/motion_{body_names}",
            )
            visualization_markers_cfg.markers["hit"].radius = 0.03
            visualization_markers_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            self.visualization_markers[f"motion_{body_names}"] = VisualizationMarkers(visualization_markers_cfg)

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/goal_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.28, 0.28, 0.28)
        self.visualization_markers["goal_robot"] = VisualizationMarkers(visualization_markers_cfg)

        if self.motion.has_object:
            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/real_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            real_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/motion_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            motion_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/goal_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.28, 0.28, 0.28)
            goal_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            self.visualization_markers["real_object"] = real_object_visualizer
            self.visualization_markers["motion_object"] = motion_object_visualizer
            self.visualization_markers["goal_object"] = goal_object_visualizer

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class HybridMotionCommand(MotionCommand):
    """Hybrid WBT command that mixes tracking and goal-generation environments."""

    def __init__(self, cfg: Any, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.hybrid_cfg: HybridMotionConfig | None = self.motion_cfg.hybrid

    def setup(self) -> None:
        self.hybrid_cfg = self.motion_cfg.hybrid
        if self.hybrid_cfg is None:
            raise ValueError("HybridMotionCommand requires motion_config.hybrid to be configured.")
        super().setup()
        self._build_fixed_env_partition()
        self._setup_fixed_size_randomization()
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self._refresh_adaptive_sampler_runtime_cache()
            self._refresh_adaptive_sampler_metrics()

    def init_buffers(self):
        track_env_mask = getattr(self, "track_env_mask", None)
        gen_env_mask = getattr(self, "gen_env_mask", None)
        object_goal_noise_scale = getattr(self, "_object_goal_noise_curriculum_scale", None)
        super().init_buffers()
        if track_env_mask is not None and gen_env_mask is not None:
            self.track_env_mask = track_env_mask.to(device=self.device, dtype=torch.bool).clone()
            self.gen_env_mask = gen_env_mask.to(device=self.device, dtype=torch.bool).clone()
        else:
            self.track_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.gen_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._exact_goal_ref_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._exact_goal_ref_quat_w = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self._exact_goal_ref_quat_w[:, 3] = 1.0
        self._active_goal_ref_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._active_goal_ref_quat_w = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self._active_goal_ref_quat_w[:, 3] = 1.0

        self._exact_goal_object_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._exact_goal_object_quat_w = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self._exact_goal_object_quat_w[:, 3] = 1.0
        self._active_goal_object_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._active_goal_object_quat_w = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self._active_goal_object_quat_w[:, 3] = 1.0
        self.set_object_goal_noise_curriculum_scale(self._restored_curriculum_scale_value(object_goal_noise_scale))

    @property
    def exact_goal_ref_pos_w(self) -> torch.Tensor:
        return self._exact_goal_ref_pos_w

    @property
    def exact_goal_ref_quat_w(self) -> torch.Tensor:
        return self._exact_goal_ref_quat_w

    @property
    def exact_goal_object_pos_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            raise RuntimeError("Exact object goal requested for a motion without object data.")
        return self._exact_goal_object_pos_w

    @property
    def exact_goal_object_quat_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            raise RuntimeError("Exact object goal requested for a motion without object data.")
        return self._exact_goal_object_quat_w

    @property
    def goal_ref_pos_w(self) -> torch.Tensor:
        return self._active_goal_ref_pos_w

    @property
    def goal_ref_quat_w(self) -> torch.Tensor:
        if self._robot_goal_yaw_only_enabled():
            return yaw_quat(self._active_goal_ref_quat_w, w_last=True)
        return self._active_goal_ref_quat_w

    @property
    def goal_object_pos_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            raise RuntimeError("Goal object pose requested for a motion without object data.")
        return self._active_goal_object_pos_w

    @property
    def goal_object_quat_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            raise RuntimeError("Goal object pose requested for a motion without object data.")
        if self._object_goal_yaw_only_enabled():
            return yaw_quat(self._active_goal_object_quat_w, w_last=True)
        return self._active_goal_object_quat_w

    @property
    def object_goal_noise_curriculum_scale(self) -> torch.Tensor:
        scale = getattr(self, "_object_goal_noise_curriculum_scale", None)
        if scale is None:
            return torch.tensor(1.0, dtype=torch.float32, device=self.device)
        return scale

    @property
    def object_goal_noise_pos_axis_scale(self) -> torch.Tensor:
        scale = getattr(self, "_object_goal_noise_pos_axis_scale", None)
        if scale is None:
            return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)
        return scale

    @property
    def object_goal_noise_rpy_axis_scale(self) -> torch.Tensor:
        scale = getattr(self, "_object_goal_noise_rpy_axis_scale", None)
        if scale is None:
            return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)
        return scale

    def update_metrics(self):
        super().update_metrics()
        self.metrics["hybrid/track_ratio"] = self.track_env_mask.float().mean()
        self.metrics["hybrid/gen_ratio"] = self.gen_env_mask.float().mean()
        if self._fixed_size_randomization_enabled():
            assert self.hybrid_cfg is not None
            size_cfg = self.hybrid_cfg.size_randomization
            self._log_fixed_size_group_metrics("track", self.track_env_mask, size_cfg.track_scale_range)
            self._log_fixed_size_group_metrics("gen", self.gen_env_mask, size_cfg.gen_scale_range)

    def _log_fixed_size_group_metrics(
        self,
        group_name: str,
        group_mask: torch.Tensor,
        configured_range: list[float],
    ) -> None:
        samples = self._fixed_size_scalars[group_mask]
        nan_value = torch.tensor(float("nan"), dtype=torch.float32, device=self.device)
        self.metrics[f"hybrid/size/{group_name}_mean"] = samples.mean() if samples.numel() else nan_value
        self.metrics[f"hybrid/size/{group_name}_min"] = samples.min() if samples.numel() else nan_value
        self.metrics[f"hybrid/size/{group_name}_max"] = samples.max() if samples.numel() else nan_value
        self.metrics[f"hybrid/size/{group_name}_configured_min"] = torch.tensor(
            float(configured_range[0]), dtype=torch.float32, device=self.device
        )
        self.metrics[f"hybrid/size/{group_name}_configured_max"] = torch.tensor(
            float(configured_range[1]), dtype=torch.float32, device=self.device
        )

    def set_object_goal_noise_curriculum_scale(self, scale: float | torch.Tensor) -> None:
        scale_value = self._curriculum_scale_tensor(scale)
        self._object_goal_noise_curriculum_scale = scale_value
        self._object_goal_noise_pos_axis_scale = torch.tensor(
            [float(scale_value.item()), float(scale_value.item()), 1.0],
            dtype=torch.float32,
            device=self.device,
        )
        self._object_goal_noise_rpy_axis_scale = torch.tensor(
            [1.0, 1.0, float(scale_value.item())],
            dtype=torch.float32,
            device=self.device,
        )

    def _apply_gen_robot_goal_noise(self, env_ids: torch.Tensor) -> None:
        assert self.hybrid_cfg is not None
        if env_ids.numel() == 0:
            return
        self._apply_gen_goal_noise(
            env_ids,
            exact_goal_pos_w=self._exact_goal_ref_pos_w[env_ids],
            exact_goal_quat_w=self._exact_goal_ref_quat_w[env_ids],
            noise_cfg=self.hybrid_cfg.robot_goal_noise,
            active_goal_pos_w=self._active_goal_ref_pos_w,
            active_goal_quat_w=self._active_goal_ref_quat_w,
        )

    def reapply_gen_object_goal_noise(self, env_ids: torch.Tensor | None) -> None:
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0 or not self.motion.has_object:
            return
        gen_env_ids = env_ids[self.gen_env_mask[env_ids]]
        if gen_env_ids.numel() == 0:
            return
        self._apply_gen_object_goal_noise(gen_env_ids)

    def _apply_gen_object_goal_noise(self, env_ids: torch.Tensor) -> None:
        assert self.hybrid_cfg is not None
        if env_ids.numel() == 0 or not self.motion.has_object:
            return
        self._apply_gen_goal_noise(
            env_ids,
            exact_goal_pos_w=self._exact_goal_object_pos_w[env_ids],
            exact_goal_quat_w=self._exact_goal_object_quat_w[env_ids],
            noise_cfg=self.hybrid_cfg.object_goal_noise,
            active_goal_pos_w=self._active_goal_object_pos_w,
            active_goal_quat_w=self._active_goal_object_quat_w,
            pos_axis_scale_xyz=self.object_goal_noise_pos_axis_scale,
            rpy_axis_scale=self.object_goal_noise_rpy_axis_scale,
        )
        if self._fixed_size_randomization_enabled():
            reference_pos, reference_quat = self._raw_object_motion_poses(self.goal_time_steps[env_ids], env_ids)
            self._active_goal_object_pos_w[env_ids] = self._correct_scaled_object_positions(
                env_ids,
                self._active_goal_object_pos_w[env_ids],
                self._active_goal_object_quat_w[env_ids],
                reference_pos,
                reference_quat,
            )

    def _restored_curriculum_scale_value(self, scale: float | torch.Tensor | None) -> float:
        if scale is None:
            return 1.0
        if isinstance(scale, torch.Tensor):
            return float(scale.detach().item())
        return float(scale)

    def _curriculum_scale_tensor(self, scale: float | torch.Tensor) -> torch.Tensor:
        scale_value = torch.as_tensor(scale, dtype=torch.float32, device=self.device).reshape(())
        return torch.clamp(scale_value, min=0.0)

    def _adaptive_sampler_group_names(self) -> tuple[str, ...]:
        return ("track", "gen")

    def _adaptive_sampler_group_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        group_ids = torch.zeros(env_ids.numel(), dtype=torch.long, device=self.device)
        group_ids[self.gen_env_mask[env_ids]] = 1
        return group_ids

    def _adaptive_sampler_group_priors(self) -> torch.Tensor:
        return torch.stack(
            [
                self.track_env_mask.float().mean(),
                self.gen_env_mask.float().mean(),
            ]
        )

    def _adaptive_sampler_enabled_group_mask(self) -> torch.Tensor:
        return torch.tensor([True, False], dtype=torch.bool, device=self.device)

    def _adaptive_sampler_group_first_frame_probs(self) -> torch.Tensor:
        probs = super()._adaptive_sampler_group_first_frame_probs()
        assert self.hybrid_cfg is not None

        gen_prob = self._validate_probability(
            self.hybrid_cfg.gen_start_at_timestep_zero_prob,
            field_name="motion_config.hybrid.gen_start_at_timestep_zero_prob",
        )
        if gen_prob is None:
            return probs

        probs = probs.clone()
        probs[1] = gen_prob
        return probs

    def _apply_step_zero_root_noise_overrides(
        self,
        env_ids: torch.Tensor,
        step_zero_mask: torch.Tensor,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        target_root_pos: torch.Tensor,
        target_root_rot: torch.Tensor,
    ) -> None:
        super()._apply_step_zero_root_noise_overrides(
            env_ids,
            step_zero_mask,
            root_pos,
            root_rot,
            target_root_pos,
            target_root_rot,
        )
        if self.hybrid_cfg is None:
            return

        gen_mask = step_zero_mask & self.gen_env_mask[env_ids]
        self._apply_step_zero_root_noise_cfg(
            mask=gen_mask,
            noise_cfg=self.hybrid_cfg.gen_step_zero_root_noise,
            root_pos=root_pos,
            root_rot=root_rot,
            target_root_pos=target_root_pos,
            target_root_rot=target_root_rot,
        )

    def _get_start_at_timestep_zero_probs(self, env_ids: torch.Tensor) -> torch.Tensor:
        probs = super()._get_start_at_timestep_zero_probs(env_ids)
        assert self.hybrid_cfg is not None

        gen_prob = self._validate_probability(
            self.hybrid_cfg.gen_start_at_timestep_zero_prob,
            field_name="motion_config.hybrid.gen_start_at_timestep_zero_prob",
        )
        if gen_prob is None:
            return probs

        gen_mask = self.gen_env_mask[env_ids]
        if not gen_mask.any():
            return probs

        probs = probs.clone()
        probs[gen_mask] = gen_prob
        return probs

    def _build_fixed_env_partition(self) -> None:
        assert self.hybrid_cfg is not None
        track_ratio = float(self.hybrid_cfg.track_ratio)
        track_mask = build_fixed_hybrid_track_mask(
            self.num_envs,
            track_ratio,
            int(self.hybrid_cfg.env_partition_seed),
        )
        self.track_env_mask = track_mask.to(device=self.device)
        self.gen_env_mask = ~self.track_env_mask

        logger.info(
            "HybridMotionCommand: fixed env partition with {} track envs and {} gen envs (track_ratio={:.3f}, seed={})",
            int(self.track_env_mask.sum().item()),
            int(self.gen_env_mask.sum().item()),
            track_ratio,
            int(self.hybrid_cfg.env_partition_seed),
        )

    def _setup_fixed_size_randomization(self) -> None:
        assert self.hybrid_cfg is not None
        size_cfg = self.hybrid_cfg.size_randomization
        if not size_cfg.enabled:
            return

        actor_name = "object" if size_cfg.mode == "uniform_xyz" else "climb_box"
        scales_xyz = self._env.simulator.get_actor_scale_factors(actor_name).to(
            device=self.device,
            dtype=torch.float32,
        )
        if scales_xyz.shape != (self.num_envs, 3):
            raise ValueError(
                f"Actor '{actor_name}' fixed scale tensor must have shape ({self.num_envs}, 3), "
                f"got {tuple(scales_xyz.shape)}."
            )

        expected_scalars = sample_fixed_hybrid_size_scales(self.track_env_mask.cpu(), size_cfg)
        expected_scales = expand_size_scalars(expected_scalars, size_cfg.mode).to(self.device)
        if not torch.equal(scales_xyz, expected_scales):
            raise RuntimeError(
                "Pre-simulation hybrid size assignment does not match the runtime track/gen partition and seed."
            )

        self._fixed_actor_name = actor_name
        self._fixed_actor_scales_xyz = scales_xyz
        self._fixed_size_scalars = scales_xyz[:, 0] if size_cfg.mode == "uniform_xyz" else scales_xyz[:, 2]
        if actor_name == "object":
            bounds = self._env.simulator.get_actor_local_collision_bounds(actor_name).to(
                device=self.device,
                dtype=torch.float32,
            )
            self._fixed_object_collision_corners = collision_bounds_corners(bounds)

    def _fixed_size_randomization_enabled(self) -> bool:
        return bool(
            self.hybrid_cfg is not None
            and self.hybrid_cfg.size_randomization.enabled
            and hasattr(self, "_fixed_actor_scales_xyz")
        )

    def _raw_object_motion_poses(
        self,
        time_steps: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.motion.object_pos_w[time_steps] + self._env.simulator.scene.env_origins[env_ids]
        quaternions = self.motion.object_quat_w[time_steps]
        return positions, quaternions

    def _correct_scaled_object_positions(
        self,
        env_ids: torch.Tensor,
        candidate_positions_w: torch.Tensor,
        candidate_quaternions_xyzw: torch.Tensor,
        reference_positions_w: torch.Tensor,
        reference_quaternions_xyzw: torch.Tensor,
    ) -> torch.Tensor:
        assert self.hybrid_cfg is not None
        ground_z = self._env.simulator.scene.env_origins[env_ids, 2]
        return correct_object_positions_for_ground(
            candidate_positions_w,
            candidate_quaternions_xyzw,
            reference_positions_w,
            reference_quaternions_xyzw,
            self._fixed_actor_scales_xyz[env_ids],
            self._fixed_object_collision_corners,
            ground_z,
            clearance=float(self.hybrid_cfg.size_randomization.ground_clearance),
            contact_tolerance=float(self.hybrid_cfg.size_randomization.ground_contact_tolerance),
        )

    @property
    def object_pos_w(self) -> torch.Tensor:
        raw_positions = super().object_pos_w
        if not self._fixed_size_randomization_enabled() or not self.motion.has_object:
            return raw_positions
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        return self._correct_scaled_object_positions(
            env_ids,
            raw_positions,
            self.motion.object_quat_w[self.time_steps],
            raw_positions,
            self.motion.object_quat_w[self.time_steps],
        )

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        if not self._fixed_size_randomization_enabled() or not self.motion.has_object:
            return super().object_lin_vel_w

        reference_velocity = super().object_lin_vel_w
        motion_start = self.motion.motion_start_idx[self.motion_ids]
        motion_end = self.motion.motion_end_idx[self.motion_ids]
        previous_steps = torch.maximum(self.time_steps - 1, motion_start)
        next_steps = torch.minimum(self.time_steps + 1, motion_end - 1)
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        previous_pos, previous_quat = self._raw_object_motion_poses(previous_steps, env_ids)
        next_pos, next_quat = self._raw_object_motion_poses(next_steps, env_ids)
        corrected_previous = self._correct_scaled_object_positions(
            env_ids,
            previous_pos,
            previous_quat,
            previous_pos,
            previous_quat,
        )
        corrected_next = self._correct_scaled_object_positions(
            env_ids,
            next_pos,
            next_quat,
            next_pos,
            next_quat,
        )
        elapsed = ((next_steps - previous_steps).to(torch.float32) / float(self.motion.fps)).clamp_min(
            1.0 / float(self.motion.fps)
        )
        previous_correction = corrected_previous - previous_pos
        next_correction = corrected_next - next_pos
        correction_velocity = (next_correction - previous_correction) / elapsed.unsqueeze(-1)
        return reference_velocity + correction_velocity

    def _object_reset_states(self, env_ids: torch.Tensor) -> torch.Tensor:
        object_states = super()._object_reset_states(env_ids)
        if not self._fixed_size_randomization_enabled() or not self.motion.has_object:
            return object_states
        reference_pos, reference_quat = self._raw_object_motion_poses(self.time_steps[env_ids], env_ids)
        object_states[:, :3] = self._correct_scaled_object_positions(
            env_ids,
            object_states[:, :3],
            object_states[:, 3:7],
            reference_pos,
            reference_quat,
        )
        return object_states

    def _on_goal_time_steps_updated(self, env_ids: torch.Tensor) -> None:
        self._refresh_active_goals(env_ids)

    def _refresh_active_goals(self, env_ids: torch.Tensor) -> None:
        assert self.hybrid_cfg is not None

        goal_time_steps = self.goal_time_steps[env_ids]
        env_origins = self._env.simulator.scene.env_origins[env_ids]

        exact_goal_ref_pos_w = self.motion.body_pos_w[goal_time_steps, self.ref_body_index] + env_origins
        exact_goal_ref_quat_w = self.motion.body_quat_w[goal_time_steps, self.ref_body_index]

        self._exact_goal_ref_pos_w[env_ids] = exact_goal_ref_pos_w
        self._exact_goal_ref_quat_w[env_ids] = exact_goal_ref_quat_w
        self._active_goal_ref_pos_w[env_ids] = exact_goal_ref_pos_w
        self._active_goal_ref_quat_w[env_ids] = exact_goal_ref_quat_w

        if self.motion.has_object:
            exact_goal_object_pos_w = self.motion.object_pos_w[goal_time_steps] + env_origins
            exact_goal_object_quat_w = self.motion.object_quat_w[goal_time_steps]
            if self._fixed_size_randomization_enabled():
                exact_goal_object_pos_w = self._correct_scaled_object_positions(
                    env_ids,
                    exact_goal_object_pos_w,
                    exact_goal_object_quat_w,
                    exact_goal_object_pos_w,
                    exact_goal_object_quat_w,
                )
            self._exact_goal_object_pos_w[env_ids] = exact_goal_object_pos_w
            self._exact_goal_object_quat_w[env_ids] = exact_goal_object_quat_w
            self._active_goal_object_pos_w[env_ids] = exact_goal_object_pos_w
            self._active_goal_object_quat_w[env_ids] = exact_goal_object_quat_w

        gen_env_ids = env_ids[self.gen_env_mask[env_ids]]
        if gen_env_ids.numel() == 0:
            return

        self._apply_gen_robot_goal_noise(gen_env_ids)

        if self.motion.has_object:
            self._apply_gen_object_goal_noise(gen_env_ids)

    def _apply_gen_goal_noise(
        self,
        env_ids: torch.Tensor,
        *,
        exact_goal_pos_w: torch.Tensor,
        exact_goal_quat_w: torch.Tensor,
        noise_cfg: GoalPoseNoiseConfig,
        active_goal_pos_w: torch.Tensor,
        active_goal_quat_w: torch.Tensor,
        pos_axis_scale_xyz: torch.Tensor | None = None,
        rpy_axis_scale: torch.Tensor | None = None,
    ) -> None:
        pos_noise, quat_noise = self._sample_goal_noise_with_axis_scale(
            env_ids.numel(),
            noise_cfg,
            pos_axis_scale_xyz=pos_axis_scale_xyz,
            rpy_axis_scale=rpy_axis_scale,
        )
        active_goal_pos_w[env_ids] = exact_goal_pos_w + pos_noise
        active_goal_quat_w[env_ids] = quat_normalize(quat_mul(quat_noise, exact_goal_quat_w, w_last=True))

    def _sample_goal_noise_with_axis_scale(
        self,
        num_envs: int,
        noise_cfg: GoalPoseNoiseConfig,
        *,
        pos_axis_scale_xyz: torch.Tensor | None = None,
        rpy_axis_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pos_axis_scale_xyz is None and rpy_axis_scale is None:
            return self._sample_goal_noise(num_envs, noise_cfg)

        pos_scale = (
            torch.ones(3, dtype=torch.float32, device=self.device)
            if pos_axis_scale_xyz is None
            else pos_axis_scale_xyz.to(device=self.device, dtype=torch.float32)
        )
        rpy_scale = (
            torch.ones(3, dtype=torch.float32, device=self.device)
            if rpy_axis_scale is None
            else rpy_axis_scale.to(device=self.device, dtype=torch.float32)
        )
        scaled_noise_cfg = GoalPoseNoiseConfig(
            pos_std_xyz=(torch.tensor(noise_cfg.pos_std_xyz, dtype=torch.float32, device=self.device) * pos_scale).tolist(),
            pos_clip_xyz=(
                torch.tensor(noise_cfg.pos_clip_xyz, dtype=torch.float32, device=self.device) * pos_scale
            ).tolist(),
            rpy_std=(torch.tensor(noise_cfg.rpy_std, dtype=torch.float32, device=self.device) * rpy_scale).tolist(),
            rpy_clip=(torch.tensor(noise_cfg.rpy_clip, dtype=torch.float32, device=self.device) * rpy_scale).tolist(),
        )
        return self._sample_goal_noise(num_envs, scaled_noise_cfg)

    def _sample_goal_noise(self, num_envs: int, noise_cfg: GoalPoseNoiseConfig) -> tuple[torch.Tensor, torch.Tensor]:
        pos_std = torch.tensor(noise_cfg.pos_std_xyz, dtype=torch.float32, device=self.device).unsqueeze(0)
        pos_clip = torch.tensor(noise_cfg.pos_clip_xyz, dtype=torch.float32, device=self.device).unsqueeze(0)
        pos_noise = torch.randn(num_envs, 3, device=self.device, dtype=torch.float32) * pos_std
        pos_noise = torch.clamp(pos_noise, min=-pos_clip, max=pos_clip)

        rpy_std = torch.tensor(noise_cfg.rpy_std, dtype=torch.float32, device=self.device).unsqueeze(0)
        rpy_clip = torch.tensor(noise_cfg.rpy_clip, dtype=torch.float32, device=self.device).unsqueeze(0)
        rpy_noise = torch.randn(num_envs, 3, device=self.device, dtype=torch.float32) * rpy_std
        rpy_noise = torch.clamp(rpy_noise, min=-rpy_clip, max=rpy_clip)

        quat_noise = quat_from_euler_xyz(rpy_noise[:, 0], rpy_noise[:, 1], rpy_noise[:, 2])
        return pos_noise, quat_normalize(quat_noise)
