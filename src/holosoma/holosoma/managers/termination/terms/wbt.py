"""Whole Body Tracking-specific termination terms."""

from __future__ import annotations

from typing import Any, List

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.terms.wbt import gravity_vector
from holosoma.managers.termination.base import TerminationTermBase
from holosoma.utils.rotations import (
    quat_error_magnitude,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch


#########################################################################################################
## Termination terms
#########################################################################################################
def motion_ends(env, **_) -> torch.Tensor:
    """Terminate if the motion ends."""
    motion_command = env.command_manager.get_state("motion_command")
    return motion_command.time_steps >= motion_command.motion.time_step_total - 2


class BadTracking(TerminationTermBase):
    """Terminate if the tracking is bad.

    - bad ref pos
    - bad ref ori
    - bad motion body pos
    if has object:
        - bad object pos
        - bad object ori

    When bad tracking is detected, the motion_commmand.AdaptiveTimestepsSampler will be updated.
    """

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self._last_breakdown: dict[str, torch.Tensor] = {}

        self.bad_ref_pos_threshold = cfg.params["bad_ref_pos_threshold"]
        self.bad_ref_ori_threshold = cfg.params["bad_ref_ori_threshold"]

        self.bad_motion_body_pos_body_names = cfg.params["bad_motion_body_pos_body_names"]

        # NOTE: body_names_to_track is shared with command_manager
        self.body_names_to_track = cfg.params["body_names_to_track"]
        self.bad_motion_body_pos_threshold = cfg.params["bad_motion_body_pos_threshold"]
        self.bad_motion_body_pos_body_indexes = self._get_index_of_a_in_b(
            self.bad_motion_body_pos_body_names, self.body_names_to_track, self.env.device
        )

        self.bad_object_pos_threshold = cfg.params["bad_object_pos_threshold"]
        self.bad_object_ori_threshold = cfg.params["bad_object_ori_threshold"]

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        assert motion_command.motion_cfg.body_names_to_track == self.body_names_to_track, (
            "body_names_to_track in motion_command and termination.params are not the same"
            f"motion_command.motion_cfg.body_names_to_track: {motion_command.motion_cfg.body_names_to_track}"
            f"termination.params['body_names_to_track']: {self.body_names_to_track}"
        )

        # return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        bad_ref_pos = self.bad_ref_pos(motion_command)
        bad_ref_ori = self.bad_ref_ori(motion_command)
        bad_motion_body_pos = self.bad_motion_body_pos(motion_command)
        bad_object_pos = torch.zeros_like(bad_tracking := bad_ref_pos | bad_ref_ori | bad_motion_body_pos)
        bad_object_ori = torch.zeros_like(bad_tracking)

        if motion_command.motion.has_object:
            bad_object_pos = self.bad_object_pos(motion_command)
            bad_object_ori = self.bad_object_ori(motion_command)
            bad_tracking |= bad_object_pos | bad_object_ori

        self._last_breakdown = {
            self._bad_ref_pos_component_name(): bad_ref_pos.detach().clone(),
            "ref_ori": bad_ref_ori.detach().clone(),
            self._bad_body_pos_component_name(): bad_motion_body_pos.detach().clone(),
            "object_pos": bad_object_pos.detach().clone(),
            "object_ori": bad_object_ori.detach().clone(),
        }

        return bad_tracking

    def get_last_breakdown(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self._last_breakdown.items()}

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference position is too far from the robot's position."""
        return torch.norm(motion_command.ref_pos_w - motion_command.robot_ref_pos_w, dim=1) > self.bad_ref_pos_threshold

    def bad_ref_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference orientation is too far from the robot's orientation."""
        motion_projected_gravity_b = quat_rotate_inverse(
            motion_command.ref_quat_w, gravity_vector(self.env), w_last=True
        )
        robot_projected_gravity_b = quat_rotate_inverse(
            motion_command.robot_ref_quat_w, gravity_vector(self.env), w_last=True
        )
        return (
            torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]) > self.bad_ref_ori_threshold
        )

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the motion body position is too far from the robot's body position."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.norm(
            motion_command.body_pos_relative_w[:, body_idx] - motion_command.robot_body_pos_w[:, body_idx], dim=-1
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)

    def bad_object_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object position is too far from the simulator's object position."""
        return (
            torch.norm(motion_command.object_pos_w - motion_command.simulator_object_pos_w, dim=-1)
            > self.bad_object_pos_threshold
        )

    def bad_object_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object orientation is too far from the simulator's object orientation."""
        return (
            quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w)
            > self.bad_object_ori_threshold
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments."""

    def _bad_ref_pos_component_name(self) -> str:
        return "ref_pos"

    def _bad_body_pos_component_name(self) -> str:
        return "body_pos"

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class BodyGroupProximity(TerminationTermBase):
    """Terminate when any pair of bodies in a configured group gets too close."""

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.min_distance = float(cfg.params.get("min_distance", 0.05))
        self._last_breakdown: dict[str, torch.Tensor] = {}
        self._group_pair_indices: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._group_breakdown_names: list[str] = []

        body_groups = cfg.params.get("body_groups", [["left_foot_contact_point", "right_foot_contact_point"]])
        simulator_body_names = list(self.env.simulator.body_names)  # type: ignore[attr-defined]
        for group_idx, body_group in enumerate(body_groups):
            if len(body_group) < 2:
                raise ValueError(f"body_groups[{group_idx}] must contain at least two body names, got {body_group}")
            body_indices = self._get_index_of_a_in_b(body_group, simulator_body_names, self.env.device)
            pair_i, pair_j = torch.triu_indices(
                len(body_group), len(body_group), offset=1, device=self.env.device
            )
            self._group_pair_indices.append((body_indices[pair_i], body_indices[pair_j]))
            self._group_breakdown_names.append(f"group_{group_idx}")

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        body_pos_w = self.env.simulator._rigid_body_pos
        terminated = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        breakdown: dict[str, torch.Tensor] = {}

        for group_name, (body_i, body_j) in zip(self._group_breakdown_names, self._group_pair_indices):
            pairwise_dist = torch.norm(body_pos_w[:, body_i, :] - body_pos_w[:, body_j, :], dim=-1)
            group_terminated = torch.any(pairwise_dist < self.min_distance, dim=-1)
            breakdown[group_name] = group_terminated.detach().clone()
            terminated |= group_terminated

        self._last_breakdown = breakdown
        return terminated
    def get_last_breakdown(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self._last_breakdown.items()}

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments."""

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class BadTrackingZOnly(BadTracking):
    """BadTracking variant using z-axis-only position checks for parity with BM Wo-State-Estimation."""

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference z position is too far from the robot's z position."""
        z_err = torch.abs(motion_command.ref_pos_w[:, -1] - motion_command.robot_ref_pos_w[:, -1])
        return z_err > self.bad_ref_pos_threshold

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if tracked bodies have too much z-axis position error."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.abs(
            motion_command.body_pos_relative_w[:, body_idx, -1] - motion_command.robot_body_pos_w[:, body_idx, -1]
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)

    def _bad_ref_pos_component_name(self) -> str:
        return "ref_pos_z"

    def _bad_body_pos_component_name(self) -> str:
        return "body_pos_z"


class HybridBadTracking(BadTracking):
    """Track envs use bad-tracking, gen envs use low-root-height plus optional gen sanity checks."""

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.min_root_height = cfg.params["min_root_height"]
        self.gen_bad_ref_pos_z_threshold = cfg.params.get("gen_bad_ref_pos_z_threshold")
        self.gen_bad_ref_pos_xyz_threshold = cfg.params.get("gen_bad_ref_pos_xyz_threshold", 1.0e6)
        self.gen_bad_object_pos_z_threshold = cfg.params.get("gen_bad_object_pos_z_threshold")
        self.gen_bad_object_ori_threshold = cfg.params.get("gen_bad_object_ori_threshold")

    def gen_bad_ref_pos_z(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate gen envs if robot torso z deviates too much from the current-step reference z."""
        if self.gen_bad_ref_pos_z_threshold is None:
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        z_err = torch.abs(motion_command.ref_pos_w[:, 2] - motion_command.robot_ref_pos_w[:, 2])
        return z_err > self.gen_bad_ref_pos_z_threshold

    def gen_bad_ref_pos_xyz(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate gen envs if robot torso xyz position deviates too much from the current-step reference."""
        if self.gen_bad_ref_pos_xyz_threshold is None:
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        pos_err = torch.norm(motion_command.ref_pos_w - motion_command.robot_ref_pos_w, dim=-1)
        return pos_err > self.gen_bad_ref_pos_xyz_threshold

    def gen_bad_object_pos_z(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate gen envs if object z deviates too much from the current-step reference z."""
        if self.gen_bad_object_pos_z_threshold is None or not motion_command.motion.has_object:
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        z_err = torch.abs(motion_command.object_pos_w[:, 2] - motion_command.simulator_object_pos_w[:, 2])
        return z_err > self.gen_bad_object_pos_z_threshold

    def gen_bad_object_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate gen envs if object orientation deviates too much from the current-step reference."""
        if self.gen_bad_object_ori_threshold is None or not motion_command.motion.has_object:
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        ori_err = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w)
        return ori_err > self.gen_bad_object_ori_threshold

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        if not hasattr(motion_command, "track_env_mask") or not hasattr(motion_command, "gen_env_mask"):
            raise RuntimeError("Hybrid termination requires HybridMotionCommand with track/gen masks.")
        assert motion_command.motion_cfg.body_names_to_track == self.body_names_to_track, (
            "body_names_to_track in motion_command and termination.params are not the same"
            f"motion_command.motion_cfg.body_names_to_track: {motion_command.motion_cfg.body_names_to_track}"
            f"termination.params['body_names_to_track']: {self.body_names_to_track}"
        )

        track_env_mask = motion_command.track_env_mask
        gen_env_mask = motion_command.gen_env_mask

        bad_ref_pos = self.bad_ref_pos(motion_command) & track_env_mask
        bad_ref_ori = self.bad_ref_ori(motion_command) & track_env_mask
        bad_motion_body_pos = self.bad_motion_body_pos(motion_command) & track_env_mask
        bad_object_pos = torch.zeros_like(track_env_mask)
        bad_object_ori = torch.zeros_like(track_env_mask)

        if motion_command.motion.has_object:
            bad_object_pos = self.bad_object_pos(motion_command) & track_env_mask
            bad_object_ori = self.bad_object_ori(motion_command) & track_env_mask

        low_root_height = (motion_command.robot_root_pos_w[:, 2] < self.min_root_height) & gen_env_mask
        gen_ref_pos_z = self.gen_bad_ref_pos_z(motion_command) & gen_env_mask
        gen_ref_pos_xyz = self.gen_bad_ref_pos_xyz(motion_command) & gen_env_mask
        gen_object_pos_z = self.gen_bad_object_pos_z(motion_command) & gen_env_mask
        gen_object_ori = self.gen_bad_object_ori(motion_command) & gen_env_mask
        terminated = (
            bad_ref_pos
            | bad_ref_ori
            | bad_motion_body_pos
            | bad_object_pos
            | bad_object_ori
            | low_root_height
            | gen_ref_pos_z
            | gen_ref_pos_xyz
            | gen_object_pos_z
            | gen_object_ori
        )

        self._last_breakdown = {
            f"track_{self._bad_ref_pos_component_name()}": bad_ref_pos.detach().clone(),
            "track_ref_ori": bad_ref_ori.detach().clone(),
            f"track_{self._bad_body_pos_component_name()}": bad_motion_body_pos.detach().clone(),
            "track_object_pos": bad_object_pos.detach().clone(),
            "track_object_ori": bad_object_ori.detach().clone(),
            "gen_low_root_height": low_root_height.detach().clone(),
            "gen_ref_pos_z": gen_ref_pos_z.detach().clone(),
            "gen_ref_pos_xyz": gen_ref_pos_xyz.detach().clone(),
            "gen_object_pos_z": gen_object_pos_z.detach().clone(),
            "gen_object_ori": gen_object_ori.detach().clone(),
        }

        return terminated
