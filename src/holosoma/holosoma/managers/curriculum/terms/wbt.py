"""Curriculum hooks for Whole Body Tracking tasks."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.managers.command.terms.wbt import HybridMotionCommand
from holosoma.managers.curriculum.base import CurriculumTermBase


class HybridGenNoiseCurriculum(CurriculumTermBase):
    """Success-driven curriculum for hybrid object-goal noise."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.enabled = bool(params.get("enabled", False))
        self.initial_scale = float(params.get("initial_scale", 0.3))
        self.min_scale = float(params.get("min_scale", 0.3))
        self.max_scale = float(params.get("max_scale", 1.0))
        self.scale_step = float(params.get("scale_step", 0.01))
        self.success_threshold_up = float(params.get("success_threshold_up", 0.90))
        self.success_threshold_down = float(params.get("success_threshold_down", 0.80))
        self.update_interval = max(1, int(params.get("update_interval", 20)))
        self.ema_alpha = float(params.get("ema_alpha", 0.5))
        # One scale per object group (motion_dir_object_groups), each stepped by its own groups' success: with one
        # shared scale a hard object holds the mean success between the thresholds and the range never widens.
        self.per_object_group = bool(params.get("per_object_group", False))

        if self.min_scale > self.max_scale:
            raise ValueError(
                f"{self._term_name()} requires min_scale <= max_scale, "
                f"got {self.min_scale} > {self.max_scale}."
            )
        if self.scale_step < 0.0:
            raise ValueError(
                f"{self._term_name()} requires non-negative scale_step, "
                f"got {self.scale_step}."
            )
        if not 0.0 <= self.success_threshold_down <= 1.0:
            raise ValueError(
                f"{self._term_name()} success_threshold_down must be in [0, 1], "
                f"got {self.success_threshold_down}."
            )
        if not 0.0 <= self.success_threshold_up <= 1.0:
            raise ValueError(
                f"{self._term_name()} success_threshold_up must be in [0, 1], "
                f"got {self.success_threshold_up}."
            )
        if self.success_threshold_down > self.success_threshold_up:
            raise ValueError(
                f"{self._term_name()} requires success_threshold_down <= success_threshold_up, "
                f"got {self.success_threshold_down} > {self.success_threshold_up}."
            )
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(
                f"{self._term_name()} ema_alpha must be in (0, 1], "
                f"got {self.ema_alpha}."
            )

        self.motion_command: HybridMotionCommand | None = None
        self.success_ema = torch.zeros((), device=env.device, dtype=torch.float32)
        self.has_success_ema = False
        self.current_scale = torch.tensor(
            min(max(self.initial_scale, self.min_scale), self.max_scale),
            device=env.device,
            dtype=torch.float32,
        )
        self.accumulated_success_sum = torch.zeros((), device=env.device, dtype=torch.float32)
        self.accumulated_success_count = torch.zeros((), device=env.device, dtype=torch.long)
        self.latest_batch_success = torch.zeros((), device=env.device, dtype=torch.float32)
        self.collect_rollout = False
        self.num_groups = 1
        self._init_group_state()

    def _init_group_state(self) -> None:
        device = self.env.device
        self.group_scale = self.current_scale.repeat(self.num_groups)
        self.group_success_ema = torch.zeros(self.num_groups, device=device, dtype=torch.float32)
        self.group_has_success_ema = torch.zeros(self.num_groups, device=device, dtype=torch.bool)
        self.group_success_sum = torch.zeros(self.num_groups, device=device, dtype=torch.float32)
        self.group_success_count = torch.zeros(self.num_groups, device=device, dtype=torch.long)

    def setup(self) -> None:
        if not self.enabled:
            return

        motion_command = self._get_motion_command()
        if not isinstance(motion_command, HybridMotionCommand):
            raise RuntimeError(f"{self._term_name()} requires HybridMotionCommand.")
        if not bool(motion_command.motion.has_object):
            raise RuntimeError(
                f"{self._term_name()} requires object-backed hybrid motions, but the active motion command "
                "has no object data."
            )

        self.motion_command = motion_command
        self._apply_scale_to_targets(self.current_scale)
        if self.per_object_group:
            self.num_groups = int(getattr(motion_command, "num_object_groups", 1))
            self._init_group_state()

    def reset(self, env_ids) -> None:
        if not self.enabled or self.motion_command is None:
            return
        if not self.collect_rollout:
            return

        success_values = self._get_gen_success_values()
        if success_values is None or success_values.numel() == 0:
            return

        success_values = success_values.to(device=self.env.device, dtype=torch.float32).flatten()
        self.accumulated_success_sum += success_values.sum()
        self.accumulated_success_count += success_values.numel()
        if self.per_object_group:
            # the gen environments whose episodes these are, in the same order (set by the env's _fill_extras)
            gen_env_ids = getattr(self.env, "hybrid_gen_reset_env_ids", None)
            if gen_env_ids is None or gen_env_ids.numel() != success_values.numel():
                return
            groups = self.motion_command.env_object_group[gen_env_ids.to(self.env.device)]
            self.group_success_sum.index_add_(0, groups, success_values)
            self.group_success_count.index_add_(0, groups, torch.ones_like(groups))

    def step(self) -> None:
        return

    def on_training_step(
        self,
        training_step: int,
        *,
        log_dir: str | None = None,
        is_main_process: bool = True,
    ) -> None:
        del log_dir, is_main_process
        if not self.enabled or self.motion_command is None:
            return
        self.collect_rollout = self._should_collect_rollout(int(training_step))
        self.accumulated_success_sum.zero_()
        self.accumulated_success_count.zero_()
        self.group_success_sum.zero_()
        self.group_success_count.zero_()

    def on_rollout_end(
        self,
        training_step: int,
        *,
        log_dir: str | None = None,
        is_main_process: bool = True,
    ) -> None:
        del training_step, log_dir, is_main_process
        if not self.enabled or self.motion_command is None:
            return
        if not self.collect_rollout:
            return

        num_gen_episodes = int(self.accumulated_success_count.item())
        if num_gen_episodes <= 0:
            self.collect_rollout = False
            return

        batch_mean = self.accumulated_success_sum / float(num_gen_episodes)
        self.latest_batch_success = batch_mean
        if self.has_success_ema:
            self.success_ema = (1.0 - self.ema_alpha) * self.success_ema + self.ema_alpha * batch_mean
        else:
            self.success_ema = batch_mean
            self.has_success_ema = True

        if self.per_object_group:
            self._update_group_scales()
            self.accumulated_success_sum.zero_()
            self.accumulated_success_count.zero_()
            self.collect_rollout = False
            return

        next_scale = self.current_scale.clone()
        if float(self.success_ema.item()) > self.success_threshold_up:
            next_scale = torch.clamp(next_scale + self.scale_step, min=self.min_scale, max=self.max_scale)
        elif float(self.success_ema.item()) < self.success_threshold_down:
            next_scale = torch.clamp(next_scale - self.scale_step, min=self.min_scale, max=self.max_scale)

        scale_changed = not torch.allclose(next_scale, self.current_scale)
        self.current_scale = next_scale
        self.accumulated_success_sum.zero_()
        self.accumulated_success_count.zero_()
        self.collect_rollout = False

        self._apply_scale_to_targets(self.current_scale)
        if scale_changed:
            self._reapply_goal_noise_targets(None)

    def _update_group_scales(self) -> None:
        """Step each object group's scale by its own success EMA; groups without finished gen episodes keep theirs."""
        assert self.motion_command is not None
        seen = self.group_success_count > 0
        batch = self.group_success_sum / self.group_success_count.clamp(min=1).to(torch.float32)
        ema = torch.where(
            self.group_has_success_ema,
            (1.0 - self.ema_alpha) * self.group_success_ema + self.ema_alpha * batch,
            batch,
        )
        self.group_success_ema = torch.where(seen, ema, self.group_success_ema)
        self.group_has_success_ema |= seen

        step = torch.zeros_like(self.group_scale)
        step[seen & (self.group_success_ema > self.success_threshold_up)] = self.scale_step
        step[seen & (self.group_success_ema < self.success_threshold_down)] = -self.scale_step
        next_scale = torch.clamp(self.group_scale + step, min=self.min_scale, max=self.max_scale)
        scale_changed = not torch.allclose(next_scale, self.group_scale)
        self.group_scale = next_scale
        self.current_scale = next_scale.mean()
        self.group_success_sum.zero_()
        self.group_success_count.zero_()

        self._apply_scale_to_targets(self.current_scale)
        self.motion_command.set_object_goal_noise_group_scales(self.group_scale)
        if scale_changed:
            self._reapply_goal_noise_targets(None)

    def _should_collect_rollout(self, training_step: int) -> bool:
        if self.update_interval <= 1:
            return True
        return (int(training_step) + 1) % self.update_interval == 0

    def _get_motion_command(self) -> Any:
        motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is None:
            raise RuntimeError(f"{self._term_name()} could not find the motion_command state.")
        return motion_command

    def _get_gen_success_values(self) -> torch.Tensor | None:
        episode_metrics = getattr(self.env, "extras", {}).get("episode", {})
        success_values = episode_metrics.get("hybrid/success/gen")
        if success_values is None:
            return None
        return torch.as_tensor(success_values, dtype=torch.float32, device=self.env.device)

    def _apply_scale_to_targets(self, scale: torch.Tensor) -> None:
        assert self.motion_command is not None
        self.motion_command.set_object_goal_noise_curriculum_scale(scale)

    def _reapply_goal_noise_targets(self, env_ids: torch.Tensor | None) -> None:
        assert self.motion_command is not None
        self.motion_command.reapply_gen_object_goal_noise(env_ids)

    def _term_name(self) -> str:
        return type(self).__name__
