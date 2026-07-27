"""Observation manager for computing and processing observations."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import torch

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.utils import resolve_callable

from .base import ObservationTermBase


class ObservationManager:
    """Manages observation computation with groups, terms, and processing.

    The manager orchestrates the computation of observations from multiple terms,
    applying scaling, noise, clipping, and history buffering to produce the final
    observation tensors. It maintains exact equivalence with the direct observation
    system while providing a more modular structure.

    Parameters
    ----------
    cfg : ObservationManagerCfg
        Configuration specifying observation groups and terms.
    env : Any
        Environment instance (typically a ``BaseTask`` subclass).
    device : str
        Device to place tensors on.
    """

    def __init__(self, cfg: ObservationManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = getattr(env, "logger", None)

        # Storage for resolved functions and stateful terms
        self._term_funcs: dict[str, dict[str, Callable]] = {}
        self._term_instances: dict[str, dict[str, ObservationTermBase]] = {}

        # History buffers: group_name -> term_name -> deque
        self._history_buffers: dict[str, dict[str, deque]] = {}
        self._history_aug_buffers: dict[str, deque] = {}

        # Initialize groups
        self._initialize_groups()

    def _initialize_groups(self) -> None:
        """Initialize observation groups and resolve term functions."""
        for group_name, group_cfg in self.cfg.groups.items():
            self._term_funcs[group_name] = {}
            self._term_instances[group_name] = {}
            self._history_buffers[group_name] = {}

            for term_name, term_cfg in group_cfg.terms.items():
                # Resolve function
                func = resolve_callable(term_cfg.func, context="observation term")

                # Check if it's a class (stateful) or function (stateless)
                if isinstance(func, type) and issubclass(func, ObservationTermBase):
                    # Stateful term - instantiate
                    instance = func(term_cfg, self.env)
                    self._term_instances[group_name][term_name] = instance
                else:
                    # Stateless function
                    self._term_funcs[group_name][term_name] = func

                # Initialize history buffer if needed (using group-level history_length)
                if group_cfg.history_length > 1:
                    self._history_buffers[group_name][term_name] = deque(maxlen=group_cfg.history_length)

            self._initialize_history_augmentation(group_name, group_cfg)

    def _initialize_history_augmentation(self, group_name: str, group_cfg: ObsGroupCfg) -> None:
        """Validate and initialize selected-term history augmentation."""
        history_cfg = group_cfg.history_augmentation
        if not self._history_augmentation_enabled(group_cfg):
            return

        if history_cfg.name in group_cfg.terms:
            raise ValueError(
                f"History augmentation name '{history_cfg.name}' for group '{group_name}' conflicts with an obs term."
            )
        if not history_cfg.source_terms:
            raise ValueError(f"History augmentation for group '{group_name}' requires at least one source term.")
        sample_offsets = self._history_augmentation_offsets(group_name, group_cfg)
        if not sample_offsets:
            raise ValueError(
                f"History augmentation for group '{group_name}' requires sample offsets or a positive duration."
            )

        missing_terms = [term_name for term_name in history_cfg.source_terms if term_name not in group_cfg.terms]
        if missing_terms:
            raise ValueError(
                f"History augmentation for group '{group_name}' references missing source terms: {missing_terms}."
            )

        invalid_offsets = [offset for offset in sample_offsets if offset < 1]
        if invalid_offsets:
            raise ValueError(
                f"History augmentation for group '{group_name}' requires positive past offsets; "
                f"got {invalid_offsets}."
            )

        self._history_aug_buffers[group_name] = deque(maxlen=max(sample_offsets) + 1)

    def _history_augmentation_offsets(self, group_name: str, group_cfg: ObsGroupCfg) -> tuple[int, ...]:
        """Return effective selected-term history sample offsets for a group."""
        history_cfg = group_cfg.history_augmentation
        duration_s = self.cfg.history_augmentation_duration_s
        if duration_s is None:
            duration_s = history_cfg.duration_s

        stride_steps = self.cfg.history_augmentation_stride_steps
        if stride_steps is None:
            stride_steps = history_cfg.stride_steps

        if duration_s is None:
            return history_cfg.sample_offsets

        if duration_s <= 0.0:
            raise ValueError(f"History augmentation for group '{group_name}' requires positive duration_s.")
        if stride_steps < 1:
            raise ValueError(f"History augmentation for group '{group_name}' requires positive stride_steps.")

        dt = getattr(self.env, "dt", None)
        if dt is None or dt <= 0.0:
            raise ValueError(
                f"History augmentation for group '{group_name}' requires env.dt to derive offsets from duration_s."
            )

        max_offset = int(round(duration_s / float(dt)))
        if max_offset < 1:
            raise ValueError(
                f"History augmentation for group '{group_name}' duration_s={duration_s} is shorter than env.dt={dt}."
            )
        return tuple(range(max_offset, 0, -stride_steps))

    def _history_augmentation_enabled(self, group_cfg: ObsGroupCfg) -> bool:
        """Return whether a group's history augmentation is effectively enabled."""
        history_cfg = group_cfg.history_augmentation
        return self.cfg.history_augmentation and bool(history_cfg.source_terms)

    def compute(self, *, modify_history: bool = True) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute all observation groups.

        Parameters
        ----------
        modify_history : bool, optional
            If ``True``, update history buffers; if ``False``, preserve them
            for bootstrapping. Defaults to ``True``.

        Returns
        -------
        dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Mapping from group names to observation tensors or dictionaries of tensors.
        """
        obs_dict = {}
        for group_name in self.cfg.groups:
            obs_dict[group_name] = self.compute_group(group_name, modify_history=modify_history)
        return obs_dict

    def compute_group(self, group_name: str, *, modify_history: bool = True) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute observations for a specific group.

        This method replicates the exact behavior of the direct
        ``parse_observation()`` function to ensure equivalence.

        Parameters
        ----------
        group_name : str
            Name of the observation group to compute.
        modify_history : bool, optional
            If ``True``, update history buffers; if ``False``, preserve them
            for bootstrapping. Defaults to ``True``.

        Returns
        -------
        torch.Tensor | dict[str, torch.Tensor]
            Concatenated tensor when ``group.concatenate`` is ``True``; otherwise
            a dictionary of tensors keyed by term name.
        """
        group_cfg = self.cfg.groups[group_name]
        obs_tensors = {}
        current_tensors = {}

        for term_name, term_cfg in group_cfg.terms.items():
            # 1. Compute base observation
            obs = self._compute_term(group_name, term_name, term_cfg)

            # 2. Apply noise (matches direct: noise before scaling)
            if group_cfg.enable_noise and term_cfg.noise > 0:
                obs = self._apply_noise(obs, term_cfg.noise)

            # 3. Apply scaling (matches direct: scale after noise)
            obs = self._apply_scale(obs, term_cfg.scale)

            # 4. Apply clipping (if specified)
            if term_cfg.clip is not None:
                obs = obs.clip(term_cfg.clip[0], term_cfg.clip[1])

            current_tensors[term_name] = obs

            # 5. Handle history buffering
            if group_cfg.history_length > 1:
                obs = self._apply_history(group_name, term_name, obs, group_cfg, modify_buffer=modify_history)

            obs_tensors[term_name] = obs

        history_augmentation_enabled = self._history_augmentation_enabled(group_cfg)
        if history_augmentation_enabled:
            history_obs = self._apply_history_augmentation(
                group_name, current_tensors, group_cfg, modify_buffer=modify_history
            )
            obs_tensors[group_cfg.history_augmentation.name] = history_obs

        # Concatenate or return dict
        if group_cfg.concatenate:
            # Concatenate real terms in alphabetically sorted order (to match direct system behavior),
            # then append the synthetic history augmentation when enabled.
            sorted_keys = sorted(group_cfg.terms.keys())
            if history_augmentation_enabled:
                sorted_keys.append(group_cfg.history_augmentation.name)
            return torch.cat([obs_tensors[key] for key in sorted_keys], dim=-1)
        return obs_tensors

    def _compute_term(self, group_name: str, term_name: str, term_cfg: ObsTermCfg) -> torch.Tensor:
        """Compute a single observation term.

        Parameters
        ----------
        group_name : str
            Name of the observation group.
        term_name : str
            Name of the observation term.
        term_cfg : ObsTermCfg
            Configuration for the observation term.

        Returns
        -------
        torch.Tensor
            Observation tensor with shape ``[num_envs, obs_dim]``.
        """
        # Check if stateful or stateless
        if term_name in self._term_instances[group_name]:
            # Stateful term
            instance = self._term_instances[group_name][term_name]
            obs = instance(self.env, **term_cfg.params)
        else:
            # Stateless function
            func = self._term_funcs[group_name][term_name]
            obs = func(self.env, **term_cfg.params)

        return obs.clone()

    def _apply_noise(self, obs: torch.Tensor, noise_scale: float) -> torch.Tensor:
        """Apply uniform observation noise.

        The implementation matches the direct path, sampling uniform noise in
        ``[-noise_scale, noise_scale]``.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor.
        noise_scale : float
            Magnitude of the uniform noise.

        Returns
        -------
        torch.Tensor
            Noisy observation tensor.
        """
        # Direct uses: (torch.rand_like(obs) * 2.0 - 1.0) * noise_scale
        noise = (torch.rand_like(obs) * 2.0 - 1.0) * noise_scale
        return obs + noise

    def _apply_scale(self, obs: torch.Tensor, scale: float | tuple) -> torch.Tensor:
        """Apply scaling to an observation tensor.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor.
        scale : float or tuple
            Scalar or per-dimension scale factors.

        Returns
        -------
        torch.Tensor
            Scaled observation tensor.
        """
        return obs * scale

    def _apply_history(
        self, group_name: str, term_name: str, obs: torch.Tensor, group_cfg: ObsGroupCfg, *, modify_buffer: bool = True
    ) -> torch.Tensor:
        """Apply history buffering to an observation term.

        Maintains a circular buffer of past observations and returns them
        concatenated along the feature dimension.

        Parameters
        ----------
        group_name : str
            Name of the observation group.
        term_name : str
            Name of the observation term.
        obs : torch.Tensor
            Current observation tensor with shape ``[num_envs, obs_dim]``.
        group_cfg : ObsGroupCfg
            Configuration for the observation group (contains ``history_length``).
        modify_buffer : bool, optional
            If ``True``, append to the history buffer; if ``False``, preserve the
            buffer contents. Defaults to ``True``.

        Returns
        -------
        torch.Tensor
            Historical observations with shape ``[num_envs, obs_dim * history_length]``.
        """
        buffer = self._history_buffers[group_name][term_name]

        if modify_buffer:
            # Add current observation to buffer
            buffer.append(obs)
            history = list(buffer)
        else:
            # Don't modify buffer - create temporary history with current obs
            history = list(buffer) + [obs]
            # Trim to history_length if needed
            if len(history) > group_cfg.history_length:
                history = history[-group_cfg.history_length :]

        # If buffer not full yet, pad with zeros (same as direct behavior)
        if len(history) < group_cfg.history_length:
            num_missing = group_cfg.history_length - len(history)
            obs_dim = obs.shape[1]
            padding = [torch.zeros(self.env.num_envs, obs_dim, device=self.device) for _ in range(num_missing)]
            history = padding + history

        # Stack along time dimension: [num_envs, history_length, obs_dim]
        stacked = torch.stack(history, dim=1)

        # Flatten to [num_envs, history_length * obs_dim]
        return stacked.reshape(self.env.num_envs, -1)

    def _apply_history_augmentation(
        self,
        group_name: str,
        current_tensors: dict[str, torch.Tensor],
        group_cfg: ObsGroupCfg,
        *,
        modify_buffer: bool = True,
    ) -> torch.Tensor:
        """Append selected, strided past snapshots for an observation group."""
        history_cfg = group_cfg.history_augmentation
        sample_offsets = self._history_augmentation_offsets(group_name, group_cfg)
        current_snapshot = torch.cat([current_tensors[term_name] for term_name in history_cfg.source_terms], dim=-1)
        buffer = self._history_aug_buffers[group_name]

        if modify_buffer:
            buffer.append(current_snapshot)
            history = list(buffer)
        else:
            history = list(buffer) + [current_snapshot]
            max_len = max(sample_offsets) + 1
            if len(history) > max_len:
                history = history[-max_len:]

        zero_snapshot = torch.zeros(
            self.env.num_envs,
            current_snapshot.shape[1],
            device=self.device,
            dtype=current_snapshot.dtype,
        )
        samples = []
        for offset in sample_offsets:
            if len(history) > offset:
                samples.append(history[-offset - 1])
            else:
                samples.append(zero_snapshot)

        return torch.cat(samples, dim=-1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset observation history and stateful terms.

        Parameters
        ----------
        env_ids : torch.Tensor or None, optional
            Environment IDs to reset. If ``None``, reset all environments.
        """
        # Normalize environment ids
        env_ids_tensor: torch.Tensor | None
        if env_ids is None:
            env_ids_tensor = None
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Reset or clear history buffers
        for group_buffers in self._history_buffers.values():
            for buffer in group_buffers.values():
                if env_ids_tensor is None:
                    buffer.clear()
                else:
                    if env_ids_tensor.numel() == 0:
                        continue
                    for history in buffer:
                        history[env_ids_tensor] = 0.0

        for buffer in self._history_aug_buffers.values():
            if env_ids_tensor is None:
                buffer.clear()
            else:
                if env_ids_tensor.numel() == 0:
                    continue
                for history in buffer:
                    history[env_ids_tensor] = 0.0

        # Reset stateful term instances
        for group_instances in self._term_instances.values():
            for instance in group_instances.values():
                instance.reset(env_ids_tensor)

    def get_obs_dims(self) -> dict[str, int | dict[str, int]]:
        """Get observation dimensions for each group.

        Returns
        -------
        dict[str, int or dict[str, int]]
            Mapping from group names to observation dimensions. Values are integers
            when the group concatenates terms, otherwise dictionaries of per-term
            dimensions.
        """
        dims: dict[str, int | dict[str, int]] = {}
        for group_name, group_cfg in self.cfg.groups.items():
            if group_cfg.concatenate:
                # Sum up all term dimensions
                total_dim = 0
                for term_name, term_cfg in group_cfg.terms.items():
                    # Compute term once to get its dimension
                    obs = self._compute_term(group_name, term_name, term_cfg)
                    term_dim = obs.shape[1]

                    # Account for history at group level
                    if group_cfg.history_length > 1:
                        term_dim *= group_cfg.history_length

                    total_dim += term_dim
                if self._history_augmentation_enabled(group_cfg):
                    total_dim += self._get_history_augmentation_dim(group_name, group_cfg)
                dims[group_name] = total_dim
            else:
                # Return dict of individual dimensions
                term_dims: dict[str, int] = {
                    term_name: self._compute_term(group_name, term_name, term_cfg).shape[1]
                    for term_name, term_cfg in group_cfg.terms.items()
                }
                if self._history_augmentation_enabled(group_cfg):
                    term_dims[group_cfg.history_augmentation.name] = self._get_history_augmentation_dim(
                        group_name, group_cfg
                    )
                dims[group_name] = term_dims
        return dims

    def _get_history_augmentation_dim(self, group_name: str, group_cfg: ObsGroupCfg) -> int:
        """Compute flattened dimension of selected-term history augmentation."""
        history_cfg = group_cfg.history_augmentation
        source_dim = 0
        for term_name in history_cfg.source_terms:
            term_cfg = group_cfg.terms[term_name]
            source_dim += self._compute_term(group_name, term_name, term_cfg).shape[1]
        return source_dim * len(self._history_augmentation_offsets(group_name, group_cfg))
