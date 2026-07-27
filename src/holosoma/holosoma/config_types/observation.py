"""Configuration types for observation manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ObsTermCfg:
    """Configuration for a single observation term."""

    func: str
    """Import path to the observation function (e.g. ``holosoma.managers.observation.terms.locomotion:base_lin_vel``)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments forwarded to ``func``."""

    scale: float | tuple[float, ...] = 1.0
    """Scaling factor(s) applied after noise and clipping."""

    noise: float = 0.0
    """Noise magnitude applied before scaling when the group enables noise."""

    clip: tuple[float, float] | None = None
    """Optional min/max clip bounds applied after scaling."""


@dataclass(frozen=True)
class ObsHistoryAugCfg:
    """Configuration for selected-term, strided observation history."""

    source_terms: tuple[str, ...] = field(default_factory=tuple)
    """Observation terms to concatenate into each history snapshot."""

    sample_offsets: tuple[int, ...] = (50, 40, 30, 20, 10)
    """Past policy-step offsets to sample, ordered as they should be concatenated."""

    duration_s: float | None = None
    """Optional past duration, in control seconds, used to derive sample offsets."""

    stride_steps: int = 10
    """Policy-step stride used when deriving sample offsets from ``duration_s``."""

    name: str = "history_state"
    """Synthetic observation term name used for the appended history block."""


@dataclass(frozen=True)
class ObsGroupCfg:
    """Configuration for an observation group (e.g. actor or critic input)."""

    terms: dict[str, ObsTermCfg] = field(default_factory=dict)
    """Mapping of term name to its configuration."""

    concatenate: bool = True
    """Whether to concatenate term outputs into a single tensor."""

    enable_noise: bool = False
    """If ``True``, apply each term's noise setting before scaling."""

    history_length: int = 1
    """Number of timesteps to retain for history stacking (``1`` disables history)."""

    history_augmentation: ObsHistoryAugCfg = field(default_factory=ObsHistoryAugCfg)
    """Optional selected-term, strided history appended to this observation group."""


@dataclass(frozen=True)
class ObservationManagerCfg:
    """Configuration for the observation manager."""

    groups: dict[str, ObsGroupCfg] = field(default_factory=dict)
    """Mapping of group name to its configuration."""

    history_augmentation: bool = False
    """If ``True``, enable configured selected-term history augmentation groups."""

    history_augmentation_duration_s: float | None = None
    """Optional global override for selected-term history duration in seconds."""

    history_augmentation_stride_steps: int | None = None
    """Optional global override for selected-term history stride in policy steps."""

    clip_observations: float = 100.0
    """Global observation clipping threshold (applied to all observations)."""
