"""Checkpoint/restore tests for ``PenaltyCurriculum`` and the locomotion manager's checkpoint state.

Uses the real term and checkpoint methods on a minimal fake env, so no simulator is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
from holosoma.managers.curriculum.terms.locomotion import PenaltyCurriculum

pytestmark = pytest.mark.no_sim

INITIAL_SCALE = 0.1
LEVEL_UP_THRESHOLD = 750.0
DEGREE = 0.00025
# Resets needed to ramp INITIAL_SCALE to 1.0: ln(10) / ln(1 + DEGREE) ~= 9211.
RAMP_RESETS = 12000

PENALTY_WEIGHTS = {"penalty_action_rate": -3.0, "penalty_feet_ori": -1.0}
UNTAGGED_WEIGHTS = {"tracking_lin_vel": 1.0}


@dataclass(frozen=True)
class _TermCfg:
    weight: float
    tags: tuple[str, ...] = ()


class _RewardManager:
    def __init__(self) -> None:
        self._terms = {name: _TermCfg(w, ("penalty_curriculum",)) for name, w in PENALTY_WEIGHTS.items()}
        self._terms.update({name: _TermCfg(w) for name, w in UNTAGGED_WEIGHTS.items()})

    @property
    def active_terms(self) -> list[str]:
        return list(self._terms)

    def get_term_cfg(self, name: str) -> _TermCfg:
        return self._terms[name]

    def set_term_cfg(self, name: str, cfg: _TermCfg) -> None:
        self._terms[name] = cfg


class _EpisodeTracker:
    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass

    def suppress_next_update(self) -> None:
        pass


class _CurriculumManager:
    def __init__(self, terms: dict[str, Any]) -> None:
        self._terms = terms

    def get_term(self, name: str) -> Any | None:
        return self._terms.get(name)


class _Env:
    """The env a curriculum term sees, with the locomotion manager's checkpoint methods bound to it."""

    _get_average_episode_tracker = LeggedRobotLocomotionManager._get_average_episode_tracker
    _get_penalty_curriculum = LeggedRobotLocomotionManager._get_penalty_curriculum
    get_checkpoint_state = LeggedRobotLocomotionManager.get_checkpoint_state
    load_checkpoint_state = LeggedRobotLocomotionManager.load_checkpoint_state

    def __init__(self, *, enabled: bool = True) -> None:
        self.reward_manager = _RewardManager()
        self.average_episode_length = 0.0
        self.log_dict: dict[str, Any] = {}
        self.reward_penalty_scale = 1.0
        self.curriculum = PenaltyCurriculum(
            SimpleNamespace(
                params={
                    "enabled": enabled,
                    "tag": "penalty_curriculum",
                    "initial_scale": INITIAL_SCALE,
                    "min_scale": 0.0,
                    "max_scale": 1.0,
                    "level_down_threshold": 150.0,
                    "level_up_threshold": LEVEL_UP_THRESHOLD,
                    "degree": DEGREE,
                }
            ),
            self,
        )
        self.curriculum.setup()
        self.curriculum_manager = _CurriculumManager(
            {"average_episode_tracker": _EpisodeTracker(), "penalty_curriculum": self.curriculum}
        )

    def weight(self, name: str) -> float:
        return self.reward_manager.get_term_cfg(name).weight


def _manager(env: _Env) -> LeggedRobotLocomotionManager:
    return cast("LeggedRobotLocomotionManager", env)


def _ramped_env() -> _Env:
    env = _Env()
    env.average_episode_length = LEVEL_UP_THRESHOLD + 150.0
    for _ in range(RAMP_RESETS):
        env.curriculum.reset(None)
    return env


def _assert_scale_applied(env: _Env, scale: float) -> None:
    assert env.curriculum.current_scale == pytest.approx(scale)
    assert env.reward_penalty_scale == pytest.approx(scale)
    for name, weight in PENALTY_WEIGHTS.items():
        assert env.weight(name) == pytest.approx(weight * scale)
    for name, weight in UNTAGGED_WEIGHTS.items():
        assert env.weight(name) == pytest.approx(weight)


def test_ramp_saturates_and_scales_only_tagged_terms() -> None:
    _assert_scale_applied(_ramped_env(), 1.0)


def test_state_dict_round_trip_restores_scale_and_weights() -> None:
    state = _ramped_env().curriculum.state_dict()

    resumed = _Env()
    _assert_scale_applied(resumed, INITIAL_SCALE)
    resumed.curriculum.load_state_dict(state)
    _assert_scale_applied(resumed, 1.0)

    resumed.average_episode_length = LEVEL_UP_THRESHOLD + 150.0
    resumed.curriculum.reset(None)
    _assert_scale_applied(resumed, 1.0)


def test_load_state_dict_clamps_to_bounds() -> None:
    env = _Env()
    env.curriculum.load_state_dict({"current_scale": 5.0})
    _assert_scale_applied(env, 1.0)


def test_manager_checkpoint_restores_curriculum() -> None:
    state = _manager(_ramped_env()).get_checkpoint_state()
    assert state["penalty_curriculum"] == {"current_scale": pytest.approx(1.0)}

    resumed = _Env()
    _manager(resumed).load_checkpoint_state(state)
    _assert_scale_applied(resumed, 1.0)

    resumed.average_episode_length = LEVEL_UP_THRESHOLD + 150.0
    resumed.curriculum.reset(None)
    _assert_scale_applied(resumed, 1.0)


def test_manager_restores_legacy_checkpoint_from_mirror() -> None:
    resumed = _Env()
    _manager(resumed).load_checkpoint_state({"reward_penalty_scale": 0.873})
    _assert_scale_applied(resumed, 0.873)


def test_manager_restores_legacy_tensor_checkpoint() -> None:
    resumed = _Env()
    _manager(resumed).load_checkpoint_state({"reward_penalty_scale": torch.tensor(0.5)})
    _assert_scale_applied(resumed, 0.5)


def test_disabled_term_saves_nothing_and_ignores_restore() -> None:
    disabled = _Env(enabled=False)
    assert disabled.curriculum.state_dict() == {}

    # A disabled run's checkpoint must not hand an enabled resume its never-applied scale.
    # (The legacy mirror is still copied over; the term's next reset rewrites it.)
    resumed = _Env()
    _manager(resumed).load_checkpoint_state(_manager(disabled).get_checkpoint_state())
    assert resumed.curriculum.current_scale == pytest.approx(INITIAL_SCALE)
    for name, weight in PENALTY_WEIGHTS.items():
        assert resumed.weight(name) == pytest.approx(weight * INITIAL_SCALE)

    # A disabled resume leaves the weights and the logged scale alone.
    disabled.curriculum.load_state_dict({"current_scale": 0.5})
    assert disabled.curriculum.current_scale == pytest.approx(INITIAL_SCALE)
    assert disabled.reward_penalty_scale == pytest.approx(1.0)
    for name, weight in PENALTY_WEIGHTS.items():
        assert disabled.weight(name) == pytest.approx(weight)
