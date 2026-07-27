from types import SimpleNamespace

import torch

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsHistoryAugCfg, ObsTermCfg
from holosoma.managers.observation.manager import ObservationManager


def term_a(env):
    return env.term_a


def term_b(env):
    return env.term_b


def _manager(env, history_augmentation, *, enable_history_augmentation=False):
    return ObservationManager(
        ObservationManagerCfg(
            history_augmentation=enable_history_augmentation,
            groups={
                "actor_obs": ObsGroupCfg(
                    concatenate=True,
                    enable_noise=False,
                    terms={
                        "a": ObsTermCfg(func=f"{__name__}:term_a"),
                        "b": ObsTermCfg(func=f"{__name__}:term_b"),
                    },
                    history_augmentation=history_augmentation,
                )
            }
        ),
        env,
        "cpu",
    )


def _set_snapshot(env, value, num_envs=1):
    rows = torch.arange(num_envs, dtype=torch.float32).unsqueeze(-1) * 100.0
    env.term_a = rows + float(value)
    env.term_b = rows + torch.tensor([[float(value + 10), float(value + 20)]])


def test_history_augmentation_disabled_preserves_observations_and_dims():
    env = SimpleNamespace(num_envs=1, device="cpu")
    _set_snapshot(env, 5)
    manager = _manager(
        env,
        ObsHistoryAugCfg(source_terms=("a", "b"), sample_offsets=(3, 2, 1)),
    )

    obs = manager.compute_group("actor_obs")

    assert torch.equal(obs, torch.tensor([[5.0, 15.0, 25.0]]))
    assert manager.get_obs_dims()["actor_obs"] == 3


def test_history_augmentation_appends_zero_padded_strided_samples():
    env = SimpleNamespace(num_envs=1, device="cpu")
    manager = _manager(
        env,
        ObsHistoryAugCfg(source_terms=("a", "b"), sample_offsets=(3, 2, 1)),
        enable_history_augmentation=True,
    )

    for step in range(4):
        _set_snapshot(env, step)
        obs = manager.compute_group("actor_obs")

    expected = torch.tensor([[3.0, 13.0, 23.0, 0.0, 10.0, 20.0, 1.0, 11.0, 21.0, 2.0, 12.0, 22.0]])
    assert torch.equal(obs, expected)
    assert manager.get_obs_dims()["actor_obs"] == 12


def test_history_augmentation_derives_offsets_from_duration_and_stride():
    env = SimpleNamespace(num_envs=1, device="cpu", dt=0.1)
    manager = _manager(
        env,
        ObsHistoryAugCfg(source_terms=("a", "b"), sample_offsets=(), duration_s=0.3, stride_steps=1),
        enable_history_augmentation=True,
    )

    for step in range(4):
        _set_snapshot(env, step)
        obs = manager.compute_group("actor_obs")

    expected = torch.tensor([[3.0, 13.0, 23.0, 0.0, 10.0, 20.0, 1.0, 11.0, 21.0, 2.0, 12.0, 22.0]])
    assert torch.equal(obs, expected)
    assert manager.get_obs_dims()["actor_obs"] == 12


def test_history_augmentation_global_duration_and_stride_override_sample_offsets():
    env = SimpleNamespace(num_envs=1, device="cpu", dt=0.1)
    cfg = ObservationManagerCfg(
        history_augmentation=True,
        history_augmentation_duration_s=0.2,
        history_augmentation_stride_steps=1,
        groups={
            "actor_obs": ObsGroupCfg(
                concatenate=True,
                terms={
                    "a": ObsTermCfg(func=f"{__name__}:term_a"),
                    "b": ObsTermCfg(func=f"{__name__}:term_b"),
                },
                history_augmentation=ObsHistoryAugCfg(source_terms=("a", "b"), sample_offsets=(1,)),
            )
        },
    )
    manager = ObservationManager(cfg, env, "cpu")

    for step in range(3):
        _set_snapshot(env, step)
        obs = manager.compute_group("actor_obs")

    expected = torch.tensor([[2.0, 12.0, 22.0, 0.0, 10.0, 20.0, 1.0, 11.0, 21.0]])
    assert torch.equal(obs, expected)
    assert manager.get_obs_dims()["actor_obs"] == 9


def test_history_augmentation_modify_history_false_does_not_mutate_buffer():
    env = SimpleNamespace(num_envs=1, device="cpu")
    manager = _manager(
        env,
        ObsHistoryAugCfg(source_terms=("a", "b"), sample_offsets=(2, 1)),
        enable_history_augmentation=True,
    )

    for step in range(2):
        _set_snapshot(env, step)
        manager.compute_group("actor_obs")

    _set_snapshot(env, 2)
    final_obs = manager.compute_group("actor_obs", modify_history=False)
    assert torch.equal(final_obs, torch.tensor([[2.0, 12.0, 22.0, 0.0, 10.0, 20.0, 1.0, 11.0, 21.0]]))

    _set_snapshot(env, 3)
    next_obs = manager.compute_group("actor_obs")
    assert torch.equal(next_obs, torch.tensor([[3.0, 13.0, 23.0, 0.0, 10.0, 20.0, 1.0, 11.0, 21.0]]))


def test_history_augmentation_reset_zeros_selected_env_history():
    env = SimpleNamespace(num_envs=2, device="cpu")
    manager = _manager(
        env,
        ObsHistoryAugCfg(source_terms=("a", "b"), sample_offsets=(1,)),
        enable_history_augmentation=True,
    )

    for step in range(2):
        _set_snapshot(env, step, num_envs=2)
        manager.compute_group("actor_obs")

    manager.reset(torch.tensor([1]))

    _set_snapshot(env, 2, num_envs=2)
    obs = manager.compute_group("actor_obs")

    expected = torch.tensor(
        [
            [2.0, 12.0, 22.0, 1.0, 11.0, 21.0],
            [102.0, 112.0, 122.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.equal(obs, expected)


def test_history_augmentation_can_be_enabled_globally_for_multiple_groups():
    env = SimpleNamespace(num_envs=1, device="cpu")
    _set_snapshot(env, 0)
    cfg = ObservationManagerCfg(
        history_augmentation=True,
        groups={
            "actor_obs": ObsGroupCfg(
                concatenate=True,
                terms={
                    "a": ObsTermCfg(func=f"{__name__}:term_a"),
                    "b": ObsTermCfg(func=f"{__name__}:term_b"),
                },
                history_augmentation=ObsHistoryAugCfg(source_terms=("a", "b"), sample_offsets=(1,)),
            ),
            "critic_obs": ObsGroupCfg(
                concatenate=True,
                terms={
                    "a": ObsTermCfg(func=f"{__name__}:term_a"),
                    "b": ObsTermCfg(func=f"{__name__}:term_b"),
                },
                history_augmentation=ObsHistoryAugCfg(
                    source_terms=("a", "b"),
                    sample_offsets=(1,),
                ),
            ),
        }
    )
    manager = ObservationManager(cfg, env, "cpu")

    assert manager.get_obs_dims() == {"actor_obs": 6, "critic_obs": 6}
    manager.compute()

    _set_snapshot(env, 1)
    obs = manager.compute()

    expected = torch.tensor([[1.0, 11.0, 21.0, 0.0, 10.0, 20.0]])
    assert torch.equal(obs["actor_obs"], expected)
    assert torch.equal(obs["critic_obs"], expected)
