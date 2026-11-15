from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.command.terms.wbt import HybridMotionCommand
from holosoma.managers.curriculum.terms.wbt import HybridGenNoiseCurriculum


def _make_hybrid_object_goal_noise_curriculum(
    *,
    params: dict | None = None,
    motion_command: HybridMotionCommand | None = None,
    extras: dict | None = None,
) -> HybridGenNoiseCurriculum:
    if motion_command is None:
        motion_command = object.__new__(HybridMotionCommand)
        motion_command.device = torch.device("cpu")
        motion_command.num_envs = 4
        motion_command.motion = SimpleNamespace(has_object=True)
        motion_command.set_object_goal_noise_curriculum_scale = lambda scale: setattr(
            motion_command,
            "applied_scale",
            float(torch.as_tensor(scale).item()),
        )
        motion_command.reapply_gen_object_goal_noise = lambda env_ids: setattr(
            motion_command,
            "last_reapplied_env_ids",
            torch.arange(motion_command.num_envs, dtype=torch.long)
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.long),
        )

    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=4,
        extras=extras or {"episode": {}},
        command_manager=SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None),
    )
    term_params = {
        "enabled": True,
        "initial_scale": 0.3,
        "min_scale": 0.3,
        "max_scale": 1.0,
        "scale_step": 0.01,
        "success_threshold_up": 0.90,
        "success_threshold_down": 0.80,
        "update_interval": 20,
        "ema_alpha": 0.5,
    }
    term_params.update(params or {})
    cfg = SimpleNamespace(params=term_params)
    return HybridGenNoiseCurriculum(cfg, env)


def test_hybrid_gen_noise_curriculum_defaults_disabled_and_noops_for_non_hybrid_command():
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=2,
        extras={"episode": {}},
        command_manager=SimpleNamespace(get_state=lambda name: SimpleNamespace() if name == "motion_command" else None),
    )
    curriculum = HybridGenNoiseCurriculum(SimpleNamespace(params={}), env)

    curriculum.setup()

    assert curriculum.enabled is False
    assert curriculum.motion_command is None


def test_hybrid_object_goal_noise_curriculum_noops_without_completed_gen_episodes():
    curriculum = _make_hybrid_object_goal_noise_curriculum()

    curriculum.setup()
    curriculum.reset(torch.tensor([0, 1], dtype=torch.long))

    assert torch.allclose(curriculum.success_ema, torch.tensor(0.0, dtype=torch.float32))
    assert not curriculum.has_success_ema
    assert int(curriculum.accumulated_success_count.item()) == 0
    assert curriculum.motion_command is not None
    assert abs(curriculum.motion_command.applied_scale - 0.3) < 1.0e-6
    assert not hasattr(curriculum.motion_command, "last_reapplied_env_ids")


def test_hybrid_object_goal_noise_curriculum_updates_ema_and_steps_up():
    motion_command = object.__new__(HybridMotionCommand)
    motion_command.device = torch.device("cpu")
    motion_command.num_envs = 4
    motion_command.motion = SimpleNamespace(has_object=True)
    motion_command.set_object_goal_noise_curriculum_scale = lambda scale: setattr(
        motion_command,
        "applied_scale",
        float(torch.as_tensor(scale).item()),
    )
    motion_command.reapply_gen_object_goal_noise = lambda env_ids: setattr(
        motion_command,
        "last_reapplied_env_ids",
        torch.arange(motion_command.num_envs, dtype=torch.long)
        if env_ids is None
        else torch.as_tensor(env_ids, dtype=torch.long),
    )
    curriculum = _make_hybrid_object_goal_noise_curriculum(
        params={
            "update_interval": 1,
            "success_threshold_up": 0.8,
            "scale_step": 0.1,
        },
        motion_command=motion_command,
        extras={"episode": {"hybrid/success/gen": torch.tensor([1.0, 1.0], dtype=torch.float32)}},
    )

    curriculum.setup()
    curriculum.on_training_step(0)
    curriculum.reset(torch.tensor([1, 3], dtype=torch.long))
    curriculum.on_rollout_end(0)

    assert torch.allclose(curriculum.success_ema, torch.tensor(1.0, dtype=torch.float32))
    assert torch.allclose(curriculum.current_scale, torch.tensor(0.4, dtype=torch.float32))
    assert torch.allclose(curriculum.latest_batch_success, torch.tensor(1.0, dtype=torch.float32))
    assert curriculum.motion_command is not None
    assert abs(curriculum.motion_command.applied_scale - 0.4) < 1.0e-6
    assert torch.equal(curriculum.motion_command.last_reapplied_env_ids, torch.tensor([0, 1, 2, 3], dtype=torch.long))


def test_hybrid_object_goal_noise_curriculum_steps_down_when_success_collapses():
    motion_command = object.__new__(HybridMotionCommand)
    motion_command.device = torch.device("cpu")
    motion_command.num_envs = 4
    motion_command.motion = SimpleNamespace(has_object=True)
    motion_command.set_object_goal_noise_curriculum_scale = lambda scale: setattr(
        motion_command,
        "applied_scale",
        float(torch.as_tensor(scale).item()),
    )
    motion_command.reapply_gen_object_goal_noise = lambda env_ids: setattr(
        motion_command,
        "last_reapplied_env_ids",
        torch.arange(motion_command.num_envs, dtype=torch.long)
        if env_ids is None
        else torch.as_tensor(env_ids, dtype=torch.long),
    )
    curriculum = _make_hybrid_object_goal_noise_curriculum(
        params={
            "update_interval": 1,
            "success_threshold_up": 0.8,
            "success_threshold_down": 0.6,
            "scale_step": 0.1,
            "ema_alpha": 0.5,
        },
        motion_command=motion_command,
        extras={"episode": {"hybrid/success/gen": torch.tensor([1.0, 1.0], dtype=torch.float32)}},
    )

    curriculum.setup()
    curriculum.on_training_step(0)
    curriculum.reset(torch.tensor([0, 2], dtype=torch.long))
    curriculum.on_rollout_end(0)
    curriculum.env.extras["episode"]["hybrid/success/gen"] = torch.tensor([0.0, 0.0], dtype=torch.float32)
    curriculum.on_training_step(1)
    curriculum.reset(torch.tensor([1, 3], dtype=torch.long))
    curriculum.on_rollout_end(1)

    assert torch.allclose(curriculum.success_ema, torch.tensor(0.5, dtype=torch.float32))
    assert torch.allclose(curriculum.current_scale, torch.tensor(0.3, dtype=torch.float32))
    assert curriculum.motion_command is not None
    assert abs(curriculum.motion_command.applied_scale - 0.3) < 1.0e-6
    assert torch.equal(curriculum.motion_command.last_reapplied_env_ids, torch.tensor([0, 1, 2, 3], dtype=torch.long))


def test_hybrid_object_goal_noise_curriculum_only_collects_on_selected_rollouts():
    curriculum = _make_hybrid_object_goal_noise_curriculum(
        params={"update_interval": 3},
        extras={"episode": {"hybrid/success/gen": torch.tensor([1.0, 1.0], dtype=torch.float32)}},
    )

    curriculum.setup()
    curriculum.on_training_step(0)
    curriculum.reset(torch.tensor([0, 1], dtype=torch.long))
    curriculum.on_rollout_end(0)

    assert int(curriculum.accumulated_success_count.item()) == 0
    assert torch.allclose(curriculum.current_scale, torch.tensor(0.3, dtype=torch.float32))
    assert not curriculum.has_success_ema

    curriculum.on_training_step(2)
    curriculum.reset(torch.tensor([0, 1], dtype=torch.long))
    assert int(curriculum.accumulated_success_count.item()) == 2
    curriculum.on_rollout_end(2)

    assert torch.allclose(curriculum.success_ema, torch.tensor(1.0, dtype=torch.float32))
    assert torch.allclose(curriculum.current_scale, torch.tensor(0.31, dtype=torch.float32))


def test_hybrid_object_goal_noise_curriculum_skips_update_when_selected_rollout_has_no_completed_gen_episodes():
    curriculum = _make_hybrid_object_goal_noise_curriculum(
        params={"update_interval": 1},
        extras={"episode": {}},
    )

    curriculum.setup()
    curriculum.on_training_step(0)
    curriculum.reset(torch.tensor([0, 1], dtype=torch.long))
    curriculum.on_rollout_end(0)

    assert not curriculum.has_success_ema
    assert torch.allclose(curriculum.current_scale, torch.tensor(0.3, dtype=torch.float32))


def test_hybrid_object_goal_noise_curriculum_requires_hybrid_command():
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=2,
        extras={"episode": {}},
        command_manager=SimpleNamespace(get_state=lambda name: SimpleNamespace() if name == "motion_command" else None),
    )
    curriculum = HybridGenNoiseCurriculum(
        SimpleNamespace(params={"enabled": True}),
        env,
    )

    try:
        curriculum.setup()
    except RuntimeError as exc:
        assert "requires HybridMotionCommand" in str(exc)
    else:
        raise AssertionError("Expected setup() to reject non-hybrid motion commands.")


def test_hybrid_object_goal_noise_curriculum_requires_object_backed_hybrid_motion():
    motion_command = object.__new__(HybridMotionCommand)
    motion_command.device = torch.device("cpu")
    motion_command.motion = SimpleNamespace(has_object=False)
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=2,
        extras={"episode": {}},
        command_manager=SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None),
    )
    curriculum = HybridGenNoiseCurriculum(
        SimpleNamespace(params={"enabled": True}),
        env,
    )

    try:
        curriculum.setup()
    except RuntimeError as exc:
        assert "no object data" in str(exc)
    else:
        raise AssertionError("Expected setup() to reject hybrid motions without objects.")


def test_hybrid_object_goal_noise_curriculum_per_object_group_steps_each_group_by_its_own_success():
    motion_command = object.__new__(HybridMotionCommand)
    motion_command.device = torch.device("cpu")
    motion_command.num_envs = 4
    motion_command.motion = SimpleNamespace(has_object=True)
    motion_command.num_object_groups = 2
    motion_command.env_object_group = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    motion_command.set_object_goal_noise_curriculum_scale = lambda scale: None
    motion_command.set_object_goal_noise_group_scales = lambda scales: setattr(
        motion_command, "applied_group_scales", scales.clone()
    )
    motion_command.reapply_gen_object_goal_noise = lambda env_ids: None
    curriculum = _make_hybrid_object_goal_noise_curriculum(
        params={
            "update_interval": 1,
            "success_threshold_up": 0.7,
            "success_threshold_down": 0.5,
            "scale_step": 0.1,
            "per_object_group": True,
        },
        motion_command=motion_command,
        extras={"episode": {"hybrid/success/gen": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)}},
    )
    curriculum.env.hybrid_gen_reset_env_ids = torch.tensor([0, 1, 3], dtype=torch.long)

    curriculum.setup()
    curriculum.on_training_step(0)
    curriculum.reset(torch.tensor([0, 1, 3], dtype=torch.long))
    curriculum.on_rollout_end(0)

    # group 0 succeeded (scale up), group 1 failed (already at min_scale); the shared EMA (2/3) alone would not move
    assert torch.allclose(curriculum.group_success_ema, torch.tensor([1.0, 0.0]))
    assert torch.allclose(curriculum.group_scale, torch.tensor([0.4, 0.3]))
    assert torch.allclose(motion_command.applied_group_scales, torch.tensor([0.4, 0.3]))

    # a rollout with episodes from group 1 only leaves group 0's EMA and scale alone
    curriculum.env.extras["episode"]["hybrid/success/gen"] = torch.tensor([1.0, 1.0], dtype=torch.float32)
    curriculum.env.hybrid_gen_reset_env_ids = torch.tensor([2, 3], dtype=torch.long)
    curriculum.on_training_step(1)
    curriculum.reset(torch.tensor([2, 3], dtype=torch.long))
    curriculum.on_rollout_end(1)
    assert torch.allclose(curriculum.group_success_ema, torch.tensor([1.0, 0.5]))
    assert torch.allclose(curriculum.group_scale, torch.tensor([0.4, 0.3]))


def test_hybrid_object_goal_noise_curriculum_initial_group_scales_apply_from_the_first_training_step():
    motion_command = object.__new__(HybridMotionCommand)
    motion_command.device = torch.device("cpu")
    motion_command.num_envs = 4
    motion_command.motion = SimpleNamespace(has_object=True)
    motion_command.num_object_groups = 2
    motion_command.env_object_group = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    motion_command.set_object_goal_noise_curriculum_scale = lambda scale: None
    motion_command.set_object_goal_noise_group_scales = lambda scales: setattr(
        motion_command, "applied_group_scales", scales.clone()
    )
    motion_command.reapply_gen_object_goal_noise = lambda env_ids: None
    curriculum = _make_hybrid_object_goal_noise_curriculum(
        params={"per_object_group": True, "initial_group_scales": "0.94,0.2"},
        motion_command=motion_command,
        extras={"episode": {}},
    )

    curriculum.setup()
    # set up, but not given to the motion command until training starts (an evaluation keeps its own scale)
    assert torch.allclose(curriculum.group_scale, torch.tensor([0.94, 0.3]))  # clamped to min_scale
    assert not hasattr(motion_command, "applied_group_scales")
    curriculum.on_training_step(0)
    assert torch.allclose(motion_command.applied_group_scales, torch.tensor([0.94, 0.3]))
    assert torch.isclose(curriculum.current_scale, torch.tensor(0.62))

    bad = _make_hybrid_object_goal_noise_curriculum(
        params={"per_object_group": True, "initial_group_scales": "0.5,0.5,0.5"},
        motion_command=motion_command,
        extras={"episode": {}},
    )
    with pytest.raises(ValueError, match="initial_group_scales"):
        bad.setup()


def test_hybrid_object_goal_noise_group_scales_scale_each_environment_goal_noise():
    from holosoma.config_types.command import GoalPoseNoiseConfig

    command = object.__new__(HybridMotionCommand)
    command.device = torch.device("cpu")
    command.num_envs = 512
    command.motion = SimpleNamespace(has_object=True)
    command.hybrid_cfg = SimpleNamespace(
        object_goal_noise=GoalPoseNoiseConfig(
            pos_std_xyz=[0.5, 0.5, 0.001], pos_clip_xyz=[1.0, 1.0, 0.001], rpy_std=[0.0, 0.0, 0.4], rpy_clip=[0.0, 0.0, 0.8]
        )
    )
    command.env_object_group = torch.arange(512) % 2
    command._fixed_size_randomization_enabled = lambda: False
    command._exact_goal_object_pos_w = torch.zeros(512, 3)
    command._exact_goal_object_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(512, 1)
    command._active_goal_object_pos_w = torch.zeros(512, 3)
    command._active_goal_object_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(512, 1)

    command.set_object_goal_noise_group_scales(torch.tensor([0.0, 0.5]))
    command._apply_gen_object_goal_noise(torch.arange(512))

    xy = command._active_goal_object_pos_w[:, :2]
    assert torch.equal(xy[0::2], torch.zeros(256, 2))  # group 0 at scale 0: the exact goal
    assert float(xy[1::2].abs().max()) <= 0.5 + 1.0e-6  # group 1: clip 1.0 m scaled by 0.5
    assert float(xy[1::2].std()) > 0.15
    assert torch.allclose(command._active_goal_object_quat_w[0::2], command._exact_goal_object_quat_w[0::2])
