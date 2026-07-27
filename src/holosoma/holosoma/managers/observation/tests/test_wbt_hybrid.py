from types import SimpleNamespace

import torch

from holosoma.managers.observation.terms import wbt as wbt_obs


def test_hybrid_motion_command_masks_gen_env_rows(monkeypatch):
    env = SimpleNamespace(num_envs=2, device="cpu")
    motion_command = SimpleNamespace(
        command=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        track_env_mask=torch.tensor([True, False]),
        gen_env_mask=torch.tensor([False, True]),
    )

    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)

    masked = wbt_obs.hybrid_motion_command(env)

    assert torch.allclose(masked, torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float32))


def test_hybrid_motion_ref_terms_mask_gen_env_rows(monkeypatch):
    env = SimpleNamespace(num_envs=2, device="cpu")
    motion_command = SimpleNamespace(
        track_env_mask=torch.tensor([True, False]),
        gen_env_mask=torch.tensor([False, True]),
    )

    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)
    monkeypatch.setattr(
        wbt_obs,
        "motion_ref_pos_b",
        lambda _: torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
    )
    monkeypatch.setattr(
        wbt_obs,
        "motion_ref_ori_b",
        lambda _: torch.tensor([[1.0] * 6, [2.0] * 6], dtype=torch.float32),
    )

    masked_pos = wbt_obs.hybrid_motion_ref_pos_b(env)
    masked_ori = wbt_obs.hybrid_motion_ref_ori_b(env)

    assert torch.allclose(masked_pos, torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=torch.float32))
    assert torch.allclose(masked_ori, torch.tensor([[1.0] * 6, [0.0] * 6], dtype=torch.float32))


def test_hybrid_object_ref_terms_mask_gen_env_rows(monkeypatch):
    env = SimpleNamespace(num_envs=2, device="cpu")
    motion_command = SimpleNamespace(
        track_env_mask=torch.tensor([False, True]),
        gen_env_mask=torch.tensor([True, False]),
    )

    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)
    monkeypatch.setattr(
        wbt_obs,
        "obj_ref_pos_b",
        lambda _: torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
    )
    monkeypatch.setattr(
        wbt_obs,
        "obj_ref_ori_b",
        lambda _: torch.tensor([[1.0] * 6, [2.0] * 6], dtype=torch.float32),
    )

    masked_pos = wbt_obs.hybrid_obj_ref_pos_b(env)
    masked_ori = wbt_obs.hybrid_obj_ref_ori_b(env)

    assert torch.allclose(masked_pos, torch.tensor([[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]], dtype=torch.float32))
    assert torch.allclose(masked_ori, torch.tensor([[0.0] * 6, [2.0] * 6], dtype=torch.float32))


def test_goal_terms_read_active_goal_buffers(monkeypatch):
    env = SimpleNamespace(num_envs=1, device="cpu")
    motion_command = SimpleNamespace(
        robot_ref_pos_w=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        goal_ref_pos_w=torch.tensor([[4.0, 6.0, 9.0]], dtype=torch.float32),
        goal_ref_quat_w=torch.tensor([[0.0, 0.0, 0.70710678, 0.70710678]], dtype=torch.float32),
    )

    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)

    pos_b = wbt_obs.motion_goal_pos_b(env)
    ori_b = wbt_obs.motion_goal_ori_b(env)

    assert torch.allclose(pos_b, torch.tensor([[3.0, 4.0, 6.0]], dtype=torch.float32), atol=1.0e-5)
    assert torch.allclose(
        ori_b,
        torch.tensor([[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        atol=1.0e-5,
    )
