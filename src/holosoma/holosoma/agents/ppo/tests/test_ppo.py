from types import SimpleNamespace

import torch

from holosoma.agents.ppo.ppo import PPO


def test_reset_policy_std_on_resume_restores_init_std_and_clears_optimizer_state():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(reset_policy_std_on_resume=True, init_noise_std=0.8)
    ppo.actor = SimpleNamespace(std=torch.nn.Parameter(torch.tensor([0.2, 0.4], dtype=torch.float32)))
    ppo.actor_optimizer = torch.optim.Adam([ppo.actor.std], lr=1e-3)

    loss = ppo.actor.std.sum()
    loss.backward()
    ppo.actor_optimizer.step()
    ppo.actor_optimizer.zero_grad()

    assert ppo.actor.std in ppo.actor_optimizer.state

    PPO._maybe_reset_policy_std_on_resume(ppo)

    assert torch.allclose(ppo.actor.std.detach(), torch.full((2,), 0.8, dtype=torch.float32))
    assert ppo.actor.std not in ppo.actor_optimizer.state


def test_reset_policy_std_on_resume_is_noop_when_disabled():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(reset_policy_std_on_resume=False, init_noise_std=0.8)
    ppo.actor = SimpleNamespace(std=torch.nn.Parameter(torch.tensor([0.2, 0.4], dtype=torch.float32)))
    ppo.actor_optimizer = torch.optim.Adam([ppo.actor.std], lr=1e-3)

    loss = ppo.actor.std.sum()
    loss.backward()
    ppo.actor_optimizer.step()
    ppo.actor_optimizer.zero_grad()

    std_before = ppo.actor.std.detach().clone()

    PPO._maybe_reset_policy_std_on_resume(ppo)

    assert torch.allclose(ppo.actor.std.detach(), std_before)
    assert ppo.actor.std in ppo.actor_optimizer.state
