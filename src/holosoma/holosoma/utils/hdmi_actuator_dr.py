"""HDMI actuator sampling and substep filter, ported from active-adaptation.

Reference: mimic_lite/tasks/actions.py JointPosition and
active_adaptation/envs/mdp/randomization.py motor_params_implicit.
Preserves integer physics-substep delay, per-substep lerp and log-uniform scales.
"""
import math
import re

import torch


def grouped_bounds(names, groups, *, device):
    bounds = []
    for name in names:
        matches = [value for pattern, value in groups.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f"Expected one actuator group for {name}, found {len(matches)}")
        low, high = matches[0]
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
            raise ValueError(f"Invalid positive range for {name}: {matches[0]}")
        bounds.append((low, high))
    return torch.tensor(bounds, dtype=torch.float32, device=device)


def log_uniform_scales(bounds, num_envs):
    low, high = bounds.unbind(-1)
    low = low.unsqueeze(0).expand(num_envs, -1)
    high = high.unsqueeze(0).expand(num_envs, -1)
    # Same operation order as legacy uniform(low.log(), high.log()).exp().
    return (low.log() + torch.rand_like(low) * (high.log() - low.log())).exp()


class HDMISubstepFilter:
    def __init__(self, num_envs, num_dof, decimation, device, *, min_delay=1, max_delay=3,
                 alpha_range=(0.8, 1.0)):
        if not (0 <= min_delay <= max_delay and decimation > 0):
            raise ValueError("Invalid physics-substep delay range")
        if not (0 < alpha_range[0] <= alpha_range[1] <= 1):
            raise ValueError("Invalid smoothing alpha range")
        self.num_envs, self.action_dim = num_envs, num_dof
        self.decimation, self.device = decimation, device
        self.min_delay, self.max_delay = min_delay, max_delay
        self.alpha_range = alpha_range
        delay_hist = max((max_delay - 1) // decimation + 1, 3)
        self.action_buf = torch.zeros(num_envs, delay_hist, num_dof, device=device)
        self.applied_action = torch.zeros(num_envs, num_dof, device=device)
        self.alpha = torch.ones(num_envs, 1, device=device)
        self.delay = torch.zeros(num_envs, 1, dtype=torch.long, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0
        self.delay[env_ids] = torch.randint(self.min_delay, self.max_delay + 1,
                                          (len(env_ids), 1), device=self.device)
        self.alpha[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            self.alpha_range[0], self.alpha_range[1])

    def process_action(self, action):
        self.action_buf = self.action_buf.roll(1, dims=1)
        self.action_buf[:, 0] = action

    def apply_action(self, substep):
        if not 0 <= substep < self.decimation:
            raise ValueError(f"Invalid physics substep {substep}")
        delay_idx = (self.delay - substep + self.decimation - 1) // self.decimation
        delay_idx = delay_idx.clamp_(0, self.action_buf.shape[1] - 1)
        delayed_action = torch.gather(
            self.action_buf, 1, delay_idx[:, :, None].expand(-1, 1, self.action_dim)).squeeze(1)
        self.applied_action.lerp_(delayed_action, self.alpha)
        return self.applied_action
