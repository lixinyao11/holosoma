"""Group-2 reset velocity noise after native reference-state initialization."""
import torch
from holosoma.managers.command.terms.wbt import HybridMotionCommand


class HDMIBodyMotionCommand(HybridMotionCommand):
    def reset(self, env_ids):
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        super().reset(env_ids)
        sim = self._env.simulator
        reference = self.joint_vel[env_ids].clone()
        if not torch.equal(sim.dof_vel[env_ids], reference):
            raise RuntimeError('Native reset no longer writes reference joint velocities')
        bounds = self.cfg.params.get('joint_velocity_noise_range', (-0.1, 0.1))
        noise = torch.empty_like(reference).uniform_(*bounds)
        actual = reference + noise
        sim.dof_vel[env_ids] = actual
        # WBT's existing _refresh_envs_after_reset writes this state to the engine.
        if not hasattr(self, 'hdmi_velocity_audit'):
            self.hdmi_velocity_audit = {
                key: torch.zeros_like(sim.dof_vel) for key in ('reference', 'noise', 'actual')}
            self.hdmi_velocity_audit['reset_count'] = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long)
        for key, value in (('reference', reference), ('noise', noise), ('actual', actual)):
            self.hdmi_velocity_audit[key][env_ids] = value
        self.hdmi_velocity_audit['reset_count'][env_ids] += 1
