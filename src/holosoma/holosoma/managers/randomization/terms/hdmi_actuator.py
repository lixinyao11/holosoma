"""Opt-in actuator-only DR; does not change materials, body dynamics or rewards."""
import torch

from holosoma.managers.randomization.terms.locomotion import ActuatorRandomizerState
from holosoma.utils.hdmi_actuator_dr import grouped_bounds, log_uniform_scales


class HDMIActuatorRandomizerState(ActuatorRandomizerState):
    def setup(self):
        super().setup()
        if self.enable_rfi_lim:
            raise ValueError("The actuator-only preset must not enable torque RFI")
        names = self.env.robot_config.dof_names
        self.kp_bounds = grouped_bounds(names, self.cfg.params['kp_groups'], device=self.env.device)
        self.kd_bounds = grouped_bounds(names, self.cfg.params['kd_groups'], device=self.env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self.kp_scale[env_ids] = log_uniform_scales(self.kp_bounds, len(env_ids))
        self.kd_scale[env_ids] = log_uniform_scales(self.kd_bounds, len(env_ids))
        self.rfi_lim_scale[env_ids] = 1.0


def randomize_armature_startup(env, *, scale_range=(0.75, 1.25)):
    """Legacy startup-only log-uniform multiplier, with explicit joint-name mapping."""
    sim = env.simulator
    if hasattr(sim, 'hdmi_armature_audit'):
        raise RuntimeError("Armature startup DR must run once, not accumulate on reset")
    names = list(env.robot_config.dof_names)
    bounds = grouped_bounds(names, {'.*': scale_range}, device=env.device)
    scale = log_uniform_scales(bounds, env.num_envs)
    robot = sim._robot
    indices = torch.tensor([robot.joint_names.index(n) for n in names], device=env.device)
    if sim.__class__.__name__ == 'IsaacSim':
        base = robot.data.joint_armature[:, indices].clone()
        actual = base * scale
        robot.write_joint_armature_to_sim(actual, joint_ids=indices)
        applied = robot.root_physx_view.get_dof_armatures()[:, indices.cpu()].to(env.device)
    else:
        raise NotImplementedError(f"Armature DR not implemented for {type(sim).__name__}")
    if not torch.equal(actual, applied):
        raise RuntimeError("Armature readback differs from requested values")
    sim.hdmi_armature_audit = {'joint_names': names, 'baseline': base.detach().clone(),
                              'scale': scale.detach().clone(), 'actual': actual.detach().clone()}
