"""Opt-in legacy HDMI substep delay/LPF, retaining Holosoma's PD and clipping."""
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm
from holosoma.utils.hdmi_actuator_dr import HDMISubstepFilter


class HDMIJointPositionActionTerm(JointPositionActionTerm):
    def setup(self):
        if getattr(self.env, "_randomize_ctrl_delay", False):
            raise ValueError("HDMI substep delay must not stack with control-step delay")
        super().setup()
        sim = self.env.simulator
        if abs(float(sim.sim_dt) - 0.005) > 1e-12:
            raise ValueError("This 5-15ms preset requires a 5ms physics timestep")
        self.hdmi_filter = HDMISubstepFilter(
            self.env.num_envs, self.action_dim,
            sim.simulator_config.sim.control_decimation, self.env.device,
            **self.cfg.params)

    def process_actions(self, actions):
        super().process_actions(actions)
        self.hdmi_filter.process_action(self._processed_actions)

    def apply_actions(self):
        self._actions_after_delay.copy_(self.hdmi_filter.apply_action(self._substep_idx))
        super().apply_actions()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.hdmi_filter.reset(env_ids)
