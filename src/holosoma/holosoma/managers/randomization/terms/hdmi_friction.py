"""Episode-wise, load-independent joint dry-friction effort for G1.

The old HDMI MJLab range is an absolute effort, not a scale or material friction.
Isaac Sim >=5 uses static/dynamic efforts; older coefficient APIs are rejected.
"""
import torch
from holosoma.managers.randomization.base import RandomizationTermBase
from holosoma.managers.randomization.terms.locomotion import _ensure_env_ids_tensor


class HDMIJointFrictionRandomizer(RandomizationTermBase):
    def setup(self):
        self.low, self.high = map(float, self.cfg.params['effort_range_nm'])
        if not 0.0 <= self.low <= self.high:
            raise ValueError('Joint friction requires a nonnegative ordered effort range')
        self.sim = self.env.simulator
        self.robot = self.sim._robot
        self.joint_names = list(self.env.robot_config.dof_names)
        self.joint_ids = torch.tensor([self.robot.joint_names.index(n) for n in self.joint_names],
                                      device=self.env.device, dtype=torch.long)
        self.sampled_effort = torch.zeros((self.env.num_envs, len(self.joint_names)), device=self.env.device)
        self.reset_count = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if self.sim.__class__.__name__ == 'IsaacSim':
            from isaacsim.core.version import get_version
            if int(get_version()[2]) < 5:
                raise RuntimeError('This effort-based profile requires Isaac Sim >=5; legacy coefficients are not equivalent')
            self.backend = 'isaacsim'
        else:
            raise NotImplementedError(f'Joint friction DR unsupported for {type(self.sim).__name__}')

    def reset(self, env_ids=None):
        env_ids = _ensure_env_ids_tensor(self.env, env_ids)
        if not env_ids.numel():
            return
        # Legacy HDMI distribution: independent uniform absolute friction per joint and episode.
        effort = torch.empty((env_ids.numel(), len(self.joint_names)), device=self.env.device).uniform_(self.low, self.high)
        # >=5 API retains the old "coefficient" name, but its first two native
        # components are static/dynamic effort. Set both for moving-joint drag.
        # None preserves the existing viscous parameter; do not add damping.
        self.robot.write_joint_friction_coefficient_to_sim(
            effort, joint_dynamic_friction_coeff=effort,
            joint_ids=self.joint_ids, env_ids=env_ids)
        self.sampled_effort[env_ids] = effort
        self.reset_count[env_ids] += 1

    def step(self):
        pass
