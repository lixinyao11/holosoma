# HDMI actuator DR (opt-in)

Current hardware baseline: DR multipliers now apply to the mode-15 nominal motors/PD; see [hdmi-mode15.md](hdmi-mode15.md). The unchanged-nominal-gains statement below describes the initial DR-only patch.

Preset: `exp:g1-29dof-wbt-hybrid-object-hdmi-actuator-dr`.
The existing `exp:g1-29dof-wbt-hybrid-object-hdmi` remains unchanged.
This branch adds no automatic training or Stage2 launcher.

| Parameter | Range | Sampling |
| --- | --- | --- |
| Arm Kp/Kd multipliers | 0.9–1.1 | Independent log-uniform per joint, each reset |
| Waist/leg Kp/Kd multipliers | 0.8–1.2 | Independent log-uniform per joint, each reset |
| Joint armature multiplier | 0.75–1.25 | Log-uniform per joint, once at startup |
| Action delay | 1–3 physics substeps (5–15 ms) | Uniform integer per environment, each reset |
| Action low-pass alpha | 0.8–1.0 | Uniform per environment, each reset; applied every physics substep |

The source of these algorithms is the preserved active-adaptation implementation:
`mimic_lite/tasks/actions.py::JointPosition`, and
`active_adaptation.envs.mdp.randomization::motor_params_implicit`.
The independent filter oracle is retained under `tests/unit/actuator_dr/legacy_actions.py`.
The log-uniform PD sampling matches the old MJLab HDMI implementation. Holosoma's
nominal PD gains, action scale, clipping, observations, rewards, PPO and existing
randomizations are unchanged. Added DR naturally changes the RNG consumption.

The new action class inherits Holosoma's original PD/torque recording and inserts
the old delay/low-pass filter before PD. The existing control-step delay must stay
disabled: its 20 ms unit is different. The 5–15 ms preset requires a 5 ms physics
timestep. The alpha coefficient is a physics-substep coefficient, not a control-step
coefficient. Partial reset clears only selected environments' filter history.

Armature uses the original baseline exactly once. MJLab updates derived constants
using the old HDMI `set_const_0` procedure. IsaacSim uses its native joint-armature
write interface. Startup checks exact readback. No joint-friction, body-mass, COM,
material, push or initial-velocity DR is added by this preset.

Focused CPU acceptance (no simulator rollout):

```sh
PYTHONPATH=src/holosoma python tests/unit/actuator_dr/verify.py
```

The checks cover exact legacy filter output across 320 substeps, selective reset,
log-uniform sampling, exact unchanged configuration outside the actuator group,
neutral-filter PD equivalence, non-neutral PD routing, and invalid/stacked delay.
GPU smoke additionally checks actual armature writeback, nonconstant PD scales,
5–15 ms sampled delay, alpha range, and finite physical state and torques.
Use the audited experiment-specific smoke runner; no smoke output is a formal
checkpoint or evidence of improved task success.
