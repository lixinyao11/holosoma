# HDMI body/reset DR (group 1 + group 2)

Use `exp:g1-29dof-wbt-hybrid-object-hdmi-body-dr` with either
`simulator:isaacsim` or `simulator:mjlab`.
The existing `exp:g1-29dof-wbt-hybrid-object-hdmi-actuator-dr` remains group 1 only.
The original `exp:g1-29dof-wbt-hybrid-object-hdmi` is unchanged.

## New perturbations

| Field | Distribution | Timing |
| --- | --- | --- |
| `torso_link` mass | uniform multiplier 0.9–1.1; inertia receives the same multiplier | once at startup, independently per environment |
| `pelvis` COM | uniform additive XYZ offsets −0.03 to +0.03 m in body frame | once at startup, independently per environment/axis |
| Initial joint velocity | reference joint velocity + independent uniform −0.1 to +0.1 rad/s | every motion reset, only selected environments |

Both backends implement the same ranges. Pelvis COM follows the old HDMI **MJLab**
range; it does not inherit the old Isaac branch's half-size X offset.
Existing leg/pelvis mass DR, torso COM ranges, object/material DR and group-1
PD/armature/substep delay/filter settings remain unchanged. No extra pushes or
contact perturbations are enabled by this preset. Extra random draws naturally
change later RNG consumption; this is not a bitwise old/new trajectory comparison.

## Implementation and evidence

`managers/randomization/terms/hdmi_body.py` uses the legacy mass/inertia scaling
and additive COM convention. Isaac uses native PhysX get/set methods; MJLab writes
its batched model and recomputes derived constants. Startup readback covers all masses, inertias and COM arrays, including non-selected
entries. MJLab is bitwise exact. PhysX inverse-mass/inertia-frame/quaternion
roundtrips use a per-body bound of eight float32 epsilon times the corresponding
parameter magnitude (COM translation and rotation checked separately); both
requested and actual values are retained. This is not a change to rollout tolerances. A second
startup invocation is rejected to prevent accumulating perturbations.

`managers/command/terms/hdmi_body.py` subclasses the existing hybrid command.
It preserves native reference reset, adds velocity noise, and uses the unchanged
WBT reset-to-simulator writeback. It retains the latest reference/noise/actual
velocity and per-environment reset counts. Empty resets consume no RNG.

`tests/unit/body_dr/verify.py` contains focused CPU checks only for this increment.
GPU smoke instrumentation additionally checks native engine joint velocities after
reset and that body parameters are not re-sampled by resets. Existing group-1
CPU acceptance is reused. Smoke is not evidence of improved sim2sim or sim2real
performance, and no formal training has been started.

Sources: prior HDMI DR audit and original `perturb_body_mass` / `perturb_body_com`
implementations, preserved under the local `holosoma-hdmi-dr-audit-v1` and
`holosoma-sim2sim-video-v1/audit` evidence directories. Existing robot asset,
observation, reward and PPO are unchanged.
