# HDMI 3A: root-velocity perturbations

Preset: `exp:g1-29dof-wbt-hybrid-object-hdmi-push-dr`.
The `hdmi-friction-dr` preset inherits this configuration, including joint friction.
Both use the same configuration with `simulator:isaacsim` or `simulator:mjlab`.

## Current settings: legacy HDMI amplitude and interval

| World-frame velocity increment | Uniform range |
| --- | --- |
| Linear X/Y | ±0.2 m/s each |
| Linear Z | ±0.1 m/s |
| Angular X/Y | ±0.2 rad/s each |
| Angular Z | ±0.4 rad/s |

Each environment samples an independent 4–6 second interval. The production
scheduler converts it to integer control steps using its existing floor rule
(20 ms control period). Reset, resampling, evaluation and disabled guards are
unchanged. These are additive velocity increments, not forces or Euler angles.

Only horizontal amplitude (previously ±0.4 m/s) and interval (previously 1–5 s)
were reduced. Groups 1/2, the vertical/angular part of 3A, joint friction in the
friction preset, and the four self-collision exclusions remain unchanged.
The earlier actuator/body-only presets are preserved as separate configurations.

## Backend semantics and existing acceptance

WBT `_push_robots` sends six-component increments through
`set_actor_root_state_tensor_robots`. MJLab converts COM linear velocity to
free-root link velocity using the current randomized COM; Isaac writes native
root COM velocity. No backend code changes are included in this adjustment.

The prior 64-environment / two-iteration native-engine smokes used the earlier
±0.4 m/s, 1–5 s settings and are retained as interface evidence. Their native
write/readback checks are reused, not described as a new smoke at the reduced
settings. The subsequent joint-friction smokes are also retained.

For this configuration-only adjustment, CPU acceptance compares complete old/new
push and friction profiles, verifies that only the two authorized fields change,
and invokes the actual production scheduler and push method against a write-call
fixture. Same random samples must produce exactly half the horizontal increment
and unchanged vertical/angular increments. A 24-second, 64-environment schedule
checks bounds, triggering and resampling. All other production Python files are
SHA-checked against the preceding source snapshot. No formal training or new GPU
simulation is started for this adjustment.

Evidence: `/data/pummpkin/holosoma-push-align-rp9-v1/acceptance.json` and the
desktop archive `/home/pummpkin/holosoma-push-align-audit-v1`.
