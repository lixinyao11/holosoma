# HDMI mode 15 asset in Holosoma

The HDMI teacher presets in this worktree now use mode 15 through
`config_values/wbt/g1/robot_mode15.py`. The upstream non-HDMI presets and old
robot assets remain available. Isaac and the MJLab adapter both import the same
vendored `data/robots/g1_mode15/g1_mode15_holosoma.urdf`; neither resolves a
moving remote asset at runtime.

## Source and adaptation

- Geometry source: `elijahgalahad/g1_xmls`, revision
  `1c1d082f299126b9272e81c04121ff347c97fa8f`, `g1-mode_13_15.urdf`.
- Original source URDF SHA256:
  `7b0838af39e9678162efc08c25143968b32a5f83d724892827df12b9e8a631a0`.
- Baseline motors and PD are taken from the desktop HDMI implementation's
  mode-15 selection: 7520-22 hip pitch and 5010 wrist pitch/yaw, including its
  reflected-inertia calculation, 10 Hz natural frequency and damping ratio 2.
  `provenance.json` records the exact source-code SHA and all mesh SHAs.
- The original mode-15 links, transforms, inertias, visual geometry and contact
  geometry are preserved. Holosoma's four existing non-contact fixed markers
  are appended with their original transforms/inertials (total 0.002000002 kg),
  preserving the 32-body motion/observation contract.
- Source ankle and waist pitch/roll URDF limit metadata (35 Nm, 30 rad/s) is
  reconciled to the old HDMI runtime actuator specification (50 Nm, 37 rad/s).
  These were already Holosoma's configured motor limits. All overrides are
  listed in the provenance file.
- The unmodified source MJCF is retained for provenance; the older standalone
  MuJoCo/run_sim path is not the backend validated here.

## Control and DR

| Joint group | Effort limit (Nm) | Speed (rad/s) | Armature | Kp | Kd |
|---|---:|---:|---:|---:|---:|
| Hip pitch, hip roll, knee | 139 | 20 | 0.025101925 | 99.0984 | 6.30880 |
| Hip yaw, waist yaw | 88 | 32 | 0.010177520 | 40.1792 | 2.55789 |
| Wrist pitch/yaw | 13.4 | 27 | 0.0021812 | 8.61103 | 0.548195 |
| Shoulder, elbow, wrist roll | 25 | 37 | 0.003609725 | 14.2506 | 0.907223 |
| Ankle, waist pitch/roll | 50 | 37 | 0.007219450 | 28.5012 | 1.81445 |

The external position-target PD, torque clipping, 200 Hz physics / 50 Hz
control, and action formula `0.25 * effort_limit / baseline_Kp` are unchanged.
Wrist pitch/yaw scale is approximately 0.389036; it is not hardcoded to 0.44.
DR multipliers act on the new baselines. Existing cumulative DR, including
joint friction, horizontal pushes ±0.2 m/s every 4–6 s, and the four explicit
self-collision exclusions, are unchanged. Nominal friction is not replaced by
the old asset's 0.01 because the current episode-wise 0.05–0.25 Nm profile is
retained.

## Validation and use

Use `exp:g1-29dof-wbt-hybrid-object-hdmi-friction-dr` with either
`simulator:isaacsim` or `simulator:mjlab` for the cumulative DR profile. The
standard 043 motion/object path arguments remain necessary. MJLab retains the
previously authorized approximate material mapping.

Only the newly changed asset import, motor/PD values, named state mapping,
native armature and collision filters, and two-iteration (64-environment)
integration smoke are checked. Previous DR distribution tests are reused.
Smoke is fresh and is not a performance evaluation. Existing 4010 checkpoints
are not evidence of mode-15 policy quality and are not resumed automatically.
No formal training or commit is part of this update.
