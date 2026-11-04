> Historical experiment specification: the sections below describe the frozen
> pre-adoption A/B/C/D runs. The candidate branch now uses upstream joint rewards
> and hand observations; see [upstream adoption](upstream_hands_profile.md).
> Old dimensions and no-reference-actor claims below do not describe the new default.

# HDMI-informed goal-conditioned state teacher

Current hardware baseline: the HDMI presets now use the pinned mode-15 asset and baseline motors/PD described in [hdmi-mode15.md](hdmi-mode15.md). The unchanged-assets/PD statement below describes the initial observation/reward-only patch, not this later hardware update.

Opt-in preset: `exp:g1-29dof-wbt-hybrid-object-hdmi`.
Branch: `feat/hdmi-goal-rl`, based on denso-hil `9a0f685`.
The original hybrid-object and hybrid-multi-object presets remain unchanged.
Only a short runtime/PPO smoke has been performed with this preset; no formal
training has started. Reward choices below are initial engineering settings,
not tuned or experimentally validated improvements.

## Actor and critic

The actor still has no motion/reference/phase/ID input. It uses:

- Root angular velocity, projected gravity, joint position and velocity at policy
  offsets `[0, 1, 2, 3, 4, 8, 16]`, following HDMI's proprioceptive-history layout.
- Last actions at offsets `[0, 1, 2]`. At observation time offset 0 is the action
  just executed, not a future action.
- Current object pose, linear and angular velocity, and goal object pose.
- Current object-origin-to-left/right-wrist displacement in the existing current-torso conditioning frame.
  Object angular velocity uses that same frame, as does existing object linear velocity.
  These are wrist landmarks, not palm points or contact measurements.

Per-term histories use ObservationManager's reset-aware buffers. Missing history
is zero-padded, including after an individual environment reset. Dimension probes
and `modify_history=False` do not advance buffers. Noise is sampled before storing
history. The existing hybrid noise magnitudes are retained; projected-gravity
noise is 0.01. HDMI's additional gravity bias noise is not ported.

The actor has 565 inputs for the present 29-DoF G1 configuration.

The critic retains privileged reference information and receives matching body
and action history. This changes network input dimensions: existing actor/critic
checkpoints are not directly load-compatible. PPO, action scales, PD gains,
assets, randomization and the track/gen schedule are unchanged. Only the four
gen reference-deviation termination checks are disabled (details below).

## Rewards

- Reuse the eight original body/object tracking formulas in track environments;
  zero their contribution in gen environments. Original hybrid wrappers add a
  constant 0.5 on gen environments; the new preset deliberately removes that
  baseline instead of treating it as tracking accuracy.
- Add HDMI joint-position and joint-velocity tracking:
  `exp(-mean(abs(reference - actual))/sigma)`, track only. Position sigma=0.25,
  velocity sigma=2.5, both weight=0.5. Waist/hip/knee/shoulder/elbow/wrist joints
  are selected; ankles are excluded, following HDMI's reward-joint selection.
- Retain existing action-rate, joint-limit and undesired-contact penalties.
  Add joint-speed squared penalty with weight=-1e-4. Do not port feet-air-time
  and survival bonuses or pretend net body contact is pairwise self-contact.
- Gen progress: reward a new best value of normalized goal error
  `position_distance/1m + orientation_error/pi`, once for each improvement.
  Retreat and re-approach cannot repeatedly earn the same progress. Reset or
  goal-version changes establish a new baseline without a teleport bonus.
  Weight=1; the term divides by dt because RewardManager multiplies by dt.
  This is shaping, not a policy-invariant potential claim.
- Replace the original pose-only event with `GoalSettledOnce`: XY distance <=0.2m,
  lowest collision vertex <=0.05m above flat ground, orientation <=30deg,
  linear speed <=0.05m/s, angular
  speed <=0.2rad/s, continuously for 0.5s. Bonus=3, once per goal, gen only.
  Terminal failures do not qualify. The existing reward key
  `goal_object_reached_once` is retained so `hybrid/success/gen` and the goal-noise
  curriculum now use this stricter stable-arrival event.

The height test uses actual object position/orientation and URDF collision
vertices (including collision origins and mesh scales), plus runtime object scale
when fixed-size randomization is enabled. It does not use a rotated AABB corner
or assume the object's origin is its bottom. Ground height is environment-origin
Z; this preset supports a single rigid-link object on flat ground, not multiple
object variants or tabletop placement. Unsupported geometry fails explicitly.
Collision vertices are cached once. The check is a one-sided upper bound; it
adds no penetration tolerance rule. It is geometric clearance, not contact force
or exact PhysX cooked-collider surface measurement.

Reference goal Z is ignored by this arrival event, allowing slightly floating or
penetrating reference poses. Goal observations and the separate 3D progress
shaping term are unchanged. Support force and hand release are not required:
a stationary held object within 5cm of the ground can still qualify. Report this
as stable near-ground arrival, not verified release. Gen reference-deviation termination checks are disabled; tracking checks remain.

No palm-landmark tracking was added: the G1 asset's wrist/hand geometry must first
be reconciled with HDMI's palm landmarks. Existing wrist/body tracking is retained.

## Source and usage

The HDMI sources inspected on desktop were:
`projects/mimic-lite/cfg/task/tracking-base.yaml`,
`projects/mimic-lite/mimic_lite/tasks/rewards/track.py`, and
`projects/mimic-lite/mimic_lite/tasks/transforms.py` under active-adaptation.
The joint error in the source is absolute error, not squared error.

Use the normal Holosoma training CLI with this new preset and explicit motion
and object-URDF overrides. Dataset paths are not embedded in the preset. The
existing independent 043 smoke configuration does not silently switch to it.
Keep the original preset as the baseline, and use fresh initialization when
training this preset. CPU tests live in `tests/unit/test_hdmi_goal.py`.

## 043 configuration and pre-training audit

Portable factory: `holosoma.config_values.wbt.g1.hdmi_goal043.make_config(asset_root)`.
It selects the new HDMI goal preset, replaces only the motion/URDF paths and run
name, and keeps the original standalone `043-v1/config043.py` unchanged.
Inspect without importing Isaac or running learning:

```bash
python -m holosoma.config_values.wbt.g1.hdmi_goal043 --asset-root /path/to/assets/suitcase
```

The supplied assets use OMOMO/UMR 043; they are not asserted byte-identical to
historical HDMI retargeted data. This CLI is configuration preparation only.
No training budget or training launch is implied by generating the config.

## Goal/gen termination separation

The HDMI preset now sets these four existing `HybridBadTracking` options to `None`:
`gen_bad_ref_pos_z_threshold`, `gen_bad_ref_pos_xyz_threshold`,
`gen_bad_object_pos_z_threshold`, `gen_bad_object_ori_threshold`.
This uses the existing implementation's disable semantics; no shared termination
algorithm or original preset is modified. Gen can choose a different route and
placement time without being terminated for reference deviation.

Unchanged: all track checks, gen low-root-height threshold 0.45m, feet-proximity
0.05m, timeout, reward terms (including 3D `GoalProgress`), goal randomization,
and the 5cm near-ground arrival event. These retained conditions are not a claim
of exhaustive safety detection. Gen initialization still uses motion states,
the critic still sees reference. The base preset defaults to a fixed 50/50 split;
it does not automatically start with tracking or switch stages. The explicit
Stage 1/2 launch recipes below implement the upstream training schedule.

The previous audit's 3D-progress/raw-goal-Z difference is acknowledged and left
unchanged by user choice. No new reward tuning or training is part of this change.
Tests exercise actual termination dispatch: reference deviations still terminate
track, do not terminate gen, and low root height still terminates gen.


## Formal training recipe: inherit upstream Object Stage 1/2

Use `scripts_run/run_hdmi_goal043_stage1.sh`, then
`scripts_run/run_hdmi_goal043_stage2.sh`, copied from README Object Stage 1/2.
Set `GPU`, `ASSET_ROOT`, and (Stage 2 only) `STAGE1_CHECKPOINT` explicitly.
These scripts have only been prepared, not launched.

- Both stages: 8192 environments, `num_learning_iterations=15000`, environment
  spacing 4.0, start-at-zero probability 0.05, size randomization disabled.
- Stage 1: fresh initialization, track ratio 1.0.
- Stage 2: resume Stage 1, track ratio 0.5; reset policy std to 0.45;
  gen start-at-zero probability 0.2. Goal noise and curriculum arguments are
  copied exactly from the original README, with no new tuning.
- PPO, network widths, learning rate, rollout length, minibatches, remaining
  randomization and defaults continue to come from the original base preset.
- The only algorithmic deviations remain the approved observation/reward changes
  and four disabled gen reference-deviation terminations. In particular, do not
  restore the README's gen reference-Z termination of 0.6 over that approved change.
- 043 motion/URDF and machine paths are task/runtime substitutions. The smoke's
  environment count, iteration limit and checkpoint are not formal settings.
- The proposed 512-environment / 2000-iteration / fresh 50:50 recipe is withdrawn.
  Do not silently reduce the upstream budget or batch configuration for hardware.

The iteration limit above is passed unchanged to the original resume logic; it
is not a separate custom interpretation of additional iterations.
