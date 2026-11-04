> Historical experiment specification: the sections below describe the frozen
> pre-adoption A/B/C/D runs. The candidate branch now uses upstream joint rewards
> and hand observations; see [upstream adoption](upstream_hands_profile.md).
> Old dimensions and no-reference-actor claims below do not describe the new default.

# M15 B/C/D configuration

Source branch: `pummpkin/dev` (formerly `pummpkin/legacy`).
Source owner: `desktop:/home/pummpkin/projects/holosoma-legacy`; the directory and
`legacy_*` Python module names are retained for launcher and API compatibility.
The former dev commit `084fe025` is archived as
`archive/dev-before-legacy-promotion-20261104`.
Base commit: `595ff7f0c8b7d4d67c2ca5756e65435ac1b203e9`.
Prepared snapshot: `legacy-m15-d40-phase-config-20261102-v1` (includes uncommitted changes; adds D to the unchanged C40 split from `legacy-m15-c40-config-20261102-v1`).

| Group | `extra_dr` | `motion_set` | Difference from A |
| --- | --- | --- | --- |
| A | `False` | `single043` | Existing combined M15 legacy baseline |
| B | `True` | `single043` | Four-item extra DR bundle |
| C | `False` | `multi40` | Reviewed UMR 40 training motions; 2 separate evaluation motions |
| D | `False` | `multi40`, `track_phase=True` | C plus one current actor phase scalar |
| Optional combination | `True` | `multi40` | Both; supported, not scheduled |

All retain the existing M15 assets, four self-collision exclusions, motor
PD/armature DR, and action delay/LPF. A/B/C keep phase off; D enables it explicitly. PPO, rewards,
terminations and the existing motion sampling policy are unchanged. D adds only
the actor phase observation; all existing observation terms remain unchanged. Common experiment settings (track ratio, number of environments,
iterations, simulator, etc.) still belong to the launcher: these new selectors
do not silently change those settings or start a run.

## B: one optional bundle

`make_legacy_m15_config(extra_dr=True)` adds:

1. Startup torso mass/inertia scale 0.9–1.1 and pelvis COM offset ±0.03 m.
2. Joint velocity noise ±0.1 rad/s relative to the reference during reset.
3. Push intervals 4–6 s, linear bounds X/Y ±0.2 m/s and Z ±0.1 m/s;
   angular bounds X/Y ±0.2 rad/s and Z ±0.4 rad/s.
4. Per-joint, per-episode static/dynamic dry-friction effort 0.05–0.25 N·m.
   This is joint effort, not contact-material friction; viscous friction is preserved.

The three runtime implementations are copied byte-for-byte from dev commit
`084fe0254245d218e5880fdadad0199c70a39e49`; configuration composition is separate.
Native body/friction writeback requires Isaac Sim 5. CPU tests validate the
configuration and runtime formulas; native writeback and training acceptance
belong to each run, not to this configuration factory.

## Configuration API and inspection

```python
from holosoma.config_values.wbt.g1.hdmi_goal043 import make_config, make_eval_config

a = make_config(single043_asset_root)
b = make_config(single043_asset_root, extra_dr=True)
c = make_config(prepared_multi40_asset_root, motion_set="multi40")
c_eval = make_eval_config(prepared_multi40_asset_root)  # Data split only; runner sets eval mode.
d = make_config(prepared_multi40_asset_root, motion_set="multi40", track_phase=True)
d_eval = make_eval_config(prepared_multi40_asset_root, track_phase=True)
```

The existing 043 module also exposes both switches. This command **only prints
the configuration** and does not start Isaac or training:

```sh
PYTHONPATH=src/holosoma /home/pummpkin/holosoma-runtime/venv/bin/python \
  -m holosoma.config_values.wbt.g1.hdmi_goal043 \
  --asset-root /home/pummpkin/holosoma-runtime/043-v1/assets/suitcase \
  --extra-dr --motion-set single043
```

## C: reviewed UMR train40 / eval2

Frozen selection: `umr-suitcase-train40-eval2-20261102-v1`.
The exact approved list is versioned in
`src/holosoma/holosoma/utils/m15_multi40_selection.json`, including source hashes,
reviewed geometry, train/eval/reserve/exclude membership and selection reasons.
Only UMR exports are accepted; no GMR, retargeting or trajectory edits.

- Training: exactly 40 files in `motions_train/`, including `sub4_suitcase_043`.
- Evaluation: exactly `sub14_suitcase_006` and `sub8_suitcase_043` in `motions_eval/`.
- `sub5_suitcase_051` is reserved; the remaining 17 reviewed exclusions stay out.
- This is a motion-level holdout, not a subject-level holdout. Two motions give a
  useful first check, not a broad generalization estimate.

Desktop prepared cache:
`/home/pummpkin/holosoma-runtime/multi40-umr-20261102-v1/assets/suitcase`.
It has shared suitcase geometry and an unmodified 043 anchor at `motions/` for
compatibility checks. Neither split loads that anchor directory: native
`MultiMotionLoader` receives only its explicit `motions_train/` or `motions_eval/`
directory. Eval's fallback `motion_file` also points to an eval motion, not 043.

```sh
# Inspection only; does not start Isaac or PPO.
PYTHONPATH=src/holosoma /home/pummpkin/holosoma-runtime/venv/bin/python \
  -m holosoma.config_values.wbt.g1.hdmi_goal043 \
  --asset-root /home/pummpkin/holosoma-runtime/multi40-umr-20261102-v1/assets/suitcase \
  --motion-set multi40 --motion-split train
# Evaluation data config: change only --motion-split to eval.
```

`make_eval_config` selects data only. The evaluation runner must load the chosen
checkpoint, enable evaluation mode and schedule each of the two held-out motions;
it must never optimize on these two motions. No policy evaluation was launched
as part of this preparation.

The factory validates both splits before the native loader can silently skip a
bad file: exact membership, no extras, manifest and pinned selection consistency,
frame counts, 50 Hz, joint/body coverage, shapes, finite values, unit quaternions,
approved file content and shared geometry. A missing eval file also blocks cache
validation. Unknown splits fail. No fallback to 35 or the historical 50 list.

To recreate the cache, use a **new** destination:

```sh
PYTHONPATH=src/holosoma /home/pummpkin/holosoma-runtime/venv/bin/python \
  scripts/prepare_m15_multi40.py \
  --source-root /mnt/nas/jyuntins/denso/data/holosoma/omomo/umr/suitcase \
  --reference-root /home/pummpkin/holosoma-runtime/043-v1/assets/suitcase \
  --destination /path/to/NEW/suitcase
```

This copies only the selected motion/geometry files to a separate cache and
validates before publishing it. It rejects an existing destination. It never
copies an environment, mutates source motions or updates an active RP directory.
The historical `multi50` validator and preparer remain available for reproducing
earlier work, but they are **not** this C experiment's selection. Its historical
35/50 export audit does not describe current train40/eval2 readiness.

## CPU validation and limits

```sh
PYTHONPATH=src/holosoma /home/pummpkin/holosoma-runtime/venv/bin/python \
  scripts/check_m15_multi40.py \
  --asset-root /home/pummpkin/holosoma-runtime/multi40-umr-20261102-v1/assets/suitcase \
  --report /path/to/cpu-data-sampling.json
```

The real-data check calls the unchanged native loader and sampler classes on CPU.
It verifies all frames against NPZ tensors after joint/body reordering and
quaternion conversion, both split counts, frame boundaries, the existing
1-second terminal hold on every clip, zero hold velocities, uniform sampling,
adaptive sampling/difficulty response and partial/empty reset selection. Fixed
reset selection is checked for 32 CPU reset states on each eval motion; these are
not policy rollouts. The adaptive sampler and 5% timestep-zero setting are
preserved, as are the default A and B configs and the native `wbt.py` implementation.

Unit tests cover split leakage, missing/extra/corrupt files, selection/geometry
tampering, no overwrite, independent copied files, selector isolation and the
existing B/default regression. CPU validation does not establish native Isaac
physics, contact quality, reset writeback, PPO behavior or successful GPU smoke.
This preparation uses the existing desktop CPU runtime; it is not a certification
that desktop package versions match the frozen RP runtime.

Snapshot evidence is under
`operations/rp-workflow/version_snapshots/legacy-m15-c40-config-20261102-v1/`:
patches, untracked originals, actual scoped files and manifest, split/config
binding, CPU tests and real-data checks. The master ledger is
`operations/rp-workflow/node_paths.json`. These snapshots record the original uncommitted preparation. Subsequent
experiments use frozen source copies; changing or committing this working tree
does not update an existing run.

## D: C plus track phase

D uses exactly C's 40 training motions and the same two held-out motions. It does
not enable B's four-item extra DR bundle. The data, sampler, PPO, reward,
termination, existing observations, M15 assets and common three enhancements
remain identical to C. Use `track_phase=True` (CLI: `--track-phase`) for both
D training and D evaluation; the default is off for existing A/B/C callers.

The runtime function is copied byte-for-byte from current commit
`084fe0254245d218e5880fdadad0199c70a39e49` at
`holosoma/managers/observation/terms/track_phase.py`. The legacy composition helper
adds its term to the selected config; it does not import current's cumulative DR
preset. Phase is the normalized local reference frame in the current motion:
start 0, original final frame 1, and the existing 1-second terminal hold stays 1.
It follows motion-ID/frame resets immediately, independently of episode elapsed
time; gen environments receive 0. Default-pose prepend/append transitions are
rejected because their phase semantics are not defined by this implementation.

This supplies time progress, not lift/carry/place labels, motion ID or future
reference poses. Adding phase may help distinguish positions within a motion;
it does not itself disambiguate different motions at the same normalized time.
D evaluation also requires reference progress. Report that conditioning when
comparing C/D; it is not evidence of reference-free goal-only deployment.

Only one current, unnoised actor scalar is added, with no extra history.
Expected dimensions are actor 566 (C 565), critic 770 (unchanged); native GPU smoke
must still read back full dimensions. D needs a fresh network matching its input
shape; do not silently load a 565-input C checkpoint or pad/truncate inputs.

The CPU suite tests configuration isolation, phase offset/length, middle-frame
reset, clip changes, single-frame clips, terminal hold, gen masking and the
native observation manager's scalar wiring. The real-data checker covers every
frame of all 42 motions after the native hold augmentation, plus 32,768 adaptive
reset samples per split and partial resets:

```sh
PYTHONPATH=src/holosoma /home/pummpkin/holosoma-runtime/venv/bin/python \
  scripts/check_m15_multi40_phase.py \
  --asset-root /home/pummpkin/holosoma-runtime/multi40-umr-20261102-v1/assets/suitcase \
  --report /path/to/phase-cpu-acceptance.json
```

D code, patches, exact C/D configs, source provenance, CPU checks and binding are
saved under `operations/rp-workflow/version_snapshots/legacy-m15-d40-phase-config-20261102-v1/`.
Those records describe preparation-time validation. See the validation status
below for subsequent runs; the data selection and prior snapshots are preserved.


## Validation status and repository contents

As recorded on 2026-11-04:

- A and C each completed their fresh 5000-iteration run after full smoke acceptance.
- D passed smoke and entered fresh training. The last retrieved evidence contains
  3718 iterations; its final run status and final artifact collection are unverified.
- B is a separate smoke32-to-fresh5000 experiment. Its native acceptance must be
  established by its own reports; code publication does not certify completion.
- C/D training results do not establish held-out policy performance. The two eval
  motions are selected by configuration, but require a separate evaluation runner.

Run the CPU regression suite in an existing compatible environment, with
`CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/holosoma`:

```sh
python -m pytest -p no:cacheprovider -q tests/unit/test_legacy_m15_profile.py tests/unit/test_hdmi_goal.py tests/unit/test_hdmi_goal043.py tests/unit/test_hdmi_goal_termination.py tests/unit/test_m15_variants.py tests/unit/test_m15_extra_dr_runtime.py tests/unit/test_m15_multi40.py tests/unit/test_m15_track_phase.py
```

The versioned selection JSON is a small provenance manifest, not the motion data.
Its status describes the freeze-time preparation and its bytes are hash-pinned;
do not rewrite that historical status to track a current run. Source datasets,
prepared motion caches, experiment logs, checkpoints, videos and operational
archives remain outside this code change.
