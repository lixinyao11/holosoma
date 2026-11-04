# Unified M15 configuration entry

The canonical entry is holosoma.config_values.wbt.g1.m15.make_config. It returns
the existing upstream ExperimentConfig, with its existing robot, observation,
reward, command, simulator and randomization types. No second schema or
parallel training configuration is introduced.

## Group parameters

| Group | extra_dr | motion_set | track_phase |
| --- | --- | --- | --- |
| A | false | single043 | false |
| B | true | single043 | false |
| C | false | multi40 | false |
| D | false | multi40 | true |

A group is a parameter preset, not a copy of a complete experiment. An explicit
group plus contradictory extra_dr, motion_set or track_phase raises ValueError.
Omit group to construct a custom combination. Hardware and upstream feature
options can modify any group; the run name records those modifications.

Python:

    from holosoma.config_values.wbt.g1.m15 import make_config, make_eval_config
    cfg = make_config(asset_root, group="C")
    phase_cfg = make_config(asset_root, group="D")
    heldout_cfg = make_eval_config(asset_root, group="D")

The default remains the latest upstream-hands profile. This refactor does not
revert the joint/hand adoption or reinterpret historical A/B/C/D results.

## Inspect/export without training

    python -m holosoma.config_values.wbt.g1.m15         --asset-root /path/to/suitcase --group D --output experiment.yaml

Omit --output to print expanded JSON. JSON/YAML exports use the upstream
ExperimentConfig serialization. Output files must not already exist.
This command only validates/builds/exports; it never starts Isaac or training.
Exported YAML is an audit artifact, not a new YAML training loader.
Existing wrappers pass the returned ExperimentConfig into the upstream runner.

## Central options

| Option | Default | Configuration target |
| --- | --- | --- |
| collision_exclusions | true | robot.asset.collision_exclusions |
| motor_dr | true | existing actuator PD/armature randomization |
| delay_filter | true | action.terms.joint_control |
| extra_dr | group-dependent | existing four-item B bundle |
| motion_set / motion_split | group-dependent / train | checked motion paths and exact UMR split |
| track_phase | group-dependent | actor phase term |
| rewind_prob | 0.0 | motion_config.rewind_prob |
| ego_camera | false | simulator.config.ego_camera.enabled |
| future_object_poses | false | upstream obj_future_terms in actor and critic |
| object_bbox | false | upstream object_bbox_terms in actor and critic |
| object_persist_steps | 1 | bad_tracking.params.object_persist_steps |
| body_pos_z_only | false | bad_tracking.params.body_pos_z_only |

CLI uses hyphens, e.g. --rewind-prob 0.8, --ego-camera, --no-motor-dr.
Options with --no-* have symmetric enable/disable forms. Rewind uses upstream's
existing 25..125 step range. The persistence option changes only the persistence
field: it does not import an entire upstream preset with additional thresholds.

All composition updates individual fields or terms. It never replaces the
whole robot/observation/command using another experiment preset. Dataset
integrity and train/eval separation are checked before constructing a run.
The current asset factory is single-suitcase: many motions do not imply many
object assets. object_per_env and multi-object asset wiring remain in the
upstream multi-object preset; they are not falsely exposed as a working
single-suitcase toggle here.

## Compatibility and validation boundary

hdmi_goal043 remains a thin Python/CLI alias to this factory. Low-level profile
helpers remain implementation details; existing frozen deployment scripts are
not changed. Real-data acceptance scripts use the canonical entry.

CPU tests verify group conflicts, optional-feature composition, exports and
legacy API compatibility. Full A/B/C/D defaults must compare equal to the
pre-refactor 48a75cf configs. Optional camera, bbox, future observations,
rewind and changed termination settings still need an isolated Isaac smoke
before a run; no new GPU validation or training is implied by config export.
