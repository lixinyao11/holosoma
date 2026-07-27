# Hybrid Whole-Body Tracking

This document describes the hybrid learning setup implemented for the two
29-DoF Unitree G1 tasks in this release:

```text
exp:g1-29dof-wbt-hybrid-climb
exp:g1-29dof-wbt-hybrid-object
```

The implementation is built on
[Holosoma](https://github.com/jiashunwang/holosama) and runs with Isaac Sim.
It is an unofficial research implementation and extension of
[Generalizing from References (GfR)](https://jiashunwang.github.io/GfR/) and
[Hybrid Imitation Learning (HIL)](https://jiashunwang.github.io/HIL/), not the
official code release for either work.

The exact release training and evaluation commands are in the repository
[README](../README.md) and in `scripts_run/run_train.sh` and
`scripts_run/run_play.sh`. This document focuses on how the hybrid tasks work.

## Core idea

A policy can learn high-quality motion from a reference without receiving that
reference as actor input. The actor sees the current state and a sparse task
goal. The reference is used during training to define goals, rewards, reset
states, and privileged critic information.

Each simulator contains two fixed groups of environments:

| | Track environments | General RL environments |
|---|---|---|
| Purpose | learn the behavior in the reference motion | transfer that behavior to varied goals and conditions |
| Goal | exact terminal goal from the reference | terminal goal perturbed by task-specific noise |
| Reset-frame sampling | adaptive | uniform |
| Reference tracking rewards | active | masked |
| Task success reward | tracking rewards | one sparse goal-reaching bonus |
| Tracking failure termination | active | relaxed gen-specific thresholds |
| Actor input | state and task goal | the same state and task-goal layout |
| Critic input | task-specific privileged signals | the same layout with hybrid masking |

The policy and value function are shared. Within each task, the actor
observation shape does not change between track and gen environments. The
actor never receives the per-step target reference.

Environment membership is sampled once with `env_partition_seed=0` and remains
fixed for the simulator lifetime. With the default `track_ratio=0.5`, half of
the environments are track and half are gen.

## Tasks and observations

The climb task demonstrates goal-conditioned locomotion. The actor receives
robot state and the robot goal.

The object task demonstrates goal-conditioned loco-manipulation. The actor
receives robot state, current object state, and the object goal. It does not
receive the per-step robot or object reference.

Both critics use privileged tracking information during training. Assistive
wrench state is not part of either actor or critic observation in this
release.

## Two-stage training

The release recipes use the same schedule for both tasks:

1. Stage 1 uses only track environments with `track_ratio=1.0`.
2. Stage 2 resumes the Stage 1 checkpoint and uses a 50/50 track/gen mix.

Stage 2 resets the learned policy action standard deviation to `0.45`. On
resume, `num-learning-iterations` is the number of additional iterations, not
the final cumulative iteration number.

The supplied recipes use:

| Task | Stage 1 | Stage 2 |
|---|---:|---:|
| climb | 20,000 track-only iterations | 20,000 additional hybrid iterations |
| object | 15,000 track-only iterations | 20,000 additional hybrid iterations |

## Episode and goal lifecycle

At reset, the command term performs these operations:

1. Select a motion and reset frame. Track uses adaptive sampling; gen uses
   uniform sampling.
2. Replace the selected frame with frame zero with probability `0.05` for
   track or `0.2` for gen.
3. Select the terminal reference frame and construct exact robot and object
   goals.
4. Copy the exact goal into the active goal buffer and add clipped Gaussian
   goal noise for gen environments only.
5. Load robot and object state from the selected reset frame.
6. Apply the common uniform initialization noise.
7. At frame zero, replace the robot root position and orientation noise with
   the applicable clipped-Gaussian step-zero configuration.
8. Write the initialized state to the simulator.

Both tasks select the terminal reference pose as the goal and hold that pose
for one second at the end of the motion.

`eval_sample_random_timestep=False` takes precedence during evaluation and
starts every reset at frame zero. It does not disable initialization,
step-zero, goal, observation, or physics noise.

## Initialization noise

The common initialization noise is sampled uniformly. Values below are
half-ranges: a value `a` means `U(-a, a)`.

| Quantity | Half-range |
|---|---:|
| joint position | `0.1 rad` |
| joint velocity | `0` |
| root position XYZ | `[0.025, 0.025, 0.01] m` |
| root rotation RPY | `[0.05, 0.05, 0.1] rad` |
| root linear velocity XYZ | `[0.25, 0.25, 0.1] m/s` |
| root angular velocity XYZ | `[0.26, 0.26, 0.39] rad/s` |
| object position XYZ | `[0.025, 0.025, 0.0] m` |
| object orientation and velocity | exact reference |

Frame-zero root noise is a clipped-Gaussian replacement for the common root
position and orientation noise. It is not an additional offset:

| Task/mode | Position std | Position clip | RPY std | RPY clip |
|---|---:|---:|---:|---:|
| climb track | `[0.05, 0.05, 0.01]` | `[0.2, 0.2, 0.02]` | `[0.05, 0.05, 0.1]` | `[0.1, 0.1, 0.3]` |
| climb gen preset | `[0.2, 0.2, 0.01]` | `[1.0, 1.0, 0.02]` | `[0.05, 0.05, 0.2]` | `[0.15, 0.15, 0.6]` |
| climb gen Stage 2 recipe | `[0.3, 0.3, 0.01]` | `[1.0, 1.0, 0.02]` | `[0.05, 0.05, 0.3]` | `[0.15, 0.15, 1.0]` |
| object track | no separate override; common uniform noise remains | — | — | — |
| object gen | `[0.01, 0.01, 0.003]` | `[0.05, 0.05, 0.01]` | `[0.01, 0.01, 0.03]` | `[0.05, 0.05, 0.1]` |

The object gen values intentionally keep the frame-zero initialization close
to the reference, similar to the earlier move-box behavior that used only the
common `±0.025 m` robot-root position reset range.

Actor observation noise is additive uniform noise with magnitudes `0.10` for
base angular velocity, `0.005` for joint position, and `0.25` for joint
velocity. Goal and object observation terms have no separate observation
noise.

## Gen goal noise

Track environments always use the exact motion-derived goal. Gen environments
apply independent clipped Gaussian position and RPY offsets:

| Goal | Position std | Position clip | RPY std | RPY clip |
|---|---:|---:|---:|---:|
| climb robot | `[0.05, 0.05, 0.001]` | `[0.10, 0.10, 0.001]` | `[0.001, 0.001, 0.03]` | `[0.001, 0.001, 0.10]` |
| object robot | `[0.05, 0.05, 0.001]` | `[0.10, 0.10, 0.001]` | `[0.001, 0.001, 0.001]` | `[0.001, 0.001, 0.001]` |
| object goal preset | `[0.2, 0.2, 0.001]` | `[0.3, 0.3, 0.001]` | `[0.001, 0.001, 0.2]` | `[0.001, 0.001, 0.4]` |
| object goal Stage 2 recipe | `[0.5, 0.5, 0.001]` | `[1.0, 1.0, 0.001]` | `[0.001, 0.001, 0.4]` | `[0.001, 0.001, 0.8]` |

Climb robot-goal noise has no curriculum and is always applied at its
configured scale. The small nonzero yaw noise lets gen environments vary
heading while keeping the goal close to the reference.

The object task applies its curriculum scale only to object-goal X/Y and yaw
noise. Robot-goal noise, object Z, roll, and pitch noise are not scaled.

## Object goal-noise curriculum

The object task adjusts goal difficulty from completed gen episodes:

| Parameter | Value |
|---|---:|
| initial and minimum scale | `0.3` |
| maximum scale | `1.0` |
| update step | `0.01` |
| increase condition | gen success EMA `> 0.90` |
| decrease condition | gen success EMA `< 0.80` |
| update interval | every `20` training steps |
| EMA coefficient | `0.5` |

When the scale changes, active gen object goals are regenerated immediately.
The curriculum state is not restored from a checkpoint for evaluation.
`run_play.sh` therefore sets `initial_scale=1.0` for the Stage 2 object
evaluation.

## Rewards

Track environments use the existing robot motion-tracking rewards. Object
track environments also use object-tracking rewards.

Gen environments mask per-step reference-tracking rewards and receive a
one-time sparse bonus for each goal:

| Task | Bonus | Success condition |
|---|---:|---|
| climb | `2.0` | robot goal position error `≤ 0.2 m` and orientation error `≤ π/6` |
| object | `3.0` | object goal position error `≤ 0.2 m` and orientation error `≤ π/6` |

Shared policy regularization rewards remain active in both modes.

## Termination

Training episodes time out after 10 seconds. Track environments also use the
ordinary motion-tracking failure checks:

| Check | Threshold |
|---|---:|
| robot root position error | full XYZ distance `> 0.4 m` |
| selected ankle or wrist position error | full XYZ distance `> 0.25 m` |
| root orientation projected-gravity Z-component difference | `> 0.4` |
| object position error | full XYZ distance `> 0.3 m` |
| object orientation error | `> 0.8 rad` |
| foot-to-foot proximity | distance `< 0.05 m` |

The body-position checks use full XYZ distance, not Z-only distance.
Object-specific checks apply only to the object task.

Gen environments use relaxed, safety-oriented tracking limits:

| Check | Threshold |
|---|---:|
| minimum robot root height | `< 0.45 m` |
| robot root reference Z error | `> 0.4 m` |
| robot root reference XYZ error | `> 100.0 m` |
| object reference Z error | `> 0.6 m` |
| object orientation error | `> 1.0 rad` |

The evaluation commands in `scripts_run/run_play.sh` replace
`body-proximity` and `bad-tracking` with timeout-only terms. This disables
early failure termination for play without changing the training
configuration saved in the checkpoint.

## Physics and optional size randomization

The robot training randomization includes material properties, base center of
mass, link mass, joint-position bias, and periodic planar pushes. The object
task additionally randomizes object material, mass, and inertia. Evaluation
suppresses training pushes after entering evaluation mode but otherwise
inherits the checkpoint's randomization unless explicitly overridden.

Geometry size randomization is disabled by default. When enabled, each
environment receives one fixed scale for the simulator lifetime:

| Mode | Scale |
|---|---:|
| track | `U(0.95, 1.05)` |
| gen | `U(0.85, 1.15)` |

The object is scaled uniformly in XYZ. The climb box is scaled only in height.
Ground-clearance corrections keep scaled geometry above the ground plane.

The two tasks intentionally load their boxes through different paths. Climb
injects a fixed kinematic `climb_box` from motion metadata. Move-box keeps a
dynamic object loaded from `robot.object.object_urdf_path`; an empty
`scene.rigid_objects` list before simulator creation is therefore expected for
move-box when size randomization is disabled.

## Evaluation behavior

`eval_agent.py` reconstructs the experiment configuration saved with the
checkpoint, so evaluation commands must not include an `exp:...` alias.

`training.num-envs` controls the number of simulated environments. Viser only
limits how many are displayed and defaults to at most 16, so the release play
commands do not set `viser.config.num-envs`.

In gen mode, `viser.config.no-show-reference` hides the displaced blue
reference. It does not change policy inputs, rewards, goals, or simulation.

## Checkpoint compatibility

Stage 2 requires a Stage 1 checkpoint with the same actor and critic
observation layouts. The Stage 1 checkpoints used by the release recipes were
verified against the current layouts. Some earlier experimental checkpoints
contained seven assistive-wrench critic observations; those checkpoints have
a different critic shape and cannot be resumed as a whole checkpoint.

## Implementation map

- `src/holosoma/holosoma/config_types/command.py`: hybrid and noise schemas.
- `src/holosoma/holosoma/config_values/wbt/g1/command.py`: task defaults.
- `src/holosoma/holosoma/config_values/wbt/g1/curriculum.py`: object
  goal-noise curriculum.
- `src/holosoma/holosoma/managers/command/terms/wbt.py`: partitioning,
  reset sampling, initialization, and goal construction.
- `src/holosoma/holosoma/managers/curriculum/terms/wbt.py`: curriculum update.
- `src/holosoma/holosoma/envs/wbt/hybrid_size_randomization.py`: fixed
  geometry scaling and ground correction.
- `src/holosoma/holosoma/envs/wbt/wbt_manager.py`: scene preparation and
  hybrid logging.

## References

- Jiashun Wang, M. Eva Mungai, He Li, Jean Pierre Sleiman, Jessica Hodgins,
  and Farbod Farshidian. “Generalizing from References using a Multi-Task
  Reference and Goal-Driven RL Framework.” RSS, 2026.
  [Project page](https://jiashunwang.github.io/GfR/) ·
  [Paper](https://arxiv.org/abs/2602.20375)
- Jiashun Wang, Yifeng Jiang, Haotian Zhang, Chen Tessler, Davis Rempe,
  Jessica Hodgins, and Xue Bin Peng. “HIL: Hybrid Imitation Learning for
  Dynamic Athletic Control.” ACM Transactions on Graphics, 2026.
  [Project page](https://jiashunwang.github.io/HIL/) ·
  [Paper](https://jiashunwang.github.io/HIL/static/mat/Hybrid_Imitation_Learning_TOG.pdf)
