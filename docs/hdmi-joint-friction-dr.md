# HDMI joint-friction DR

New experiment: `exp:g1-29dof-wbt-hybrid-object-hdmi-friction-dr`.
It extends `hdmi-push-dr`, retaining groups 1, 2, 3A and the four self-collision exclusions. Existing presets remain unchanged.

Each episode reset independently samples an absolute **0.05–0.25 N·m** uniform dry-friction effort for each of the 29 named robot joints. Values overwrite the previous sample; they do not accumulate. Only the selected environments reset. The floating base and object are not randomized by this term.

## Backend semantics

- **MJLab:** write native `dof_frictionloss` at the robot's velocity-DOF addresses. This is the dry-friction force/torque bound used by the constraint solver, not a coefficient multiplying contact load.
- **Isaac Sim 5.1:** the API retains the name `write_joint_friction_coefficient_to_sim`, but current native properties are static effort, dynamic effort, and viscous coefficient. Set static **and** dynamic effort to the same sample; preserve the existing viscous parameter. Setting only the static argument, as the legacy helper did, would leave the current zero dynamic friction unchanged.
- **Isaac <5:** rejected by this profile. Its legacy load-dependent friction coefficient is not the same physical quantity and must not receive the effort values blindly.

The numerical range comes from the saved HDMI593-associated MJLab training configuration. This is not a claim that historical PhysX coefficients, current PhysX joint friction, and MuJoCo dry-friction constraints are physically identical. No new viscous damping, geometry, contact-material randomization, observation, reward, or PPO changes are included.

Source references:
- Legacy configuration: `holosoma-sim2sim-video-v1/audit/hdmi593_training_config.yaml`, SHA256 `41fa24bfd7b4669b07a5043b1c8b49735ee5eddf5db700a32e075f8ef3b31ae3`.
- Preserved legacy randomizer: `holosoma-sim2sim-video-v1/audit/hdmi_randomization.py`, SHA256 `e0aecb364aa5f047fe04a41e42f75247234b86086bab3390f80f7c52a2b40a4a`.
- [PhysX articulation friction](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html).
- [MuJoCo friction loss](https://mujoco.readthedocs.io/en/stable/computation/#friction-loss).

## Validation scope

The separate audit checks the configuration delta, native parameter readback, subset/empty resets, unchanged other DOFs and viscous properties, and actual MuJoCo solver friction rows. Each backend receives a 64-environment, two-PPO-iteration smoke (3,072 transitions), with finite physics and saved-checkpoint checks. Audit data includes actual sampled and native values. Existing DR/collision tests are not repeated. These short checks validate integration, not learned robustness or sim-to-sim performance.

GPU runs and original files are preserved under `/data/pummpkin/holosoma-friction-dr-rp9-v1`; the desktop source remains in `/home/pummpkin/projects/holosoma-hdmi-actuator-dr` on `feat/hdmi-actuator-dr`. No formal training or commit is part of this change.
