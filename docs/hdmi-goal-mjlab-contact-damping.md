# MJLab normal-contact and object-damping correction

Independent branch: `feat/hdmi-goal-mjlab`. No commit. Original C32 and Isaac training worktrees are unchanged.

## Runtime changes

- Return net normal contact force **on** each robot body, in world coordinates, including the physical-step history. Exclude tangential friction. Decode the native solver's actual contact force; do not alter collision rules or friction coefficients.
- Apply object linear/angular damping rates of 0.01, using the actual randomized mass and rotated inertia about the COM. Implement as force/torque, with the gravity-order correction for the linear decay. These are velocity-decay rates, not raw joint damping coefficients.
- Keep the native solver, approved material approximation, reset configuration, observations, rewards, policy, PPO and assets unchanged. This is not a claim of exact PhysX dynamics.

Rate reference: https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/lowleveldynamics/src/DyBodyCoreIntegrator.h

## Acceptance and frozen comparison

Evidence on RP6: `/data/pummpkin/holosoma-mjlab-goal043-v1/contact-damping-v2`.

- `smoke_v3`: GPU force-projection checks for elliptic/pyramidal cones; damping-rate check; 64 environments, two actual control steps, checkpoint3500 frozen, no optimizer updates. All passed.
- `pair_v1/legacy`: 512 initial states, first episode only, at most1000 control steps (20 seconds), stochastic action policy with a separate reproducible Gaussian stream.
- `pair_v1/fixed`: preserved initial pairing failure. Actual qpos/qvel/materials and actor observation matched, but warm-up qacc, critic observation and transient buffers were not all bitwise equal. No comparison rollout was accepted from this attempt.
- `pair_v2/fixed`: explicitly restore the legacy snapshot's recorded tensor fields, then require exact equality across442 fields. First actor observation, action noise and action match exactly. Policy and normalizer digests are unchanged before/after; optimizer updates remain zero.
- Shared low-load GPU4 was explicitly authorized. Existing processes were preserved and preflight recorded. This run is not a throughput benchmark.

Results: mean first-episode survival172.61 ->166.46 steps; median101 ->99; orientation termination377/512 ->379/512. Common-alive orientation error0.15449 ->0.15501rad. Paired mean difference-6.15 steps, bootstrap95% interval[-22.68,10.49]. Five versus seven episodes are right-censored at1000 steps. No clear transfer improvement was demonstrated.

These compare pre-fix versus fixed MJLab with a frozen policy. They are not a paired Isaac/MJLab test and cannot be directly compared with prior40-update PPO logger windows. The force-interface fix affects reward meaning; its learning effect has not been tested by this frozen-policy probe.

Previous backend source: `/home/pummpkin/projects/holosoma-mjlab-audits/contact-damping-v2/backend_before.py`.
