# Experimental MJLab backend for Holosoma 043

Branch: `feat/hdmi-goal-mjlab`. This worktree preserves the inherited HDMI goal-RL changes. The original worktree and the running Isaac training are unchanged.

## Scope

`simulator:mjlab` connects Holosoma's existing observation, action/PD, reward, termination, motion and PPO code to MJLab's Scene/Simulation/Entity APIs. It follows the desktop active-adaptation MJLab runtime and uses MJLab 1.6.0, MuJoCo/MuJoCo-Warp 3.11.0 and Warp 1.14.0. Tested with Python 3.11 and the existing Torch 2.7.0+cu128; the RP dependencies are installed into an isolated target directory, not the production environment.

Implemented interfaces: robot/object root states, joint/body ordering, link poses with COM velocities, indexed state writes/reset, world-frame net body contact forces and history, direct torque application, and joint state refresh after every physics substep for the unchanged PD controller. Supports the current plane plus one suitcase URDF. Rendering, cameras, arbitrary scenes and general object variants are not implemented by this adapter.

The original robot/object URDFs and referenced mesh bytes are reused. The robot's 21 collision cylinders are converted to capsules with the radius and height verified against the frozen Isaac USD. The native solvers, contact generation and materials are not physically equivalent.

## Explicit material approximation

Enable `HOLOSOMA_MJLAB_APPROXIMATE_MATERIALS=1` only for the authorized backend comparison. Sliding friction uses the original sampled dynamic friction. Restitution is approximately mapped to contact damping ratio. Static friction is sampled and saved but is not separately applied. Other supported mass, COM and inertia randomization retains the original configured ranges. Actual samples are saved in `randomization_actual.pt`.

Required runtime settings:

- `simulator:mjlab`
- `--logger.video.enabled=False --logger.headless-recording=False`
- `HOLOSOMA_MJLAB_AUDIT_DIR=<independent output directory>`
- `HOLOSOMA_MJLAB_APPROXIMATE_MATERIALS=1`
- `HOLOSOMA_MJLAB_STATE_GATE=1` for the short interface smoke only; omit during timing.

Use the existing task and asset arguments. Do not substitute generic G1 assets or change policy/reward/PPO configuration to accommodate the backend.

## Acceptance and timing evidence

RP evidence root: `/data/pummpkin/holosoma-mjlab-goal043-v1/records` on rp-4090-6.

- `smoke07`: final 64-environment, 2-update smoke, after matching physics-rate PD state refresh. State write/read error <= 1e-6, joint ordering exact, untouched environments unchanged, checkpoint finite; observation/reward/PPO and asset SHA checks preserved.
- `benchmark512_v2_mjlab` and `benchmark512_isaacsim`: same physical GPU7, 512 environments, 24 steps/update, 40 updates from the same frozen iteration3500 checkpoint. Exact restore checks before optimization. Discard the first10 updates and compare3510..3539. Neither records video or consumes images.
- `comparison.json`: timings and episode statistics, including limitations. Collection time includes simulation, policy inference, observations, rewards and resets, rather than pure physics alone.
- `benchmark_mjlab`: preserved 2048-environment OOM before PPO; the present buffers do not fit that scale on a 24GiB4090.
- `benchmark512_mjlab`: preserved initial timing, excluded from the final comparison because it lacked physics-rate joint-state refresh. The original adapter is retained as `backend_before_pd_refresh.py`.

These checks establish interface functionality and a short timing comparison. They do not establish matching PhysX contact behavior, successful policy transfer, equivalent task success, visual throughput or convergence. No formal MJLab training or Stage2 run is launched.
