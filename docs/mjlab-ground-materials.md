# Ground material combination repair

Development branch `pummpkin/mjlab`, based on desktop commit 34e4f9c. No running experiment was modified.

The legacy Isaac plane uses multiply friction and restitution, with ground restitution zero. Previously the MJLab adapter retained native max-friction and mixed-solref rules: a sampled dynamic friction 0.2 against ground1 became effective contact friction1.0. This was an explicitly accepted approximation for initial backend timing, but creates a training-domain discrepancy.

This change adds explicit pairs only between the plane and already collision-enabled geoms. Pair sliding friction is the product of the existing sampled dynamic friction and ground friction. The existing restitution-to-solref approximation maps the zero ground restitution to damping ratio1. Time constant, solimp, collision dimension, and native solver remain unchanged for these equal-priority assets. Non-ground pairs and all DR sampling ranges are unchanged. Static and dynamic friction are still not separately modeled; zero restitution through solref is approximate, not exact PhysX dynamics.

Per-world pair fields are expanded before graph capture and updated after material sampling, including subset-world updates. No new random draws are added. Existing collision exclusions and visual-only geoms remain untouched.

Validation: native CPU MuJoCo contact readback 1.0 ->0.2; sampled0.1/0.3/0.6 values and untouched-world isolation; actual MuJoCo-Warp per-world field shapes; legacy robot/suitcase compile with69 ground pairs and unchanged body/joint/filter fields. No GPU rollout or learning-performance test has been performed. Do not claim the episode-length gap is resolved.

Run the CPU check using an existing environment with torch, mujoco3.11, mujoco-warp3.11 and mjlab1.6:
`CUDA_VISIBLE_DEVICES="" python tests/unit/mjlab_ground_materials/check_cpu.py /path/to/new-audit-output`

Existing training snapshots remain frozen. Further validation should use an isolated short paired rollout before any new training.
