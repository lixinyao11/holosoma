# Validated Isaac legacy / mode15 asset comparison

This records the asset-only switch used by `holosoma-isaac-asset-ab2000-v1`.
Both conditions completed fresh smoke32 and fresh2000, with strict configuration
and four-rank actual-physics acceptance. It does not import the current branch's
new DR, phase observation, collision exclusions, or action processing.

The 41 regular overlay files and one relative symlink are byte-identical to the
validated package from dev commit `084fe0254245d218e5880fdadad0199c70a39e49`.
The original 434-file experiment source and existing default presets are unchanged.
The original baseline remains available at `c1f6176eb3e8dd7b58ffa8b63da7b4dad43b8ef8`.

Use the same explicit asset switch as the experiment, before simulator creation:

```python
from holosoma.config_values.wbt.g1.hdmi_goal043 import make_config
from reproduction.isaac_asset_ab2000.config import configure

cfg = configure(make_config(asset_root), "mode15")  # B: M15 asset + motor/PD
# cfg = configure(make_config(asset_root), "legacy")  # A: original robot
```

`asset_root` contains `suitcase.urdf` and `motions/omomo__sub4_suitcase_043.npz`.
The portable factory only sets data paths and the run name. Apply the original
experiment's common runtime settings separately: 4 ranks, 2048 environments/rank,
seed 42+rank, 24 rollout steps, 50 Hz control / 200 Hz physics, 10-second episodes,
track ratio 1, and a fresh checkpoint. The source experiment's launch manifests
and resolved configurations retain the full runtime settings and acceptance evidence.
This module is a configuration switch, not a training launcher.

Mode15 changes the robot assets (including hand collision geometry) and matched
motor limits, armature and PD gains as a whole; it is not a geometry-only study.
The original restitution and startup bias remain enabled. Episode duration can
span motion resets and timeout is not success rate. Replay videos use recorded
training states, not paired evaluations with identical initial states.

See `provenance.json` for the fixed versions, acceptance scope and evidence path.
Training logs, checkpoints, videos, generated USD files and environment packages
are retained outside Git.
