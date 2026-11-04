# denso-hil integration

Integration branch: pummpkin/dev-denso-rebase.
Upstream: 6a8e070c7022fcffe84816ac45dea9498f6c0dc7 (2026-11-02).
Original published dev: 13c136868b5320433c5bc9421312dc29f3b576e5.
The six dev commits were rebased from their shared ancestor 9a0f685.
This is a denso-hil rebase, not a main rebase. The published dev branch and
the running B experiment's frozen source remain unchanged.

## Upstream ownership

Keep upstream multi-object scene construction, object group indexing, bbox and
future-reference observations, rewind, persistent termination, height-only
body checks, optional ego camera and launcher behavior. These capabilities
remain opt-in for the existing A/B/C/D factory.

Local video filenames now use the upstream UUID implementation. Keep our
global-rank-zero recording guard and FFmpeg error evidence around that code.

Keep the upstream contact patch buffer increase (327680 to 1048576), which is
a simulator resource limit in sim_utils, outside the serialized experiment
config. Do not describe the entire runtime as byte-identical to the frozen
experiments. Upstream launch-script NCCL settings do not replace the running
AWS experiment's validated NCCL_PROTO=LL environment.

## Extensions that are not equivalent duplicates

- Our joint reward selects a subset of joints, includes velocity tracking and
  removes the generation group's baseline. Upstream position tracking uses all
  joints with different group masking. Keep the existing experiment reward.
- Our wrist displacement observation uses the current torso conditioning frame;
  upstream hand/object features use a different frame/reference contract.
- Our validated M15 URDF, actuator bounds, four collision exclusions, motor DR,
  delay/filter, extra DR and 40/2 UMR split remain intact. Identical mesh bytes
  do not establish equivalence of an entire robot configuration.
- Track phase remains an optional actor scalar. It is not future-reference
  observations or semantic stage input.

## Collision integration

Preserve upstream ordinary and per-object-group clone paths. Author source
robot filters before cloning. Resolve each cloned robot's own filter targets
after its prim exists; for grouped objects this happens inside group cloning,
before registering PhysX replication. No filters are applied to uncreated
environment prims.

## Validation and limits

90 CPU regression tests passed. Real UMR 40-train/2-held-out loading/sampling and phase
checks are recorded in the owner's untracked
operations/rp-workflow/denso_rebase_20261104 directory.
Full A/B/C/D serialized configs were compared with the original dev: the only
new fields are disabled rewind, disabled object-per-env, unset object weights
and a disabled camera configuration.

New CPU tests exercise opt-in defaults, upstream preset registration, actual
video encode/decode uniqueness, rank-zero recording and failure diagnostics.
The existing desktop CPU environment has no pxr/Isaac runtime. New scene
cloning/collision integration and enabled camera/multi-object modes still need
an isolated Isaac smoke before deployment. No GPU experiment was launched or
changed as part of this integration.
