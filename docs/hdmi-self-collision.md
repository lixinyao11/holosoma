# HDMI G1 self-collision exclusions

The HDMI-informed G1 presets, including the three DR variants, share the original
HDMI four body-pair exclusions:

- `left_elbow_link` / `left_wrist_pitch_link`
- `right_elbow_link` / `right_wrist_pitch_link`
- `pelvis` / `right_hip_roll_link`
- `pelvis` / `left_hip_roll_link`

`RobotAssetConfig.collision_exclusions` defaults to an empty tuple. Unrelated
Holosoma presets retain their previous behavior. The HDMI preset sets the four
pairs; both backends consume the same configuration.

MJLab adds missing native MJCF exclusions before compilation. Isaac authors USD
`FilteredPairs` on the source articulation before cloning and resolves the pairs
on every environment. Existing filters are retained. Missing or ambiguous body
names are errors rather than silent omissions.

This change does not disable pelvis or torso collision geometry, turn off global
self-collision, modify joint friction, or change collision shapes, material DR,
control, rewards, observations, or PPO settings. Historical frozen source copies
and results remain unchanged.

The opt-in audit compares the loaded MJLab exclusion signatures and the loaded
Isaac USD relationships before and after the change. It checks geometry/enable
flags, existing filters, per-environment targets and unchanged joint friction.
It is followed by a 64-environment, two-iteration smoke in each backend. These
checks are outside normal training and do not measure improved task performance.
