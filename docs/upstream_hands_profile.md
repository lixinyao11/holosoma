# Upstream joint reward and hand observations

This candidate replaces the earlier HDMI-specific joint tracking and wrist
observation with denso-hil 6a8e070's existing definitions. No upstream math is
copied or modified: configuration imports the actual upstream preset terms.

## Reward

Use motion_joint_pos_error_exp from g1_29dof_wbt_reward_hybrid_object_hands:
hybrid_motion_joint_pos_error_exp, sigma 0.25, weight 0.5. It uses all 29 joint
positions, including ankles. Remove joint_pos_tracking and joint_vel_tracking.
The upstream hybrid wrapper returns a constant 0.5 in generation environments,
before the reward weight and manager dt scaling. This is deliberately retained.
The separate joint-speed penalty, other rewards, PPO, DR and terminations are
unchanged. This change does not add upstream hand-position rewards.

## Observations

Replace object_to_hands_b (six torso-frame values) in actor and critic with
the two upstream hand_obs_terms, using the two wrist-yaw link landmarks:

- hand_obj_pos: six current wrist coordinates in the simulated object's frame.
- hybrid_hand_obj_pos_error: six simulated-minus-reference wrist coordinates,
  each measured in its own object's frame; zero in generation environments.

Thus tracking actors now receive reference information. This is neither a
contact measurement nor a grasp/release detector. Other history and observation
terms remain unchanged. Future object trajectories, bbox and rewind are not
enabled by this change.

Expected dimensions: A/B/C actor 571, critic 776; D actor 572, critic 776.
The actor 571 dimension is checked through the real CPU ObservationManager;
full native GPU actor/critic dimensions still require a fresh Isaac smoke.
Existing 565/566-input checkpoints are incompatible: do not pad/truncate or
resume them into this profile. Run names carry _upstream_hands to distinguish
new runs; D still adds only _track_phase and its one scalar.

## Experiment provenance

Previously collected A/B/C/D results describe the old profile. They do not
measure this adoption. Source at 07742a4 preserves the compatibility rebase;
published dev and the active B experiment remain frozen. No new training or
deployment is performed by changing these candidate configurations.

Validation covers direct upstream preset equality, all-joint reward including
ankles and generation masking, rotated object-frame hand features, reference
masking, translation invariance, actor dimensions and the existing M15/DR/
multi40/phase regressions. GPU smoke and learning benefit are not yet verified.
