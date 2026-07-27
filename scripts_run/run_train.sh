#!/usr/bin/env bash

source scripts/source_isaacsim_setup.sh

# Copy or uncomment one command at a time.
# Stage 2 checkpoints must use the same actor and critic observation layout.


# =============================================================================
# HYBRID CLIMB — STAGE 1: track only
# =============================================================================

# CUDA_VISIBLE_DEVICES=0 \
# python src/holosoma/holosoma/train_agent.py \
#   exp:g1-29dof-wbt-hybrid-climb \
#   logger:wandb \
#   --training.num-envs=8192 \
#   --algo.config.num-learning-iterations=20000 \
#   --simulator.config.scene.env-spacing=5.0 \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=1.0 \
#   --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/climbox1' \
#   --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.05 \
#   --command.setup-terms.motion-command.params.motion-config.goal-orientation.robot-yaw-only=True \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False


# =============================================================================
# HYBRID CLIMB — STAGE 2: resume Stage 1, 50% track / 50% gen
# =============================================================================

CUDA_VISIBLE_DEVICES=0 \
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-hybrid-climb \
  logger:wandb \
  --training.checkpoint=logs/WholeBodyTracking/20260726_073420-g1_29dof_wbt_hybrid_climb-locomotion/model.pt \
  --training.num-envs=8192 \
  --algo.config.num-learning-iterations=20000 \
  --algo.config.reset-policy-std-on-resume=True \
  --algo.config.init-noise-std=0.45 \
  --simulator.config.scene.env-spacing=5.0 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=0.5 \
  --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/climbox1' \
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.05 \
  --command.setup-terms.motion-command.params.motion-config.goal-orientation.robot-yaw-only=True \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-start-at-timestep-zero-prob=0.2 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.pos-std-xyz='[0.3, 0.3, 0.01]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.pos-clip-xyz='[1.0, 1.0, 0.02]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.rpy-std='[0.05, 0.05, 0.3]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.rpy-clip='[0.15, 0.15, 1.0]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False


# =============================================================================
# HYBRID OBJECT — STAGE 1: track only
# =============================================================================

# CUDA_VISIBLE_DEVICES=0 \
# python src/holosoma/holosoma/train_agent.py \
#   exp:g1-29dof-wbt-hybrid-object \
#   logger:wandb \
#   --training.num-envs=8192 \
#   --algo.config.num-learning-iterations=15000 \
#   --simulator.config.scene.env-spacing=4.0 \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=1.0 \
#   --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/movebox' \
#   --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.05 \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False


# =============================================================================
# HYBRID OBJECT — STAGE 2: resume Stage 1, 50% track / 50% gen
# =============================================================================

# CUDA_VISIBLE_DEVICES=0 \
# python src/holosoma/holosoma/train_agent.py \
#   exp:g1-29dof-wbt-hybrid-object \
#   logger:wandb \
#   --training.checkpoint=logs/WholeBodyTracking/20260724_050832-g1_29dof_wbt_hybrid_object-locomotion/model.pt \
#   --training.num-envs=8192 \
#   --algo.config.num-learning-iterations=20000 \
#   --algo.config.reset-policy-std-on-resume=True \
#   --algo.config.init-noise-std=0.45 \
#   --simulator.config.scene.env-spacing=4.0 \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=0.5 \
#   --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/movebox' \
#   --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.05 \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.gen-start-at-timestep-zero-prob=0.2 \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.pos-std-xyz='[0.5, 0.5, 0.001]' \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.pos-clip-xyz='[1.0, 1.0, 0.001]' \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.rpy-std='[0.001, 0.001, 0.4]' \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.rpy-clip='[0.001, 0.001, 0.8]' \
#   --curriculum.setup-terms.hybrid-gen-noise-curriculum.params.enabled=True \
#   --curriculum.setup-terms.hybrid-gen-noise-curriculum.params.initial-scale=0.3 \
#   --termination.terms.bad-tracking.params.gen-bad-object-pos-z-threshold=0.6 \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False
