#!/usr/bin/env bash

source scripts/source_isaacsim_setup.sh

# Copy or uncomment one command. eval_agent.py restores the experiment from
# the checkpoint, so do not add an exp:... alias. Every command explicitly
# keeps the default evaluation behavior: start at motion timestep 0 on every
# reset. Open http://localhost:8012.


# =============================================================================
# HYBRID CLIMB — Stage 1 checkpoint, track-only Viser test
# =============================================================================

# CUDA_VISIBLE_DEVICES=0 \
# python src/holosoma/holosoma/eval_agent.py \
#   --checkpoint=logs/WholeBodyTracking/20260726_073420-g1_29dof_wbt_hybrid_climb-locomotion/model.pt \
#   --viser.config.enabled \
#   --viser.config.port=8012 \
#   --training.num-envs=2 \
#   --training.headless=True \
#   --training.max-eval-steps=10000 \
#   --training.export-onnx=False \
#   --simulator.config.scene.env-spacing=5.0 \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=1.0 \
#   --command.setup-terms.motion-command.params.motion-config.motion-file='' \
#   --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/climbox1' \
#   --command.setup-terms.motion-command.params.motion-config.eval-sample-random-timestep=False \
#   --command.setup-terms.motion-command.params.motion-config.goal-orientation.robot-yaw-only=True \
#   --termination.terms.body-proximity.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
#   --termination.terms.body-proximity.is-timeout=True \
#   --termination.terms.bad-tracking.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
#   --termination.terms.bad-tracking.is-timeout=True \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False


# =============================================================================
# HYBRID CLIMB — Stage 2 checkpoint, gen-only Viser play
# =============================================================================

CUDA_VISIBLE_DEVICES=0 \
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=logs/WholeBodyTracking/20260726_231144-g1_29dof_wbt_hybrid_climb-locomotion/model.pt \
  --viser.config.enabled \
  --viser.config.port=8012 \
  --viser.config.no-show-reference \
  --training.num-envs=4 \
  --training.headless=True \
  --training.max-eval-steps=10000 \
  --training.export-onnx=False \
  --simulator.config.scene.env-spacing=0.0 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=0.0 \
  --command.setup-terms.motion-command.params.motion-config.motion-file='' \
  --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/climbox1' \
  --command.setup-terms.motion-command.params.motion-config.eval-sample-random-timestep=False \
  --command.setup-terms.motion-command.params.motion-config.goal-orientation.robot-yaw-only=True \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.pos-std-xyz='[0.4, 0.4, 0.01]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.pos-clip-xyz='[1.0, 1.0, 0.02]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.rpy-std='[0.05, 0.05, 0.3]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.rpy-clip='[0.15, 0.15, 1.0]' \
  --termination.terms.body-proximity.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
  --termination.terms.body-proximity.is-timeout=True \
  --termination.terms.bad-tracking.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
  --termination.terms.bad-tracking.is-timeout=True \
  --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False


# =============================================================================
# HYBRID OBJECT — Stage 1 checkpoint, track-only Viser test
# =============================================================================

# CUDA_VISIBLE_DEVICES=0 \
# python src/holosoma/holosoma/eval_agent.py \
#   --checkpoint=logs/WholeBodyTracking/20260724_050832-g1_29dof_wbt_hybrid_object-locomotion/model.pt \
#   --viser.config.enabled \
#   --viser.config.port=8012 \
#   --training.num-envs=2 \
#   --training.headless=True \
#   --training.max-eval-steps=10000 \
#   --training.export-onnx=False \
#   --simulator.config.scene.env-spacing=0.0 \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=1.0 \
#   --command.setup-terms.motion-command.params.motion-config.motion-file='' \
#   --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/movebox' \
#   --command.setup-terms.motion-command.params.motion-config.eval-sample-random-timestep=False \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False \
#   --termination.terms.bad-tracking.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
#   --termination.terms.bad-tracking.is-timeout=True \
#   --termination.terms.body-proximity.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
#   --termination.terms.body-proximity.is-timeout=True


# =============================================================================
# HYBRID OBJECT — Stage 2 checkpoint, gen-only Viser play
# =============================================================================

# CUDA_VISIBLE_DEVICES=0 \
# python src/holosoma/holosoma/eval_agent.py \
#   --checkpoint=logs/WholeBodyTracking/20260725_184354-g1_29dof_wbt_hybrid_object-locomotion/model.pt \
#   --viser.config.enabled \
#   --viser.config.port=8012 \
#   --viser.config.no-show-reference \
#   --training.num-envs=4 \
#   --training.headless=True \
#   --training.max-eval-steps=10000 \
#   --training.export-onnx=False \
#   --simulator.config.scene.env-spacing=0.0 \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=0.0 \
#   --command.setup-terms.motion-command.params.motion-config.motion-file='' \
#   --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/movebox' \
#   --command.setup-terms.motion-command.params.motion-config.eval-sample-random-timestep=False \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.pos-std-xyz='[0.5, 0.5, 0.001]' \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.pos-clip-xyz='[1.0, 1.0, 0.001]' \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.rpy-std='[0.001, 0.001, 0.4]' \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.rpy-clip='[0.001, 0.001, 0.8]' \
#   --curriculum.setup-terms.hybrid-gen-noise-curriculum.params.initial-scale=1.0 \
#   --termination.terms.body-proximity.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
#   --termination.terms.body-proximity.is-timeout=True \
#   --termination.terms.bad-tracking.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
#   --termination.terms.bad-tracking.is-timeout=True \
#   --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False
