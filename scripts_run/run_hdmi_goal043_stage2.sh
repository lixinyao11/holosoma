#!/usr/bin/env bash
set -euo pipefail
# Exact README Object Stage recipe except approved preset, 043 assets, and paths.
# Activate the isolated Holosoma environment before running. No smoke checkpoint reuse.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CUDA_VISIBLE_DEVICES="${GPU:?Set GPU after checking availability}" \
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-hybrid-object-hdmi \
  logger:wandb \
  --training.checkpoint="${STAGE1_CHECKPOINT:?Set Stage 1 checkpoint}" \
  --training.num-envs=8192 \
  --algo.config.num-learning-iterations=15000 \
  --algo.config.reset-policy-std-on-resume=True \
  --algo.config.init-noise-std=0.45 \
  --simulator.config.scene.env-spacing=4.0 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=0.5 \
  --command.setup-terms.motion-command.params.motion-config.motion-file="${ASSET_ROOT:?Set suitcase asset directory}/motions/omomo__sub4_suitcase_043.npz" \
  --robot.object.object-urdf-path="${ASSET_ROOT}/suitcase.urdf" \
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.05 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-start-at-timestep-zero-prob=0.2 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.pos-std-xyz='[0.5, 0.5, 0.001]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.pos-clip-xyz='[1.0, 1.0, 0.001]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.rpy-std='[0.001, 0.001, 0.4]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.rpy-clip='[0.001, 0.001, 0.8]' \
  --curriculum.setup-terms.hybrid-gen-noise-curriculum.params.enabled=True \
  --curriculum.setup-terms.hybrid-gen-noise-curriculum.params.initial-scale=0.3 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False
