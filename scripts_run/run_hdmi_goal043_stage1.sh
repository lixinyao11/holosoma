#!/usr/bin/env bash
set -euo pipefail
# Exact README Object Stage recipe except approved preset, 043 assets, and paths.
# Activate the isolated Holosoma environment before running. No smoke checkpoint reuse.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CUDA_VISIBLE_DEVICES="${GPU:?Set GPU after checking availability}" \
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-hybrid-object-hdmi \
  logger:wandb \
  --training.num-envs=8192 \
  --algo.config.num-learning-iterations=15000 \
  --simulator.config.scene.env-spacing=4.0 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=1.0 \
  --command.setup-terms.motion-command.params.motion-config.motion-file="${ASSET_ROOT:?Set suitcase asset directory}/motions/omomo__sub4_suitcase_043.npz" \
  --robot.object.object-urdf-path="${ASSET_ROOT}/suitcase.urdf" \
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.05 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False
