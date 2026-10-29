#!/usr/bin/env bash
# HIL training (hybrid tracking + goal envs) on several Denso holosoma datasets at once
# (exp:g1-29dof-wbt-hybrid-multi-object): environment e holds object e % N and samples only that object's clips.
# Stage arguments as in the README (track ratio, goal noise, curriculum, checkpoint) go in "$@":
#
#   bash scripts/denso_hil_multi_object.sh logger:wandb --training.num-envs=8192 \
#       --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=1.0 ...
#
# DENSO_HOLOSOMA_DATA: dataset root (default ~/Code/Denso/data/holosoma/omomo/umr, from Denso's
# `scripts/export.py --format holosoma`); OBJECTS: object datasets to use, in order.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DATA=${DENSO_HOLOSOMA_DATA:-$HOME/Code/Denso/data/holosoma/omomo/umr}
OBJECTS=${OBJECTS:-"largebox plasticbox smallbox suitcase"}
dirs=()
urdfs=()
for o in $OBJECTS; do
  dirs+=("$DATA/$o/motions")
  urdfs+=("'$DATA/$o/$o.urdf'")
done
source "$SCRIPT_DIR/source_isaacsim_setup.sh"    # also puts this checkout's src/holosoma first on PYTHONPATH
cd "$SCRIPT_DIR/.."                               # holosoma writes logs/ relative to the working directory
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt-hybrid-multi-object "$@" \
  --command.setup-terms.motion-command.params.motion-config.motion-dir="$(IFS=,; echo "${dirs[*]}")" \
  --robot.object.object-urdf-paths "[$(IFS=,; echo "${urdfs[*]}")]"     # Python list literal (holosoma's tyro)
