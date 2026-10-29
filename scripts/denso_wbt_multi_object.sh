#!/usr/bin/env bash
# Whole-body tracking on several Denso holosoma datasets at once (one object per environment group: environment e
# holds object e % N and tracks only that object's clips; exp:g1-29dof-wbt-w-multi-object).
#
#   bash scripts/denso_wbt_multi_object.sh logger:wandb --training.num-envs=4096 ...
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
source "$SCRIPT_DIR/source_isaacsim_setup.sh"
python "$SCRIPT_DIR/../src/holosoma/holosoma/train_agent.py" exp:g1-29dof-wbt-w-multi-object "$@" \
  --command.setup-terms.motion-command.params.motion-config.motion-dir="$(IFS=,; echo "${dirs[*]}")" \
  --scene.rigid-objects.object.urdf-files "[$(IFS=,; echo "${urdfs[*]}")]"     # Python list literal (holosoma's tyro)
