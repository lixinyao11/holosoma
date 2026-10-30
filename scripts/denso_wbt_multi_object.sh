#!/usr/bin/env bash
# Whole-body tracking on several Denso holosoma datasets at once (one object per environment group: environment e
# holds object e % N and tracks only that object's clips; exp:g1-29dof-wbt-w-multi-object).
#
#   bash scripts/denso_wbt_multi_object.sh logger:wandb --training.num-envs=4096 ...
#
# DENSO_HOLOSOMA_DATA: dataset root (default ~/Code/Denso/data/holosoma/omomo/umr, from Denso's
# `scripts/export.py --format holosoma`); OBJECTS: object datasets to use, in order; EXP: experiment preset
# (default g1-29dof-wbt-w-multi-object; g1-29dof-wbt-w-multi-object-state gives the actor the object and reference).
# EXCLUDE: optional file of clip names (npz stems, one per line) to leave out; the remaining clips of each object are
# linked into a temporary <object>/motions directory.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DATA=${DENSO_HOLOSOMA_DATA:-$HOME/Code/Denso/data/holosoma/omomo/umr}
OBJECTS=${OBJECTS:-"largebox plasticbox smallbox suitcase"}
EXP=${EXP:-g1-29dof-wbt-w-multi-object}
dirs=()
urdfs=()
[[ -n "${EXCLUDE:-}" ]] && KEEP=$(mktemp -d)
for o in $OBJECTS; do
  if [[ -n "${EXCLUDE:-}" ]]; then
    mkdir -p "$KEEP/$o/motions"
    for f in "$DATA/$o/motions"/*.npz; do
      grep -qxF "$(basename "$f" .npz)" "$EXCLUDE" || ln -s "$f" "$KEEP/$o/motions/"
    done
    echo "$o: $(ls "$KEEP/$o/motions" | wc -l) of $(ls "$DATA/$o/motions" | wc -l) clips (EXCLUDE=$EXCLUDE)"
    dirs+=("$KEEP/$o/motions")
  else
    dirs+=("$DATA/$o/motions")
  fi
  urdfs+=("'$DATA/$o/$o.urdf'")
done
source "$SCRIPT_DIR/source_isaacsim_setup.sh"
# NPROC > 1: one training process per GPU (torchrun); --training.num-envs is then the total over all GPUs
LAUNCH=(python)
(( ${NPROC:-1} > 1 )) && LAUNCH=(python -m torch.distributed.run --standalone --nproc_per_node="$NPROC")
"${LAUNCH[@]}" "$SCRIPT_DIR/../src/holosoma/holosoma/train_agent.py" "exp:$EXP" "$@" \
  --command.setup-terms.motion-command.params.motion-config.motion-dir="$(IFS=,; echo "${dirs[*]}")" \
  --scene.rigid-objects.object.urdf-files "[$(IFS=,; echo "${urdfs[*]}")]"     # Python list literal (holosoma's tyro)
