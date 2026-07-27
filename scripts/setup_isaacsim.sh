#!/usr/bin/env bash
# Exit on error, undefined variables, and failed pipelines; print commands.
set -euxo pipefail

# Clear LD_LIBRARY_PATH to prevent host conda libraries (e.g. libstdc++, libtinfo)
# from breaking system tools like cmake during the build.
unset LD_LIBRARY_PATH

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(realpath "$(dirname "$SCRIPT_DIR")")

ISAACSIM_VERSION=5.1.0
ISAACSIM_PACKAGE_VERSION=5.1.0.0
ISAACLAB_TAG=v2.3.0
ISAACLAB_COMMIT=3c6e67bb5c7ada942a6d1884ab69338f57596f77
SETUP_SCHEMA_VERSION=2
RUNTIME_CONSTRAINTS_FILE=$SCRIPT_DIR/constraints-isaacsim-5.1.txt

if ! command -v sudo &> /dev/null; then
  # In docker builds sudo is not available, but commands already run as root.
  echo "Warning: sudo could not be found, you may need to run this script with sudo"
  function sudo { "$@"; }
  export -f sudo
fi

# Use CONDA_ENV_NAME if provided, otherwise default to "hssim"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hssim}
echo "conda environment name is set to: $CONDA_ENV_NAME"

# Create overall workspace
source "$SCRIPT_DIR/source_common.sh"
ENV_ROOT=$CONDA_ROOT/envs/$CONDA_ENV_NAME
SENTINEL_FILE=$WORKSPACE_DIR/.env_setup_finished_${CONDA_ENV_NAME}_isaacsim-${ISAACSIM_VERSION}_isaaclab-${ISAACLAB_TAG}_schema-${SETUP_SCHEMA_VERSION}
echo "SENTINEL_FILE: $SENTINEL_FILE"

mkdir -p "$WORKSPACE_DIR"

ISAACSIM_READY=false
if [[ -x "$ENV_ROOT/bin/python" ]] && \
  "$ENV_ROOT/bin/python" -c \
    "import importlib.metadata as m; assert m.version('isaacsim') == '$ISAACSIM_PACKAGE_VERSION'"; then
  ISAACSIM_READY=true
fi

if [[ ! -f "$SENTINEL_FILE" || ! -d "$ENV_ROOT" || ! -d "$WORKSPACE_DIR/IsaacLab/.git" || "$ISAACSIM_READY" != true ]]; then
  # Install miniconda
  if [[ ! -d "$CONDA_ROOT" ]]; then
    if command -v conda &> /dev/null; then
      CONDA_ROOT=$(conda info --base)
      ENV_ROOT=$CONDA_ROOT/envs/$CONDA_ENV_NAME
    else
      mkdir -p "$CONDA_ROOT"
      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$CONDA_ROOT/miniconda.sh"
      bash "$CONDA_ROOT/miniconda.sh" -b -u -p "$CONDA_ROOT"
      rm "$CONDA_ROOT/miniconda.sh"
    fi
  fi

  # Create the conda environment
  if [[ ! -d "$ENV_ROOT" ]]; then
    accept_conda_tos_if_available
    install_mamba_if_needed
    MAMBA_ROOT_PREFIX=$CONDA_ROOT "$CONDA_ROOT/bin/mamba" create -y -n "$CONDA_ENV_NAME" \
      python=3.11 -c conda-forge --override-channels
  fi

  source "$CONDA_ROOT/bin/activate" "$CONDA_ENV_NAME"

  # Install ffmpeg for video encoding
  conda install -c conda-forge -y ffmpeg
  conda install -c conda-forge -y libiconv
  conda install -c conda-forge -y libglu

  # Below follows https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
  # Install IsaacSim
  python -m pip install --upgrade pip
  python -m pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

  # Install dependencies from PyPI first
  python -m pip install pyperclip
  # Then install isaacsim from NVIDIA index only
  python -m pip install "isaacsim[all,extscache]==$ISAACSIM_VERSION" --extra-index-url https://pypi.nvidia.com

  if [[ ! -d "$WORKSPACE_DIR/IsaacLab" ]]; then
    git clone https://github.com/isaac-sim/IsaacLab.git --branch "$ISAACLAB_TAG" "$WORKSPACE_DIR/IsaacLab"
  fi

  if [[ ! -d "$WORKSPACE_DIR/IsaacLab/.git" ]]; then
    echo "Error: $WORKSPACE_DIR/IsaacLab exists but is not an IsaacLab git checkout." >&2
    exit 1
  fi
  ACTUAL_ISAACLAB_COMMIT=$(git -C "$WORKSPACE_DIR/IsaacLab" rev-parse HEAD)
  if [[ "$ACTUAL_ISAACLAB_COMMIT" != "$ISAACLAB_COMMIT" ]]; then
    echo "Error: expected IsaacLab $ISAACLAB_TAG at $ISAACLAB_COMMIT, found $ACTUAL_ISAACLAB_COMMIT." >&2
    echo "Move the existing $WORKSPACE_DIR/IsaacLab checkout aside and rerun setup." >&2
    exit 1
  fi

  MISSING_BUILD_TOOLS=()
  for tool in cmake make g++; do
    if ! command -v "$tool" &> /dev/null; then
      MISSING_BUILD_TOOLS+=("$tool")
    fi
  done
  if (( ${#MISSING_BUILD_TOOLS[@]} > 0 )); then
    echo "Installing missing build tools: ${MISSING_BUILD_TOOLS[*]}"
    sudo apt-get update
    sudo apt-get install -y cmake build-essential
  else
    echo "Required build tools are already available; skipping apt installation."
  fi

  cd "$WORKSPACE_DIR/IsaacLab"
  # setuptools 81 removes pkg_resources, which an IsaacLab dependency still needs.
  # see https://github.com/isaac-sim/IsaacLab/pull/4585
  python -m pip install 'setuptools<81'
  BUILD_CONSTRAINTS_FILE=$(mktemp)
  printf '%s\n' 'setuptools<81' > "$BUILD_CONSTRAINTS_FILE"
  cleanup_build_constraints() {
    rm -f "$BUILD_CONSTRAINTS_FILE"
  }
  trap cleanup_build_constraints EXIT
  export PIP_BUILD_CONSTRAINT=$BUILD_CONSTRAINTS_FILE
  # Fix upstream bug: should use flatdict 4.1.0 (https://github.com/isaac-sim/IsaacLab/issues/4576)
  if grep -q 'flatdict==4.0.1' source/isaaclab/setup.py; then
    sed -i 's/flatdict==4.0.1/flatdict==4.1.0/' source/isaaclab/setup.py
  elif ! grep -q 'flatdict==4.1.0' source/isaaclab/setup.py; then
    echo "Error: could not find the expected flatdict pin in IsaacLab." >&2
    exit 1
  fi
  # work-around for egl_probe cmake max version issue
  export CMAKE_POLICY_VERSION_MINIMUM=3.5
  export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-1}
  # IsaacLab's wrapper calls `tabs` before doing any work. `tabs` fails when
  # automation supplies TERM=dumb, so provide a capable terminal definition
  # for this invocation.
  ISAACLAB_TERM=${TERM:-xterm}
  if [[ "$ISAACLAB_TERM" == "dumb" ]]; then
    ISAACLAB_TERM=xterm
  fi
  # IsaacLab 2.3 leaves several framework dependencies open-ended. Apply the
  # tested runtime constraints so a future resolver cannot replace Torch 2.7
  # and CUDA 12.8 with a newer incompatible stack.
  export PIP_CONSTRAINT=$RUNTIME_CONSTRAINTS_FILE
  TERM=$ISAACLAB_TERM ./isaaclab.sh --install
  unset PIP_CONSTRAINT
  unset PIP_BUILD_CONSTRAINT
  cleanup_build_constraints
  trap - EXIT

  cd "$ROOT_DIR"
fi

# This intentionally runs even when the expensive simulator setup is skipped.
# It prevents a shared sentinel from leaving this environment attached to a
# different Holosoma checkout.
source "$CONDA_ROOT/bin/activate" "$CONDA_ENV_NAME"
python -m pip install --constraint "$RUNTIME_CONSTRAINTS_FILE" 'setuptools<81' 'wheel==0.45.1'
python -m pip install --constraint "$RUNTIME_CONSTRAINTS_FILE" -e "$ROOT_DIR/src/holosoma[unitree,booster]"

ACTUAL_ISAACLAB_COMMIT=$(git -C "$WORKSPACE_DIR/IsaacLab" rev-parse HEAD)
if [[ "$ACTUAL_ISAACLAB_COMMIT" != "$ISAACLAB_COMMIT" ]]; then
  echo "Error: expected IsaacLab $ISAACLAB_TAG at $ISAACLAB_COMMIT, found $ACTUAL_ISAACLAB_COMMIT." >&2
  exit 1
fi

EXPECTED_HOLOSOMA_ROOT=$ROOT_DIR/src/holosoma \
EXPECTED_ISAACSIM_VERSION=$ISAACSIM_PACKAGE_VERSION \
python - <<'PY'
import importlib.metadata
import os
from pathlib import Path

import holosoma

expected_root = Path(os.environ["EXPECTED_HOLOSOMA_ROOT"]).resolve()
actual_module = Path(holosoma.__file__).resolve()
if not actual_module.is_relative_to(expected_root):
    raise RuntimeError(f"holosoma imports from {actual_module}, expected a path under {expected_root}")

expected_versions = {
    "isaacsim": os.environ["EXPECTED_ISAACSIM_VERSION"],
    "viser": "1.0.21",
    "websockets": "15.0.1",
    "wandb": "0.22.0",
    "tyro": "1.0.12",
    "click": "8.2.1",
    "typing-extensions": "4.15.0",
    "psutil": "5.9.8",
    "packaging": "23.0",
    "requests": "2.32.3",
    "charset-normalizer": "3.3.2",
    "chardet": "5.2.0",
    "wheel": "0.45.1",
    "torch": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
    "torchaudio": "2.7.0",
    "warp-lang": "1.10.0",
    "transformers": "5.5.0",
    "huggingface-hub": "1.9.0",
    "ipython": "9.10.1",
    "pytest": "9.0.2",
    "skrl": "1.4.3",
    "stable-baselines3": "2.8.0",
    "ray": "2.52.1",
    "typer": "0.24.1",
}
for distribution, expected in expected_versions.items():
    actual = importlib.metadata.version(distribution)
    if actual != expected:
        raise RuntimeError(f"{distribution}=={actual}, expected {distribution}=={expected}")

print(f"Validated current editable Holosoma install: {actual_module}")
print("Validated pinned IsaacSim/Holosoma runtime dependency versions.")
PY

# Viser 1.0.21 + websockets 15.0.1 is the combination exercised by the
# evaluation smoke tests. IsaacSim 5.1 works with it, but its package metadata
# declares click==8.1.7, websockets==12.0, and typing-extensions==4.12.2.
# Allow only those known metadata mismatches; fail on every other broken
# requirement.
if ! PIP_CHECK_OUTPUT=$(python -m pip check 2>&1); then
  PIP_CHECK_PROBLEMS=$(printf '%s\n' "$PIP_CHECK_OUTPUT" | grep -E 'has requirement|is not installed' || true)
  if [[ -z "$PIP_CHECK_PROBLEMS" ]]; then
    echo "Error: pip check failed unexpectedly:" >&2
    printf '%s\n' "$PIP_CHECK_OUTPUT" >&2
    exit 1
  fi
  UNEXPECTED_PIP_CHECK_PROBLEMS=$(printf '%s\n' "$PIP_CHECK_PROBLEMS" | grep -Ev \
    -e '^isaacsim-kernel 5\.1\.0\.0 has requirement click==8\.1\.7, but you have click 8\.2\.1\.$' \
    -e '^isaacsim-kernel 5\.1\.0\.0 has requirement websockets==12\.0, but you have websockets 15\.0\.1\.$' \
    -e '^isaacsim-kernel 5\.1\.0\.0 has requirement typing_extensions==4\.12\.2, but you have typing-extensions 4\.15\.0\.$' || true)
  if [[ -n "$UNEXPECTED_PIP_CHECK_PROBLEMS" ]]; then
    echo "Error: unexpected dependency conflicts:" >&2
    printf '%s\n' "$UNEXPECTED_PIP_CHECK_PROBLEMS" >&2
    exit 1
  fi
  echo "Warning: ignoring only the documented metadata mismatches for the tested runtime stack." >&2
  printf '%s\n' "$PIP_CHECK_PROBLEMS" >&2
fi

printf 'schema=%s\nrepo=%s\nisaacsim=%s\nisaaclab_tag=%s\nisaaclab_commit=%s\n' \
  "$SETUP_SCHEMA_VERSION" \
  "$ROOT_DIR" \
  "$ISAACSIM_VERSION" \
  "$ISAACLAB_TAG" \
  "$ISAACLAB_COMMIT" > "$SENTINEL_FILE"
