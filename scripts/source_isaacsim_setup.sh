# Detect script directory (works in both bash and zsh)
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi

# Use CONDA_ENV_NAME if provided, otherwise default to "hssim"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hssim}
echo "conda environment name is set to: $CONDA_ENV_NAME"

source "$SCRIPT_DIR/source_common.sh"
source "$CONDA_ROOT/bin/activate" "$CONDA_ENV_NAME"
export OMNI_KIT_ACCEPT_EULA=1

# Prefer the checkout containing this setup script, even when the shared conda
# environment was previously used by another Holosoma checkout.
REPO_ROOT=$(realpath "$(dirname "$SCRIPT_DIR")")
export PYTHONPATH="$REPO_ROOT/src/holosoma${PYTHONPATH:+:$PYTHONPATH}"
