WORKSPACE_DIR=$HOME/.holosoma_deps
if [[ -d $WORKSPACE_DIR/miniconda3 ]]; then
  CONDA_ROOT=$WORKSPACE_DIR/miniconda3
elif command -v conda &> /dev/null; then
  CONDA_ROOT=$(conda info --base)
else
  CONDA_ROOT=$WORKSPACE_DIR/miniconda3
fi

accept_conda_tos_if_available() {
  if "$CONDA_ROOT/bin/conda" tos --help &> /dev/null; then
    "$CONDA_ROOT/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    "$CONDA_ROOT/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  else
    echo "conda tos subcommand not available; skipping Anaconda Terms of Service acceptance"
  fi
}

install_mamba_if_needed() {
  if [[ ! -f "$CONDA_ROOT/bin/mamba" ]]; then
    CONDA_SOLVER=classic "$CONDA_ROOT/bin/conda" install -y mamba -c conda-forge -n base --override-channels
  fi
}
