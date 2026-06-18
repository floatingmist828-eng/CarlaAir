#!/usr/bin/env bash
set -euo pipefail

ROOT="${GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}"
CONDA_HOME="${GRIFFIN_CONDA_HOME:-$HOME/miniconda3}"
ENV_NAME="${GRIFFIN_ENV_NAME:-griffin}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MMDET3D_SRC="${GRIFFIN_MMDET3D_SRC:-$HOME/.cache/griffin/mmdetection3d-v0.17.1}"

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PIP_PROGRESS_BAR="${PIP_PROGRESS_BAR:-off}"

LOG_DIR="${GRIFFIN_ENV_LOG_DIR:-$ROOT/griffin_repro/artifacts/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup_griffin_env_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Griffin environment setup log: $LOG_FILE"
echo "Repository root: $ROOT"
echo "Conda home: $CONDA_HOME"
echo "Environment: $ENV_NAME"
echo "CUDA_HOME: $CUDA_HOME"

if [ ! -x "$CONDA_HOME/bin/conda" ]; then
  installer="$LOG_DIR/Miniconda3-latest-Linux-x86_64.sh"
  echo "Installing Miniconda from https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  curl -fL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o "$installer"
  bash "$installer" -b -p "$CONDA_HOME"
fi

# shellcheck disable=SC1091
source "$CONDA_HOME/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.8 pip -y
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install "setuptools==59.5.0" wheel "numpy<1.24"
python -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
python -m pip install mmdet==2.14.0 mmsegmentation==0.14.1

install_mmdet3d_source() {
  if [ -f "$MMDET3D_SRC/setup.py" ]; then
    if [ -d "$MMDET3D_SRC/.git" ]; then
      git -C "$MMDET3D_SRC" checkout v0.17.1 || true
    fi
    return 0
  fi

  rm -rf "$MMDET3D_SRC"
  mkdir -p "$(dirname "$MMDET3D_SRC")"
  if timeout 180 git -c http.version=HTTP/1.1 clone --depth 1 --branch v0.17.1 \
    https://github.com/open-mmlab/mmdetection3d.git "$MMDET3D_SRC"; then
    return 0
  fi

  rm -rf "$MMDET3D_SRC"
  archive="$LOG_DIR/mmdetection3d-v0.17.1.tar.gz"
  if ! timeout 180 curl --retry 2 --connect-timeout 20 -fL \
    https://github.com/open-mmlab/mmdetection3d/archive/refs/tags/v0.17.1.tar.gz -o "$archive"; then
    return 1
  fi
  mkdir -p "$MMDET3D_SRC"
  tar -xzf "$archive" --strip-components=1 -C "$MMDET3D_SRC"
  return 0
}

install_mmdet3d_from_pypi_sdist() {
  rm -rf "$MMDET3D_SRC"
  pypi_dir="$LOG_DIR/mmdet3d_pypi"
  rm -rf "$pypi_dir"
  mkdir -p "$pypi_dir" "$MMDET3D_SRC"
  python -m pip download mmdet3d==0.17.1 --no-deps --no-build-isolation -d "$pypi_dir"
  archive="$(find "$pypi_dir" -maxdepth 1 -name 'mmdet3d-0.17.1*.tar.gz' | head -n 1)"
  if [ -z "$archive" ]; then
    echo "Unable to locate downloaded mmdet3d 0.17.1 source archive." >&2
    return 1
  fi
  tar -xzf "$archive" --strip-components=1 -C "$MMDET3D_SRC"
}

patch_mmdet3d_setup() {
  python - "$MMDET3D_SRC/setup.py" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
pattern = (
    r"\n\s+make_cuda_ext\(\n"
    r"\s+name='sparse_conv_ext',.*?"
    r"extra_args=\['-w', '-std=c\+\+14'\]\),"
)
patched, count = re.subn(pattern, "", text, count=1, flags=re.S)
if count:
    path.write_text(patched, encoding="utf-8")
    print("Skipping mmdet3d spconv extension for Griffin camera/BEV reproduction.")
else:
    print("mmdet3d spconv extension block was not present or was already skipped.")
PY
}

if ! install_mmdet3d_source; then
  install_mmdet3d_from_pypi_sdist
fi
patch_mmdet3d_setup
python -m pip install -r "$MMDET3D_SRC/requirements/runtime.txt"
python -m pip install -v -e "$MMDET3D_SRC" --no-deps

python -m pip install -r "$ROOT/griffin_repro/official/requirements.txt"
cd "$ROOT"
python scripts/griffin_repro.py env-check --strict --json
echo "Griffin environment is ready. Activate it with: source $CONDA_HOME/etc/profile.d/conda.sh && conda activate $ENV_NAME"
