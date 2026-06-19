#!/usr/bin/env bash
set -euo pipefail

CONDA_HOME="${GRIFFIN_CONDA_HOME:-$HOME/miniconda3}"
GRIFFIN_ENV_NAME="${GRIFFIN_ENV_NAME:-griffin}"
MMDET3D_ROOT="${GRIFFIN_MMDET3D_ROOT:-$HOME/.cache/griffin/mmdetection3d-v0.17.1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
MAX_JOBS="${MAX_JOBS:-2}"

if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$GRIFFIN_ENV_NAME"
fi

if [ ! -d "$MMDET3D_ROOT/mmdet3d/ops/spconv" ]; then
  echo "Missing mmdet3d spconv source directory: $MMDET3D_ROOT/mmdet3d/ops/spconv" >&2
  exit 2
fi

export CUDA_HOME
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS
export PATH="$CUDA_HOME/bin:$PATH"

cd "$MMDET3D_ROOT"
cat > /tmp/build_mmdet3d_spconv_ext.py <<'PY'
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mmdet3d_spconv_sparse_ext",
    ext_modules=[
        CUDAExtension(
            name="mmdet3d.ops.spconv.sparse_conv_ext",
            sources=[
                "mmdet3d/ops/spconv/src/all.cc",
                "mmdet3d/ops/spconv/src/reordering.cc",
                "mmdet3d/ops/spconv/src/reordering_cuda.cu",
                "mmdet3d/ops/spconv/src/indice.cc",
                "mmdet3d/ops/spconv/src/indice_cuda.cu",
                "mmdet3d/ops/spconv/src/maxpool.cc",
                "mmdet3d/ops/spconv/src/maxpool_cuda.cu",
            ],
            include_dirs=["mmdet3d/ops/spconv/include"],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": [
                    "-O2",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
PY

python /tmp/build_mmdet3d_spconv_ext.py build_ext --inplace
python - <<'PY'
import glob
import mmdet3d.ops.spconv as spconv

print("spconv_import_ok=", spconv.__file__)
print("sparse_conv_ext=", glob.glob("mmdet3d/ops/spconv/sparse_conv_ext*.so"))
PY
