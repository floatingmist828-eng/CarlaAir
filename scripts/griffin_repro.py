#!/usr/bin/env python3
"""Utilities for the isolated Griffin paper reproduction package."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib.metadata
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import struct
import zlib
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = REPO_ROOT / "griffin_repro"
OFFICIAL_ROOT = REPRO_ROOT / "official"
MANIFEST_PATH = REPRO_ROOT / "manifest.json"
RESULTS_CSV = OFFICIAL_ROOT / "docs" / "detailed_results.csv"
CONDA_INSTALLER_URL = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
_ZIP_CENTRAL_DIRECTORY_CACHE: dict[tuple[str, int], tuple[int, bytes]] = {}

DATASETS = {
    "50scenes_25m": {
        "dataset_prefix": "griffin_50scenes_25m",
        "scene_count": 47,
        "altitude": "25 +/- 2 m",
    },
    "50scenes_40m": {
        "dataset_prefix": "griffin_50scenes_40m",
        "scene_count": 54,
        "altitude": "40 +/- 2 m",
    },
    "50scenes_55m": {
        "dataset_prefix": "griffin_50scenes_55m",
        "scene_count": 50,
        "altitude": "55 +/- 2 m",
    },
    "100scenes_random": {
        "dataset_prefix": "griffin_100scenes_random",
        "scene_count": 104,
        "altitude": "20-60 m",
    },
}

FUSION_METHODS = [
    "0-no fusion",
    "1-early fusion",
    "2a1-v2x-vit",
    "2a2-where2comm",
    "2b1-cooptrack",
    "2b2-univ2x",
    "3-late fusion",
]

METRICS = {
    "AP": "3D detection average precision from the Griffin benchmark table.",
    "AMOTA": "Tracking accuracy metric used as the main multi-object tracking score.",
    "BPS": "Communication bytes per sample, computed by the official analysis tooling when feature payloads exist.",
    "FPS": "Runtime throughput reported by the paper for efficiency comparison.",
}

ROBUSTNESS = {
    "communication_latency_ms": [100, 200, 300, 400],
    "packet_loss": [0.1, 0.2, 0.3, 0.4, 0.5],
    "translation_error_m": [0.5, 1.0, 1.5, 2.0, 2.5],
    "rotation_error_deg": [1, 2, 3, 4, 5],
}

CHECKPOINT_ITERS = {
    "100scenes_random": "iter_36072.pth",
    "50scenes_25m": "iter_33024.pth",
    "50scenes_40m": "iter_38784.pth",
    "50scenes_55m": "iter_35760.pth",
}

RUNNABLE_METHODS = {
    "0-no fusion",
    "1-early fusion",
    "2b1-cooptrack",
    "3-late fusion",
}

RUNNABLE_METHOD_ORDER = [
    "0-no fusion",
    "1-early fusion",
    "2b1-cooptrack",
    "3-late fusion",
]

DATA_PACKAGES = {
    "50scenes_25m": [
        ("datasets/griffin_50scenes_25m/drone_camera_back.zip", 19492671867),
        ("datasets/griffin_50scenes_25m/drone_camera_bottom.zip", 19047808871),
        ("datasets/griffin_50scenes_25m/drone_camera_front.zip", 19486880375),
        ("datasets/griffin_50scenes_25m/drone_camera_instance_segmentation.zip", 3558624019),
        ("datasets/griffin_50scenes_25m/drone_camera_left.zip", 19309526941),
        ("datasets/griffin_50scenes_25m/drone_camera_right.zip", 19669453631),
        ("datasets/griffin_50scenes_25m/drone_metadata.zip", 14384997),
        ("datasets/griffin_50scenes_25m/md5.txt", 898),
        ("datasets/griffin_50scenes_25m/vehicle_camera_back.zip", 17138643853),
        ("datasets/griffin_50scenes_25m/vehicle_camera_front.zip", 16257201694),
        ("datasets/griffin_50scenes_25m/vehicle_camera_instance_segmentation.zip", 1116380149),
        ("datasets/griffin_50scenes_25m/vehicle_camera_left.zip", 15554668466),
        ("datasets/griffin_50scenes_25m/vehicle_camera_right.zip", 16318407065),
        ("datasets/griffin_50scenes_25m/vehicle_lidar.zip", 214487013),
        ("datasets/griffin_50scenes_25m/vehicle_metadata.zip", 10876283),
    ],
}

SMOKE_25M_INSTANCE_PACKAGES = {
    "datasets/griffin_50scenes_25m/drone_camera_back.zip",
    "datasets/griffin_50scenes_25m/drone_camera_bottom.zip",
    "datasets/griffin_50scenes_25m/drone_camera_front.zip",
    "datasets/griffin_50scenes_25m/drone_camera_left.zip",
    "datasets/griffin_50scenes_25m/drone_camera_right.zip",
    "datasets/griffin_50scenes_25m/drone_metadata.zip",
    "datasets/griffin_50scenes_25m/md5.txt",
    "datasets/griffin_50scenes_25m/vehicle_camera_back.zip",
    "datasets/griffin_50scenes_25m/vehicle_camera_front.zip",
    "datasets/griffin_50scenes_25m/vehicle_camera_left.zip",
    "datasets/griffin_50scenes_25m/vehicle_camera_right.zip",
    "datasets/griffin_50scenes_25m/vehicle_metadata.zip",
}

SMOKE_25M_VEHICLE_PACKAGES = {
    "datasets/griffin_50scenes_25m/md5.txt",
    "datasets/griffin_50scenes_25m/vehicle_camera_back.zip",
    "datasets/griffin_50scenes_25m/vehicle_camera_front.zip",
    "datasets/griffin_50scenes_25m/vehicle_camera_left.zip",
    "datasets/griffin_50scenes_25m/vehicle_camera_right.zip",
    "datasets/griffin_50scenes_25m/vehicle_metadata.zip",
}

DATA_PACKAGE_PROFILES = {
    "full": None,
    "smoke_25m_instance": SMOKE_25M_INSTANCE_PACKAGES,
    "smoke_25m_vehicle": SMOKE_25M_VEHICLE_PACKAGES,
}

CAMERA_DIRECTIONS = {
    "CAM_FRONT": "front",
    "CAM_BACK": "back",
    "CAM_LEFT": "left",
    "CAM_RIGHT": "right",
    "CAM_BOTTOM": "bottom",
}

EXPECTED_PYTHON_MODULES = {
    "torch": {"import": "torch", "package": "torch", "version": "1.9.1"},
    "mmcv": {"import": "mmcv", "package": "mmcv-full", "version": "1.4.0"},
    "mmdet": {"import": "mmdet", "package": "mmdet", "version": "2.14.0"},
    "mmseg": {"import": "mmseg", "package": "mmsegmentation", "version": "0.14.1"},
    "mmdet3d": {"import": "mmdet3d", "package": "mmdet3d", "version": "0.17.1"},
}


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_results() -> list[dict[str, str]]:
    with RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def is_zero_condition(row: dict[str, str]) -> bool:
    return all(
        float(row[name]) == 0.0
        for name in (
            "communication latency",
            "packet loss",
            "translation error",
            "rotation error",
        )
    )


def config_files() -> list[Path]:
    projects = OFFICIAL_ROOT / "projects"
    if not projects.exists():
        return []
    files = []
    for path in projects.rglob("*.py"):
        if any(part.startswith("configs_griffin_") for part in path.parts):
            files.append(path)
    return sorted(files)


def result_summary() -> dict[str, Any]:
    rows = load_results()
    baseline = []
    for row in rows:
        if is_zero_condition(row):
            baseline.append(
                {
                    "dataset": row["dataset"],
                    "method": row["methods"],
                    "AP": float(row["AP"]),
                    "AMOTA": float(row["AMOTA"]),
                }
            )

    robustness = {
        "communication_latency": sorted(
            {float(row["communication latency"]) for row in rows if float(row["communication latency"]) > 0}
        ),
        "packet_loss": sorted({float(row["packet loss"]) for row in rows if float(row["packet loss"]) > 0}),
        "translation_error": sorted(
            {float(row["translation error"]) for row in rows if float(row["translation error"]) > 0}
        ),
        "rotation_error": sorted({float(row["rotation error"]) for row in rows if float(row["rotation error"]) > 0}),
    }
    return {"baseline": baseline, "robustness": robustness, "rows": len(rows)}


def verify_layout() -> dict[str, Any]:
    manifest = load_manifest()
    rows = load_results() if RESULTS_CSV.exists() else []
    missing = []
    required = [
        OFFICIAL_ROOT / "README.md",
        OFFICIAL_ROOT / "tools" / "dist_eval.sh",
        OFFICIAL_ROOT / "tools" / "analysis_tools" / "compute_BPS.py",
        OFFICIAL_ROOT / "projects" / "mmdet3d_plugin" / "datasets" / "griffin_dataset.py",
    ]
    for path in required:
        if not path.exists():
            missing.append(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))
    return {
        "official_exists": OFFICIAL_ROOT.exists(),
        "manifest_source": manifest["source"]["repository"],
        "config_files": len(config_files()),
        "detailed_result_rows": len(rows),
        "baseline_rows": sum(1 for row in rows if is_zero_condition(row)),
        "experiment_profiles": len(manifest["profiles"]),
        "missing": missing,
    }


def dataset_prefix(dataset: str) -> str:
    if dataset not in DATASETS:
        raise SystemExit(f"Unknown dataset {dataset!r}")
    return DATASETS[dataset]["dataset_prefix"]


def profile_payload(profile_name: str) -> dict[str, Any]:
    manifest = load_manifest()
    profiles = manifest["profiles"]
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise SystemExit(f"Unknown profile {profile_name!r}. Available profiles: {available}")
    profile = profiles[profile_name]
    command = (
        f"cd griffin_repro/official && "
        f"{dist_eval_command(profile['config'], profile['checkpoint'], profile['gpus'])}"
    )
    return {
        "profile": profile_name,
        "description": profile["description"],
        "dataset": profile["dataset"],
        "method": profile["method"],
        "config": profile["config"],
        "checkpoint": profile["checkpoint"],
        "gpus": profile["gpus"],
        "expected": profile["expected"],
        "commands": [command],
        "asset_checks": [f"griffin_repro/official/{path}" for path in profile.get("required_paths", [])],
    }


def dist_eval_command(config: str, checkpoint: str, gpus: int | str) -> str:
    return f"CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-0}} bash tools/dist_eval.sh {config} {checkpoint} {gpus}"


def list_profiles() -> dict[str, Any]:
    manifest = load_manifest()
    profiles = {}
    for name, profile in sorted(manifest["profiles"].items()):
        config = OFFICIAL_ROOT / profile["config"]
        checkpoint = OFFICIAL_ROOT / profile["checkpoint"]
        checks = check_assets(name)["checks"]
        profiles[name] = {
            "dataset": profile["dataset"],
            "method": profile["method"],
            "description": profile["description"],
            "config": profile["config"],
            "checkpoint": profile["checkpoint"],
            "config_exists": config.exists(),
            "checkpoint_exists": checkpoint.exists(),
            "ready": all(item["exists"] for item in checks),
            "expected": profile["expected"],
        }
    return {"profiles": profiles}


def check_assets(profile_name: str) -> dict[str, Any]:
    payload = profile_payload(profile_name)
    checks = []
    for rel_path in payload["asset_checks"]:
        path = REPO_ROOT / rel_path
        checks.append({"path": rel_path, "exists": path.exists()})
    return {"profile": profile_name, "checks": checks, "ready": all(item["exists"] for item in checks)}


def check_path_list(paths: list[str]) -> dict[str, Any]:
    checks = [{"path": path, "exists": (REPO_ROOT / path).exists()} for path in paths]
    return {"checks": checks, "ready": all(item["exists"] for item in checks)}


def module_status(name: str, spec: dict[str, str]) -> dict[str, Any]:
    import_name = spec["import"]
    package_name = spec["package"]
    expected_version = spec["version"]
    available = importlib.util.find_spec(import_name) is not None
    version = None
    if available:
        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            version = None
    version_matches = bool(
        available and (expected_version is None or (version is not None and version.startswith(expected_version)))
    )
    return {
        "available": available,
        "import": import_name,
        "package": package_name,
        "version": version,
        "expected_version": expected_version,
        "version_matches": version_matches,
        "ok": available and version_matches,
    }


def env_check() -> dict[str, Any]:
    modules = {
        name: module_status(name, spec)
        for name, spec in EXPECTED_PYTHON_MODULES.items()
    }
    nvidia_smi_path = shutil.which("nvidia-smi")
    nvidia_smi = {"available": nvidia_smi_path is not None, "path": nvidia_smi_path, "gpus": []}
    if nvidia_smi_path:
        try:
            result = subprocess.run(
                [
                    nvidia_smi_path,
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                nvidia_smi["gpus"] = [
                    {"raw": line.strip()} for line in result.stdout.splitlines() if line.strip()
                ]
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "python_modules": modules,
        "nvidia_smi": nvidia_smi,
        "ready": all(item["ok"] for item in modules.values()),
    }


def conda_activation_block() -> str:
    return """CONDA_HOME="${GRIFFIN_CONDA_HOME:-$HOME/miniconda3}"
GRIFFIN_ENV_NAME="${GRIFFIN_ENV_NAME:-griffin}"
if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$GRIFFIN_ENV_NAME"
fi
"""


def env_setup_script() -> str:
    installer_name = Path(CONDA_INSTALLER_URL).name
    return f"""#!/usr/bin/env bash
set -euo pipefail

ROOT="${{GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}}"
CONDA_HOME="${{GRIFFIN_CONDA_HOME:-$HOME/miniconda3}}"
ENV_NAME="${{GRIFFIN_ENV_NAME:-griffin}}"
CUDA_HOME="${{CUDA_HOME:-/usr/local/cuda}}"
MMDET3D_SRC="${{GRIFFIN_MMDET3D_SRC:-$HOME/.cache/griffin/mmdetection3d-v0.17.1}}"

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${{LD_LIBRARY_PATH:-}}"
export PIP_PROGRESS_BAR="${{PIP_PROGRESS_BAR:-off}}"

LOG_DIR="${{GRIFFIN_ENV_LOG_DIR:-$ROOT/griffin_repro/artifacts/logs}}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup_griffin_env_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Griffin environment setup log: $LOG_FILE"
echo "Repository root: $ROOT"
echo "Conda home: $CONDA_HOME"
echo "Environment: $ENV_NAME"
echo "CUDA_HOME: $CUDA_HOME"

if [ ! -x "$CONDA_HOME/bin/conda" ]; then
  installer="$LOG_DIR/{installer_name}"
  echo "Installing Miniconda from {CONDA_INSTALLER_URL}"
  curl -fL "{CONDA_INSTALLER_URL}" -o "$installer"
  bash "$installer" -b -p "$CONDA_HOME"
fi

# shellcheck disable=SC1091
source "$CONDA_HOME/etc/profile.d/conda.sh"

if ! conda env list | awk '{{print $1}}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.8 pip -y
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install "setuptools==59.5.0" wheel "numpy<1.24"
python -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \\
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install mmcv-full==1.4.0 \\
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
python -m pip install mmdet==2.14.0 mmsegmentation==0.14.1

install_mmdet3d_source() {{
  if [ -f "$MMDET3D_SRC/setup.py" ]; then
    if [ -d "$MMDET3D_SRC/.git" ]; then
      git -C "$MMDET3D_SRC" checkout v0.17.1 || true
    fi
    return 0
  fi

  rm -rf "$MMDET3D_SRC"
  mkdir -p "$(dirname "$MMDET3D_SRC")"
  if timeout 180 git -c http.version=HTTP/1.1 clone --depth 1 --branch v0.17.1 \\
    https://github.com/open-mmlab/mmdetection3d.git "$MMDET3D_SRC"; then
    return 0
  fi

  rm -rf "$MMDET3D_SRC"
  archive="$LOG_DIR/mmdetection3d-v0.17.1.tar.gz"
  if ! timeout 180 curl --retry 2 --connect-timeout 20 -fL \\
    https://github.com/open-mmlab/mmdetection3d/archive/refs/tags/v0.17.1.tar.gz -o "$archive"; then
    return 1
  fi
  mkdir -p "$MMDET3D_SRC"
  tar -xzf "$archive" --strip-components=1 -C "$MMDET3D_SRC"
  return 0
}}

install_mmdet3d_from_pypi_sdist() {{
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
}}

patch_mmdet3d_setup() {{
  python - "$MMDET3D_SRC/setup.py" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
pattern = (
    r"\\n\\s+make_cuda_ext\\(\\n"
    r"\\s+name='sparse_conv_ext',.*?"
    r"extra_args=\\['-w', '-std=c\\+\\+14'\\]\\),"
)
patched, count = re.subn(pattern, "", text, count=1, flags=re.S)
if count:
    path.write_text(patched, encoding="utf-8")
    print("Skipping mmdet3d spconv extension for Griffin camera/BEV reproduction.")
else:
    print("mmdet3d spconv extension block was not present or was already skipped.")
PY
}}

if ! install_mmdet3d_source; then
  install_mmdet3d_from_pypi_sdist
fi
patch_mmdet3d_setup
python -m pip install -r "$MMDET3D_SRC/requirements/runtime.txt"
python -m pip install -v -e "$MMDET3D_SRC" --no-deps

python -m pip install "filterpy==1.4.5"
python -m pip install -r "$ROOT/griffin_repro/official/requirements.txt"
cd "$ROOT"
python scripts/griffin_repro.py env-check --strict --json
echo "Griffin environment is ready. Activate it with: source $CONDA_HOME/etc/profile.d/conda.sh && conda activate $ENV_NAME"
"""


def write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_env_script(out: str) -> dict[str, Any]:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(path, env_setup_script())
    return {"path": str(path), "bytes": path.stat().st_size}


def data_packages(dataset: str, package_profile: str = "full") -> dict[str, Any]:
    if dataset not in DATA_PACKAGES:
        raise SystemExit(f"No data package manifest is recorded for dataset {dataset!r}")
    if package_profile not in DATA_PACKAGE_PROFILES:
        known = ", ".join(sorted(DATA_PACKAGE_PROFILES))
        raise SystemExit(f"Unknown data package profile {package_profile!r}; expected one of: {known}")
    prefix = dataset_prefix(dataset)
    selected_paths = DATA_PACKAGE_PROFILES[package_profile]
    packages = [
        {
            "path": path,
            "size_bytes": size,
            "url": f"https://huggingface.co/datasets/wjh-svm/Griffin/resolve/main/{path}",
        }
        for path, size in DATA_PACKAGES[dataset]
        if selected_paths is None or path in selected_paths
    ]
    full_total = sum(size for _, size in DATA_PACKAGES[dataset])
    return {
        "dataset": dataset,
        "dataset_prefix": prefix,
        "package_profile": package_profile,
        "source": "https://huggingface.co/datasets/wjh-svm/Griffin",
        "package_count": len(packages),
        "full_package_count": len(DATA_PACKAGES[dataset]),
        "total_size_bytes": sum(item["size_bytes"] for item in packages),
        "full_total_size_bytes": full_total,
        "packages": packages,
    }


def data_download_script(dataset: str, package_profile: str = "smoke_25m_instance") -> str:
    package_payload = data_packages(dataset, package_profile)
    prefix = package_payload["dataset_prefix"]
    rows = "\n".join(
        f'  "{item["path"]}|{item["size_bytes"]}"'
        for item in package_payload["packages"]
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail

ROOT="${{GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}}"
BASE_URL="${{GRIFFIN_DATA_BASE_URL:-https://hf-mirror.com/datasets/wjh-svm/Griffin/resolve/main}}"
DATA_PARENT="$ROOT/griffin_repro/official/datasets"
DATA_ROOT="$ROOT/griffin_repro/official/datasets/{prefix}"
ARCHIVE_DIR="${{GRIFFIN_ARCHIVE_DIR:-$DATA_ROOT/archives}}"
PACKAGE_PROFILE="{package_profile}"
TOTAL_SIZE_BYTES={package_payload["total_size_bytes"]}
FULL_TOTAL_SIZE_BYTES={package_payload["full_total_size_bytes"]}
DOWNLOAD_JOBS="${{GRIFFIN_DOWNLOAD_JOBS:-3}}"
DOWNLOAD_MAX_PASSES="${{GRIFFIN_DOWNLOAD_MAX_PASSES:-12}}"
LOCK_FILE="$ARCHIVE_DIR/.download.lock"

if ! help wait 2>/dev/null | grep -q -- "-n"; then
  DOWNLOAD_JOBS=1
fi

mkdir -p "$ARCHIVE_DIR" "$DATA_PARENT"
exec 9>"$LOCK_FILE"
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    echo "Another Griffin data download is already active for $ARCHIVE_DIR." >&2
    exit 75
  fi
fi
cd "$ROOT"

packages=(
{rows}
)

download_one() {{
  local rel_path="$1"
  local expected_size="$2"
  local name
  local output
  local url
  local actual_size
  name="$(basename "$rel_path")"
  output="$ARCHIVE_DIR/$name"
  url="$BASE_URL/$rel_path"

  if [ -f "$output" ]; then
    actual_size="$(stat -c%s "$output")"
    if [ "$actual_size" = "$expected_size" ]; then
      echo "OK size: $name"
      return
    fi
    if [ "$actual_size" -gt "$expected_size" ]; then
      echo "$name is larger than expected; deleting corrupt partial archive before retry"
      rm -f "$output"
    fi
  fi

  echo "Downloading $name from $url"
  curl --retry 5 --connect-timeout 30 -L -C - -o "$output" "$url"
  actual_size="$(stat -c%s "$output")"
  if [ "$actual_size" != "$expected_size" ]; then
    echo "Size mismatch for $name: expected $expected_size, got $actual_size" >&2
    exit 4
  fi
}}

all_selected_complete() {{
  local item
  local path
  local expected_size
  local actual_size
  for item in "${{packages[@]}}"; do
    path="$ARCHIVE_DIR/$(basename "${{item%%|*}}")"
    expected_size="${{item##*|}}"
    if [ ! -f "$path" ]; then
      return 1
    fi
    actual_size="$(stat -c%s "$path")"
    if [ "$actual_size" != "$expected_size" ]; then
      return 1
    fi
  done
  return 0
}}

download_pass() {{
  local download_fail=0
  local active_jobs=0
  local item
  for item in "${{packages[@]}}"; do
    download_one "${{item%%|*}}" "${{item##*|}}" &
    active_jobs=$((active_jobs + 1))
    if [ "$active_jobs" -ge "$DOWNLOAD_JOBS" ]; then
      if ! wait -n; then
        download_fail=1
      fi
      active_jobs=$((active_jobs - 1))
    fi
  done

  while [ "$active_jobs" -gt 0 ]; do
    if ! wait -n; then
      download_fail=1
    fi
    active_jobs=$((active_jobs - 1))
  done

  return "$download_fail"
}}

for pass in $(seq 1 "$DOWNLOAD_MAX_PASSES"); do
  echo "Download pass $pass/$DOWNLOAD_MAX_PASSES for $PACKAGE_PROFILE package set"
  if download_pass && all_selected_complete; then
    break
  fi
  if [ "$pass" -lt "$DOWNLOAD_MAX_PASSES" ]; then
    echo "Download pass $pass did not complete all selected Griffin archives; retrying with resume."
    sleep 15
  fi
done

if ! all_selected_complete; then
  echo "Selected Griffin archive set is incomplete after $DOWNLOAD_MAX_PASSES pass(es)." >&2
  exit 4
fi

cd "$ARCHIVE_DIR"
rm -f md5.selected.txt
for item in "${{packages[@]}}"; do
  name="$(basename "${{item%%|*}}")"
  grep -F " ./$name" md5.txt >> md5.selected.txt
done
md5sum -c md5.selected.txt

mkdir -p "$DATA_ROOT"
for item in "${{packages[@]}}"; do
  archive="$(basename "${{item%%|*}}")"
  if [ "$archive" = "md5.txt" ]; then
    continue
  fi
  marker="$archive.extracted.to-data-parent"
  if [ -f "$marker" ]; then
    echo "Already extracted: $archive"
    continue
  fi
  echo "Extracting $archive"
  unzip -oq "$archive" -d "$DATA_PARENT"
  touch "$marker"
done

cd "$ROOT"
python scripts/griffin_repro.py check-partial-assets --profile smoke_25m_instance --json
"""


def write_data_script(dataset: str, out: str, package_profile: str = "smoke_25m_instance") -> dict[str, Any]:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(path, data_download_script(dataset, package_profile))
    payload = data_packages(dataset, package_profile)
    return {
        "dataset": dataset,
        "dataset_prefix": payload["dataset_prefix"],
        "package_profile": package_profile,
        "path": str(path),
        "bytes": path.stat().st_size,
        "total_size_bytes": payload["total_size_bytes"],
        "full_total_size_bytes": payload["full_total_size_bytes"],
    }


def check_data_packages(dataset: str, package_profile: str = "smoke_25m_instance") -> dict[str, Any]:
    payload = data_packages(dataset, package_profile)
    archive_dir = OFFICIAL_ROOT / "datasets" / payload["dataset_prefix"] / "archives"
    checks = []
    complete_size = 0
    for item in payload["packages"]:
        name = Path(item["path"]).name
        path = archive_dir / name
        actual_size = path.stat().st_size if path.exists() else 0
        size_delta = actual_size - item["size_bytes"]
        complete = actual_size == item["size_bytes"]
        if complete:
            complete_size += actual_size
        checks.append(
            {
                "path": item["path"],
                "archive": str(path.relative_to(REPO_ROOT)),
                "expected_size_bytes": item["size_bytes"],
                "actual_size_bytes": actual_size,
                "missing_size_bytes": max(item["size_bytes"] - actual_size, 0),
                "oversize_size_bytes": max(size_delta, 0),
                "size_delta_bytes": size_delta,
                "complete": complete,
            }
        )
    return {
        "dataset": dataset,
        "dataset_prefix": payload["dataset_prefix"],
        "package_profile": package_profile,
        "archive_dir": str(archive_dir.relative_to(REPO_ROOT)),
        "package_count": len(checks),
        "complete_count": sum(1 for item in checks if item["complete"]),
        "total_size_bytes": payload["total_size_bytes"],
        "complete_size_bytes": complete_size,
        "ready": all(item["complete"] for item in checks),
        "checks": checks,
    }


def paper_matrix() -> dict[str, Any]:
    manifest = load_manifest()
    rows = load_results()
    baseline = {(row["dataset"], row["methods"]) for row in rows if is_zero_condition(row)}
    expected_baseline = {(dataset, method) for dataset in DATASETS for method in FUSION_METHODS}
    return {
        "source": manifest["source"],
        "datasets": DATASETS,
        "fusion_methods": FUSION_METHODS,
        "metrics": METRICS,
        "robustness": ROBUSTNESS,
        "result_rows": len(rows),
        "baseline_rows": len(baseline),
        "baseline_complete": expected_baseline <= baseline,
        "missing_baselines": [
            {"dataset": dataset, "method": method}
            for dataset, method in sorted(expected_baseline - baseline)
        ],
    }


def config_path_exists(path: str | None) -> bool:
    return bool(path and (OFFICIAL_ROOT / path).exists())


def checkpoint_path_exists(path: str | None) -> bool:
    return bool(path and (OFFICIAL_ROOT / path).exists())


def first_existing_config(candidates: list[str]) -> str:
    for candidate in candidates:
        if config_path_exists(candidate):
            return candidate
    return candidates[0]


def base_config_for(dataset: str, method: str) -> str | None:
    prefix = dataset_prefix(dataset)
    if method == "0-no fusion":
        return first_existing_config(
            [
                f"projects/configs_{prefix}/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls.py",
                f"projects/configs_{prefix}/vehicle-side/tiny_track_r50_stream_bs8_24epoch_3cls.py",
            ]
        )
    if method == "1-early fusion":
        return first_existing_config(
            [
                f"projects/configs_{prefix}/early-fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py",
                f"projects/configs_{prefix}/early-fusion/tiny_track_r50_stream_bs8_24epoch_3cls.py",
            ]
        )
    if method == "2b1-cooptrack":
        return first_existing_config(
            [
                f"projects/configs_{prefix}/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py",
                f"projects/configs_{prefix}/cooperative/tiny_track_r50_stream_bs8_48epoch_3cls.py",
                f"projects/configs_{prefix}/cooperative/tiny_track_r50_stream_bs8_24epoch_3cls.py",
            ]
        )
    if method == "3-late fusion":
        return first_existing_config(
            [
                f"projects/configs_{prefix}/cooperative/late_fusion/tiny_track_r50_stream_bs1_3cls_late_fusion.py",
                f"projects/configs_{prefix}/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion.py",
            ]
        )
    return None


def checkpoint_for(dataset: str, method: str) -> str | None:
    prefix = dataset_prefix(dataset)
    iteration = CHECKPOINT_ITERS[dataset]
    if method == "0-no fusion":
        return f"ckpts/{prefix}/vehicle-side/{iteration}"
    if method == "1-early fusion":
        return f"ckpts/{prefix}/early-fusion/{iteration}"
    if method == "2b1-cooptrack":
        return f"ckpts/{prefix}/cooperative/instance_fusion/{iteration}"
    return None


def condition_from_row(row: dict[str, str]) -> tuple[str, dict[str, float]]:
    latency = float(row["communication latency"])
    packet_loss = float(row["packet loss"])
    translation = float(row["translation error"])
    rotation = float(row["rotation error"])
    if latency:
        return f"communication_latency_ms_{int(latency)}", {"communication_latency_ms": int(latency)}
    if packet_loss:
        return f"packet_loss_{packet_loss:g}", {"packet_loss": packet_loss}
    if translation:
        return f"translation_error_m_{translation:g}", {"translation_error_m": translation}
    if rotation:
        return f"rotation_error_deg_{int(rotation)}", {"rotation_error_deg": int(rotation)}
    return "baseline", {}


def robustness_config(base_config: str | None, method: str, condition: dict[str, float]) -> str | None:
    if not base_config or not condition:
        return base_config

    path = Path(base_config)
    stem = path.stem
    if "packet_loss" in condition:
        value = int(round(float(condition["packet_loss"]) * 100))
        folder = "drop_noised"
        suffix = f"drop{value}"
    elif "communication_latency_ms" in condition:
        value = int(condition["communication_latency_ms"])
        folder = "latency"
        suffix = f"{value}latency"
    elif "translation_error_m" in condition:
        value = int(round(float(condition["translation_error_m"]) * 10))
        folder = "loc_noised"
        suffix = f"loc{value:02d}"
    elif "rotation_error_deg" in condition:
        value = int(condition["rotation_error_deg"])
        folder = "orien_noised"
        suffix = f"orien{value * 10}" if method == "3-late fusion" else f"orien{value}"
    else:
        return base_config

    return str(path.with_name(folder) / f"{stem}_{suffix}.py").replace("\\", "/")


def late_fusion_track_config(config: str | None) -> str | None:
    if not config:
        return None
    path = Path(config)
    candidate = str(path.with_name(f"{path.stem}_ab3dmot.py")).replace("\\", "/")
    if config_path_exists(candidate):
        return candidate
    config_group = config.split("/")[1]
    dataset = config_group.replace("configs_griffin_", "", 1)
    base = base_config_for(dataset, "3-late fusion")
    if not base:
        return None
    base_path = Path(base)
    return str(base_path.with_name(f"{base_path.stem}_ab3dmot.py")).replace("\\", "/")


def command_for_row(dataset: str, method: str, config: str | None, checkpoint: str | None) -> tuple[str | None, str | None]:
    prefix = dataset_prefix(dataset)
    if not config:
        return None, None
    if method == "3-late fusion":
        track_config = late_fusion_track_config(config)
        veh_pkl = f"projects/work_dirs_{prefix}/vehicle-side/results.pkl"
        drone_pkl = f"projects/work_dirs_{prefix}/drone-side/results.pkl"
        return (
            "late_fusion_pipeline",
            "cd griffin_repro/official && "
            f"bash tools/eval_late_fusion.sh {veh_pkl} {drone_pkl} {config} {track_config}",
        )
    if checkpoint:
        return (
            "dist_eval",
            f"cd griffin_repro/official && {dist_eval_command(config, checkpoint, 1)}",
        )
    return None, None


def numeric_metrics(row: dict[str, str]) -> dict[str, float]:
    values = {}
    for key, value in row.items():
        if key in {"dataset", "methods"} or value == "":
            continue
        try:
            values[key] = float(value)
        except ValueError:
            continue
    return values


def paper_run_matrix(dataset: str | None = None, include_robustness: bool = False) -> dict[str, Any]:
    rows = load_results()
    emitted = []
    for row in rows:
        if dataset and row["dataset"] != dataset:
            continue
        condition_id, condition = condition_from_row(row)
        if condition and not include_robustness:
            continue

        method = row["methods"]
        base_config = base_config_for(row["dataset"], method)
        config = robustness_config(base_config, method, condition)
        checkpoint = checkpoint_for(row["dataset"], method)
        command_kind, command = command_for_row(row["dataset"], method, config, checkpoint)
        config_exists = config_path_exists(config)
        checkpoint_exists = checkpoint_path_exists(checkpoint)
        if method not in RUNNABLE_METHODS:
            status = "paper_result_only"
            config = None
            checkpoint = None
            command_kind = None
            command = None
            config_exists = False
            checkpoint_exists = False
        elif method == "3-late fusion" and config_exists:
            status = "pipeline_inputs_required"
        elif config_exists:
            status = "runnable_config"
        else:
            status = "missing_config"

        metrics = numeric_metrics(row)
        emitted.append(
            {
                "dataset": row["dataset"],
                "dataset_prefix": dataset_prefix(row["dataset"]),
                "method": method,
                "condition_id": condition_id,
                "condition": condition,
                "AP": metrics.get("AP"),
                "AMOTA": metrics.get("AMOTA"),
                "metrics": metrics,
                "config": config,
                "checkpoint": checkpoint,
                "command_kind": command_kind,
                "command": command,
                "config_exists": config_exists,
                "checkpoint_exists": checkpoint_exists,
                "status": status,
            }
        )

    baseline = [row for row in emitted if row["condition_id"] == "baseline"]
    return {
        "rows": emitted,
        "summary": {
            "result_rows": len(rows),
            "paper_result_rows": len(rows),
            "emitted_rows": len(emitted),
            "baseline_rows": len(baseline),
            "baseline_complete": len(baseline) == len(DATASETS) * len(FUSION_METHODS) if dataset is None else True,
            "runnable_config_rows": sum(row["status"] in {"runnable_config", "pipeline_inputs_required"} for row in emitted),
            "paper_result_only_rows": sum(row["status"] == "paper_result_only" for row in emitted),
        },
    }


def partial_run_plan(profile_name: str) -> dict[str, Any]:
    payload = profile_payload(profile_name)
    prefix = dataset_prefix(payload["dataset"])
    final_eval_command = payload["commands"][0].split("&&", 1)[1].strip()
    if payload["method"] == "0-no fusion":
        vehicle_preprocess_command = f"""python - <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "tools/griffin_data_converter"))
from tools.griffin_data_converter.trans_kitti2nuscenes import GriffinKittiToNuScenesConverter
from tools.griffin_data_converter.generate_nuscenes_pkl import create_nuscenes_infos

prefix = "{prefix}"
split_file = Path(f"data/split_datas/{{prefix}}.json")
with split_file.open("r", encoding="utf-8") as handle:
    split_info = json.load(handle)["batch_split"]
converter = GriffinKittiToNuScenesConverter(
    source_dir=f"datasets/{{prefix}}/griffin-release/vehicle-side",
    target_dir=f"datasets/{{prefix}}/griffin-nuscenes/vehicle-side",
    side="vehicle",
)
converter.convert({{}})
create_nuscenes_infos(
    f"datasets/{{prefix}}/griffin-nuscenes/vehicle-side",
    f"data/infos/{{prefix}}/vehicle-side",
    "griffin",
    "v1.0-trainval",
    side="vehicle",
    split_info=split_info,
)
PY"""
        commands = [
            "cd griffin_repro/official",
            vehicle_preprocess_command,
            final_eval_command,
        ]
        preprocess_assets = [
            f"griffin_repro/official/datasets/{prefix}/griffin-release/vehicle-side",
            f"griffin_repro/official/data/split_datas/{prefix}.json",
            f"griffin_repro/official/{payload['checkpoint']}",
        ]
    else:
        drone_checkpoint = f"ckpts/{prefix}/drone-side/{Path(payload['checkpoint']).name}"
        commands = [
            "cd griffin_repro/official",
            f"bash tools/griffin_converter.sh {prefix}",
            (
                dist_eval_command(
                    f"projects/configs_{prefix}/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval_train.py",
                    drone_checkpoint,
                    1,
                )
            ),
            (
                dist_eval_command(
                    f"projects/configs_{prefix}/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval.py",
                    drone_checkpoint,
                    1,
                )
            ),
            final_eval_command,
        ]
        preprocess_assets = [
            f"griffin_repro/official/datasets/{prefix}/griffin-release/vehicle-side",
            f"griffin_repro/official/datasets/{prefix}/griffin-release/drone-side",
            f"griffin_repro/official/data/split_datas/{prefix}.json",
            f"griffin_repro/official/{drone_checkpoint}",
            f"griffin_repro/official/{payload['checkpoint']}",
        ]
    evaluation_assets = [
        *payload["asset_checks"],
    ]
    return {
        "profile": profile_name,
        "dataset": payload["dataset"],
        "dataset_prefix": prefix,
        "method": payload["method"],
        "expected": payload["expected"],
        "commands": commands,
        "preprocess_assets": list(dict.fromkeys(preprocess_assets)),
        "evaluation_assets": list(dict.fromkeys(evaluation_assets)),
        "required_assets": list(dict.fromkeys([*preprocess_assets, *evaluation_assets])),
    }


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def _profile_v2x_side(profile: dict[str, Any]) -> str:
    config = profile["config"]
    for side in ("cooperative", "vehicle-side", "early-fusion", "drone-side"):
        if f"/{side}/" in config:
            return side
    raise SystemExit(f"Unable to infer Griffin v2x side from config: {config}")


def _config_path_literal(path: str) -> str:
    if path.startswith("/") or re.match(r"^[A-Za-z]:/", path):
        return path
    return f"./{path}"


def _select_infos_for_partial_eval(
    infos: list[dict[str, Any]],
    scene_limit: int,
    max_samples: int | None,
    samples_per_scene: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if scene_limit < 1:
        raise SystemExit("--scene-limit must be at least 1")
    if max_samples is not None and max_samples < 1:
        raise SystemExit("--max-samples must be at least 1 when provided")
    if samples_per_scene is not None and samples_per_scene < 1:
        raise SystemExit("--samples-per-scene must be at least 1 when provided")

    sorted_infos = sorted(infos, key=lambda item: item.get("timestamp", 0))
    selected_scenes = []
    for info in sorted_infos:
        scene_token = info.get("scene_token")
        if scene_token is None:
            continue
        if scene_token not in selected_scenes:
            selected_scenes.append(scene_token)
            if len(selected_scenes) == scene_limit:
                break
    selected_scene_set = set(selected_scenes)
    if samples_per_scene is None:
        selected_infos = [info for info in sorted_infos if info.get("scene_token") in selected_scene_set]
    else:
        grouped_infos: dict[str, list[dict[str, Any]]] = {scene: [] for scene in selected_scenes}
        for info in sorted_infos:
            scene_token = info.get("scene_token")
            if scene_token in grouped_infos and len(grouped_infos[scene_token]) < samples_per_scene:
                grouped_infos[scene_token].append(info)
        selected_infos = [
            info
            for scene in selected_scenes
            for info in grouped_infos.get(scene, [])
        ]
    if max_samples is not None:
        selected_infos = selected_infos[:max_samples]
    selected_scenes = list(dict.fromkeys(info["scene_token"] for info in selected_infos if "scene_token" in info))
    if not selected_infos:
        raise SystemExit("No Griffin infos matched the requested partial evaluation subset")
    return selected_infos, selected_scenes


def _annotation_class_names(info: dict[str, Any]) -> list[str]:
    names = []
    for key in ("gt_names", "gt_names_3d"):
        raw_names = info.get(key)
        if hasattr(raw_names, "tolist"):
            raw_names = raw_names.tolist()
        if isinstance(raw_names, (list, tuple)):
            names.extend(str(name) for name in raw_names)
    ann_infos = info.get("ann_infos")
    if isinstance(ann_infos, (list, tuple)) and len(ann_infos) >= 2:
        raw_names = ann_infos[1]
        if hasattr(raw_names, "tolist"):
            raw_names = raw_names.tolist()
        if isinstance(raw_names, (list, tuple)):
            names.extend(str(name) for name in raw_names)
    return names


def _describe_subset_infos(
    infos: list[dict[str, Any]],
    scene_limit: int,
    max_samples: int | None,
    samples_per_scene: int | None,
) -> dict[str, Any]:
    scene_frame_counts = Counter(str(info.get("scene_token")) for info in infos if info.get("scene_token") is not None)
    selected_infos, selected_scenes = _select_infos_for_partial_eval(
        infos,
        scene_limit,
        max_samples,
        samples_per_scene,
    )
    selected_scene_frame_counts = Counter(
        str(info.get("scene_token")) for info in selected_infos if info.get("scene_token") is not None
    )
    class_annotation_counts: Counter[str] = Counter()
    class_frame_presence: Counter[str] = Counter()
    frames_with_any_gt_names = 0
    for info in selected_infos:
        names = _annotation_class_names(info)
        if names:
            frames_with_any_gt_names += 1
        class_annotation_counts.update(names)
        class_frame_presence.update(set(names))
    return {
        "total_samples": len(infos),
        "scene_count": len(scene_frame_counts),
        "scene_frame_counts": dict(sorted(scene_frame_counts.items())),
        "selected_scene_count": len(selected_scenes),
        "selected_sample_count": len(selected_infos),
        "selected_scenes": selected_scenes,
        "selected_scene_frame_counts": dict(sorted(selected_scene_frame_counts.items())),
        "frames_with_any_gt_names": frames_with_any_gt_names,
        "class_annotation_counts": dict(sorted(class_annotation_counts.items())),
        "class_frame_presence": dict(sorted(class_frame_presence.items())),
    }


def describe_partial_subset(
    profile_name: str,
    scene_limit: int = 1,
    max_samples: int | None = None,
    samples_per_scene: int | None = None,
) -> dict[str, Any]:
    profile = profile_payload(profile_name)
    dataset = profile["dataset"]
    prefix = dataset_prefix(dataset)
    sides = {}
    for side in ("cooperative", "vehicle-side", "early-fusion", "drone-side"):
        ann_path = OFFICIAL_ROOT / "data" / "infos" / prefix / side / "griffin_infos_val.pkl"
        data = _load_pickle(ann_path)
        sides[side] = {
            "ann_file": _relative_posix(ann_path, OFFICIAL_ROOT),
            **_describe_subset_infos(
                list(data.get("infos", [])),
                scene_limit,
                max_samples,
                samples_per_scene,
            ),
        }
    return {
        "profile": profile_name,
        "dataset": dataset,
        "dataset_prefix": prefix,
        "paper_scene_count": DATASETS[dataset]["scene_count"],
        "scene_limit": scene_limit,
        "max_samples": max_samples,
        "samples_per_scene": samples_per_scene,
        "sides": sides,
    }


def prepare_partial_eval(
    profile_name: str,
    scene_limit: int = 1,
    max_samples: int | None = None,
    samples_per_scene: int | None = None,
    source_ann: str | None = None,
    out_ann: str | None = None,
    out_config: str | None = None,
    out_tag: str | None = None,
) -> dict[str, Any]:
    profile = profile_payload(profile_name)
    prefix = dataset_prefix(profile["dataset"])
    v2x_side = _profile_v2x_side(profile)
    source_ann_path = (
        Path(source_ann)
        if source_ann
        else OFFICIAL_ROOT / "data" / "infos" / prefix / v2x_side / "griffin_infos_val.pkl"
    )
    tag = out_tag or f"partial_{scene_limit}scene"
    out_ann_path = (
        Path(out_ann)
        if out_ann
        else source_ann_path.with_name(f"griffin_infos_val_{tag}.pkl")
    )
    base_config = OFFICIAL_ROOT / profile["config"]
    out_config_path = (
        Path(out_config)
        if out_config
        else base_config.with_name(f"{base_config.stem}_{tag}.py")
    )
    if not source_ann_path.exists():
        raise SystemExit(f"Missing source annotation file: {source_ann_path}")
    if not base_config.exists():
        raise SystemExit(f"Missing base Griffin config: {base_config}")

    with source_ann_path.open("rb") as handle:
        data = pickle.load(handle)
    selected_infos, selected_scenes = _select_infos_for_partial_eval(
        list(data.get("infos", [])),
        scene_limit,
        max_samples,
        samples_per_scene,
    )
    out_payload = dict(data)
    out_payload["infos"] = selected_infos
    out_ann_path.parent.mkdir(parents=True, exist_ok=True)
    with out_ann_path.open("wb") as handle:
        pickle.dump(out_payload, handle)

    base_ref = Path(os.path.relpath(base_config, out_config_path.parent)).as_posix()
    base_ref_literal = _config_path_literal(base_ref)
    ann_ref = _relative_posix(out_ann_path, OFFICIAL_ROOT)
    ann_ref_literal = _config_path_literal(ann_ref)
    config_text = f"""# Generated by scripts/griffin_repro.py prepare-partial-eval.
_base_ = '{base_ref_literal}'

ann_file_val = '{ann_ref_literal}'
data = dict(
    workers_per_gpu=0,
    val=dict(ann_file=ann_file_val),
    test=dict(ann_file=ann_file_val),
)
"""
    out_config_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(out_config_path, config_text)

    config_rel = _relative_posix(out_config_path, OFFICIAL_ROOT)
    ann_rel = _relative_posix(out_ann_path, OFFICIAL_ROOT)
    command = (
        "cd griffin_repro/official && "
        f"{dist_eval_command(config_rel, profile['checkpoint'], profile['gpus'])}"
    )
    return {
        "profile": profile_name,
        "dataset": profile["dataset"],
        "dataset_prefix": prefix,
        "method": profile["method"],
        "v2x_side": v2x_side,
        "expected": profile["expected"],
        "source_ann": _relative_posix(source_ann_path, OFFICIAL_ROOT),
        "ann_file": ann_rel,
        "config": config_rel,
        "checkpoint": profile["checkpoint"],
        "selected_scene_count": len(selected_scenes),
        "selected_sample_count": len(selected_infos),
        "selected_scenes": selected_scenes,
        "command": command,
    }


def _load_pickle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing Griffin annotation file: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_partial_config(base_config: Path, out_config: Path, ann_path: Path, generated_by: str) -> str:
    if not base_config.exists():
        raise SystemExit(f"Missing base Griffin config: {base_config}")
    base_ref = Path(os.path.relpath(base_config, out_config.parent)).as_posix()
    base_ref_literal = _config_path_literal(base_ref)
    ann_ref = _relative_posix(ann_path, OFFICIAL_ROOT)
    ann_ref_literal = _config_path_literal(ann_ref)
    config_text = f"""# Generated by scripts/griffin_repro.py {generated_by}.
_base_ = '{base_ref_literal}'

ann_file_val = '{ann_ref_literal}'
data = dict(
    workers_per_gpu=0,
    val=dict(ann_file=ann_file_val),
    test=dict(ann_file=ann_file_val),
)
"""
    out_config.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(out_config, config_text)
    return ann_ref


def _selected_cooperative_infos(
    profile_name: str,
    scene_limit: int,
    max_samples: int | None,
    samples_per_scene: int | None = None,
    source_ann: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], Path]:
    profile = profile_payload(profile_name)
    prefix = dataset_prefix(profile["dataset"])
    source_ann_path = (
        Path(source_ann)
        if source_ann
        else OFFICIAL_ROOT / "data" / "infos" / prefix / "cooperative" / "griffin_infos_val.pkl"
    )
    data = _load_pickle(source_ann_path)
    selected_infos, selected_scenes = _select_infos_for_partial_eval(
        list(data.get("infos", [])),
        scene_limit,
        max_samples,
        samples_per_scene,
    )
    return data, selected_infos, selected_scenes, source_ann_path


def prepare_drone_query_partial_eval(
    profile_name: str,
    scene_limit: int = 1,
    max_samples: int | None = None,
    samples_per_scene: int | None = None,
    source_ann: str | None = None,
    out_ann: str | None = None,
    out_config: str | None = None,
    out_tag: str | None = None,
) -> dict[str, Any]:
    profile = profile_payload(profile_name)
    prefix = dataset_prefix(profile["dataset"])
    source_data, selected_coop_infos, selected_scenes, source_ann_path = _selected_cooperative_infos(
        profile_name,
        scene_limit,
        max_samples,
        samples_per_scene,
        source_ann,
    )
    drone_ann_path = OFFICIAL_ROOT / "data" / "infos" / prefix / "drone-side" / "griffin_infos_val.pkl"
    drone_data = _load_pickle(drone_ann_path)
    drone_by_token = {info.get("token"): info for info in drone_data.get("infos", [])}
    selected_drone_infos = []
    missing_tokens = []
    for info in selected_coop_infos:
        air_token = info.get("air_sample_token")
        if air_token in drone_by_token:
            selected_drone_infos.append(drone_by_token[air_token])
        else:
            missing_tokens.append(air_token)
    if missing_tokens:
        raise SystemExit(f"Missing drone-side infos for air_sample_token values: {missing_tokens}")

    tag = out_tag or f"partial_{scene_limit}scene"
    out_ann_path = (
        Path(out_ann)
        if out_ann
        else drone_ann_path.with_name(f"griffin_infos_val_{tag}.pkl")
    )
    out_payload = dict(drone_data)
    out_payload["infos"] = selected_drone_infos
    out_ann_path.parent.mkdir(parents=True, exist_ok=True)
    with out_ann_path.open("wb") as handle:
        pickle.dump(out_payload, handle)

    base_config = OFFICIAL_ROOT / "projects" / f"configs_{prefix}" / "drone-side" / "tiny_track_r50_stream_bs8_24epoch_3cls_eval.py"
    out_config_path = (
        Path(out_config)
        if out_config
        else base_config.with_name(f"{base_config.stem}_{tag}.py")
    )
    ann_rel = _write_partial_config(
        base_config,
        out_config_path,
        out_ann_path,
        "prepare-drone-query-partial-eval",
    )
    config_rel = _relative_posix(out_config_path, OFFICIAL_ROOT)
    drone_checkpoint = f"ckpts/{prefix}/drone-side/{Path(profile['checkpoint']).name}"
    command = (
        "cd griffin_repro/official && "
        f"{dist_eval_command(config_rel, drone_checkpoint, profile['gpus'])}"
    )
    return {
        "profile": profile_name,
        "dataset": profile["dataset"],
        "dataset_prefix": prefix,
        "method": profile["method"],
        "v2x_side": "drone-side",
        "source_ann": _relative_posix(source_ann_path, OFFICIAL_ROOT),
        "cooperative_source_sample_count": len(source_data.get("infos", [])),
        "ann_file": ann_rel,
        "config": config_rel,
        "checkpoint": drone_checkpoint,
        "selected_scene_count": len(selected_scenes),
        "selected_sample_count": len(selected_drone_infos),
        "selected_scenes": selected_scenes,
        "command": command,
    }


def _archive_size(dataset: str, archive_name: str) -> int:
    for package_path, size in DATA_PACKAGES[dataset]:
        if Path(package_path).name == archive_name:
            return size
    raise SystemExit(f"Unknown Griffin archive {archive_name!r} for dataset {dataset!r}")


def _package_path(dataset: str, archive_name: str) -> str:
    for package_path, _ in DATA_PACKAGES[dataset]:
        if Path(package_path).name == archive_name:
            return package_path
    raise SystemExit(f"Unknown Griffin archive {archive_name!r} for dataset {dataset!r}")


def _frame_id_from_camera(camera: dict[str, Any]) -> str | None:
    data_path = str(camera.get("data_path") or camera.get("filename") or "")
    if not data_path:
        return None
    return Path(data_path).stem


def _infos_for_partial_images(
    profile_name: str,
    image_side: str,
    scene_limit: int,
    max_samples: int | None,
    samples_per_scene: int | None = None,
    source_ann: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], Path]:
    profile = profile_payload(profile_name)
    prefix = dataset_prefix(profile["dataset"])
    if image_side not in {"vehicle-side", "drone-side"}:
        raise SystemExit("--image-side must be vehicle-side or drone-side")
    if source_ann:
        source_ann_path = Path(source_ann)
        data = _load_pickle(source_ann_path)
        selected_infos, selected_scenes = _select_infos_for_partial_eval(
            list(data.get("infos", [])),
            scene_limit,
            max_samples,
            samples_per_scene,
        )
        return data, selected_infos, selected_scenes, source_ann_path
    if profile["method"] == "0-no fusion":
        source_ann_path = OFFICIAL_ROOT / "data" / "infos" / prefix / "vehicle-side" / "griffin_infos_val.pkl"
        data = _load_pickle(source_ann_path)
        selected_infos, selected_scenes = _select_infos_for_partial_eval(
            list(data.get("infos", [])),
            scene_limit,
            max_samples,
            samples_per_scene,
        )
        return data, selected_infos, selected_scenes, source_ann_path

    coop_data, selected_coop_infos, selected_scenes, coop_ann_path = _selected_cooperative_infos(
        profile_name,
        scene_limit,
        max_samples,
        samples_per_scene,
    )
    if image_side == "vehicle-side":
        return coop_data, selected_coop_infos, selected_scenes, coop_ann_path

    drone_ann_path = OFFICIAL_ROOT / "data" / "infos" / prefix / "drone-side" / "griffin_infos_val.pkl"
    drone_data = _load_pickle(drone_ann_path)
    drone_by_token = {info.get("token"): info for info in drone_data.get("infos", [])}
    selected_drone_infos = []
    missing_tokens = []
    for info in selected_coop_infos:
        air_token = info.get("air_sample_token")
        if air_token in drone_by_token:
            selected_drone_infos.append(drone_by_token[air_token])
        else:
            missing_tokens.append(air_token)
    if missing_tokens:
        raise SystemExit(f"Missing drone-side infos for air_sample_token values: {missing_tokens}")
    return drone_data, selected_drone_infos, selected_scenes, drone_ann_path


def partial_image_materialization_plan(
    profile_name: str,
    image_side: str,
    scene_limit: int = 1,
    max_samples: int | None = None,
    samples_per_scene: int | None = None,
    source_ann: str | None = None,
) -> dict[str, Any]:
    profile = profile_payload(profile_name)
    dataset = profile["dataset"]
    prefix = dataset_prefix(dataset)
    _, selected_infos, selected_scenes, source_ann_path = _infos_for_partial_images(
        profile_name,
        image_side,
        scene_limit,
        max_samples,
        samples_per_scene,
        source_ann,
    )
    archive_prefix = "drone" if image_side == "drone-side" else "vehicle"
    base_url = os.environ.get(
        "GRIFFIN_DATA_BASE_URL",
        "https://hf-mirror.com/datasets/wjh-svm/Griffin/resolve/main",
    ).rstrip("/")
    frames = []
    directions = []
    items = []
    for info in selected_infos:
        for cam_name, camera in sorted(info.get("cams", {}).items()):
            direction = CAMERA_DIRECTIONS.get(cam_name)
            frame = _frame_id_from_camera(camera)
            if not direction or not frame:
                continue
            archive = f"{archive_prefix}_camera_{direction}.zip"
            package_path = _package_path(dataset, archive)
            member = f"{prefix}/griffin-release/{image_side}/camera/{direction}/{frame}.png"
            dest = OFFICIAL_ROOT / "datasets" / prefix / "griffin-release" / image_side / "camera" / direction / f"{frame}.png"
            frames.append(frame)
            directions.append(direction)
            items.append(
                {
                    "direction": direction,
                    "frame": frame,
                    "archive": archive,
                    "package_path": package_path,
                    "archive_size_bytes": _archive_size(dataset, archive),
                    "url": f"{base_url}/{package_path}",
                    "member": member,
                    "dest": dest.as_posix(),
                }
            )
    items = list({f"{item['archive']}|{item['member']}": item for item in items}.values())
    return {
        "profile": profile_name,
        "dataset": dataset,
        "dataset_prefix": prefix,
        "image_side": image_side,
        "source_ann": _relative_posix(source_ann_path, OFFICIAL_ROOT),
        "selected_scene_count": len(selected_scenes),
        "selected_sample_count": len(selected_infos),
        "selected_scenes": selected_scenes,
        "frames": sorted(set(frames)),
        "directions": sorted(set(directions)),
        "items": items,
    }


def _curl_range(url: str, start: int, end: int) -> bytes:
    result = subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--http1.1",
            "--retry",
            "3",
            "--retry-all-errors",
            "--connect-timeout",
            "20",
            "--max-time",
            "180",
            "--range",
            f"{start}-{end}",
            url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def _zip64_central_directory(url: str, tail: bytes, eocd_at: int) -> tuple[int, int, int]:
    locator_at = tail.rfind(b"PK\x06\x07", 0, eocd_at)
    if locator_at < 0:
        raise SystemExit(f"ZIP64 central directory locator missing in {url}")
    locator = tail[locator_at : locator_at + 20]
    if len(locator) < 20:
        raise SystemExit(f"Truncated ZIP64 central directory locator in {url}")
    _, _, zip64_eocd_offset, _ = struct.unpack("<4sIQI", locator)

    record = _curl_range(url, zip64_eocd_offset, zip64_eocd_offset + 55)
    if len(record) < 56 or record[:4] != b"PK\x06\x06":
        raise SystemExit(f"Invalid ZIP64 central directory record in {url}")
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        total_entries,
        central_size,
        central_offset,
    ) = struct.unpack("<4sQ2H2I4Q", record[:56])
    return total_entries, central_size, central_offset


def _central_directory_from_url(url: str, archive_size: int) -> tuple[int, bytes]:
    cache_key = (url, archive_size)
    if cache_key in _ZIP_CENTRAL_DIRECTORY_CACHE:
        return _ZIP_CENTRAL_DIRECTORY_CACHE[cache_key]

    tail_start = max(0, archive_size - 66000)
    tail = _curl_range(url, tail_start, archive_size - 1)
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0:
        raise SystemExit(f"Unable to locate ZIP central directory in {url}")
    eocd = tail[eocd_at : eocd_at + 22]
    if len(eocd) < 22:
        raise SystemExit(f"Truncated ZIP end record in {url}")
    _, _, _, _, total_entries, central_size, central_offset, _ = struct.unpack("<4s4H2IH", eocd)
    if total_entries == 0xFFFF or central_size == 0xFFFFFFFF or central_offset == 0xFFFFFFFF:
        total_entries, central_size, central_offset = _zip64_central_directory(url, tail, eocd_at)
    central = _curl_range(url, central_offset, central_offset + central_size - 1)
    _ZIP_CENTRAL_DIRECTORY_CACHE[cache_key] = (total_entries, central)
    return total_entries, central


def _apply_zip64_entry_extra(
    extra: bytes,
    compressed_size: int,
    uncompressed_size: int,
    local_offset: int,
) -> tuple[int, int, int]:
    needs_uncompressed = uncompressed_size == 0xFFFFFFFF
    needs_compressed = compressed_size == 0xFFFFFFFF
    needs_offset = local_offset == 0xFFFFFFFF
    if not (needs_uncompressed or needs_compressed or needs_offset):
        return compressed_size, uncompressed_size, local_offset

    pos = 0
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack("<HH", extra[pos : pos + 4])
        data = extra[pos + 4 : pos + 4 + data_size]
        if header_id == 0x0001:
            cursor = 0
            if needs_uncompressed:
                uncompressed_size = struct.unpack("<Q", data[cursor : cursor + 8])[0]
                cursor += 8
            if needs_compressed:
                compressed_size = struct.unpack("<Q", data[cursor : cursor + 8])[0]
                cursor += 8
            if needs_offset:
                local_offset = struct.unpack("<Q", data[cursor : cursor + 8])[0]
            return compressed_size, uncompressed_size, local_offset
        pos += 4 + data_size
    raise SystemExit("ZIP64 entry metadata missing for large archive member")


def _zip_entry_from_url(url: str, archive_size: int, member: str) -> bytes:
    total_entries, central = _central_directory_from_url(url, archive_size)
    pos = 0
    target = None
    while pos + 46 <= len(central):
        header = central[pos : pos + 46]
        if header[:4] != b"PK\x01\x02":
            break
        (
            _,
            _,
            _,
            flags,
            method,
            _,
            _,
            crc,
            compressed_size,
            uncompressed_size,
            name_len,
            extra_len,
            comment_len,
            _,
            _,
            _,
            local_offset,
        ) = struct.unpack("<4s6H3I5H2I", header)
        name_start = pos + 46
        name = central[name_start : name_start + name_len].decode("utf-8")
        if name == member:
            extra = central[name_start + name_len : name_start + name_len + extra_len]
            compressed_size, uncompressed_size, local_offset = _apply_zip64_entry_extra(
                extra,
                compressed_size,
                uncompressed_size,
                local_offset,
            )
            target = {
                "flags": flags,
                "method": method,
                "crc": crc,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_offset": local_offset,
            }
            break
        pos = name_start + name_len + extra_len + comment_len
    if target is None:
        raise SystemExit(f"Missing {member} in {url} across {total_entries} central-directory entries")

    local_header = _curl_range(url, target["local_offset"], target["local_offset"] + 29)
    if local_header[:4] != b"PK\x03\x04":
        raise SystemExit(f"Invalid local ZIP header for {member}")
    _, _, _, method, _, _, _, _, _, local_name_len, local_extra_len = struct.unpack("<4s5H3I2H", local_header)
    if method != target["method"]:
        raise SystemExit(f"ZIP method mismatch for {member}")
    data_start = target["local_offset"] + 30 + local_name_len + local_extra_len
    compressed = _curl_range(url, data_start, data_start + target["compressed_size"] - 1)
    if target["method"] == 0:
        data = compressed
    elif target["method"] == 8:
        data = zlib.decompress(compressed, -15)
    else:
        raise SystemExit(f"Unsupported ZIP compression method {target['method']} for {member}")
    if len(data) != target["uncompressed_size"]:
        raise SystemExit(f"Size mismatch for {member}: expected {target['uncompressed_size']}, got {len(data)}")
    if zlib.crc32(data) & 0xFFFFFFFF != target["crc"]:
        raise SystemExit(f"CRC mismatch for {member}")
    return data


def _zip_entry_from_local_archive(archive_path: Path, member: str) -> bytes:
    import zipfile

    with zipfile.ZipFile(archive_path) as archive:
        return archive.read(member)


def _materialize_partial_image_item(plan: dict[str, Any], item: dict[str, Any]) -> None:
    dest = Path(item["dest"])
    archive_path = OFFICIAL_ROOT / "datasets" / plan["dataset_prefix"] / "archives" / item["archive"]
    if archive_path.exists() and archive_path.stat().st_size == item["archive_size_bytes"]:
        data = _zip_entry_from_local_archive(archive_path, item["member"])
    else:
        data = _zip_entry_from_url(item["url"], item["archive_size_bytes"], item["member"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)


def _materialize_jobs_from_env() -> int:
    raw_value = os.environ.get("GRIFFIN_MATERIALIZE_JOBS", "1")
    try:
        return max(1, int(raw_value))
    except ValueError:
        raise SystemExit(f"GRIFFIN_MATERIALIZE_JOBS must be an integer, got {raw_value!r}")


def materialize_partial_images(
    profile_name: str,
    image_side: str,
    scene_limit: int = 1,
    max_samples: int | None = None,
    samples_per_scene: int | None = None,
    source_ann: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    plan = partial_image_materialization_plan(
        profile_name,
        image_side,
        scene_limit,
        max_samples,
        samples_per_scene,
        source_ann,
    )
    written = 0
    skipped = 0
    if not dry_run:
        total = len(plan["items"])
        missing_items = []
        for index, item in enumerate(plan["items"], start=1):
            dest = Path(item["dest"])
            print(
                f"Materializing {image_side} images: {index}/{total} {dest.name}",
                file=sys.stderr,
                flush=True,
            )
            if dest.exists() and dest.stat().st_size > 0:
                skipped += 1
                continue
            missing_items.append(item)
        materialize_jobs = _materialize_jobs_from_env()
        print(
            f"Materializing {image_side} images: submitted {len(missing_items)}/{total} missing downloads with {materialize_jobs} job(s)",
            file=sys.stderr,
            flush=True,
        )
        if materialize_jobs == 1:
            for item in missing_items:
                _materialize_partial_image_item(plan, item)
                written += 1
                print(
                    f"Materializing {image_side} images: completed {written}/{len(missing_items)}",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            remote_archives = {}
            for item in missing_items:
                archive_path = OFFICIAL_ROOT / "datasets" / plan["dataset_prefix"] / "archives" / item["archive"]
                if not (archive_path.exists() and archive_path.stat().st_size == item["archive_size_bytes"]):
                    remote_archives[(item["url"], item["archive_size_bytes"])] = None
            for url, archive_size in remote_archives:
                _central_directory_from_url(url, archive_size)
            with concurrent.futures.ThreadPoolExecutor(max_workers=materialize_jobs) as executor:
                future_to_item = {
                    executor.submit(_materialize_partial_image_item, plan, item): item
                    for item in missing_items
                }
                for future in concurrent.futures.as_completed(future_to_item):
                    future.result()
                    written += 1
                    print(
                        f"Materializing {image_side} images: completed {written}/{len(missing_items)}",
                        file=sys.stderr,
                        flush=True,
                    )
    return {
        **plan,
        "dry_run": dry_run,
        "planned_items": len(plan["items"]),
        "written": written,
        "skipped_existing": skipped,
        "materialize_jobs": _materialize_jobs_from_env(),
    }


def mobaxterm_script(profile_name: str) -> str:
    plan = partial_run_plan(profile_name)
    preprocess_assets = "\n".join(f'  "{path}"' for path in plan["preprocess_assets"])
    evaluation_assets = "\n".join(f'  "{path}"' for path in plan["evaluation_assets"])
    commands = plan["commands"]
    if len(commands) == 3:
        _, convert_command, final_eval_command = commands
        drone_train_command = ""
        drone_val_command = ""
    else:
        _, convert_command, drone_train_command, drone_val_command, final_eval_command = commands
    activation = conda_activation_block().rstrip()
    partial_image_commands = f'''  python scripts/griffin_repro.py materialize-partial-images --profile {profile_name} --image-side vehicle-side "${{partial_args[@]}}" --json | tee "$LOG_DIR/{profile_name}_vehicle_partial_images.json"'''
    partial_drone_query_commands = ""
    full_drone_commands = "  :"
    if drone_val_command:
        partial_image_commands += f'''
  python scripts/griffin_repro.py materialize-partial-images --profile {profile_name} --image-side drone-side "${{partial_args[@]}}" --json | tee "$LOG_DIR/{profile_name}_drone_partial_images.json"'''
        partial_drone_query_commands = f'''
  drone_query_json="$LOG_DIR/{profile_name}_drone_query_partial_eval.json"
  python scripts/griffin_repro.py prepare-drone-query-partial-eval --profile {profile_name} "${{partial_args[@]}}" --json | tee "$drone_query_json"
  drone_query_eval_command=$(python - "$drone_query_json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(payload["command"].split("&&", 1)[1].strip())
PY
)
  cd griffin_repro/official
  eval "$drone_query_eval_command"
  cd "$ROOT"'''
        full_drone_commands = f"""
  cd griffin_repro/official
  {drone_train_command}
  {drone_val_command}
  cd ../.."""
    return f"""#!/usr/bin/env bash
set -euo pipefail

ROOT="${{GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}}"
cd "$ROOT"
{activation}
python scripts/griffin_repro.py env-check --strict --json
set +e
spconv_error=$(python - <<'PY' 2>&1
import mmdet3d.ops.spconv
PY
)
spconv_status=$?
set -e
if [ "$spconv_status" -ne 0 ]; then
  printf '%s\\n' "$spconv_error" >&2
  if printf '%s' "$spconv_error" | grep -q 'sparse_conv_ext'; then
    bash griffin_repro/build_mmdet3d_spconv_ext_mobaxterm.sh
  else
    exit "$spconv_status"
  fi
fi
LOG_DIR="${{GRIFFIN_SMOKE_LOG_DIR:-$ROOT/griffin_repro/artifacts/logs}}"
mkdir -p "$LOG_DIR"

check_assets() {{
  local group_name="$1"
  shift
  missing_assets=0
  for asset in "$@"; do
    if [ ! -e "$asset" ]; then
      echo "MISSING ($group_name): $asset" >&2
      missing_assets=1
    fi
  done
  if [ "$missing_assets" -ne 0 ]; then
    echo "Install the listed Griffin assets before running $group_name." >&2
    exit 2
  fi
}}

preprocess_assets=(
{preprocess_assets}
)
check_assets "preprocess" "${{preprocess_assets[@]}}"

partial_scene_limit="${{GRIFFIN_PARTIAL_SCENE_LIMIT:-1}}"
partial_max_samples="${{GRIFFIN_PARTIAL_MAX_SAMPLES:-20}}"
partial_samples_per_scene="${{GRIFFIN_PARTIAL_SAMPLES_PER_SCENE:-}}"
partial_metric_tolerance="${{GRIFFIN_PARTIAL_METRIC_TOLERANCE:-1.0}}"
skip_converter="${{GRIFFIN_SKIP_CONVERTER:-0}}"
partial_args=()
if [ "$partial_scene_limit" -gt 0 ]; then
  partial_args=(--scene-limit "$partial_scene_limit" --out-tag "partial_${{partial_scene_limit}}scene")
  if [ -n "$partial_samples_per_scene" ]; then
    partial_args+=(--samples-per-scene "$partial_samples_per_scene" --out-tag "partial_${{partial_scene_limit}}scene_${{partial_samples_per_scene}}per_scene")
  elif [ -n "$partial_max_samples" ]; then
    partial_args+=(--max-samples "$partial_max_samples" --out-tag "partial_${{partial_scene_limit}}scene_${{partial_max_samples}}samples")
  fi
fi

evaluation_assets=(
{evaluation_assets}
)

cd griffin_repro/official
if [ "$skip_converter" = "1" ]; then
  cd ../..
  check_assets "converted data" "${{evaluation_assets[@]}}"
  echo "Skipping Griffin converter because GRIFFIN_SKIP_CONVERTER=1"
  cd griffin_repro/official
else
  {convert_command}
fi
cd ../..

if [ "$partial_scene_limit" -gt 0 ]; then
{partial_image_commands}
{partial_drone_query_commands}
else
{full_drone_commands}
fi

check_assets "evaluation" "${{evaluation_assets[@]}}"

if [ "$partial_scene_limit" -gt 0 ]; then
  partial_json="$LOG_DIR/{profile_name}_partial_eval.json"
  python scripts/griffin_repro.py prepare-partial-eval --profile {profile_name} "${{partial_args[@]}}" --json | tee "$partial_json"
  partial_eval_command=$(python - "$partial_json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(payload["command"].split("&&", 1)[1].strip())
PY
)
  final_eval_to_run="$partial_eval_command"
  validation_tolerance="$partial_metric_tolerance"
else
  final_eval_to_run="{final_eval_command}"
  validation_tolerance="0.02"
fi

cd griffin_repro/official
eval "$final_eval_to_run"
latest_log=$(find projects -path '*/logs/test_*.log' -type f -printf '%T@ %p\\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
if [ -z "$latest_log" ]; then
  echo "No Griffin eval log found under griffin_repro/official/projects." >&2
  exit 3
fi
cd "$ROOT"
python scripts/griffin_repro.py validate-run --profile {profile_name} --log "griffin_repro/official/${{latest_log}}" --tolerance "$validation_tolerance" --json
"""


def write_mobaxterm_script(profile_name: str, out: str) -> dict[str, Any]:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(path, mobaxterm_script(profile_name))
    return {"profile": profile_name, "path": str(path), "bytes": path.stat().st_size}


def data_script_name(dataset: str, package_profile: str) -> str:
    if package_profile == "smoke_25m_instance":
        return f"download_{dataset}_mobaxterm.sh"
    prefix = "smoke_25m_"
    suffix = package_profile[len(prefix) :] if package_profile.startswith(prefix) else package_profile
    return f"download_{dataset}_{suffix}_mobaxterm.sh"


def run_script_name(profile_name: str) -> str:
    return f"run_{profile_name}_mobaxterm.sh"


def supervisor_script(profile_name: str, dataset: str, package_profile: str) -> str:
    profile_payload(profile_name)
    data_packages(dataset, package_profile)
    activation = conda_activation_block().rstrip()
    download_script = data_script_name(dataset, package_profile)
    smoke_script = run_script_name(profile_name)
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd "${{GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}}"
{activation}

LOG_DIR="${{GRIFFIN_SUPERVISOR_LOG_DIR:-griffin_repro/artifacts/logs}}"
SUPERVISOR_SLEEP_SEC="${{GRIFFIN_SUPERVISOR_SLEEP_SEC:-300}}"
SUPERVISOR_MAX_ATTEMPTS="${{GRIFFIN_SUPERVISOR_MAX_ATTEMPTS:-0}}"
mkdir -p "$LOG_DIR"

data_status="$LOG_DIR/{profile_name}_supervisor_data_status.json"
latest_log="$LOG_DIR/{profile_name}_supervisor.latest"

check_data_ready() {{
  python scripts/griffin_repro.py check-data-packages --dataset {dataset} --package-profile {package_profile} --json > "$data_status"
  python - "$data_status" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
missing = sum(item["missing_size_bytes"] for item in payload["checks"] if not item["complete"])
oversize = sum(item.get("oversize_size_bytes", 0) for item in payload["checks"] if not item["complete"])
print(
    f"data_ready={{payload['ready']}} complete={{payload['complete_count']}}/{{payload['package_count']}} "
    f"missing={{missing}} oversize={{oversize}}",
    flush=True,
)
raise SystemExit(0 if payload["ready"] else 1)
PY
}}

cleanup_stale_downloads() {{
  pkill -f "curl .*griffin_50scenes_25m/archives" 2>/dev/null || true
  pkill -f "bash griffin_repro/{download_script}" 2>/dev/null || true
}}

attempt=1
while true; do
  echo "[$(date -Is)] Griffin supervisor attempt $attempt"
  if check_data_ready; then
    break
  fi

  if [ "$SUPERVISOR_MAX_ATTEMPTS" -gt 0 ] && [ "$attempt" -gt "$SUPERVISOR_MAX_ATTEMPTS" ]; then
    echo "Griffin data is still incomplete after $SUPERVISOR_MAX_ATTEMPTS supervisor attempts." >&2
    exit 4
  fi

  cleanup_stale_downloads
  set +e
  bash griffin_repro/{download_script}
  download_status=$?
  set -e
  if [ "$download_status" -ne 0 ]; then
    echo "Download script exited with $download_status; retrying after $SUPERVISOR_SLEEP_SEC seconds." >&2
  fi
  attempt=$((attempt + 1))
  sleep "$SUPERVISOR_SLEEP_SEC"
done

smoke_log="$LOG_DIR/{profile_name}_supervisor_$(date +%Y%m%d_%H%M%S).log"
echo "$smoke_log" > "$latest_log"
bash griffin_repro/{smoke_script} 2>&1 | tee "$smoke_log"
"""


def write_supervisor_script(profile_name: str, dataset: str, package_profile: str, out: str) -> dict[str, Any]:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(path, supervisor_script(profile_name, dataset, package_profile))
    return {
        "profile": profile_name,
        "dataset": dataset,
        "package_profile": package_profile,
        "path": str(path),
        "bytes": path.stat().st_size,
    }


def check_partial_assets(profile_name: str) -> dict[str, Any]:
    plan = partial_run_plan(profile_name)
    stages = []
    for stage_name, key in (("preprocess", "preprocess_assets"), ("evaluation", "evaluation_assets")):
        stage = check_path_list(plan[key])
        stage["stage"] = stage_name
        stages.append(stage)
    return {
        "profile": profile_name,
        "dataset": plan["dataset"],
        "method": plan["method"],
        "stages": stages,
        "ready": all(stage["ready"] for stage in stages),
    }


def _metric_number(value: str) -> float:
    return float(value)


def parse_paper_class_metrics(text: str, class_name: str = "car") -> dict[str, float]:
    lines = text.splitlines()
    metrics = {}
    for index, line in enumerate(lines):
        if re.match(r"^\s*Object Class\s+AP\b", line):
            for row in lines[index + 1 :]:
                if not row.strip() or row.strip().startswith("="):
                    break
                parts = row.split()
                if parts and parts[0] == class_name and len(parts) > 1:
                    metrics["AP"] = _metric_number(parts[1])
                    break
        if re.search(r"\bAMOTA\b", line) and re.search(r"\bGT\b", line):
            for row in lines[index + 1 :]:
                if not row.strip() or row.strip().startswith("="):
                    break
                parts = row.split()
                if parts and parts[0] == class_name and len(parts) > 1:
                    metrics["AMOTA"] = _metric_number(parts[1])
                    break
    return metrics


def parse_run_metrics(text: str, metric_scope: str = "aggregate") -> dict[str, float]:
    if metric_scope == "paper":
        return parse_paper_class_metrics(text)

    patterns = {
        "AP": [
            r"(?:pts_bbox_NuScenes/)?mAP\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[\s\"'])AP[\"']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        ],
        "AMOTA": [
            r"(?:pts_bbox_NuScenes/)?amota\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[\s\"'])AMOTA[\"']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            r"^AMOTA\s+([0-9]+(?:\.[0-9]+)?)\s*$",
        ],
    }
    metrics = {}
    for name, metric_patterns in patterns.items():
        for pattern in metric_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                metrics[name] = float(match.group(1))
                break
    return metrics


def _flat_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return [value]
    flattened: list[Any] = []
    for item in value:
        if isinstance(item, list):
            flattened.extend(_flat_values(item))
        else:
            flattened.append(item)
    return flattened


def _threshold_bins(values: list[float], thresholds: list[float]) -> dict[str, int]:
    return {
        "all": len(values),
        **{f">={threshold:g}": sum(1 for value in values if value >= threshold) for threshold in thresholds},
    }


def _track_id_summary(samples: list[dict[str, Any]], id_key: str = "track_ids") -> dict[str, Any]:
    lifetimes: Counter[int] = Counter()
    negative_ids = 0
    duplicate_id_frames = 0
    present_frames = 0
    missing_key_frames = 0
    ids_per_frame = _running_int_stat()

    for sample in samples:
        payload = sample.get("pts_bbox", sample) if isinstance(sample, dict) else {}
        if id_key not in payload:
            missing_key_frames += 1
            continue
        present_frames += 1
        ids = [int(track_id) for track_id in _flat_values(payload.get(id_key))]
        valid_ids = [track_id for track_id in ids if track_id >= 0]
        negative_ids += sum(1 for track_id in ids if track_id < 0)
        _add_int_stat(ids_per_frame, len(valid_ids))
        if len(valid_ids) != len(set(valid_ids)):
            duplicate_id_frames += 1
        lifetimes.update(valid_ids)

    _finish_int_stat(ids_per_frame)
    lifetime_stats = _running_int_stat()
    for value in lifetimes.values():
        _add_int_stat(lifetime_stats, value)
    _finish_int_stat(lifetime_stats)
    return {
        "present_frames": present_frames,
        "missing_key_frames": missing_key_frames,
        "negative_ids": negative_ids,
        "duplicate_id_frames": duplicate_id_frames,
        "unique_id_count": len(lifetimes),
        "ids_per_frame": ids_per_frame,
        "id_lifetime_frames": lifetime_stats,
    }


def _prediction_set_summary(
    samples: list[dict[str, Any]],
    label_key: str,
    score_key: str,
    id_key: str | None = None,
) -> dict[str, Any]:
    class_names = ["car", "bicycle", "pedestrian"]
    thresholds = [0.05, 0.1, 0.3, 0.35, 0.4, 0.5, 0.7, 0.9]
    classes = {
        name: {
            "count": 0,
            "frames": 0,
            "score_bins": {"all": 0, **{f">={threshold:g}": 0 for threshold in thresholds}},
            "mean_score": None,
            "max_score": None,
        }
        for name in class_names
    }
    score_lists = {name: [] for name in class_names}
    missing_key_frames = 0
    empty_frames = 0
    total_predictions = 0

    for sample in samples:
        payload = sample.get("pts_bbox", sample) if isinstance(sample, dict) else {}
        if label_key not in payload:
            missing_key_frames += 1
            continue
        labels = [int(label) for label in _flat_values(payload.get(label_key))]
        scores = [float(score) for score in _flat_values(payload.get(score_key))]
        if len(scores) != len(labels):
            scores = [1.0] * len(labels)
        if not labels:
            empty_frames += 1
        total_predictions += len(labels)

        frame_labels = set(labels)
        for class_id, class_name in enumerate(class_names):
            if class_id in frame_labels:
                classes[class_name]["frames"] += 1
            for label, score in zip(labels, scores):
                if label != class_id:
                    continue
                classes[class_name]["count"] += 1
                classes[class_name]["score_bins"]["all"] += 1
                score_lists[class_name].append(score)
                for threshold in thresholds:
                    if score >= threshold:
                        classes[class_name]["score_bins"][f">={threshold:g}"] += 1

    for class_name, scores in score_lists.items():
        if not scores:
            continue
        classes[class_name]["mean_score"] = round(sum(scores) / len(scores), 4)
        classes[class_name]["max_score"] = round(max(scores), 4)

    summary = {
        "total_predictions": total_predictions,
        "empty_frames": empty_frames,
        "missing_key_frames": missing_key_frames,
        "classes": classes,
    }
    if id_key:
        summary["track_ids"] = _track_id_summary(samples, id_key)
    return summary


def analyze_result_pkl(path: str) -> dict[str, Any]:
    result_path = Path(path)
    payload = _load_pickle(result_path)
    samples = payload.get("bbox_results", payload.get("outputs", payload))
    if not isinstance(samples, list):
        raise ValueError(f"Unsupported result pkl structure in {result_path}")
    return {
        "path": str(result_path),
        "samples": len(samples),
        "prediction_sets": {
            "tracking": _prediction_set_summary(samples, "labels_3d", "scores_3d", "track_ids"),
            "detection": _prediction_set_summary(samples, "labels_3d_det", "scores_3d_det"),
        },
    }


def _object_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    if hasattr(obj, "get"):
        try:
            return obj.get(key)
        except Exception:
            pass
    return getattr(obj, key, None)


def _shape_of(value: Any) -> list[int] | None:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return [int(dim) for dim in value.shape]
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if not value:
            return [0]
        child_shapes = [_shape_of(item) for item in value]
        first = child_shapes[0]
        if first is not None and all(shape == first for shape in child_shapes):
            return [len(value), *first]
        return [len(value)]
    return []


def _flat_numbers(value: Any) -> list[float]:
    numbers = []
    for item in _flat_values(value):
        try:
            numbers.append(float(item))
        except (TypeError, ValueError):
            continue
    return numbers


def _running_int_stat() -> dict[str, Any]:
    return {"count": 0, "max": None, "mean": None, "min": None, "sum": 0}


def _add_int_stat(stats: dict[str, Any], value: int) -> None:
    stats["count"] += 1
    stats["sum"] += value
    stats["min"] = value if stats["min"] is None else min(stats["min"], value)
    stats["max"] = value if stats["max"] is None else max(stats["max"], value)


def _finish_int_stat(stats: dict[str, Any]) -> None:
    if stats["count"]:
        stats["mean"] = round(stats["sum"] / stats["count"], 4)


def _annotation_tokens(ann_file: str | None) -> tuple[list[str], int | None]:
    if not ann_file:
        return [], None
    payload = _load_pickle(Path(ann_file))
    infos = payload if isinstance(payload, list) else payload.get("infos", [])
    tokens = [
        str(info.get("air_sample_token", info.get("token")))
        for info in infos
        if isinstance(info, dict) and info.get("air_sample_token", info.get("token")) is not None
    ]
    return tokens, len(infos)


def analyze_track_query_cache(
    query_dir: str,
    ann_file: str | None = None,
    keys: list[str] | None = None,
    sample_limit: int = 3,
) -> dict[str, Any]:
    official_root = str(OFFICIAL_ROOT)
    if official_root not in sys.path:
        sys.path.insert(0, official_root)
    root = Path(query_dir)
    files = sorted(root.glob("*.pkl"))
    expected_tokens, ann_samples = _annotation_tokens(ann_file)
    expected_set = set(expected_tokens)
    file_stems = {path.stem for path in files}
    keys = keys or ["query_feats", "query_embeds", "obj_idxes", "ref_pts", "scores"]
    summary: dict[str, Any] = {
        "query_dir": str(root),
        "ann_file": ann_file,
        "ann_samples": ann_samples,
        "track_query_files": len(files),
        "expected_coverage": sum(1 for token in expected_tokens if token in file_stems),
        "missing_expected": [token for token in expected_tokens if token not in file_stems][:20],
        "extra_files": sorted(file_stems - expected_set)[:20] if expected_tokens else [],
        "keys": {key: {"present_files": 0, "shapes": {}} for key in keys},
        "rows": _running_int_stat(),
        "valid_obj_idx_ge0": _running_int_stat(),
        "active_score_bins": _threshold_bins([], [0.05, 0.1, 0.3, 0.35, 0.4, 0.5]),
        "query_timing": {
            "expected_sequence_count": len(expected_tokens),
            "first_missing_index": None,
            "missing_expected_count": 0,
            "extra_file_count": len(file_stems - expected_set) if expected_tokens else 0,
        },
        "nan_or_inf_files": [],
        "ref_pts_outside_0_1_files": [],
        "samples": [],
    }
    active_scores: list[float] = []
    if expected_tokens:
        missing_indexes = [index for index, token in enumerate(expected_tokens) if token not in file_stems]
        summary["query_timing"]["missing_expected_count"] = len(missing_indexes)
        summary["query_timing"]["first_missing_index"] = missing_indexes[0] if missing_indexes else None

    for index, path in enumerate(files):
        payload = _load_pickle(path)
        sample = {"file": path.name, "type": type(payload).__name__} if index < sample_limit else None
        obj_numbers = _flat_numbers(_object_value(payload, "obj_idxes"))
        score_numbers = _flat_numbers(_object_value(payload, "scores"))
        if len(score_numbers) == len(obj_numbers):
            active_scores.extend(score for obj_idx, score in zip(obj_numbers, score_numbers) if obj_idx >= 0)
        for key in keys:
            value = _object_value(payload, key)
            shape = _shape_of(value)
            if sample is not None:
                sample[key] = shape
            if value is None:
                continue
            key_summary = summary["keys"][key]
            key_summary["present_files"] += 1
            shape_key = str(shape)
            key_summary["shapes"][shape_key] = key_summary["shapes"].get(shape_key, 0) + 1
            numbers = _flat_numbers(value)
            if any(not math.isfinite(number) for number in numbers):
                summary["nan_or_inf_files"].append(path.name)
            if key == "query_feats" and shape:
                _add_int_stat(summary["rows"], int(shape[0]))
            if key == "obj_idxes":
                _add_int_stat(summary["valid_obj_idx_ge0"], sum(1 for number in numbers if number >= 0))
            if key == "ref_pts" and any(number < 0 or number > 1 for number in numbers):
                summary["ref_pts_outside_0_1_files"].append(path.name)
        if sample is not None:
            summary["samples"].append(sample)

    _finish_int_stat(summary["rows"])
    _finish_int_stat(summary["valid_obj_idx_ge0"])
    summary["active_score_bins"] = _threshold_bins(active_scores, [0.05, 0.1, 0.3, 0.35, 0.4, 0.5])
    summary["nan_or_inf_files"] = summary["nan_or_inf_files"][:20]
    summary["ref_pts_outside_0_1_files"] = summary["ref_pts_outside_0_1_files"][:20]
    return summary


def validate_run(profile_name: str, log_path: str, tolerance: float, metric_scope: str = "aggregate") -> dict[str, Any]:
    payload = profile_payload(profile_name)
    path = Path(log_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    metrics = parse_run_metrics(text, metric_scope)
    checks = {}
    missing = []
    for name, expected in payload["expected"].items():
        if name not in metrics:
            missing.append(name)
            continue
        actual = metrics[name]
        delta = actual - expected
        checks[name] = {
            "actual": actual,
            "expected": expected,
            "delta": delta,
            "abs_delta": abs(delta),
            "tolerance": tolerance,
            "passed": abs(delta) <= tolerance,
        }
    passed = not missing and all(item["passed"] for item in checks.values())
    return {
        "profile": profile_name,
        "dataset": payload["dataset"],
        "method": payload["method"],
        "log": str(path),
        "metric_scope": metric_scope,
        "metrics": metrics,
        "checks": checks,
        "missing_metrics": missing,
        "passed": passed,
    }


def baseline_expected_metrics(dataset: str, method: str) -> dict[str, float]:
    return paper_expected_metrics(dataset, method, "baseline")


def paper_expected_metrics(dataset: str, method: str, condition_id: str = "baseline") -> dict[str, float]:
    for row in load_results():
        row_condition_id, _ = condition_from_row(row)
        if row["dataset"] == dataset and row["methods"] == method and row_condition_id == condition_id:
            metrics = numeric_metrics(row)
            return {"AP": metrics["AP"], "AMOTA": metrics["AMOTA"]}
    raise SystemExit(
        f"No paper metrics found for dataset={dataset!r}, method={method!r}, condition_id={condition_id!r}"
    )


def paper_row_metrics(dataset: str, method: str, condition_id: str = "baseline") -> dict[str, float]:
    for row in load_results():
        row_condition_id, _ = condition_from_row(row)
        if row["dataset"] == dataset and row["methods"] == method and row_condition_id == condition_id:
            return numeric_metrics(row)
    raise SystemExit(
        f"No paper row found for dataset={dataset!r}, method={method!r}, condition_id={condition_id!r}"
    )


def metric_validation_entry(
    dataset: str,
    method: str,
    metrics: dict[str, float],
    tolerance: float,
    profile: str = "log_section",
    metric_scope: str = "aggregate",
    log: str | None = None,
    condition_id: str = "baseline",
) -> dict[str, Any]:
    expected_metrics = paper_expected_metrics(dataset, method, condition_id)
    checks = {}
    missing = []
    for name, expected in expected_metrics.items():
        if name not in metrics:
            missing.append(name)
            continue
        actual = metrics[name]
        delta = actual - expected
        checks[name] = {
            "actual": actual,
            "expected": expected,
            "delta": delta,
            "abs_delta": abs(delta),
            "tolerance": tolerance,
            "passed": abs(delta) <= tolerance,
        }
    entry = {
        "profile": profile,
        "dataset": dataset,
        "method": method,
        "condition_id": condition_id,
        "metric_scope": metric_scope,
        "metrics": metrics,
        "checks": checks,
        "missing_metrics": missing,
        "passed": not missing and all(item["passed"] for item in checks.values()),
    }
    if log:
        entry["log"] = log
    return entry


def _summary_file(eval_dir: Path, relative: str) -> Path:
    direct = eval_dir / relative
    if direct.exists():
        return direct
    matches = sorted(eval_dir.glob(f"*/{relative}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Missing {relative} under {eval_dir}")
    raise ValueError(f"Multiple {relative} files under {eval_dir}; pass the exact run directory")


def _class_value(mapping: dict[str, Any], metric: str, class_name: str) -> float:
    return float(mapping[metric][class_name])


def summarize_eval_json(
    eval_dir: str,
    dataset: str,
    method: str,
    paper_tolerance: float = 0.02,
    class_name: str = "car",
    condition_id: str = "baseline",
) -> dict[str, Any]:
    root = Path(eval_dir)
    det_path = _summary_file(root, Path("det/metrics_summary.json").as_posix())
    track_path = _summary_file(root, Path("track/metrics_summary.json").as_posix())
    det = json.loads(det_path.read_text(encoding="utf-8"))
    track = json.loads(track_path.read_text(encoding="utf-8"))
    label_metrics = track["label_metrics"]
    metrics = {
        "AP": float(det["mean_dist_aps"][class_name]),
        "AMOTA": _class_value(label_metrics, "amota", class_name),
    }
    for output_name, metric_name in (
        ("GT", "gt"),
        ("TP", "tp"),
        ("FP", "fp"),
        ("FN", "fn"),
        ("IDS", "ids"),
        ("FRAG", "frag"),
    ):
        if metric_name in label_metrics and class_name in label_metrics[metric_name]:
            metrics[output_name] = _class_value(label_metrics, metric_name, class_name)
    entry = metric_validation_entry(
        dataset,
        method,
        metrics,
        paper_tolerance,
        "eval_json",
        "paper",
        condition_id=condition_id,
    )
    entry.update(
        {
            "eval_dir": str(root),
            "class_name": class_name,
            "paper_metrics": paper_row_metrics(dataset, method, condition_id),
            "source_files": {
                "detection": str(det_path),
                "tracking": str(track_path),
            },
            "ap_distances": det.get("label_aps", {}).get(class_name, {}),
        }
    )
    return entry


def config_thresholds(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8", errors="replace")

    def number(pattern: str) -> float | None:
        match = re.search(pattern, text, flags=re.MULTILINE)
        return float(match.group(1)) if match else None

    def string(pattern: str) -> str | None:
        match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
        return match.group(1) if match else None

    return {
        "config": str(config_path),
        "score_thresh": number(r"(?m)^\s*score_thresh\s*=\s*([0-9]+(?:\.[0-9]+)?)"),
        "filter_score_thresh": number(r"(?m)^\s*filter_score_thresh\s*=\s*([0-9]+(?:\.[0-9]+)?)"),
        "train_gt_iou_threshold": number(r"(?m)^\s*train_gt_iou_threshold\s*=\s*([0-9]+(?:\.[0-9]+)?)"),
        "bbox_coder_type": string(r"bbox_coder\s*=\s*dict\([\s\S]*?type\s*=\s*[\"']([^\"']+)[\"']"),
        "bbox_coder_max_num": number(r"bbox_coder\s*=\s*dict\([\s\S]*?max_num\s*=\s*([0-9]+)"),
    }


def audit_cooptrack_gap(
    result_pkl: str,
    query_dir: str,
    ann_file: str,
    eval_dir: str,
    config: str | None,
    dataset: str = "50scenes_25m",
    paper_tolerance: float = 0.02,
    class_name: str = "car",
    condition_id: str = "baseline",
) -> dict[str, Any]:
    return {
        "method": "2b1-cooptrack",
        "dataset": dataset,
        "condition_id": condition_id,
        "summary": summarize_eval_json(
            eval_dir,
            dataset,
            "2b1-cooptrack",
            paper_tolerance,
            class_name,
            condition_id,
        ),
        "result_pkl": analyze_result_pkl(result_pkl),
        "track_query": analyze_track_query_cache(
            query_dir,
            ann_file,
            ["query_feats", "query_embeds", "obj_idxes", "ref_pts", "scores", "cache_motion_feats"],
        ),
        "config_thresholds": config_thresholds(config),
    }


def summarize_official_log(
    log: str,
    dataset: str,
    method: str,
    paper_tolerance: float = 0.02,
    class_name: str = "car",
    condition_id: str = "baseline",
) -> dict[str, Any]:
    path = Path(log)
    metrics = parse_paper_class_metrics(path.read_text(encoding="utf-8", errors="replace"), class_name)
    entry = metric_validation_entry(
        dataset,
        method,
        metrics,
        paper_tolerance,
        "official_log",
        "paper",
        str(path),
        condition_id,
    )
    entry.update({"class_name": class_name, "paper_metrics": paper_row_metrics(dataset, method, condition_id)})
    return entry


def _resolve_log_path(log_value: str | None) -> Path | None:
    if not log_value:
        return None
    path = Path(log_value)
    candidates = [path] if path.is_absolute() else [REPO_ROOT / path, OFFICIAL_ROOT / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _entry_for_metric_scope(
    entry: dict[str, Any],
    dataset: str,
    paper_tolerance: float,
    metric_scope: str,
) -> dict[str, Any]:
    if metric_scope == entry.get("metric_scope", "aggregate"):
        return entry
    log_path = _resolve_log_path(entry.get("log"))
    if not log_path:
        return entry
    metrics = parse_run_metrics(log_path.read_text(encoding="utf-8", errors="replace"), metric_scope)
    if not metrics:
        return entry
    return metric_validation_entry(
        dataset,
        entry["method"],
        metrics,
        paper_tolerance,
        entry.get("profile", "log_section"),
        metric_scope,
        entry.get("log"),
    )


def _late_fusion_section(text: str) -> str | None:
    match = re.search(r"(?im)^run late\s*$", text)
    if not match:
        return None
    next_run = re.search(r"(?im)^run\s+", text[match.end() :])
    if next_run:
        return text[match.end() : match.end() + next_run.start()]
    return text[match.end() :]


def parse_late_fusion_metrics(text: str) -> dict[str, float]:
    metrics = {}
    ap_match = re.search(r"(?im)^mAP:\s*([0-9]+(?:\.[0-9]+)?)\s*$", text)
    if ap_match:
        metrics["AP"] = float(ap_match.group(1))

    amota_matches = re.findall(r"['\"]pts_bbox/amota['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if amota_matches:
        metrics["AMOTA"] = float(amota_matches[-1])
    else:
        table_matches = re.findall(r"(?im)^AMOTA\s+([0-9]+(?:\.[0-9]+)?)\s*$", text)
        if table_matches:
            metrics["AMOTA"] = float(table_matches[-1])
    return metrics


def summarize_run_log(
    log_path: str,
    paper_tolerance: float = 0.02,
    dataset: str = "50scenes_25m",
    metric_scope: str = "aggregate",
) -> dict[str, Any]:
    path = Path(log_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    entries = []
    for line in text.splitlines():
        json_start = line.find("{")
        if json_start < 0:
            continue
        try:
            payload = json.loads(line[json_start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if {"profile", "method", "metrics", "checks"} <= set(payload):
            entries.append(_entry_for_metric_scope(payload, dataset, paper_tolerance, metric_scope))

    methods = [entry["method"] for entry in entries]
    late_section = _late_fusion_section(text)
    if late_section and "3-late fusion" not in methods:
        late_metrics = (
            parse_run_metrics(late_section, metric_scope)
            if metric_scope == "paper"
            else parse_late_fusion_metrics(late_section)
        )
        if late_metrics:
            entries.append(
                metric_validation_entry(
                    dataset,
                    "3-late fusion",
                    late_metrics,
                    paper_tolerance,
                    metric_scope=metric_scope,
                )
            )
            methods.append("3-late fusion")
    missing_methods = sorted(RUNNABLE_METHODS - set(methods))
    paper_mismatches = []
    for entry in entries:
        for metric, check in entry.get("checks", {}).items():
            abs_delta = float(check.get("abs_delta", abs(check.get("delta", 0.0))))
            if abs_delta <= paper_tolerance:
                continue
            paper_mismatches.append(
                {
                    "method": entry["method"],
                    "metric": metric,
                    "actual": check.get("actual"),
                    "expected": check.get("expected"),
                    "abs_delta": abs_delta,
                }
            )
    return {
        "log": str(path),
        "method_count": len(entries),
        "methods": methods,
        "missing_runnable_methods": missing_methods,
        "all_passed": bool(entries) and all(entry.get("passed") for entry in entries),
        "metric_scope": metric_scope,
        "paper_tolerance": paper_tolerance,
        "all_within_paper_tolerance": bool(entries) and not paper_mismatches,
        "paper_mismatches": paper_mismatches,
        "entries": entries,
    }


def summarize_run_logs(
    log_paths: list[str],
    paper_tolerance: float = 0.02,
    dataset: str = "50scenes_25m",
    metric_scope: str = "aggregate",
) -> dict[str, Any]:
    paths = [Path(log_path) for log_path in log_paths]
    entries_by_method = {}
    input_summaries = []
    for path in paths:
        summary = summarize_run_log(str(path), paper_tolerance, dataset, metric_scope)
        input_summaries.append(summary)
        for entry in summary["entries"]:
            entries_by_method[entry["method"]] = entry

        if "3-late fusion" not in entries_by_method:
            text = path.read_text(encoding="utf-8", errors="replace")
            late_metrics = (
                parse_run_metrics(text, metric_scope)
                if metric_scope == "paper"
                else parse_late_fusion_metrics(text)
            )
            if late_metrics:
                entries_by_method["3-late fusion"] = metric_validation_entry(
                    dataset,
                    "3-late fusion",
                    late_metrics,
                    paper_tolerance,
                    metric_scope=metric_scope,
                )

    methods = [method for method in RUNNABLE_METHOD_ORDER if method in entries_by_method]
    entries = [entries_by_method[method] for method in methods]
    missing_methods = [method for method in RUNNABLE_METHOD_ORDER if method not in entries_by_method]
    paper_mismatches = []
    for entry in entries:
        for metric, check in entry.get("checks", {}).items():
            abs_delta = float(check.get("abs_delta", abs(check.get("delta", 0.0))))
            if abs_delta <= paper_tolerance:
                continue
            paper_mismatches.append(
                {
                    "method": entry["method"],
                    "metric": metric,
                    "actual": check.get("actual"),
                    "expected": check.get("expected"),
                    "abs_delta": abs_delta,
                }
            )

    return {
        "logs": [str(path) for path in paths],
        "method_count": len(entries),
        "methods": methods,
        "missing_runnable_methods": missing_methods,
        "all_passed": bool(entries) and all(entry.get("passed") for entry in entries),
        "metric_scope": metric_scope,
        "paper_tolerance": paper_tolerance,
        "all_within_paper_tolerance": bool(entries) and not paper_mismatches,
        "paper_mismatches": paper_mismatches,
        "entries": entries,
        "input_summaries": input_summaries,
    }


def run_profile(profile_name: str, dry_run: bool) -> int:
    payload = profile_payload(profile_name)
    print(json.dumps(payload, indent=2))
    if dry_run:
        return 0

    assets = check_assets(profile_name)
    if not assets["ready"]:
        print(json.dumps(assets, indent=2), file=sys.stderr)
        return 2

    command = payload["commands"][0].split("&&", 1)[1].strip()
    return subprocess.call(command, cwd=OFFICIAL_ROOT, shell=True)


def emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify-layout")
    verify_parser.add_argument("--json", action="store_true")

    summary_parser = subparsers.add_parser("summarize-results")
    summary_parser.add_argument("--json", action="store_true")

    env_parser = subparsers.add_parser("env-check")
    env_parser.add_argument("--strict", action="store_true")
    env_parser.add_argument("--json", action="store_true")

    data_parser = subparsers.add_parser("data-packages")
    data_parser.add_argument("--dataset", required=True)
    data_parser.add_argument("--package-profile", default="full", choices=sorted(DATA_PACKAGE_PROFILES))
    data_parser.add_argument("--json", action="store_true")

    data_script_parser = subparsers.add_parser("write-data-script")
    data_script_parser.add_argument("--dataset", required=True)
    data_script_parser.add_argument("--out", required=True)
    data_script_parser.add_argument("--package-profile", default="smoke_25m_instance", choices=sorted(DATA_PACKAGE_PROFILES))
    data_script_parser.add_argument("--json", action="store_true")

    data_check_parser = subparsers.add_parser("check-data-packages")
    data_check_parser.add_argument("--dataset", required=True)
    data_check_parser.add_argument("--package-profile", default="smoke_25m_instance", choices=sorted(DATA_PACKAGE_PROFILES))
    data_check_parser.add_argument("--json", action="store_true")

    env_script_parser = subparsers.add_parser("write-env-script")
    env_script_parser.add_argument("--out", required=True)
    env_script_parser.add_argument("--json", action="store_true")

    paper_parser = subparsers.add_parser("paper-matrix")
    paper_parser.add_argument("--json", action="store_true")

    paper_run_parser = subparsers.add_parser("paper-run-matrix")
    paper_run_parser.add_argument("--dataset", choices=sorted(DATASETS))
    paper_run_parser.add_argument("--include-robustness", action="store_true")
    paper_run_parser.add_argument("--json", action="store_true")

    profiles_parser = subparsers.add_parser("list-profiles")
    profiles_parser.add_argument("--json", action="store_true")

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("--profile", required=True)
    matrix_parser.add_argument("--json", action="store_true")

    partial_parser = subparsers.add_parser("plan-partial-run")
    partial_parser.add_argument("--profile", required=True)
    partial_parser.add_argument("--json", action="store_true")

    prepare_partial_parser = subparsers.add_parser("prepare-partial-eval")
    prepare_partial_parser.add_argument("--profile", required=True)
    prepare_partial_parser.add_argument("--scene-limit", type=int, default=1)
    prepare_partial_parser.add_argument("--max-samples", type=int)
    prepare_partial_parser.add_argument("--samples-per-scene", type=int)
    prepare_partial_parser.add_argument("--source-ann")
    prepare_partial_parser.add_argument("--out-ann")
    prepare_partial_parser.add_argument("--out-config")
    prepare_partial_parser.add_argument("--out-tag")
    prepare_partial_parser.add_argument("--json", action="store_true")

    drone_query_parser = subparsers.add_parser("prepare-drone-query-partial-eval")
    drone_query_parser.add_argument("--profile", required=True)
    drone_query_parser.add_argument("--scene-limit", type=int, default=1)
    drone_query_parser.add_argument("--max-samples", type=int)
    drone_query_parser.add_argument("--samples-per-scene", type=int)
    drone_query_parser.add_argument("--source-ann")
    drone_query_parser.add_argument("--out-ann")
    drone_query_parser.add_argument("--out-config")
    drone_query_parser.add_argument("--out-tag")
    drone_query_parser.add_argument("--json", action="store_true")

    materialize_parser = subparsers.add_parser("materialize-partial-images")
    materialize_parser.add_argument("--profile", required=True)
    materialize_parser.add_argument("--image-side", required=True, choices=["vehicle-side", "drone-side"])
    materialize_parser.add_argument("--scene-limit", type=int, default=1)
    materialize_parser.add_argument("--max-samples", type=int)
    materialize_parser.add_argument("--samples-per-scene", type=int)
    materialize_parser.add_argument("--source-ann")
    materialize_parser.add_argument("--out-tag")
    materialize_parser.add_argument("--dry-run", action="store_true")
    materialize_parser.add_argument("--json", action="store_true")

    describe_subset_parser = subparsers.add_parser("describe-partial-subset")
    describe_subset_parser.add_argument("--profile", required=True)
    describe_subset_parser.add_argument("--scene-limit", type=int, default=1)
    describe_subset_parser.add_argument("--max-samples", type=int)
    describe_subset_parser.add_argument("--samples-per-scene", type=int)
    describe_subset_parser.add_argument("--json", action="store_true")

    script_parser = subparsers.add_parser("write-mobaxterm-script")
    script_parser.add_argument("--profile", required=True)
    script_parser.add_argument("--out", required=True)
    script_parser.add_argument("--json", action="store_true")

    supervisor_parser = subparsers.add_parser("write-supervisor-script")
    supervisor_parser.add_argument("--profile", required=True)
    supervisor_parser.add_argument("--dataset", required=True)
    supervisor_parser.add_argument("--package-profile", default="smoke_25m_instance", choices=sorted(DATA_PACKAGE_PROFILES))
    supervisor_parser.add_argument("--out", required=True)
    supervisor_parser.add_argument("--json", action="store_true")

    assets_parser = subparsers.add_parser("check-assets")
    assets_parser.add_argument("--profile", required=True)
    assets_parser.add_argument("--json", action="store_true")

    partial_assets_parser = subparsers.add_parser("check-partial-assets")
    partial_assets_parser.add_argument("--profile", required=True)
    partial_assets_parser.add_argument("--json", action="store_true")

    validate_parser = subparsers.add_parser("validate-run")
    validate_parser.add_argument("--profile", required=True)
    validate_parser.add_argument("--log", required=True)
    validate_parser.add_argument("--tolerance", type=float, default=0.02)
    validate_parser.add_argument("--metric-scope", default="aggregate", choices=["aggregate", "paper"])
    validate_parser.add_argument("--json", action="store_true")

    result_pkl_parser = subparsers.add_parser("analyze-result-pkl")
    result_pkl_parser.add_argument("--path", required=True)
    result_pkl_parser.add_argument("--json", action="store_true")

    track_query_parser = subparsers.add_parser("analyze-track-query-cache")
    track_query_parser.add_argument("--query-dir", required=True)
    track_query_parser.add_argument("--ann-file")
    track_query_parser.add_argument("--key", action="append")
    track_query_parser.add_argument("--sample-limit", type=int, default=3)
    track_query_parser.add_argument("--json", action="store_true")

    run_log_parser = subparsers.add_parser("summarize-run-log")
    run_log_parser.add_argument("--log", required=True)
    run_log_parser.add_argument("--paper-tolerance", type=float, default=0.02)
    run_log_parser.add_argument("--dataset", default="50scenes_25m", choices=sorted(DATASETS))
    run_log_parser.add_argument("--metric-scope", default="aggregate", choices=["aggregate", "paper"])
    run_log_parser.add_argument("--json", action="store_true")

    run_logs_parser = subparsers.add_parser("summarize-run-logs")
    run_logs_parser.add_argument("--log", action="append", required=True)
    run_logs_parser.add_argument("--paper-tolerance", type=float, default=0.02)
    run_logs_parser.add_argument("--dataset", default="50scenes_25m", choices=sorted(DATASETS))
    run_logs_parser.add_argument("--metric-scope", default="aggregate", choices=["aggregate", "paper"])
    run_logs_parser.add_argument("--json", action="store_true")

    eval_json_parser = subparsers.add_parser("summarize-eval-json")
    eval_json_parser.add_argument("--eval-dir", required=True)
    eval_json_parser.add_argument("--dataset", default="50scenes_25m", choices=sorted(DATASETS))
    eval_json_parser.add_argument("--method", required=True)
    eval_json_parser.add_argument("--paper-tolerance", type=float, default=0.02)
    eval_json_parser.add_argument("--class-name", default="car")
    eval_json_parser.add_argument("--condition-id", default="baseline")
    eval_json_parser.add_argument("--json", action="store_true")

    coop_audit_parser = subparsers.add_parser("audit-cooptrack-gap")
    coop_audit_parser.add_argument("--result-pkl", required=True)
    coop_audit_parser.add_argument("--query-dir", required=True)
    coop_audit_parser.add_argument("--ann-file", required=True)
    coop_audit_parser.add_argument("--eval-dir", required=True)
    coop_audit_parser.add_argument("--config")
    coop_audit_parser.add_argument("--dataset", default="50scenes_25m", choices=sorted(DATASETS))
    coop_audit_parser.add_argument("--paper-tolerance", type=float, default=0.02)
    coop_audit_parser.add_argument("--class-name", default="car")
    coop_audit_parser.add_argument("--condition-id", default="baseline")
    coop_audit_parser.add_argument("--json", action="store_true")

    official_log_parser = subparsers.add_parser("summarize-official-log")
    official_log_parser.add_argument("--log", required=True)
    official_log_parser.add_argument("--dataset", default="50scenes_25m", choices=sorted(DATASETS))
    official_log_parser.add_argument("--method", required=True)
    official_log_parser.add_argument("--paper-tolerance", type=float, default=0.02)
    official_log_parser.add_argument("--class-name", default="car")
    official_log_parser.add_argument("--condition-id", default="baseline")
    official_log_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser("run-profile")
    run_parser.add_argument("--profile", required=True)
    run_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "verify-layout":
        emit(verify_layout(), args.json)
    elif args.command == "summarize-results":
        emit(result_summary(), args.json)
    elif args.command == "env-check":
        environment = env_check()
        emit(environment, args.json)
        if args.strict and not environment["ready"]:
            return 4
    elif args.command == "data-packages":
        emit(data_packages(args.dataset, args.package_profile), args.json)
    elif args.command == "write-data-script":
        emit(write_data_script(args.dataset, args.out, args.package_profile), args.json)
    elif args.command == "check-data-packages":
        emit(check_data_packages(args.dataset, args.package_profile), args.json)
    elif args.command == "write-env-script":
        emit(write_env_script(args.out), args.json)
    elif args.command == "paper-matrix":
        emit(paper_matrix(), args.json)
    elif args.command == "paper-run-matrix":
        emit(paper_run_matrix(args.dataset, args.include_robustness), args.json)
    elif args.command == "list-profiles":
        emit(list_profiles(), args.json)
    elif args.command == "matrix":
        emit(profile_payload(args.profile), args.json)
    elif args.command == "plan-partial-run":
        emit(partial_run_plan(args.profile), args.json)
    elif args.command == "prepare-partial-eval":
        emit(
            prepare_partial_eval(
                args.profile,
                args.scene_limit,
                args.max_samples,
                args.samples_per_scene,
                args.source_ann,
                args.out_ann,
                args.out_config,
                args.out_tag,
            ),
            args.json,
        )
    elif args.command == "prepare-drone-query-partial-eval":
        emit(
            prepare_drone_query_partial_eval(
                args.profile,
                args.scene_limit,
                args.max_samples,
                args.samples_per_scene,
                args.source_ann,
                args.out_ann,
                args.out_config,
                args.out_tag,
            ),
            args.json,
        )
    elif args.command == "materialize-partial-images":
        emit(
            materialize_partial_images(
                args.profile,
                args.image_side,
                args.scene_limit,
                args.max_samples,
                args.samples_per_scene,
                args.source_ann,
                args.dry_run,
            ),
            args.json,
        )
    elif args.command == "describe-partial-subset":
        emit(
            describe_partial_subset(
                args.profile,
                args.scene_limit,
                args.max_samples,
                args.samples_per_scene,
            ),
            args.json,
        )
    elif args.command == "write-mobaxterm-script":
        emit(write_mobaxterm_script(args.profile, args.out), args.json)
    elif args.command == "write-supervisor-script":
        emit(write_supervisor_script(args.profile, args.dataset, args.package_profile, args.out), args.json)
    elif args.command == "check-assets":
        emit(check_assets(args.profile), args.json)
    elif args.command == "check-partial-assets":
        emit(check_partial_assets(args.profile), args.json)
    elif args.command == "validate-run":
        validation = validate_run(args.profile, args.log, args.tolerance, args.metric_scope)
        emit(validation, args.json)
        if not validation["passed"]:
            return 3
    elif args.command == "analyze-result-pkl":
        emit(analyze_result_pkl(args.path), args.json)
    elif args.command == "analyze-track-query-cache":
        emit(analyze_track_query_cache(args.query_dir, args.ann_file, args.key, args.sample_limit), args.json)
    elif args.command == "summarize-run-log":
        emit(summarize_run_log(args.log, args.paper_tolerance, args.dataset, args.metric_scope), args.json)
    elif args.command == "summarize-run-logs":
        emit(summarize_run_logs(args.log, args.paper_tolerance, args.dataset, args.metric_scope), args.json)
    elif args.command == "summarize-eval-json":
        emit(
            summarize_eval_json(
                args.eval_dir,
                args.dataset,
                args.method,
                args.paper_tolerance,
                args.class_name,
                args.condition_id,
            ),
            args.json,
        )
    elif args.command == "audit-cooptrack-gap":
        emit(
            audit_cooptrack_gap(
                args.result_pkl,
                args.query_dir,
                args.ann_file,
                args.eval_dir,
                args.config,
                args.dataset,
                args.paper_tolerance,
                args.class_name,
                args.condition_id,
            ),
            args.json,
        )
    elif args.command == "summarize-official-log":
        emit(
            summarize_official_log(
                args.log,
                args.dataset,
                args.method,
                args.paper_tolerance,
                args.class_name,
                args.condition_id,
            ),
            args.json,
        )
    elif args.command == "run-profile":
        return run_profile(args.profile, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
