#!/usr/bin/env python3
"""Utilities for the isolated Griffin paper reproduction package."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = REPO_ROOT / "griffin_repro"
OFFICIAL_ROOT = REPRO_ROOT / "official"
MANIFEST_PATH = REPRO_ROOT / "manifest.json"
RESULTS_CSV = OFFICIAL_ROOT / "docs" / "detailed_results.csv"
CONDA_INSTALLER_URL = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

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
        f"CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-0}} "
        f"./tools/dist_eval.sh {profile['config']} {profile['checkpoint']} {profile['gpus']}"
    )
    return {
        "profile": profile_name,
        "description": profile["description"],
        "dataset": profile["dataset"],
        "method": profile["method"],
        "config": profile["config"],
        "checkpoint": profile["checkpoint"],
        "expected": profile["expected"],
        "commands": [command],
        "asset_checks": [f"griffin_repro/official/{path}" for path in profile.get("required_paths", [])],
    }


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

python -m pip install -r "$ROOT/griffin_repro/official/requirements.txt"
cd "$ROOT"
python scripts/griffin_repro.py env-check --strict --json
echo "Griffin environment is ready. Activate it with: source $CONDA_HOME/etc/profile.d/conda.sh && conda activate $ENV_NAME"
"""


def write_env_script(out: str) -> dict[str, Any]:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(env_setup_script(), encoding="utf-8", newline="\n")
    return {"path": str(path), "bytes": path.stat().st_size}


def data_packages(dataset: str) -> dict[str, Any]:
    if dataset not in DATA_PACKAGES:
        raise SystemExit(f"No data package manifest is recorded for dataset {dataset!r}")
    prefix = dataset_prefix(dataset)
    packages = [
        {
            "path": path,
            "size_bytes": size,
            "url": f"https://huggingface.co/datasets/wjh-svm/Griffin/resolve/main/{path}",
        }
        for path, size in DATA_PACKAGES[dataset]
    ]
    return {
        "dataset": dataset,
        "dataset_prefix": prefix,
        "source": "https://huggingface.co/datasets/wjh-svm/Griffin",
        "package_count": len(packages),
        "total_size_bytes": sum(item["size_bytes"] for item in packages),
        "packages": packages,
    }


def data_download_script(dataset: str) -> str:
    package_payload = data_packages(dataset)
    prefix = package_payload["dataset_prefix"]
    rows = "\n".join(
        f'  "{item["path"]}|{item["size_bytes"]}"'
        for item in package_payload["packages"]
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail

ROOT="${{GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}}"
BASE_URL="${{GRIFFIN_DATA_BASE_URL:-https://hf-mirror.com/datasets/wjh-svm/Griffin/resolve/main}}"
DATA_ROOT="$ROOT/griffin_repro/official/datasets/{prefix}"
ARCHIVE_DIR="${{GRIFFIN_ARCHIVE_DIR:-$DATA_ROOT/archives}}"
TOTAL_SIZE_BYTES={package_payload["total_size_bytes"]}
DOWNLOAD_JOBS="${{GRIFFIN_DOWNLOAD_JOBS:-3}}"

if ! help wait 2>/dev/null | grep -q -- "-n"; then
  DOWNLOAD_JOBS=1
fi

mkdir -p "$ARCHIVE_DIR"
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
  fi

  echo "Downloading $name from $url"
  curl --retry 5 --connect-timeout 30 -L -C - -o "$output" "$url"
  actual_size="$(stat -c%s "$output")"
  if [ "$actual_size" != "$expected_size" ]; then
    echo "Size mismatch for $name: expected $expected_size, got $actual_size" >&2
    exit 4
  fi
}}

download_fail=0
active_jobs=0
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

if [ "$download_fail" -ne 0 ]; then
  echo "At least one Griffin archive failed to download." >&2
  exit 4
fi

cd "$ARCHIVE_DIR"
md5sum -c md5.txt

mkdir -p "$DATA_ROOT"
for archive in *.zip; do
  marker="$archive.extracted"
  if [ -f "$marker" ]; then
    echo "Already extracted: $archive"
    continue
  fi
  echo "Extracting $archive"
  unzip -oq "$archive" -d "$DATA_ROOT"
  touch "$marker"
done

cd "$ROOT"
python scripts/griffin_repro.py check-partial-assets --profile smoke_25m_instance --json
"""


def write_data_script(dataset: str, out: str) -> dict[str, Any]:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data_download_script(dataset), encoding="utf-8", newline="\n")
    payload = data_packages(dataset)
    return {
        "dataset": dataset,
        "dataset_prefix": payload["dataset_prefix"],
        "path": str(path),
        "bytes": path.stat().st_size,
        "total_size_bytes": payload["total_size_bytes"],
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


def partial_run_plan(profile_name: str) -> dict[str, Any]:
    payload = profile_payload(profile_name)
    prefix = dataset_prefix(payload["dataset"])
    drone_checkpoint = f"ckpts/{prefix}/drone-side/{Path(payload['checkpoint']).name}"
    commands = [
        "cd griffin_repro/official",
        f"bash tools/griffin_converter.sh {prefix}",
        (
            "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ./tools/dist_eval.sh "
            f"projects/configs_{prefix}/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval_train.py "
            f"{drone_checkpoint} 1"
        ),
        (
            "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ./tools/dist_eval.sh "
            f"projects/configs_{prefix}/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval.py "
            f"{drone_checkpoint} 1"
        ),
        payload["commands"][0].split("&&", 1)[1].strip(),
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


def mobaxterm_script(profile_name: str) -> str:
    plan = partial_run_plan(profile_name)
    preprocess_assets = "\n".join(f'  "{path}"' for path in plan["preprocess_assets"])
    evaluation_assets = "\n".join(f'  "{path}"' for path in plan["evaluation_assets"])
    _, convert_command, drone_train_command, drone_val_command, final_eval_command = plan["commands"]
    activation = conda_activation_block().rstrip()
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd "${{GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}}"
{activation}
python scripts/griffin_repro.py env-check --strict --json

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

cd griffin_repro/official
{convert_command}
{drone_train_command}
{drone_val_command}
cd ../..

evaluation_assets=(
{evaluation_assets}
)
check_assets "evaluation" "${{evaluation_assets[@]}}"

cd griffin_repro/official
{final_eval_command}
latest_log=$(find projects -path '*/logs/test_*.log' -type f -printf '%T@ %p\\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
if [ -z "$latest_log" ]; then
  echo "No Griffin eval log found under griffin_repro/official/projects." >&2
  exit 3
fi
cd ../..
python scripts/griffin_repro.py validate-run --profile {profile_name} --log "griffin_repro/official/${{latest_log}}" --json
"""


def write_mobaxterm_script(profile_name: str, out: str) -> dict[str, Any]:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mobaxterm_script(profile_name), encoding="utf-8", newline="\n")
    return {"profile": profile_name, "path": str(path), "bytes": path.stat().st_size}


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


def parse_run_metrics(text: str) -> dict[str, float]:
    patterns = {
        "AP": [
            r"(?:pts_bbox_NuScenes/)?mAP\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[\s\"'])AP[\"']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        ],
        "AMOTA": [
            r"(?:pts_bbox_NuScenes/)?amota\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[\s\"'])AMOTA[\"']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
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


def validate_run(profile_name: str, log_path: str, tolerance: float) -> dict[str, Any]:
    payload = profile_payload(profile_name)
    path = Path(log_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    metrics = parse_run_metrics(text)
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
        "metrics": metrics,
        "checks": checks,
        "missing_metrics": missing,
        "passed": passed,
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
    data_parser.add_argument("--json", action="store_true")

    data_script_parser = subparsers.add_parser("write-data-script")
    data_script_parser.add_argument("--dataset", required=True)
    data_script_parser.add_argument("--out", required=True)
    data_script_parser.add_argument("--json", action="store_true")

    env_script_parser = subparsers.add_parser("write-env-script")
    env_script_parser.add_argument("--out", required=True)
    env_script_parser.add_argument("--json", action="store_true")

    paper_parser = subparsers.add_parser("paper-matrix")
    paper_parser.add_argument("--json", action="store_true")

    profiles_parser = subparsers.add_parser("list-profiles")
    profiles_parser.add_argument("--json", action="store_true")

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("--profile", required=True)
    matrix_parser.add_argument("--json", action="store_true")

    partial_parser = subparsers.add_parser("plan-partial-run")
    partial_parser.add_argument("--profile", required=True)
    partial_parser.add_argument("--json", action="store_true")

    script_parser = subparsers.add_parser("write-mobaxterm-script")
    script_parser.add_argument("--profile", required=True)
    script_parser.add_argument("--out", required=True)
    script_parser.add_argument("--json", action="store_true")

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
    validate_parser.add_argument("--json", action="store_true")

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
        emit(data_packages(args.dataset), args.json)
    elif args.command == "write-data-script":
        emit(write_data_script(args.dataset, args.out), args.json)
    elif args.command == "write-env-script":
        emit(write_env_script(args.out), args.json)
    elif args.command == "paper-matrix":
        emit(paper_matrix(), args.json)
    elif args.command == "list-profiles":
        emit(list_profiles(), args.json)
    elif args.command == "matrix":
        emit(profile_payload(args.profile), args.json)
    elif args.command == "plan-partial-run":
        emit(partial_run_plan(args.profile), args.json)
    elif args.command == "write-mobaxterm-script":
        emit(write_mobaxterm_script(args.profile, args.out), args.json)
    elif args.command == "check-assets":
        emit(check_assets(args.profile), args.json)
    elif args.command == "check-partial-assets":
        emit(check_partial_assets(args.profile), args.json)
    elif args.command == "validate-run":
        validation = validate_run(args.profile, args.log, args.tolerance)
        emit(validation, args.json)
        if not validation["passed"]:
            return 3
    elif args.command == "run-profile":
        return run_profile(args.profile, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
