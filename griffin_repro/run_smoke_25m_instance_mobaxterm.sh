#!/usr/bin/env bash
set -euo pipefail

ROOT="${GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}"
cd "$ROOT"
CONDA_HOME="${GRIFFIN_CONDA_HOME:-$HOME/miniconda3}"
GRIFFIN_ENV_NAME="${GRIFFIN_ENV_NAME:-griffin}"
if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$GRIFFIN_ENV_NAME"
fi
python scripts/griffin_repro.py env-check --strict --json
set +e
spconv_error=$(python - <<'PY' 2>&1
import mmdet3d.ops.spconv
PY
)
spconv_status=$?
set -e
if [ "$spconv_status" -ne 0 ]; then
  printf '%s\n' "$spconv_error" >&2
  if printf '%s' "$spconv_error" | grep -q 'sparse_conv_ext'; then
    bash griffin_repro/build_mmdet3d_spconv_ext_mobaxterm.sh
  else
    exit "$spconv_status"
  fi
fi
LOG_DIR="${GRIFFIN_SMOKE_LOG_DIR:-$ROOT/griffin_repro/artifacts/logs}"
mkdir -p "$LOG_DIR"

check_assets() {
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
}

preprocess_assets=(
  "griffin_repro/official/datasets/griffin_50scenes_25m/griffin-release/vehicle-side"
  "griffin_repro/official/datasets/griffin_50scenes_25m/griffin-release/drone-side"
  "griffin_repro/official/data/split_datas/griffin_50scenes_25m.json"
  "griffin_repro/official/ckpts/griffin_50scenes_25m/drone-side/iter_33024.pth"
  "griffin_repro/official/ckpts/griffin_50scenes_25m/cooperative/instance_fusion/iter_33024.pth"
)
check_assets "preprocess" "${preprocess_assets[@]}"

cd griffin_repro/official
bash tools/griffin_converter.sh griffin_50scenes_25m
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} bash tools/dist_eval.sh projects/configs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval_train.py ckpts/griffin_50scenes_25m/drone-side/iter_33024.pth 1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} bash tools/dist_eval.sh projects/configs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval.py ckpts/griffin_50scenes_25m/drone-side/iter_33024.pth 1
cd ../..

evaluation_assets=(
  "griffin_repro/official/projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py"
  "griffin_repro/official/ckpts/griffin_50scenes_25m/cooperative/instance_fusion/iter_33024.pth"
  "griffin_repro/official/datasets/griffin_50scenes_25m/griffin-nuscenes/cooperative"
  "griffin_repro/official/data/infos/griffin_50scenes_25m/cooperative/griffin_infos_val.pkl"
  "griffin_repro/official/data/infos/griffin_50scenes_25m/drone-side/track_query"
)
check_assets "evaluation" "${evaluation_assets[@]}"

partial_scene_limit="${GRIFFIN_PARTIAL_SCENE_LIMIT:-1}"
partial_max_samples="${GRIFFIN_PARTIAL_MAX_SAMPLES:-20}"
partial_metric_tolerance="${GRIFFIN_PARTIAL_METRIC_TOLERANCE:-1.0}"
if [ "$partial_scene_limit" -gt 0 ]; then
  partial_json="$LOG_DIR/smoke_25m_instance_partial_eval.json"
  partial_args=(--scene-limit "$partial_scene_limit" --out-tag "partial_${partial_scene_limit}scene")
  if [ -n "$partial_max_samples" ]; then
    partial_args+=(--max-samples "$partial_max_samples" --out-tag "partial_${partial_scene_limit}scene_${partial_max_samples}samples")
  fi
  python scripts/griffin_repro.py prepare-partial-eval --profile smoke_25m_instance "${partial_args[@]}" --json | tee "$partial_json"
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
  final_eval_to_run="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} bash tools/dist_eval.sh projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py ckpts/griffin_50scenes_25m/cooperative/instance_fusion/iter_33024.pth 1"
  validation_tolerance="0.02"
fi

cd griffin_repro/official
eval "$final_eval_to_run"
latest_log=$(find projects -path '*/logs/test_*.log' -type f -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
if [ -z "$latest_log" ]; then
  echo "No Griffin eval log found under griffin_repro/official/projects." >&2
  exit 3
fi
cd "$ROOT"
python scripts/griffin_repro.py validate-run --profile smoke_25m_instance --log "griffin_repro/official/${latest_log}" --tolerance "$validation_tolerance" --json
