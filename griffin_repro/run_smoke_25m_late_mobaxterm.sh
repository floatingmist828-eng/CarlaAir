#!/usr/bin/env bash
set -euo pipefail

ROOT="${GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}"
cd "$ROOT"
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${GRIFFIN_CONDA_ENV:-griffin}"
fi

python scripts/griffin_repro.py env-check --strict --json

LOG_DIR="${GRIFFIN_SMOKE_LOG_DIR:-$ROOT/griffin_repro/artifacts/logs}"
mkdir -p "$LOG_DIR"

partial_scene_limit="${GRIFFIN_PARTIAL_SCENE_LIMIT:-10}"
partial_max_samples="${GRIFFIN_PARTIAL_MAX_SAMPLES:-20}"
partial_samples_per_scene="${GRIFFIN_PARTIAL_SAMPLES_PER_SCENE:-20}"
partial_args=()
if [ "$partial_scene_limit" -le 0 ]; then
  echo "Late-fusion smoke expects a partial subset; set GRIFFIN_PARTIAL_SCENE_LIMIT > 0." >&2
  exit 2
fi

partial_tag="partial_${partial_scene_limit}scene"
partial_args=(--scene-limit "$partial_scene_limit" --out-tag "$partial_tag")
if [ -n "$partial_samples_per_scene" ]; then
  partial_tag="partial_${partial_scene_limit}scene_${partial_samples_per_scene}per_scene"
  partial_args=(--scene-limit "$partial_scene_limit" --samples-per-scene "$partial_samples_per_scene" --out-tag "$partial_tag")
elif [ -n "$partial_max_samples" ]; then
  partial_tag="partial_${partial_scene_limit}scene_${partial_max_samples}samples"
  partial_args=(--scene-limit "$partial_scene_limit" --max-samples "$partial_max_samples" --out-tag "$partial_tag")
fi

partial_json="$LOG_DIR/smoke_25m_late_partial_eval.json"
python scripts/griffin_repro.py prepare-partial-eval --profile smoke_25m_instance "${partial_args[@]}" --json | tee "$partial_json"

cd griffin_repro/official
python - "$partial_json" "$partial_tag" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
partial_tag = sys.argv[2]
base = Path("projects/configs_griffin_50scenes_25m/cooperative/late_fusion")
ann = "./" + payload["ann_file"].lstrip("./")
for name, base_name in [
    (f"tiny_track_r50_stream_bs1_3cls_late_fusion_{partial_tag}.py", "tiny_track_r50_stream_bs1_3cls_late_fusion.py"),
    (f"tiny_track_r50_stream_bs1_3cls_late_fusion_ab3dmot_{partial_tag}.py", "tiny_track_r50_stream_bs1_3cls_late_fusion_ab3dmot.py"),
]:
    (base / name).write_text(
        "# Generated for Griffin partial late-fusion validation.\n"
        f"_base_ = '{base_name}'\n\n"
        f"ann_file_val = '{ann}'\n"
        "data = dict(\n"
        "    workers_per_gpu=0,\n"
        "    val=dict(ann_file=ann_file_val),\n"
        "    test=dict(ann_file=ann_file_val),\n"
        ")\n",
        encoding="utf-8",
    )
PY

latest_file() {
  local pattern="$1"
  local label="$2"
  local file
  file=$(ls -t $pattern 2>/dev/null | head -n 1 || true)
  if [ -z "$file" ]; then
    echo "Missing $label pkl for tag $partial_tag. Run the corresponding vehicle and instance smoke scripts first." >&2
    exit 3
  fi
  printf '%s\n' "$file"
}

veh=$(latest_file "projects/work_dirs_griffin_50scenes_25m/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls_${partial_tag}/results-*.pkl" "vehicle-side")
drone=$(latest_file "projects/work_dirs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval_${partial_tag}/results-*.pkl" "drone-side")
det_cfg="projects/configs_griffin_50scenes_25m/cooperative/late_fusion/tiny_track_r50_stream_bs1_3cls_late_fusion_${partial_tag}.py"
track_cfg="projects/configs_griffin_50scenes_25m/cooperative/late_fusion/tiny_track_r50_stream_bs1_3cls_late_fusion_ab3dmot_${partial_tag}.py"

echo "vehicle_pkl=$veh"
echo "drone_pkl=$drone"
echo "det_cfg=$det_cfg"
echo "track_cfg=$track_cfg"
bash tools/eval_late_fusion.sh "$veh" "$drone" "$det_cfg" "$track_cfg"
