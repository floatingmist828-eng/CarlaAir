#!/usr/bin/env bash
set -euo pipefail

ROOT="${GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}"
PYTHON="${GRIFFIN_PYTHON:-/home/fp/miniconda3/envs/griffin/bin/python}"
REFERENCE_ROOT="${GRIFFIN_REFERENCE_ROOT:-/tmp/griffin-official-9c02ba4}"
POLL_SECONDS="${GRIFFIN_FINALIZE_POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${GRIFFIN_FINALIZE_MAX_WAIT_SECONDS:-28800}"

cd "$ROOT"
LOG_DIR="$ROOT/griffin_repro/artifacts/logs"
mkdir -p "$LOG_DIR"
STAMP="${GRIFFIN_FINALIZE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
FINAL_LOG="$LOG_DIR/official_25m_full_finalize_${STAMP}.log"
exec > >(tee "$FINAL_LOG") 2>&1

echo "Finalizing Griffin 50scenes 25m full validation at $STAMP"
echo "ROOT=$ROOT"
echo "PYTHON=$PYTHON"
echo "REFERENCE_ROOT=$REFERENCE_ROOT"

download_active() {
  pgrep -af 'download_50scenes_25m_full|curl .*griffin_50scenes_25m' | grep -v grep >/dev/null 2>&1
}

package_ready() {
  "$PYTHON" scripts/griffin_repro.py check-data-packages --dataset 50scenes_25m --package-profile full --json \
    > "$LOG_DIR/official_25m_package_full_${STAMP}.json"
  "$PYTHON" - "$LOG_DIR/official_25m_package_full_${STAMP}.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
sys.exit(0 if payload.get("ready") else 1)
PY
}

deadline=$((SECONDS + MAX_WAIT_SECONDS))
while true; do
  if package_ready && ! download_active; then
    echo "Full 25m package is complete and no data download process is active."
    break
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "Timed out waiting for full 25m package completion." >&2
    exit 75
  fi
  echo "Waiting for full 25m package completion..."
  sleep "$POLL_SECONDS"
done

run_json() {
  local name="$1"
  shift
  local artifact="$LOG_DIR/official_25m_${name}_${STAMP}.json"
  echo "Running $name -> $artifact"
  "$@" > "$artifact"
  echo "$artifact"
}

run_log() {
  local name="$1"
  shift
  local artifact="$LOG_DIR/official_25m_${name}_${STAMP}.log"
  echo "Running $name -> $artifact"
  "$@" 2>&1 | tee "$artifact"
}

run_json package_full "$PYTHON" scripts/griffin_repro.py check-data-packages --dataset 50scenes_25m --package-profile full --json
run_json md5_full "$PYTHON" scripts/griffin_repro.py verify-data-md5 --dataset 50scenes_25m --package-profile full --json
run_json checkpoint_25m "$PYTHON" scripts/griffin_repro.py check-checkpoint-packages --dataset 50scenes_25m --json
run_json audit_25m_assets "$PYTHON" scripts/griffin_repro.py audit-25m-assets --json
run_json source_diff "$PYTHON" scripts/griffin_repro.py official-source-diff --reference-root "$REFERENCE_ROOT" --json
run_log pytest_full "$PYTHON" -m pytest tests/test_griffin_repro.py -q

echo "Griffin 50scenes 25m full validation artifacts use stamp $STAMP"
