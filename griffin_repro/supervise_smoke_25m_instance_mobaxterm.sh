#!/usr/bin/env bash
set -euo pipefail

cd "${GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}"
CONDA_HOME="${GRIFFIN_CONDA_HOME:-$HOME/miniconda3}"
GRIFFIN_ENV_NAME="${GRIFFIN_ENV_NAME:-griffin}"
if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$GRIFFIN_ENV_NAME"
fi

LOG_DIR="${GRIFFIN_SUPERVISOR_LOG_DIR:-griffin_repro/artifacts/logs}"
SUPERVISOR_SLEEP_SEC="${GRIFFIN_SUPERVISOR_SLEEP_SEC:-300}"
SUPERVISOR_MAX_ATTEMPTS="${GRIFFIN_SUPERVISOR_MAX_ATTEMPTS:-0}"
mkdir -p "$LOG_DIR"

data_status="$LOG_DIR/smoke_25m_instance_supervisor_data_status.json"
latest_log="$LOG_DIR/smoke_25m_instance_supervisor.latest"

check_data_ready() {
  python scripts/griffin_repro.py check-data-packages --dataset 50scenes_25m --package-profile smoke_25m_instance --json > "$data_status"
  python - "$data_status" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
missing = sum(item["missing_size_bytes"] for item in payload["checks"] if not item["complete"])
oversize = sum(item.get("oversize_size_bytes", 0) for item in payload["checks"] if not item["complete"])
print(
    f"data_ready={payload['ready']} complete={payload['complete_count']}/{payload['package_count']} "
    f"missing={missing} oversize={oversize}",
    flush=True,
)
raise SystemExit(0 if payload["ready"] else 1)
PY
}

cleanup_stale_downloads() {
  pkill -f "curl .*griffin_50scenes_25m/archives" 2>/dev/null || true
  pkill -f "bash griffin_repro/download_50scenes_25m_mobaxterm.sh" 2>/dev/null || true
}

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
  bash griffin_repro/download_50scenes_25m_mobaxterm.sh
  download_status=$?
  set -e
  if [ "$download_status" -ne 0 ]; then
    echo "Download script exited with $download_status; retrying after $SUPERVISOR_SLEEP_SEC seconds." >&2
  fi
  attempt=$((attempt + 1))
  sleep "$SUPERVISOR_SLEEP_SEC"
done

smoke_log="$LOG_DIR/smoke_25m_instance_supervisor_$(date +%Y%m%d_%H%M%S).log"
echo "$smoke_log" > "$latest_log"
bash griffin_repro/run_smoke_25m_instance_mobaxterm.sh 2>&1 | tee "$smoke_log"
