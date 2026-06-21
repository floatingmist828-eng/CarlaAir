#!/usr/bin/env bash
set -euo pipefail

ROOT="${GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}"
PYTHON="${GRIFFIN_PYTHON:-/home/fp/miniconda3/envs/griffin/bin/python}"
PRIMARY_BASE_URL="${GRIFFIN_REPAIR_BASE_URL:-https://huggingface.co/datasets/wjh-svm/Griffin/resolve/main}"
FALLBACK_BASE_URL="${GRIFFIN_REPAIR_FALLBACK_BASE_URL:-https://hf-mirror.com/datasets/wjh-svm/Griffin/resolve/main}"
DATA_PARENT="$ROOT/griffin_repro/official/datasets"
DATA_ROOT="$DATA_PARENT/griffin_50scenes_25m"
ARCHIVE_DIR="${GRIFFIN_ARCHIVE_DIR:-$DATA_ROOT/archives}"
LOG_DIR="$ROOT/griffin_repro/artifacts/logs"
STAMP="${GRIFFIN_REPAIR_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOCK_FILE="$ARCHIVE_DIR/.md5-repair.lock"
KEEP_CORRUPT="${GRIFFIN_REPAIR_KEEP_CORRUPT:-1}"
REPAIR_JOBS="${GRIFFIN_REPAIR_JOBS:-3}"
REPAIR_PARTS="${GRIFFIN_REPAIR_PARTS:-1}"

case "$REPAIR_JOBS" in
  ''|*[!0-9]*)
    echo "GRIFFIN_REPAIR_JOBS must be a positive integer, got: $REPAIR_JOBS" >&2
    exit 2
    ;;
esac
case "$REPAIR_PARTS" in
  ''|*[!0-9]*)
    echo "GRIFFIN_REPAIR_PARTS must be a positive integer, got: $REPAIR_PARTS" >&2
    exit 2
    ;;
esac
if [ "$REPAIR_JOBS" -lt 1 ] || [ "$REPAIR_PARTS" -lt 1 ]; then
  echo "GRIFFIN_REPAIR_JOBS and GRIFFIN_REPAIR_PARTS must be >= 1." >&2
  exit 2
fi

if ! help wait 2>/dev/null | grep -q -- "-n"; then
  REPAIR_JOBS=1
  REPAIR_PARTS=1
fi

mkdir -p "$ARCHIVE_DIR" "$LOG_DIR" "$DATA_PARENT"
exec 9>"$LOCK_FILE"
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    echo "Another Griffin MD5 repair is already active for $ARCHIVE_DIR." >&2
    exit 75
  fi
fi

cd "$ROOT"
before_json="$LOG_DIR/official_25m_md5_repair_before_${STAMP}.json"
after_json="$LOG_DIR/official_25m_md5_repair_after_${STAMP}.json"

"$PYTHON" scripts/griffin_repro.py verify-data-md5 \
  --dataset 50scenes_25m \
  --package-profile full \
  --json > "$before_json"

mapfile -t mismatches < <("$PYTHON" - "$before_json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for item in payload.get("checks", []):
    if item.get("status") == "mismatch":
        print("|".join([
            item["path"],
            str(item["expected_size_bytes"]),
            item["expected_md5"],
        ]))
PY
)

if [ "${#mismatches[@]}" -eq 0 ]; then
  echo "No MD5 mismatches found. Before audit: $before_json"
  exit 0
fi

status_dir="$ARCHIVE_DIR/.md5-repair-$STAMP"
rm -rf "$status_dir"
mkdir -p "$status_dir"

download_verified() {
  local rel_path="$1"
  local expected_size="$2"
  local expected_md5="$3"
  local base_url="$4"
  local name
  local target
  local tmp
  local url
  local actual_size
  local actual_md5
  local part_dir
  local part_fail
  local part_jobs
  local part
  local start
  local end
  local part_path
  local part_size
  local expected_part_size

  name="$(basename "$rel_path")"
  target="$ARCHIVE_DIR/$name"
  tmp="$ARCHIVE_DIR/$name.redownload.$STAMP"
  url="$base_url/$rel_path"

  rm -f "$tmp"
  if [ "$REPAIR_PARTS" -le 1 ]; then
    echo "Redownloading $name from $url"
    curl --silent --show-error --fail --retry 5 --retry-all-errors --connect-timeout 30 --speed-limit 1024 --speed-time 120 -L -o "$tmp" "$url"
  else
    part_dir="$tmp.parts"
    rm -rf "$part_dir"
    mkdir -p "$part_dir"
    echo "Redownloading $name from $url with $REPAIR_PARTS range parts"
    part_fail=0
    part_jobs=0
    for part in $(seq 0 $((REPAIR_PARTS - 1))); do
      start=$((expected_size * part / REPAIR_PARTS))
      end=$((expected_size * (part + 1) / REPAIR_PARTS - 1))
      part_path="$part_dir/part_$(printf '%03d' "$part")"
      curl --silent --show-error --fail --retry 5 --retry-all-errors --connect-timeout 30 --speed-limit 1024 --speed-time 120 -L -r "$start-$end" -o "$part_path" "$url" &
      part_jobs=$((part_jobs + 1))
    done
    while [ "$part_jobs" -gt 0 ]; do
      if ! wait -n; then
        part_fail=1
      fi
      part_jobs=$((part_jobs - 1))
    done
    if [ "$part_fail" != "0" ]; then
      rm -rf "$part_dir"
      return 1
    fi
    for part in $(seq 0 $((REPAIR_PARTS - 1))); do
      start=$((expected_size * part / REPAIR_PARTS))
      end=$((expected_size * (part + 1) / REPAIR_PARTS - 1))
      part_path="$part_dir/part_$(printf '%03d' "$part")"
      expected_part_size=$((end - start + 1))
      part_size="$(stat -c%s "$part_path")"
      if [ "$part_size" != "$expected_part_size" ]; then
        echo "Range part size mismatch for $name part $part: expected $expected_part_size, got $part_size" >&2
        rm -rf "$part_dir"
        return 1
      fi
    done
    cat "$part_dir"/part_* > "$tmp"
    rm -rf "$part_dir"
  fi

  actual_size="$(stat -c%s "$tmp")"
  if [ "$actual_size" != "$expected_size" ]; then
    echo "Size mismatch after redownload for $name: expected $expected_size, got $actual_size" >&2
    return 1
  fi

  actual_md5="$(md5sum "$tmp" | awk '{print $1}')"
  if [ "$actual_md5" != "$expected_md5" ]; then
    echo "MD5 mismatch after redownload for $name: expected $expected_md5, got $actual_md5" >&2
    return 1
  fi

  if [ -f "$target" ]; then
    if [ "$KEEP_CORRUPT" = "1" ]; then
      mv -f "$target" "$target.corrupt.$STAMP"
    else
      rm -f "$target"
    fi
  fi
  mv -f "$tmp" "$target"
  rm -f "$target.extracted.to-data-parent"
}

repair_one() {
  local item="$1"
  local rel_path
  local rest
  local expected_size
  local expected_md5
  local name

  rel_path="${item%%|*}"
  rest="${item#*|}"
  expected_size="${rest%%|*}"
  expected_md5="${rest##*|}"
  name="$(basename "$rel_path")"

  if ! download_verified "$rel_path" "$expected_size" "$expected_md5" "$PRIMARY_BASE_URL"; then
    echo "Primary redownload failed for $name; trying fallback mirror." >&2
    if ! download_verified "$rel_path" "$expected_size" "$expected_md5" "$FALLBACK_BASE_URL"; then
      echo "Both redownload sources failed for $name." >&2
      exit 4
    fi
  fi
  printf '%s\n' "$name" > "$status_dir/$name.repaired"
}

repair_fail=0
active_jobs=0
for item in "${mismatches[@]}"; do
  repair_one "$item" &
  active_jobs=$((active_jobs + 1))
  if [ "$active_jobs" -ge "$REPAIR_JOBS" ]; then
    if ! wait -n; then
      repair_fail=1
    fi
    active_jobs=$((active_jobs - 1))
  fi
done

while [ "$active_jobs" -gt 0 ]; do
  if ! wait -n; then
    repair_fail=1
  fi
  active_jobs=$((active_jobs - 1))
done

if [ "$repair_fail" != "0" ]; then
  echo "One or more MD5 repair downloads failed." >&2
  exit 4
fi

"$PYTHON" scripts/griffin_repro.py verify-data-md5 \
  --dataset 50scenes_25m \
  --package-profile full \
  --json > "$after_json"

"$PYTHON" - "$after_json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if not payload.get("ready"):
    bad = [item for item in payload.get("checks", []) if item.get("status") != "matched"]
    raise SystemExit(f"MD5 repair did not converge; remaining bad archives: {bad}")
PY

mapfile -t repaired < <(find "$status_dir" -name '*.repaired' -type f -print0 | xargs -0 -r cat | sort)
for name in "${repaired[@]}"; do
  echo "Extracting repaired $name"
  unzip -oq "$ARCHIVE_DIR/$name" -d "$DATA_PARENT"
  touch "$ARCHIVE_DIR/$name.extracted.to-data-parent"
done

echo "MD5 repair completed. Before: $before_json"
echo "MD5 repair completed. After:  $after_json"
