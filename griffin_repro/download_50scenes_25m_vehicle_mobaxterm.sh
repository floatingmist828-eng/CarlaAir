#!/usr/bin/env bash
set -euo pipefail

ROOT="${GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}"
BASE_URL="${GRIFFIN_DATA_BASE_URL:-https://hf-mirror.com/datasets/wjh-svm/Griffin/resolve/main}"
DATA_PARENT="$ROOT/griffin_repro/official/datasets"
DATA_ROOT="$ROOT/griffin_repro/official/datasets/griffin_50scenes_25m"
ARCHIVE_DIR="${GRIFFIN_ARCHIVE_DIR:-$DATA_ROOT/archives}"
PACKAGE_PROFILE="smoke_25m_vehicle"
TOTAL_SIZE_BYTES=65279798259
FULL_TOTAL_SIZE_BYTES=167190016122
DOWNLOAD_JOBS="${GRIFFIN_DOWNLOAD_JOBS:-3}"
DOWNLOAD_MAX_PASSES="${GRIFFIN_DOWNLOAD_MAX_PASSES:-12}"
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
  "datasets/griffin_50scenes_25m/md5.txt|898"
  "datasets/griffin_50scenes_25m/vehicle_camera_back.zip|17138643853"
  "datasets/griffin_50scenes_25m/vehicle_camera_front.zip|16257201694"
  "datasets/griffin_50scenes_25m/vehicle_camera_left.zip|15554668466"
  "datasets/griffin_50scenes_25m/vehicle_camera_right.zip|16318407065"
  "datasets/griffin_50scenes_25m/vehicle_metadata.zip|10876283"
)

download_one() {
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
}

all_selected_complete() {
  local item
  local path
  local expected_size
  local actual_size
  for item in "${packages[@]}"; do
    path="$ARCHIVE_DIR/$(basename "${item%%|*}")"
    expected_size="${item##*|}"
    if [ ! -f "$path" ]; then
      return 1
    fi
    actual_size="$(stat -c%s "$path")"
    if [ "$actual_size" != "$expected_size" ]; then
      return 1
    fi
  done
  return 0
}

download_pass() {
  local download_fail=0
  local active_jobs=0
  local item
  for item in "${packages[@]}"; do
    download_one "${item%%|*}" "${item##*|}" &
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
}

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
for item in "${packages[@]}"; do
  name="$(basename "${item%%|*}")"
  grep -F " ./$name" md5.txt >> md5.selected.txt
done
md5sum -c md5.selected.txt

mkdir -p "$DATA_ROOT"
for item in "${packages[@]}"; do
  archive="$(basename "${item%%|*}")"
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
python scripts/griffin_repro.py check-data-packages --dataset 50scenes_25m --package-profile smoke_25m_vehicle --json
python scripts/griffin_repro.py check-partial-assets --profile smoke_25m_instance --json
