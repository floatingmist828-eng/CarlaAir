#!/usr/bin/env bash
set -euo pipefail

ROOT="${GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}"
BASE_URL="${GRIFFIN_DATA_BASE_URL:-https://hf-mirror.com/datasets/wjh-svm/Griffin/resolve/main}"
DATA_ROOT="$ROOT/griffin_repro/official/datasets/griffin_50scenes_25m"
ARCHIVE_DIR="${GRIFFIN_ARCHIVE_DIR:-$DATA_ROOT/archives}"
TOTAL_SIZE_BYTES=167190016122
DOWNLOAD_JOBS="${GRIFFIN_DOWNLOAD_JOBS:-3}"

if ! help wait 2>/dev/null | grep -q -- "-n"; then
  DOWNLOAD_JOBS=1
fi

mkdir -p "$ARCHIVE_DIR"
cd "$ROOT"

packages=(
  "datasets/griffin_50scenes_25m/drone_camera_back.zip|19492671867"
  "datasets/griffin_50scenes_25m/drone_camera_bottom.zip|19047808871"
  "datasets/griffin_50scenes_25m/drone_camera_front.zip|19486880375"
  "datasets/griffin_50scenes_25m/drone_camera_instance_segmentation.zip|3558624019"
  "datasets/griffin_50scenes_25m/drone_camera_left.zip|19309526941"
  "datasets/griffin_50scenes_25m/drone_camera_right.zip|19669453631"
  "datasets/griffin_50scenes_25m/drone_metadata.zip|14384997"
  "datasets/griffin_50scenes_25m/md5.txt|898"
  "datasets/griffin_50scenes_25m/vehicle_camera_back.zip|17138643853"
  "datasets/griffin_50scenes_25m/vehicle_camera_front.zip|16257201694"
  "datasets/griffin_50scenes_25m/vehicle_camera_instance_segmentation.zip|1116380149"
  "datasets/griffin_50scenes_25m/vehicle_camera_left.zip|15554668466"
  "datasets/griffin_50scenes_25m/vehicle_camera_right.zip|16318407065"
  "datasets/griffin_50scenes_25m/vehicle_lidar.zip|214487013"
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
  fi

  echo "Downloading $name from $url"
  curl --retry 5 --connect-timeout 30 -L -C - -o "$output" "$url"
  actual_size="$(stat -c%s "$output")"
  if [ "$actual_size" != "$expected_size" ]; then
    echo "Size mismatch for $name: expected $expected_size, got $actual_size" >&2
    exit 4
  fi
}

download_fail=0
active_jobs=0
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
