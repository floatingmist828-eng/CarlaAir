#!/usr/bin/env bash
set -euo pipefail

REMOTE_CODE_DIR="/home/fp/CARLA/CarlaAir-v0.1.7/code/"
TARGET_INPUT="${1:-${CARLAAIR_REMOTE_TARGET:-}}"

if [[ -z "${TARGET_INPUT}" ]]; then
  cat >&2 <<USAGE
Usage:
  scripts/sync_remote_code.sh fp@your-server
  scripts/sync_remote_code.sh fp@your-server:/home/fp/CARLA/CarlaAir-v0.1.7/code/
  CARLAAIR_REMOTE_TARGET=fp@your-server:${REMOTE_CODE_DIR} scripts/sync_remote_code.sh
USAGE
  exit 2
fi

if [[ "${TARGET_INPUT}" == *:* ]]; then
  TARGET="${TARGET_INPUT}"
else
  TARGET="${TARGET_INPUT}:${REMOTE_CODE_DIR}"
fi

cd "$(dirname "$0")/.."

rsync -avz --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.cache/' \
  --exclude '.worktrees/' \
  --exclude 'recordings/' \
  --exclude 'models/*.pt' \
  --exclude 'models/*.pth' \
  --exclude 'models/*.onnx' \
  --exclude 'models/*.engine' \
  --exclude 'models/*.bin' \
  ./ "${TARGET}"
