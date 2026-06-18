#!/usr/bin/env bash
set -euo pipefail

cd "${GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}"

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
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ./tools/dist_eval.sh projects/configs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval_train.py ckpts/griffin_50scenes_25m/drone-side/iter_33024.pth 1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ./tools/dist_eval.sh projects/configs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval.py ckpts/griffin_50scenes_25m/drone-side/iter_33024.pth 1
cd ../..

evaluation_assets=(
  "griffin_repro/official/projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py"
  "griffin_repro/official/ckpts/griffin_50scenes_25m/cooperative/instance_fusion/iter_33024.pth"
  "griffin_repro/official/datasets/griffin_50scenes_25m/griffin-nuscenes/cooperative"
  "griffin_repro/official/data/infos/griffin_50scenes_25m/cooperative/griffin_infos_val.pkl"
  "griffin_repro/official/data/infos/griffin_50scenes_25m/drone-side/track_query"
)
check_assets "evaluation" "${evaluation_assets[@]}"

cd griffin_repro/official
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ./tools/dist_eval.sh projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py ckpts/griffin_50scenes_25m/cooperative/instance_fusion/iter_33024.pth 1
