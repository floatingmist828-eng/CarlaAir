#!/bin/bash

# Check if correct number of arguments was provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 <veh-det-pkl> <inf-det-pkl> <detection-config-path> <track-config-path>"
    echo "Example: $0 projects/work_dirs_griffin_50scenes_25m/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls/results-07030943.pkl projects/work_dirs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval/results-07031922.pkl projects/configs_griffin_50scenes_25m/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion.py projects/configs_griffin_50scenes_25m/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion_ab3dmot.py"
    exit 1
fi

# Assign arguments to variables
veh_det_pkl=$1
inf_det_pkl=$2
det_config_path=$3
track_config_path=$4

# Validate that config paths contain "config"
if [[ "$det_config_path" != *"config"* ]]; then
    echo "Error: Detection config path does not contain 'config'"
    echo "Detection config path: $det_config_path"
    exit 1
fi

if [[ "$track_config_path" != *"config"* ]]; then
    echo "Error: Track config path does not contain 'config'"
    echo "Track config path: $track_config_path"
    exit 1
fi

# Check if config files exist
if [ ! -f "$det_config_path" ]; then
    echo "Error: Detection config file does not exist: $det_config_path"
    exit 1
fi

if [ ! -f "$track_config_path" ]; then
    echo "Error: Track config file does not exist: $track_config_path"
    exit 1
fi

# Validate that agent side is included in pkl paths
if [[ "$veh_det_pkl" != *"vehicle-side"* ]]; then
    echo "Error: Vehicle detection pkl path does not contain 'vehicle-side'"
    echo "Vehicle pkl path: $veh_det_pkl"
    exit 1
fi

if [[ "$inf_det_pkl" != *"drone-side"* ]]; then
    echo "Error: Drone detection pkl path does not contain 'drone-side'"
    echo "Drone pkl path: $inf_det_pkl"
    exit 1
fi

# Check if pkl files exist
if [ ! -f "$veh_det_pkl" ]; then
    echo "Error: Vehicle detection pkl file does not exist: $veh_det_pkl"
    exit 1
fi

if [ ! -f "$inf_det_pkl" ]; then
    echo "Error: Drone detection pkl file does not exist: $inf_det_pkl"
    exit 1
fi

T=`date +%m%d%H%M`

# Generate output_dir from detection config path
# Remove .py extension and change configs_ to work_dirs_
output_dir="${det_config_path%.py}"
output_dir="${output_dir/configs_/work_dirs_}"
det_output_pkl="${output_dir}/results.pkl"
logs_dir="${output_dir}/logs"
log_file="${logs_dir}/eval.${T}"

# Create output and logs directories if they don't exist
mkdir -p "$output_dir"
mkdir -p "$logs_dir"

# Start logging
echo "Starting evaluation at $(date)" | tee "$log_file"
echo "Output directory: $output_dir" | tee -a "$log_file"
echo "Log file: $log_file" | tee -a "$log_file"
echo "Vehicle detection pkl: $veh_det_pkl" | tee -a "$log_file"
echo "Drone detection pkl: $inf_det_pkl" | tee -a "$log_file"
echo "Detection Config: $det_config_path" | tee -a "$log_file"
echo "Tracking Config: $track_config_path" | tee -a "$log_file"
echo "============================================" | tee -a "$log_file"

# Detection Late Fusion with Hungarian Matching
echo "Step 1: Running detection late fusion..." | tee -a "$log_file"
python tools/result_converter/det_result_late_fusion.py \
    --config "$det_config_path" \
    --veh-det-pkl "$veh_det_pkl" \
    --inf-det-pkl "$inf_det_pkl" \
    --out "$det_output_pkl" 2>&1 | tee -a "$log_file"

# Check if the first step was successful
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Error: Detection late fusion failed!" | tee -a "$log_file"
    exit 1
fi

echo "Step 1 completed successfully" | tee -a "$log_file"
echo "============================================" | tee -a "$log_file"

# Tracking with AB3DMOT
echo "Step 2: Running AB3DMOT tracking..." | tee -a "$log_file"
bash tools/eval_track_ab3dmot.sh "$det_output_pkl" "$track_config_path" 2>&1 | tee -a "$log_file"

# Check if the second step was successful
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Error: AB3DMOT tracking failed!" | tee -a "$log_file"
    exit 1
fi

echo "Step 2 completed successfully" | tee -a "$log_file"
echo "============================================" | tee -a "$log_file"
echo "Evaluation completed at $(date)" | tee -a "$log_file"
echo "All logs saved to: $log_file"