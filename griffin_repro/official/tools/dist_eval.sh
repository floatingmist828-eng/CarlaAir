#!/usr/bin/env bash

# Check if required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <config_file> <checkpoint_file> <num_gpus>"
    echo "Example: $0 projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py projects/work_dirs_griffin_50scenes_25m/cooperative/tiny_track_r50_stream_bs8_48epoch_3cls/iter_33024.pth 1"
    exit 1
fi

T=`date +%m%d%H%M`

# -------------------------------------------------- #
CFG=$1                                               #
CKPT=$2                                              #
GPUS=$3                                              #    
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28594}
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

checkpoint_name=$(basename ${CKPT} .pth)
config_name=$(basename ${CFG} .py)
log_file="${WORK_DIR}logs/test_${T}_${checkpoint_name}_${config_name}.log"

# Create log file with config info header
echo "===== Testing Log =====" > ${log_file}
echo "Config file: ${CFG}" >> ${log_file}
echo "Checkpoint: ${CKPT}" >> ${log_file}
echo "Timestamp: $(date)" >> ${log_file}
echo "Work directory: ${WORK_DIR}" >> ${log_file}
echo "========================" >> ${log_file}
echo "" >> ${log_file}

echo "Starting evaluation with ${GPUS_PER_NODE} GPUs..."
echo "Config: ${CFG}"
echo "Checkpoint: ${CKPT}"
echo "Output directory: ${WORK_DIR}"
echo "Log file: ${log_file}"
echo ""

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/test.py \
    $CFG \
    $CKPT \
    --launcher pytorch ${@:4} \
    --eval bbox \
    --show-dir ${WORK_DIR} \
    --out ${WORK_DIR}results-$T.pkl \
    2>&1 | tee -a ${log_file}

echo "===== Testing Done =====" >> ${log_file}
echo "Timestamp: $(date)" >> ${log_file}
echo "=========================" >> ${log_file}
echo "" >> ${log_file}

echo "Find Log file at: ${log_file}"