#!/bin/bash

set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <DATASET> <SHOTS> <RATE> <TYPE> <CLASS>"
    exit 1
fi

# 基础配置（可通过环境变量覆盖）
DATA=${DATA:-/path/to/datasets}
TRAINER=${TRAINER:-NLPrompt}

DATASET=$1
CFG=${CFG:-rn50}   # config file
SHOTS=$2   # number of shots (1, 2, 4, 8, 16)
RATE=$3
TYPE=$4
CLASS=$5

# 定义 8 个不同的 REG_E 值（对应 8 张卡）
REG_E_VALUES=(${REG_E_VALUES:-"0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2"})

# 默认跑 8 张卡，可通过 GPU_IDS 覆盖
GPU_IDS=(${GPU_IDS:-"0 1 2 3 4 5 6 7"})
SEED=${SEED:-2}

if [ "${#REG_E_VALUES[@]}" -lt "${#GPU_IDS[@]}" ]; then
    echo "Error: REG_E_VALUES 数量少于 GPU_IDS 数量"
    exit 1
fi

echo "Starting ${#GPU_IDS[@]}-GPU parallel jobs..."

for i in "${!GPU_IDS[@]}"
do
    # 设置当前使用的显卡 ID
    GPU_ID=${GPU_IDS[$i]}
    # 获取当前显卡对应的 REG_E 值
    CURRENT_REG_E=${REG_E_VALUES[$i]}
    
    # 重新定义输出目录，增加 REG_E 标识以防结果覆盖
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/noise_${TYPE}_${RATE}/seed${SEED}_regE${CURRENT_REG_E}

    echo "Launching job on GPU ${GPU_ID} with REG_E=${CURRENT_REG_E}"

    # 使用 CUDA_VISIBLE_DEVICES 指定显卡，并在命令末尾添加 & 放入后台执行
    CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.NOISE_RATE ${RATE} \
        DATASET.NOISE_TYPE ${TYPE} \
        DATASET.num_class ${CLASS} \
        DATASET.REG_E ${CURRENT_REG_E} & 

done

# 等待所有后台任务完成
wait
echo "All jobs finished."
