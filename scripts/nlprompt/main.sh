#!/bin/bash

set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <DATASET> <SHOTS> <RATE> <TYPE> <CLASS>"
    exit 1
fi

# custom config (override via environment variables)
DATA=${DATA:-/path/to/datasets}
TRAINER=${TRAINER:-NLPrompt}

DATASET=$1
CFG=${CFG:-rn50}  # config file
SHOTS=$2  # number of shots (1, 2, 4, 8, 16)
RATE=$3
TYPE=$4
CLASS=$5

# Default sweep (override by providing SEED_LIST/REG_E_LIST/LR_LIST env vars)
REG_E_LIST=${REG_E_LIST:-"0.001"}
SEED_LIST=${SEED_LIST:-"0 1 2 3 4 5 6 7 8 9"}
LR_LIST=${LR_LIST:-"0.001 0.0006 0.0007"}

for REG_E in ${REG_E_LIST}
do
    for LR in ${LR_LIST}
    do
        for SEED in ${SEED_LIST}
        do
            DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/noise_${TYPE}_${RATE}/lr${LR}/seed${SEED}_regE${REG_E}
            # if [ -d "$DIR" ]; then
            #     echo "Results are available in ${DIR}. Skip this job"
            # else
                echo "Run this job and save the output to ${DIR}"
                python train.py \
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
                DATASET.REG_E ${REG_E} \
                OPTIM.LR ${LR}
            # fi
        done
    done
done
