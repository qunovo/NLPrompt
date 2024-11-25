#!/bin/bash

#cd ../..

# custom config
DATA=/path/to/datasets
TRAINER=NLPrompt

DATASET=$1
CFG=rn50  # config file
SHOTS=$2  # number of shots (1, 2, 4, 8, 16)
RATE=$3
TYPE=$4
CLASS=$5

for SEED in 1
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/noise_${TYPE}_${RATE}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
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
        DATASET.num_class ${CLASS}
    fi
done