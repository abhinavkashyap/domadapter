#!/usr/bin/env bash
# Train domain (frozen), task adapter for 5 domains "fiction" "travel" "slate" "government" "telephone"
# losses to choose from coral, cmd, mkmmd

TRAIN_PROP=1.0
DEV_PROP=1.0
TEST_PROP=1.0
EXP_DIR=${OUTPUT_DIR}
SEED=(100 1000)
DIVERGENCE=mkmmd
BSZ=32
DATA_MODULE=mnli
EPOCHS=20
MAX_SEQ_LENGTH=128
# SKIP_LAYERS="None"
SKIP_LAYERS=(0 0,1 0,1,2 0,1,2,3 0,1,2,3,4 0,1,2,3,4,5 0,1,2,3,4,5,6 0,1,2,3,4,5,6,7 0,1,2,3,4,5,6,7,8 0,1,2,3,4,5,6,7,8,9)
PADDING=max_length
NUM_CLASSES=3
LR=1e-04
REDUCTION_FACTOR="None"
GPU=1
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/ablations/train_joint_domain_task_adapter.py"
DOMAINS=("slate_travel")

for domain in "${DOMAINS[@]}"; do
    for skip in "${SKIP_LAYERS[@]}"; do
        for seed in "${SEED[@]}"; do
            python ${PYTHON_FILE} \
                --dataset-cache-dir ${DATASET_CACHE_DIR} \
                --source-target  "${domain}" \
                --pretrained-model-name "bert-base-uncased" \
                --seed "${seed}" \
                --divergence ${DIVERGENCE} \
                --data-module ${DATA_MODULE} \
                --reduction-factor ${REDUCTION_FACTOR} \
                --train-proportion ${TRAIN_PROP} \
                --dev-proportion ${DEV_PROP} \
                --test-proportion ${TEST_PROP} \
                --skip-layers "${skip}" \
                --gpu ${GPU} \
                --num-classes ${NUM_CLASSES} \
                --max-seq-length ${MAX_SEQ_LENGTH} \
                --padding ${PADDING} \
                --lr ${LR} \
                --log-freq 5 \
                --epochs ${EPOCHS} \
                --bsz ${BSZ} \
                --exp-dir ${EXP_DIR}
        done
    done
done