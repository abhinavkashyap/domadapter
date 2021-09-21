#!/usr/bin/env bash
# Train domain (frozen), task adapter for 5 domains "fiction" "travel" "slate" "government" "telephone"

TRAIN_PROP=0.001
DEV_PROP=0.005
TEST_PROP=0.05
EXP_DIR=${OUTPUT_DIR}
SEED=17
MODE=domain
BSZ=8
EPOCHS=2
MAX_SEQ_LENGTH=128
PADDING=max_length
NUM_CLASSES=3
LR=0.0001
GPU=0
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/train_domain_task_adapter.py"

for i in "fiction" "travel"; do
    for j in "slate"; do
        python ${PYTHON_FILE} \
            --dataset-cache-dir ${DATASET_CACHE_DIR} \
            --source-target  "${i}_${j}" \
            --pretrained-model-name "bert-base-uncased" \
            --seed ${SEED} \
            --train-proportion ${TRAIN_PROP} \
            --dev-proportion ${DEV_PROP} \
            --test-proportion ${TEST_PROP} \
            --mode ${MODE} \
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