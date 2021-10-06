#!/usr/bin/env bash
# Train domain (frozen), task adapter for 5 domains "fiction" "travel" "slate" "government" "telephone"

TRAIN_PROP=0.001
DEV_PROP=0.001
TEST_PROP=0.005
EXP_DIR=${OUTPUT_DIR}
SEED=1729
HIDDEN_SIZE=100
BSZ=6
EPOCHS=3
MAX_SEQ_LENGTH=128
PADDING=max_length
NUM_CLASSES=3
LR=1e-05
GPU=0
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/train_dann_adapter.py"

for i in "fiction"; do
    for j in "slate"; do
        # for LR in ${LRS[@]};
        # do
            python ${PYTHON_FILE} \
                --dataset-cache-dir ${DATASET_CACHE_DIR} \
                --source-target  "${i}_${j}" \
                --pretrained-model-name "bert-base-uncased" \
                --seed ${SEED} \
                --train-proportion ${TRAIN_PROP} \
                --dev-proportion ${DEV_PROP} \
                --test-proportion ${TEST_PROP} \
                --gpu ${GPU} \
                --hidden-size ${HIDDEN_SIZE} \
                --num-classes ${NUM_CLASSES} \
                --max-seq-length ${MAX_SEQ_LENGTH} \
                --padding ${PADDING} \
                --lr ${LR} \
                --log-freq 5 \
                --epochs ${EPOCHS} \
                --bsz ${BSZ} \
                --exp-dir ${EXP_DIR} \
                --dann_alpha 0.08
        # done
    done
done