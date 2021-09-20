#!/usr/bin/env bash
# Tran a deep adaptation networks given a divergence measure

TRAIN_PROP=0.01
DEV_PROP=0.01
EXP_DIR=${OUTPUT_DIR}
SEED=666
BSZ=4
EPOCHS=1
MAX_SEQ_LENGTH=128
PAD_TO_MAX_LENGTH=True
LR=0.0001
GPU=0
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/train_domain_adapter.py"

for i in "fiction" "travel"; do
    for j in  "slate" "government"; do
        python ${PYTHON_FILE} \
            --dataset-cache-dir ${DATASET_CACHE_DIR} \
            --source-target  "${i}_${j}" \
            --pretrained-model-name "bert-base-uncased" \
            --seed ${SEED} \
            --train-proportion ${TRAIN_PROP} \
            --dev-proportion ${DEV_PROP} \
            --max-seq-length ${MAX_SEQ_LENGTH} \
            --pad-to-max-length ${PAD_TO_MAX_LENGTH} \
            --lr ${LR} \
            --log-freq 10 \
            --epochs ${EPOCHS} \
            --bsz ${BSZ} \
            --exp-dir ${EXP_DIR}
    done
done