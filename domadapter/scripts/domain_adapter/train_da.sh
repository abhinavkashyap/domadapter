#!/usr/bin/env bash
# Train domain adapter for 5 domains "fiction" "travel" "slate" "government" "telephone"

TRAIN_PROP=0.0003
DEV_PROP=0.005
EXP_DIR=${OUTPUT_DIR}
SEED=666
BSZ=4
EPOCHS=2
MAX_SEQ_LENGTH=128
PAD_TO_MAX_LENGTH=True
LR=0.0001
GPU=0
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/train_domain_adapter.py"

for i in "fiction" "travel"; do
    for j in  "slate"; do
        python ${PYTHON_FILE} \
            --dataset-cache-dir ${DATASET_CACHE_DIR} \
            --source-target  "${i}_${j}" \
            --pretrained-model-name "bert-base-uncased" \
            --seed ${SEED} \
            --train-proportion ${TRAIN_PROP} \
            --dev-proportion ${DEV_PROP} \
            --gpu ${GPU} \
            --max-seq-length ${MAX_SEQ_LENGTH} \
            --pad-to-max-length ${PAD_TO_MAX_LENGTH} \
            --lr ${LR} \
            --log-freq 5 \
            --epochs ${EPOCHS} \
            --bsz ${BSZ} \
            --exp-dir ${EXP_DIR}
    done
done