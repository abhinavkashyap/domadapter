#!/usr/bin/env bash
# Train domain adapter for 5 domains "fiction" "travel" "slate" "government" "telephone"
# divergences to choose from coral, cmd, mkmmd

TRAIN_PROP=0.0001
DEV_PROP=0.001
EXP_DIR=${OUTPUT_DIR}
SEED=1729
BSZ=4
DIVERGENCE=mkmmd
EPOCHS=3
MAX_SEQ_LENGTH=128
REDUCTION_FACTOR=None
SKIP_LAYERS=1,2,3,4
PADDING=max_length
LR=1e-05
GPU=0
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/ablations/train_domain_adapter.py"

for i in "fiction"; do
    for j in "slate"; do
        python ${PYTHON_FILE} \
            --dataset-cache-dir ${DATASET_CACHE_DIR} \
            --source-target  "${i}_${j}" \
            --pretrained-model-name "bert-base-uncased" \
            --seed ${SEED} \
            --divergence ${DIVERGENCE} \
            --train-proportion ${TRAIN_PROP} \
            --reduction-factor ${REDUCTION_FACTOR} \
            --skip-layers ${SKIP_LAYERS} \
            --dev-proportion ${DEV_PROP} \
            --gpu ${GPU} \
            --max-seq-length ${MAX_SEQ_LENGTH} \
            --padding ${PADDING} \
            --lr ${LR} \
            --log-freq 5 \
            --epochs ${EPOCHS} \
            --bsz ${BSZ} \
            --exp-dir ${EXP_DIR}
    done
done