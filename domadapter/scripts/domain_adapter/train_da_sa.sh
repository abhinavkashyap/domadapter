#!/usr/bin/env bash
# Train domain adapter for 5 domains "fiction" "travel" "slate" "government" "telephone"
# divergences to choose from coral, cmd, mkmmd

TRAIN_PROP=1.0
DEV_PROP=1.0
EXP_DIR=${OUTPUT_DIR}
SEEDS=(1729 100 1000)
BSZ=32
DIVERGENCE=mkmmd
EPOCHS=10
DATA_MODULE=sa
MAX_SEQ_LENGTH=128
PADDING=max_length
LR=1e-05
GPU=0
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/train_domain_adapter.py"
SRC_DOMAINS=("apparel" "baby" "books" "camera_photo" "MR")
TRG_DOMAINS=("apparel" "baby" "books" "camera_photo" "MR")

for src in "${SRC_DOMAINS[@]}"; do
    for trg in "${TRG_DOMAINS[@]}"; do
      for SEED in ${SEEDS[@]}; do
          if [ ${src} = ${trg} ]; then
            echo "SKIPPING ${src}-${trg}";
            continue
          else
            python ${PYTHON_FILE} \
                --dataset-cache-dir ${DATASET_CACHE_DIR} \
                --source-target  "${src}_${trg}" \
                --pretrained-model-name "bert-base-uncased" \
                --seed ${SEED} \
                --data-module ${DATA_MODULE} \
                --divergence ${DIVERGENCE} \
                --train-proportion ${TRAIN_PROP} \
                --dev-proportion ${DEV_PROP} \
                --gpu ${GPU} \
                --max-seq-length ${MAX_SEQ_LENGTH} \
                --padding ${PADDING} \
                --lr ${LR} \
                --log-freq 5 \
                --epochs ${EPOCHS} \
                --bsz ${BSZ} \
                --exp-dir ${EXP_DIR}
            fi
      done
    done
done