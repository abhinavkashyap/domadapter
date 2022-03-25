#!/usr/bin/env bash
# Train domain (frozen), task adapter for 5 domains "fiction" "travel" "slate" "government" "telephone"
# losses to choose from coral, cmd, mkmmd
# data_module to choose from mnli, sa (num_classes 2)

TRAIN_PROP=1.0
DEV_PROP=1.0
TEST_PROP=1.0
EXP_DIR=${OUTPUT_DIR}
SEEDS=(1729 100 1000)
DIVERGENCE=mkmmd
BSZ=32
DATA_MODULE=sa
EPOCHS=10
MAX_SEQ_LENGTH=128
PADDING=max_length
NUM_CLASSES=3
LR=1e-04
GPU=0
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/train_joint_domain_task_adapter.py"
SRC_DOMAINS=("fiction" "travel" "slate" "government" "telephone")
TRG_DOMAINS=("fiction" "travel" "slate" "government" "telephone")

for src in "${SRC_DOMAINS[@]}"; do
    for trg in "${TRG_DOMAINS[@]}"; do
      for SEED in ${SEEDS[@]}; do
        if [ ${src} = ${trg} ]; then
          echo "SKIPPING ${src}-${trg}";
          continue
        elif [ ${src} = "fiction" ] && [ ${trg} = "slate" ]; then
          echo "SKIPPING ${src}-${trg}";
          continue
        else
          python ${PYTHON_FILE} \
              --dataset-cache-dir ${DATASET_CACHE_DIR} \
              --source-target  "${src}_${trg}" \
              --pretrained-model-name "bert-base-uncased" \
              --seed ${SEED} \
              --divergence ${DIVERGENCE} \
              --data-module ${DATA_MODULE} \
              --train-proportion ${TRAIN_PROP} \
              --dev-proportion ${DEV_PROP} \
              --test-proportion ${TEST_PROP} \
              --gpu ${GPU} \
              --num-classes ${NUM_CLASSES} \
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