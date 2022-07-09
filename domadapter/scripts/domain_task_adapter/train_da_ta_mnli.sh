#!/usr/bin/env bash
# Train domain (frozen), task adapter for 5 domains "fiction" "travel" "slate" "government" "telephone"
# losses to choose from coral, cmd, mkmmd

TRAIN_PROP=1.0
DEV_PROP=1.0
TEST_PROP=1.0
EXP_DIR=${OUTPUT_DIR}
SEEDS=(1729 100 1000)
DIVERGENCE=mkmmd
MODE=domain
BSZ=32
DATA_MODULE=mnli
EPOCHS=10
MAX_SEQ_LENGTH=128
REDUCTION_FACTOR=32
PADDING=max_length
NUM_CLASSES=3
LR=1e-04
GPU=0
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/train_domain_task_adapter.py"
SRC_DOMAINS=("slate")
TRG_DOMAINS=("travel")
DOMAIN_ADAPTER_WANDB_id=(2e8e35pc 1wnnonzk 1itswhu7)
COUNTER=0

for src in "${SRC_DOMAINS[@]}"; do
    for trg in "${TRG_DOMAINS[@]}"; do
      for SEED in ${SEEDS[@]}; do
          if [ ${src} = ${trg} ]; then
            echo "SKIPPING ${src}-${trg}";
            continue
          else
            echo ${COUNTER}
            python ${PYTHON_FILE} \
                --dataset-cache-dir ${DATASET_CACHE_DIR} \
                --source-target  "${src}_${trg}" \
                --pretrained-model-name "bert-base-uncased" \
                --seed ${SEED} \
                --reduction-factor ${REDUCTION_FACTOR} \
                --data-module ${DATA_MODULE} \
                --divergence ${DIVERGENCE} \
                --train-proportion ${TRAIN_PROP} \
                --dev-proportion ${DEV_PROP} \
                --test-proportion ${TEST_PROP} \
                --domain-adapter-id ${DOMAIN_ADAPTER_WANDB_id[COUNTER]} \
                --gpu ${GPU} \
                --mode ${MODE} \
                --num-classes ${NUM_CLASSES} \
                --max-seq-length ${MAX_SEQ_LENGTH} \
                --padding ${PADDING} \
                --lr ${LR} \
                --log-freq 5 \
                --epochs ${EPOCHS} \
                --bsz ${BSZ} \
                --exp-dir ${EXP_DIR}
            COUNTER=$[$COUNTER +1]
          fi
      done
    done
done