#!/usr/bin/env bash
# Train domain (frozen), task adapter for 5 domains "fiction" "travel" "slate" "government" "telephone"
# losses to choose from coral, cmd, mkmmd

TRAIN_PROP=1.0
DEV_PROP=1.0
TEST_PROP=1.0
EXP_DIR=${OUTPUT_DIR}
SEED=1729
DIVERGENCE=mkmmd
MODE=domain
DOMAIN_ADAPTER_WANDB_id=jqjlsvhw
BSZ=32
EPOCHS=10
REDUCTION_FACTOR=16
SKIP_LAYERS=0,1,2,3,4,5,6,7
MAX_SEQ_LENGTH=128
PADDING=max_length
NUM_CLASSES=3
LR=1e-04
GPU=0
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/ablations/train_domain_task_adapter.py"

for i in "fiction"; do
    for j in "slate"; do
        python ${PYTHON_FILE} \
            --dataset-cache-dir ${DATASET_CACHE_DIR} \
            --source-target  "${i}_${j}" \
            --pretrained-model-name "bert-base-uncased" \
            --seed ${SEED} \
            --divergence ${DIVERGENCE} \
            --train-proportion ${TRAIN_PROP} \
            --dev-proportion ${DEV_PROP} \
            --test-proportion ${TEST_PROP} \
            --domain-adapter-id ${DOMAIN_ADAPTER_WANDB_id} \
            --reduction-factor ${REDUCTION_FACTOR} \
            --skip-layers ${SKIP_LAYERS} \
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
    done
done