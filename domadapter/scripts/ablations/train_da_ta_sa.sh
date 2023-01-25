#!/usr/bin/env bash
# Train domain (frozen), task adapter for 5 domains "fiction" "travel" "slate" "government" "telephone"
# losses to choose from coral, cmd, mkmmd
# ABLATION: Reduction Factor Ablation

TRAIN_PROP=1.0
DEV_PROP=1.0
TEST_PROP=1.0
EXP_DIR=${OUTPUT_DIR}
SEED=(100 1000)
DIVERGENCE=mkmmd
MODE=domain
BSZ=32
DATA_MODULE=sa
EPOCHS=10
REDUCTION_FACTOR=(2 4 8 16 32 64 128)
SKIP_LAYERS="None"
MAX_SEQ_LENGTH=128
PADDING=max_length
NUM_CLASSES=2
LR=1e-04
GPU=0
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/ablations/train_domain_task_adapter.py"
DOMAINS=("camera_photo_baby")
DOMAIN_ADAPTER_WANDB_id=(1889z6y6 jy8w2flf 2zb6frrr 27mkjolx 169bhr85 18tib1sg 1sx1hmr4 3k9m4dgr 26c1jyrh 8rbtwzht 1yc88soo 1p1hj1lk 1dxej9aa 1fl2is84)
COUNTER=0

for domain in "${DOMAINS[@]}"; do
    for red in "${REDUCTION_FACTOR[@]}"; do
        for seed in "${SEED[@]}"; do
            echo ${COUNTER}
            python ${PYTHON_FILE} \
                --dataset-cache-dir ${DATASET_CACHE_DIR} \
                --source-target "${domain}" \
                --pretrained-model-name "bert-base-uncased" \
                --seed ${seed} \
                --divergence ${DIVERGENCE} \
                --data-module ${DATA_MODULE} \
                --train-proportion ${TRAIN_PROP} \
                --dev-proportion ${DEV_PROP} \
                --test-proportion ${TEST_PROP} \
                --domain-adapter-id ${DOMAIN_ADAPTER_WANDB_id[COUNTER]} \
                --reduction-factor ${red} \
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
            COUNTER=$(($COUNTER + 1))
        done
    done
done
