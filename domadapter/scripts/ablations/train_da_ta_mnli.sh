#!/usr/bin/env bash
# Train domain (frozen), task adapter for 5 domains "fiction" "travel" "slate" "government" "telephone"
# losses to choose from coral, cmd, mkmmd

TRAIN_PROP=1.0
DEV_PROP=1.0
TEST_PROP=1.0
EXP_DIR=${OUTPUT_DIR}
SEED=(100 1000)
DIVERGENCE=mkmmd
MODE=domain
BSZ=32
DATA_MODULE=mnli
EPOCHS=10
# REDUCTION_FACTOR=(2 4 8 16 32 64 128)
REDUCTION_FACTOR="None"
SKIP_LAYERS=(0 0,1 0,1,2 0,1,2,3 0,1,2,3,4 0,1,2,3,4,5 0,1,2,3,4,5,6 0,1,2,3,4,5,6,7 0,1,2,3,4,5,6,7,8 0,1,2,3,4,5,6,7,8,9)
# SKIP_LAYERS="None"
MAX_SEQ_LENGTH=128
PADDING=max_length
NUM_CLASSES=3
LR=1e-04
GPU=0
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/ablations/train_domain_task_adapter.py"
DOMAINS=("slate_travel")
DOMAIN_ADAPTER_WANDB_id=(1lf0h6w5 1wc1bu4v 13nej0n7 8itsa3kg 2b0423hg 1n3qutm0 1mti017g 29b1epx8 1k27lezb 39ln5xo1 3t955844 30k8zyd0 3h6e4oaz dy1qemzp 1juz049w 3pz2jf6z 3ocp5jxv 3usimd51 16uwycuc 18trv17u)
COUNTER=0

for domain in "${DOMAINS[@]}"; do
    for skip in "${SKIP_LAYERS[@]}"; do
        for seed in "${SEED[@]}"; do
            echo ${COUNTER}
            python ${PYTHON_FILE} \
                --dataset-cache-dir ${DATASET_CACHE_DIR} \
                --source-target  "${domain}" \
                --pretrained-model-name "bert-base-uncased" \
                --seed ${seed} \
                --divergence ${DIVERGENCE} \
                --data-module ${DATA_MODULE} \
                --train-proportion ${TRAIN_PROP} \
                --dev-proportion ${DEV_PROP} \
                --test-proportion ${TEST_PROP} \
                --domain-adapter-id ${DOMAIN_ADAPTER_WANDB_id[COUNTER]} \
                --reduction-factor ${REDUCTION_FACTOR} \
                --skip-layers ${skip} \
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
        done
    done
done