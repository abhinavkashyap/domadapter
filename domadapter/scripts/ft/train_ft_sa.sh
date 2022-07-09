#!/usr/bin/env bash
# Train domain (frozen), task adapter for 5 domains "fiction" "travel" "slate" "government" "telephone"

TRAIN_PROP=1.0
DEV_PROP=1.0
TEST_PROP=1.0
EXP_DIR=${OUTPUT_DIR}
SEEDS=(1729 100 1000)
DATA_MODULE=sa
BSZ=32
EPOCHS=5
MAX_SEQ_LENGTH=128
PADDING=max_length
NUM_CLASSES=2
LR=2e-05
GPU=0
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/train_ft.py"
# only first domain before "_" in DOMAINS will be considered for training.
# For eg, in "fiction_travel" model will be finetuned on "fiction" domain
DOMAINS=("apparel_baby" "baby_books" "books_camera_photo" "camera_photo_MR" "MR_apparel")

for src in "${DOMAINS[@]}"; do
    for SEED in ${SEEDS[@]}; do
        python ${PYTHON_FILE} \
            --dataset-cache-dir ${DATASET_CACHE_DIR} \
            --source-target  ${src} \
            --pretrained-model-name "bert-base-uncased" \
            --seed ${SEED} \
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
    done
done