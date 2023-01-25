#!/usr/bin/env bash
# Train DSN Adapter for SA

TRAIN_PROP=1.0
DEV_PROP=1.0
TEST_PROP=1.0
EXP_DIR=${OUTPUT_DIR}
SEEDS=(1729 100 1000)
HIDDEN_SIZE=128
BSZ=32
EPOCHS=10
DATA_MODULE=sa
MAX_SEQ_LENGTH=128
PADDING=max_length
NUM_CLASSES=2
DIFF_WEIGHT=0.1
SIM_WEIGHT=0.1
RECON_WEIGHT=0.1
LRS=(3e-05)
GPU=1
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/train_dsn_adapter.py"
SRC_DOMAINS=("apparel" "baby" "books" "camera_photo" "MR")
TRG_DOMAINS=("apparel" "baby" "books" "camera_photo" "MR")

for src in "${SRC_DOMAINS[@]}"; do
    for trg in "${TRG_DOMAINS[@]}"; do
        for LR in "${LRS[@]}"; do
            for SEED in ${SEEDS[@]}; do
                if [ ${src} = ${trg} ]; then
                    echo "SKIPPING ${src}-${trg}"
                    continue
                else
                    python "${PYTHON_FILE}" \
                        --dataset-cache-dir "${DATASET_CACHE_DIR}" \
                        --source-target "${src}_${trg}" \
                        --pretrained-model-name "bert-base-uncased" \
                        --seed ${SEED} \
                        --train-proportion ${TRAIN_PROP} \
                        --dev-proportion ${DEV_PROP} \
                        --test-proportion ${TEST_PROP} \
                        --data-module ${DATA_MODULE} \
                        --gpu ${GPU} \
                        --hidden-size ${HIDDEN_SIZE} \
                        --num-classes ${NUM_CLASSES} \
                        --max-seq-length ${MAX_SEQ_LENGTH} \
                        --padding ${PADDING} \
                        --lr "${LR}" \
                        --log-freq 5 \
                        --epochs ${EPOCHS} \
                        --bsz ${BSZ} \
                        --diff-weight ${DIFF_WEIGHT} \
                        --sim-weight ${SIM_WEIGHT} \
                        --recon-weight ${RECON_WEIGHT} \
                        --exp-dir "${EXP_DIR}"
                fi
            done
        done
    done
done
