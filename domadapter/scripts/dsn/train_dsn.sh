#!/usr/bin/env bash
# Train domain (frozen), task adapter for 5 domains "fiction" "travel" "slate" "government" "telephone"
# Keep HIDDEN_SIZE 128 instead of 100

TRAIN_PROP=1.0
DEV_PROP=1.0
TEST_PROP=1.0
EXP_DIR=${OUTPUT_DIR}
SEEDS=(1729 100 1000)
HIDDEN_SIZE=768
BSZ=32
EPOCHS=10
MAX_SEQ_LENGTH=128
PADDING=max_length
NUM_CLASSES=3
DIFF_WEIGHT=0.3
SIM_WEIGHT=0.1
RECON_WEIGHT=0.1
LRS=(1e-05)
GPU=0
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/train_dsn.py"
SRC_DOMAINS=("fiction" "travel" "slate" "government" "telephone")
TRG_DOMAINS=("fiction" "travel" "slate" "government" "telephone")

for src in "${SRC_DOMAINS[@]}"; do
    for trg in "${TRG_DOMAINS[@]}"; do
         for LR in "${LRS[@]}";
         do
           for SEED in ${SEEDS[@]}; do
             if [ ${src} = ${trg} ]; then
                echo "SKIPPING ${src}-${trg}";
                continue
              elif [ ${src} = "fiction" ] && [ ${trg} = "slate" ]; then
                echo "SKIPPING ${src}-${trg}";
                continue
            else
              python "${PYTHON_FILE}" \
                  --dataset-cache-dir "${DATASET_CACHE_DIR}" \
                  --source-target  "${src}_${trg}" \
                  --pretrained-model-name "bert-base-uncased" \
                  --seed ${SEED} \
                  --train-proportion ${TRAIN_PROP} \
                  --dev-proportion ${DEV_PROP} \
                  --test-proportion ${TEST_PROP} \
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