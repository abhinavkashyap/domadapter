#!/usr/bin/env bash
# Tran a deep adaptation networks given a divergence measure

TRAIN_PROP=1
DEV_PROP=1
TEST_PROP=1
EXP_NAME="[TEST_DAN]"
EXP_DIR=${OUTPUT_DIR}
SEED=1729
BSZ=8
EPOCHS=1
LR=0.0001
TOKENIZER_TYPE="bert"
TRAIN_AT_LAYER=11
NUM_CLF_LAYERS=3
CLF_HIDDEN_SIZE=200
FREEZE_UPTO=0
GRAD_CLIP_NORM=1.0
GPU=0
DIVERGENCE="rbf"
DIV_REG_PARAM=0.2
WANDB_PROJ_NAME="DAN"
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/train_dan.py"

python ${PYTHON_FILE} \
--src-train-file ${DATASET_CACHE_DIR}/"sa/camera_photo.task.train" \
--src-dev-file ${DATASET_CACHE_DIR}/"sa/camera_photo.task.dev" \
--src-test-file ${DATASET_CACHE_DIR}/"sa/camera_photo.task.test" \
--trg-train-file ${DATASET_CACHE_DIR}/"sa/apparel.task.train" \
--trg-dev-file ${DATASET_CACHE_DIR}/"sa/apparel.task.dev" \
--trg-test-file ${DATASET_CACHE_DIR}/"sa/apparel.task.test" \
--label-file ${DATASET_CACHE_DIR}/"sa/labels.txt" \
--tokenizer-type ${TOKENIZER_TYPE} \
--bsz ${BSZ} \
--dataset-cache-dir ${DATASET_CACHE_DIR}/"features" \
--train-bert-at-layer ${TRAIN_AT_LAYER} \
--num-clf-layers ${NUM_CLF_LAYERS} \
--clf-hidden-size ${CLF_HIDDEN_SIZE} \
--freeze-upto ${FREEZE_UPTO} \
--train-proportion ${TRAIN_PROP} \
--dev-proportion ${DEV_PROP} \
--test-proportion ${TEST_PROP} \
--exp-name ${EXP_NAME} \
--exp-dir ${EXP_DIR} \
--seed ${SEED} \
--lr ${LR} \
--epochs ${EPOCHS} \
--gpu ${GPU} \
--grad-clip-norm ${GRAD_CLIP_NORM} \
--is-divergence-reduced \
--div-reg-param ${DIV_REG_PARAM} \
--divergence-reduced ${DIVERGENCE} \
--wandb-proj-name ${WANDB_PROJ_NAME}
