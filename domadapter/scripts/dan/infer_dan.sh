#!/usr/bin/env bash

EXP_DIR=${OUTPUT_DIR}
EXP_NAME="[TEST_DAN]"
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/infer/dan_infer.py"

python ${PYTHON_FILE} \
--experiment-dir ${EXP_DIR}/${EXP_NAME} \
--infer-filename ${DATASET_CACHE_DIR}/"data/apparel.task.test" \
--use-infer-branch src