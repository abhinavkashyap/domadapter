#!/usr/bin/env bash

SCRIPT_FILE=${PROJECT_ROOT}/domadapter/orchestration/train_glue_ft.py
GLUE_TASK_NAME="mnli"
TOKENIZER_NAME="bert-base-uncased"
MAX_SEQ_LENGTH=128
MODEL_NAME="bert-base-uncased"
BSZ=32
EXPERIMENT_NAME="[MNLI_FT]"
WANDB_PROJ_NAME="MNLI_${MODEL_NAME}"
SEED=1729
TRAIN_PROPORTION=1.0
VALIDATION_PROPORTION=1.0
TEST_PROPORTION=1.0
GRADIENT_CLIP_VAL=5.0
EPOCHS=1
ADAM_BETA1=0.99
ADAM_BETA2=0.999
ADAM_EPSILON=1e-8
LEARNING_RATE=1e-4
GPUS="0"
NUM_PROCESSES=32
MONITOR_METRIC="accuracy"
MNLI_GENRE="fiction"

python ${SCRIPT_FILE} \
--task_name ${GLUE_TASK_NAME} \
--tokenizer_name ${TOKENIZER_NAME} \
--pad_to_max_length \
--max_seq_length ${MAX_SEQ_LENGTH} \
--model_name ${MODEL_NAME} \
--batch_size ${BSZ} \
--dataset_cache_dir ${DATASET_CACHE_DIR} \
--cache_dir ${PT_MODELS_CACHE_DIR} \
--exp_name ${EXPERIMENT_NAME} \
--wandb_proj_name ${WANDB_PROJ_NAME} \
--seed ${SEED} \
--train_data_proportion ${TRAIN_PROPORTION} \
--validation_data_proportion ${VALIDATION_PROPORTION} \
--test_data_proportion ${TEST_PROPORTION} \
--gradient_clip_val ${GRADIENT_CLIP_VAL} \
--num_epochs ${EPOCHS} \
--adam_beta1 ${ADAM_BETA1} \
--adam_beta2 ${ADAM_BETA2} \
--adam_epsilon ${ADAM_EPSILON} \
--learning_rate ${LEARNING_RATE} \
--gpus ${GPUS} \
--num_processes ${NUM_PROCESSES} \
--monitor_metric ${MONITOR_METRIC} \
--mnli_genre ${MNLI_GENRE}