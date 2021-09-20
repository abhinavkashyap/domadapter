#!/usr/bin/env bash

SCRIPT_FILE=${PROJECT_ROOT}/domadapter/orchestration/task_adapter.py
GLUE_TASK_NAME="sst2"
TOKENIZER_NAME="bert-base-uncased"
MAX_SEQ_LENGTH=128
MODEL_NAME="bert-base-uncased"
BSZ=32
ADAPTER_NAME="dummy_adapter"
EXPERIMENT_NAME="[DEBUG_TASK_ADAPTERS]"
WANDB_PROJ_NAME="ADAPTERS"
SEED=1729
TRAIN_PROPORTION=0.1
VALIDATION_PROPORTION=1.0
TEST_PROPORTION=1.0
GRADIENT_CLIP_VAL=5.0
EPOCHS=2
ADAM_BETA1=0.99
ADAM_BETA2=0.999
ADAM_EPSILON=1e-8
LEARNING_RATE=1e-4
ADAPTER_REDUCTION_FACTOR=32
GPUS="0"


python ${SCRIPT_FILE} \
--task_name ${GLUE_TASK_NAME} \
--tokenizer_name ${TOKENIZER_NAME} \
--pad_to_max_length \
--max_seq_length ${MAX_SEQ_LENGTH} \
--model_name ${MODEL_NAME} \
--batch_size ${BSZ} \
--dataset_cache_dir ${DATASET_CACHE_DIR} \
--cache_dir ${PT_MODELS_CACHE_DIR} \
--adapter_name ${ADAPTER_NAME} \
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
--adapter_reduction_factor ${ADAPTER_REDUCTION_FACTOR} \
--gpus ${GPUS}

