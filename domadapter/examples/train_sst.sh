#!/usr/bin/env bash

DATASET_CACHE="/ssd1/abhinav/domadapter/data/"
PT_MODELS_CACHE="/ssd1/abhinav/domadapter/pretrained_models_cache/"
OUTPUT_DIR="/ssd1/abhinav/domadapter/experiments/trial_run"

python train_sst.py \
--model_name_or_path bert-base-uncased \
--task_name sst2 \
--do_train \
--do_eval \
--cache_dir ${PT_MODELS_CACHE} \
--max_seq_length 128 \
--per_gpu_train_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 1.0 \
--output_dir ${OUTPUT_DIR} \
--dataset_cache_dir ${DATASET_CACHE}