#!/usr/bin/env bash

SCRIPT_FILE=${PROJECT_ROOT}/domadapter/orchestration/task_adapter.py

python ${SCRIPT_FILE} \
--task_name "cola" \
--tokenizer_name "bert-base-uncased" \
--pad_to_max_length \
--max_seq_length 128 \
--model_name "bert-base-uncased" \
--batch_size 32
