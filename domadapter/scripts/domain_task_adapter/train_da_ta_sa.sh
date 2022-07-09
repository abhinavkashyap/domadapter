#!/usr/bin/env bash
# Train domain (frozen), task adapter for 5 domains "fiction" "travel" "slate" "government" "telephone"
# losses to choose from coral, cmd, mkmmd

TRAIN_PROP=1.0
DEV_PROP=1.0
TEST_PROP=1.0
EXP_DIR=${OUTPUT_DIR}
SEEDS=(1729 100 1000)
DIVERGENCE=mkmmd
MODE=domain
BSZ=32
DATA_MODULE=sa
EPOCHS=10
MAX_SEQ_LENGTH=128
PADDING=max_length
REDUCTION_FACTOR=32
NUM_CLASSES=2
LR=1e-04
GPU=1
PYTHON_FILE=${PROJECT_ROOT}/"domadapter/orchestration/train_domain_task_adapter.py"
SRC_DOMAINS=("apparel" "baby" "books" "camera_photo" "MR")
TRG_DOMAINS=("apparel" "baby" "books" "camera_photo" "MR")
DOMAIN_ADAPTER_WANDB_id=(1q46nbuw 2hm53tyg 212yhirc 1no49m56 27efiigz 392kngy3 1ta8oiqr 2bc2me6x 1nvw5pnh 2j6ff13c 2l7gq6de 2vtkcyg3 1pb53tgd 38ohni88 3hpr6nqf 335aad8u 31orafiu 3ozkvfbu 2vybdw6j 3p7g6997 3ulv23a3 x5a2p3ly 3qnvz142 1lkg9jbs dm9zvye6 3pupofrx gh8weev5 2hfbgaqv 2583d06l 3dx2x09d 2t7p8mb2 3oh4utca 3megivpw 56g93ojx 1lugmav2 39f54jnf 8npxjztj 1buv1whl 2zx7dn10 16nlp83v qgqj7wqo 1yhpdivm 2dl5vh06 tcrc0wxm 2awiy2rl 2g7yr9oj 1ej1w0ap e9s7dpht 1xirez89 1u4xu3l5 cifmd3me 189bu86l w0empf9h 1c03rwct 1tcqy1ga 3h6bfy89 10nhhgkf 215rrjpn 10mu5j4c 2tted8ku)
COUNTER=0

for src in "${SRC_DOMAINS[@]}"; do
    for trg in "${TRG_DOMAINS[@]}"; do
      for SEED in ${SEEDS[@]}; do
          if [ ${src} = ${trg} ]; then
            echo "SKIPPING ${src}-${trg}";
            continue
          else
            echo ${COUNTER}
            python ${PYTHON_FILE} \
                --dataset-cache-dir ${DATASET_CACHE_DIR} \
                --source-target  "${src}_${trg}" \
                --pretrained-model-name "bert-base-uncased" \
                --seed ${SEED} \
                --reduction-factor ${REDUCTION_FACTOR} \
                --data-module ${DATA_MODULE} \
                --divergence ${DIVERGENCE} \
                --train-proportion ${TRAIN_PROP} \
                --dev-proportion ${DEV_PROP} \
                --test-proportion ${TEST_PROP} \
                --domain-adapter-id ${DOMAIN_ADAPTER_WANDB_id[COUNTER]} \
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
          fi
      done
    done
done