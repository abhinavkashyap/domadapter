# UDAPTER - Efficient Domain Adaptation Using Adapters

`Domadapter` trains adapters for Domain Adaptation in NLP.

## Development Prerequisites

- Python >= 3.8
- [Poetry](https://python-poetry.org/) for dependency and environment management

We use environment variables to store paths to store

- Pretrained Transformer Models
- Datasets
- Experiments
- Results

## Installation

**Creating a virtualenv**

- run `poetry install` to install the dependencies
- Use `poetry shell` to create a virtual environment

Commit your `poetry.lock` file if you install any new library.

**Export Environment Variable**

- `export DATASET_CACHE_DIR="/content/domadapter/data/"`
- `export PT_MODELS_CACHE_DIR="/content/domadapter/models"`
- `export PROJECT_ROOT="/content/domadapter/"`
- `export OUTPUT_DIR="/content/domadapter/experiments/`

**Downloading Datasets**

- Run `domadapter download mnli` to download the `mnli` dataset
- Run `domadapter download sa` to download the `amazon` dataset. This will not run in this version as Google Drive folder id is censored to maintain anonymity.

If running this on Google Colab, then:
`!pip install pytorch-lightning==1.4.2 datasets transformers pandas click wandb numpy rich` should suffice`

---
## Relevant files and folders
```
|- domadapter
|   |-- commands
|   |   |-- download.py (Module to download datasets)
|   |-- datamodules
|   |   |-- mnli_dm.py (Data module for MNLI dataset)
|   |   |-- sa_dm.py (Data module for Amazon dataset)
|   |-- divergences
|   |   |-- mkmmd_divergence.py (Module for calculating MK-MMD divergence)
|   |-- models
|   |   |-- ablations (Model files for `ablation` experiments)
|   |   |   |-- domain_adapter.py
|   |   |   |-- domain_task_adapter.py
|   |   |   |-- joint_domain_task_adapter.py
|   |   |-- adapters **(Model files for `adapter-based` experiments)**
|   |   |   |-- dann_adapter.py
|   |   |   |-- dann_adapter_multiple_classifier.py
|   |   |   |-- domain_adapter.py
|   |   |   |-- domain_task_adapter.py
|   |   |   |-- joint_domain_task_adapter.py
|   |   |-- ft (Model file for `finetuning` experiment)
|   |   |   |-- finetune.py
|   |   |-- uda (Model files for `uda` experiments)
|   |   |   |-- dann.py
|   |   |   |-- dsn.py
|   |-- orchestration (main python training file. train_*.py where "*" is method name)
|   |   |-- ablations
|   |   |   |-- train_domain_adapter.py
|   |   |   |-- train_domain_task_adapter.py
|   |   |   |-- train_joint_domain_task_adapter.py
|   |   |-- train_dann.py
|   |   |-- train_dann_adapter.py
|   |   |-- ...
|   |   |-- ...
|   |-- scripts (contains bash files to run training scripts. *_sa.sh is for Amazon dataset, *_mnli.sh for MNLI)
|   |   |-- ablations
|   |   |   |-- ...
|   |   |   |-- ...
|   |   |-- dann
|   |   |   |-- train_dann_mnli.sh
|   |   |   |-- train_dann_sa.sh
|   |   |-- dann_adapter
|   |   |   |-- train_dann_adapter_mnli.sh
|   |   |   |-- train_dann_adapter_sa.sh
|   |   |-- dann_adapter_multiple_classifier
|   |   |   |-- train_dann_adapter_mnli.sh
|   |   |   |-- train_dann_adapter_sa.sh
|   |   |-- domain_adapter
|   |   |   |-- train_da_mnli.sh
|   |   |   |-- train_da_sa.sh
|   |   |-- domain_task_adapter
|   |   |   |-- train_da_ta_mnli.sh
|   |   |   |-- train_da_ta_sa.sh
|   |   |-- dsn
|   |   |   |-- train_dsn_mnli.sh
|   |   |   |-- train_dsn_sa.sh
|   |   |-- ft
|   |   |   |-- train_ft_mnli.sh
|   |   |   |-- train_ft_sa.sh
|   |   |-- joint_domain_task_adapter
|   |       |-- train_joint_da_ta_mnli.sh
|   |       |-- train_joint_da_ta_sa.sh
|   |-- utils
|   |   |-- plotting
|   |   |   |-- adapter_representations.py (script for plotting t-SNE representation from adapters)
|   |   |-- run_prediction.sh (bash file for running adapter_representations.py)
|-- README.md
```
---

**Parameters**

The values in the brackets are suggested/possible defaults.

- SCRIPT_FILE - Scripts that train the model
- `--dataset_cache_dir` (DATASET_CACHE_DIR) - Directory to store datasets. Defined in environments
- `--source-target` (`fiction_slate`, `apparel_baby` etc) - source and target domain separated by "_". For MNLI, domains are ["fiction", "slate", "government", "travel", "telephone"] and for Amazon, domains are ["apparel", "baby", "books", "camera_photo", "MR"]
- `--padding`(max_length) - Pad sentences to maximum length
- `--max-seq-length` (128) - Maximum Length of tokens
- `--pretrained-model-name` ("bert-base-uncased") - HF Pretrained Model Name
- `--bsz` (32) - Batch Size
- `--exp-dir` - Experiment directory
- `--seed` - Seed to reproduce experiments
- `--data-module`(mnli/sa) - MNLI dataset or Amazon dataset
- `--train-proportion` (1.0) - Proportion of train data used during training
- `--dev-proportion` (1.0) - Proportion of validation data used during validation
- `--test-proportion` (1.0) - Proportion of test data used for testing
- `--gradient_clip_val` (5.0) - Gradient norms greater than this value will be clipped during training
- `--epochs` (10) - Number of total number of epochs
- `--lr` - Learning rate for the optimizer
- `--gpus\` (0) - A GPU string which mentions the GPU number used for training
- `--num-classes` (2/3) - 2 for Amazon dataset; 3 for MNLI dataset
- `--divergence` (mkmmd) - Multiple Kernel Maximum Mean Discrepancy to be used for training adapter
- `--mode` (task/domain) - train TASK-🔌 or train TS-DT-🔌
- `--reduction-factor` (2,4,8,32,64,128) - String value for bottleneck size of adapters
- `--skip-layers` (0,1,2) - Layers to be skipped from adding adapters separated by ","
