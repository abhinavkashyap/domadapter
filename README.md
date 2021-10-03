# Project Title

`Domadapter` aims to train adapters for NLP domain adaptation.

_This is a work in progress. Contact the authors if you face difficulties using it_

## Development Prerequisites

- Python >= 3.8
- [Poetry](https://python-poetry.org/) for dependency and environment management
- [Direnv](https://direnv.net/) to automatically export environment vairables

We use environment variables to store paths to store

- Pretrained Transformer Models
- Datasets
- Experiments
- Results

## Installation

**_Creating a virtualenv_**

- Use `poetry shell` to create a virtual environment
- run `poetry install` to install the dependencies

Commit your `poetry.lock` file if you install any new library.

**Downloading Datasets**

Run `domadapter download mnli` to download the `mnli` datasets

Run `domadapter download sa` to download the `sa` datasets

## Running Tests (Tentative)

We are using [https://ward.readthedocs.io](https://ward.readthedocs.io) to run our tests.
Run `ward --path tests` to run all the tests. Currently the tests are really slow.
It takes a lot of time.

## Common Problems

- (September 29th) - `cucumber-tag-expressions` install failing on newer versions of setuptools.
  Remove `ward` from `pyproject.toml`file and run `poetry install`. Cannot run the tests for now
- Refer to [Common Build Problems](https://github.com/pyenv/pyenv/wiki/Common-build-problems) if you
  run into problems installing pyenv

---

## Baselines

### Finetuning

**MNLI**

- Finetunes a pretrained model on MNLI for every genre. We sample 90% of data from every genre for fnetuning

Goto `domadapter/scripts/glue_ft` and run `bash mnlit_ft.sh`

**Parameters**

The values in the brackets are suggeted defaults.

`Tip`: To train a model on a particular genre, change the value you pass to `--multinli-genre`.
It can be one of `["fiction", "slate", "government", "travel", "telephone"]`

- SCRIPT_FILE - Scripts trains the model
- `--task-name` ("mnli") - Name of the GLUE task. One of `["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli" ]`
- `--tokenizer-name` ("bert-base-uncased") - The name of pretrained tokenizer
- `--pad_to_max_length` - Pad sentences to maximum length
- `--max_seq_length` (128) - Maximum Length of tokens
- `--model_name` ("bert-base-uncased") - HF Pretrained Model Name
- `--batch_size` (32) - Batch Size
- `--dataset_cache_dir` (DATASET_CACHE_DIR) - Directory to store datasets. Defined in .envrc file
- `--cache_dir` - (PT_MODELS_CACHE_DIR) - Directory to store pretrained models cache directory. Defined in .envrc file
- `--exp_name` - Experiment name
- `--wandb_proj_name` - Weights and Biases Project Name.
- `--seed` - Seed to reproduce experiments
- `--train_data_proportion` (1.0) - Proportion of train data used during training
- `--validation_data_proportion` (1.0) - Proportion of validation data used during validation
- `--test_data_proportion` (1.0) - Proportion of test data used for testing
- `--gradient_clip_val` (5.0) - Gradient norms greater than this value will be clipped during training
- `--num_epochs` (5) - Number of total number of epochs
- `--adam_beta_1` (0.9) - Adam optimizer beta 1 value.
- `--adam_beta_2` (0.999) - Adam optimizer beta 2 value
- `--adam_epsilon` (1e-8) - Adam epsilon values
- `--learning_rate` - Learning rate for the optimizer
- `--gpus` (0) - A GPU string which mentions the GPU number used for training
- `--num_processes` (16) - The number of processes
- `--monitor_metric` (f1) - One of `["f1", "accuracy"]`. Saves the best model based on development `f1`
  score
- `--multinli_genre` ("travel") - One of the 5 genres. If nothing is mentioned then we train on 100%
- `--sample_proportion` (0.9) - The sample of training data considered for finetuning

### Training Domain Adapters

Domain adapters are aimed at extracting domain invariant representations.

**Directly Reducing Domain Divergences**

Goto `domadapters/domain_adapter/train_da.sh`

### Stacking Domain and Task Adapters

### Collating Results

We use [`luigi`](https://github.com/spotify/luigi) to collect results.
Result collation code is available in `domadapter/utils/results`.

**MNLI Fine Tuning**

Pipelines for MNLI FT result collation is available under `mnli_ft` folder.

`MNLIResultForOneExperiment`

- creates a CSV result file for one MNLI experiment.
- For example, we store experiments for `travel` is under `/path/to/travel_experiments_folder/`. Inside this folder are folders for `experiment 1`, `experiment 2`. This class considers one experiment, say `experiment 1`, runs inference on them and writes a file in the results_folder

`CollateMNLIMeanStdAccForOneDomain`

- Aggreates results for all experiments stored under `/path/to/travel_experiment_folder`.
- Collects the results produced by `MNLIResulForOneExperiment` and writes another csv file with the mean and standard deviation of results for the domain

> Note: We can add more reports later. For example, if we want to write all the predictions of an experiment to a file, collate compare, build introspection utilities on top of them etc.
