# UDAPTER - Efficient Domain Adaptation Using Adapters

`Domadapter` trains adapters for Domain Adaptation in NLP. The idea is to use principles 
of unsupervised domain adaptation and parameter efficient fine-tuning to make domain 
adaptation more efficient. 

# Training Requirements


- Python >= 3.8
- [Poetry](https://python-poetry.org/) for dependency and environment management

## Environment Variables 
We use environment variables to store certain paths

- Pretrained Transformer Models (PT_MODEL_CACHE_DIR)
- Datasets (DATASET_CACHE_DIR)
- Experiments (OUTPUT_DIR)
- Results (OUTPUT_DIR)

Change the following variables in the .envrc file.

- `export DATASET_CACHE_DIR=""`
- `export PT_MODELS_CACHE_DIR=""`
- `export OUTPUT_DIR=""`



## Installation
---
### Creating a virtualenv

- Run `poetry install` to install the dependencies
- Use `poetry shell` to create a virtual environment

> Commit your `poetry.lock` file if you install any new library.

### Download Datasets

- Run `domadapter download mnli` to download the `mnli` dataset
- Run `domadapter download sa` to download the `amazon` dataset.

> Using Google Colab?
`!pip install pytorch-lightning==1.4.2 datasets transformers pandas click wandb numpy rich`


### Train Models 

See the `scripts` folder to train models. 
For example to train the ***Joint-DT-:electric_plug:*** on the MNLI dataset run 

`bash train_joint_da_ta_mnli.sh`



---
## Relevant folders
```
.
├── commands (domadapter terminal commands)
├── datamodules (Pytorch Lightning DataModules to load the SA and MNLI Dataset)
├── divergences (Different Divergence measures)
├── models (All the models listed in the paper)
├── orchestration (Instantiates the model, dataset, trainer and runs the experiments)
├── scripts (Bash scripts to run experiments)
├── utils (Useful utilities)
└── console.py (Python `Rich` console to pretty print everything)
```

---
