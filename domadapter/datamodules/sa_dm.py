from typing import Optional, Dict, List, Any
from rich.traceback import install
import ntpath
import os
import torch

import pytorch_lightning as pl
import pandas as pd

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

install(show_locals=True)


class SADataModuleSourceTarget(pl.LightningDataModule):
    def __init__(
        self,
        hparams: Dict[str, Any],
    ):

        """
        Use the torch Datasets to load SA datasets
        hparams: Dict[str, Any]
        hyper-parameters for the classification data module
        the following keys shold be present in the dictionary:
        ----------
        dataset_cache_dir: str
            Folder name stores the SA dataset downloaded from the web.
            If downloaded already, use it from the directory
        source_target: str
            Indicates the source and target domain of dataset.
            Should be of the form {source domain}_{target domain}
        pretrained_model_name: str
            Name of pretrained to be used from the transformer library
        pad_to_max_length: bool
            Sets padding to True for the  hugging face tokenizer
            https://huggingface.co/transformers/internal/tokenization_utils.html
            Sets the padding to the longest sequence in the batch
        max_seq_length: int
            Controls the max_length parameter of the hugging face tokenizer
            Controls the maximum length to use by one of the truncation/padding parameters.
            If left unset or set to None, this will use the predefined model maximum length if a
            maximum length is required by one of the truncation/padding parameters. If the model
            has no specific maximum input length (like XLNet) truncation/padding to a maximum length
            will be deactivated.
        batch_size: int
            Batch size of inputs
        """

        super(SADataModuleSourceTarget, self).__init__()

        os.environ["TOKENIZERS_PARALLELISM"] = "True"

        self.dataset_cache_dir = hparams["dataset_cache_dir"]
        self.source_target = hparams["source_target"]
        self.pretrained_model_name = hparams["pretrained_model_name"]
        self.padding = hparams["padding"]
        self.max_seq_length = hparams["max_seq_length"]

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.batch_size = hparams["bsz"]

        # get the tokenizer using the pretrained_model_name that is required for transformers
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name, usefast=True
        )

    def prepare_data(self):
        SourceTargetDataset(
            source_filepath=os.path.join(
                os.environ["DATASET_CACHE_DIR"],
                "sa",
                self.source_target,
                "train_source.csv",
            ),
            target_filepath=os.path.join(
                os.environ["DATASET_CACHE_DIR"],
                "sa",
                self.source_target,
                "target_unlabelled.csv",
            ),
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_seq_length=self.max_seq_length,
        )
        SourceTargetDataset(
            source_filepath=os.path.join(
                os.environ["DATASET_CACHE_DIR"],
                "sa",
                self.source_target,
                "dev_source.csv",
            ),
            target_filepath=os.path.join(
                os.environ["DATASET_CACHE_DIR"],
                "sa",
                self.source_target,
                "dev_target.csv",
            ),
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_seq_length=self.max_seq_length,
        )
        SourceTargetDataset(
            source_filepath=os.path.join(
                os.environ["DATASET_CACHE_DIR"],
                "sa",
                self.source_target,
                "test_source.csv",
            ),
            target_filepath=os.path.join(
                os.environ["DATASET_CACHE_DIR"],
                "sa",
                self.source_target,
                "test_target.csv",
            ),
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_seq_length=self.max_seq_length,
        )

    def setup(self, stage: Optional[str] = None):
        train_dataset = SourceTargetDataset(
            source_filepath=os.path.join(
                os.environ["DATASET_CACHE_DIR"],
                "sa",
                self.source_target,
                "train_source.csv",
            ),
            target_filepath=os.path.join(
                os.environ["DATASET_CACHE_DIR"],
                "sa",
                self.source_target,
                "target_unlabelled.csv",
            ),
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_seq_length=self.max_seq_length,
        )
        val_dataset = SourceTargetDataset(
            source_filepath=os.path.join(
                os.environ["DATASET_CACHE_DIR"],
                "sa",
                self.source_target,
                "dev_source.csv",
            ),
            target_filepath=os.path.join(
                os.environ["DATASET_CACHE_DIR"],
                "sa",
                self.source_target,
                "dev_target.csv",
            ),
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_seq_length=self.max_seq_length,
        )
        test_dataset = SourceTargetDataset(
            source_filepath=os.path.join(
                os.environ["DATASET_CACHE_DIR"],
                "sa",
                self.source_target,
                "test_source.csv",
            ),
            target_filepath=os.path.join(
                os.environ["DATASET_CACHE_DIR"],
                "sa",
                self.source_target,
                "test_target.csv",
            ),
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_seq_length=self.max_seq_length,
        )

        if stage == "fit":
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
        elif stage == "test":
            self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16)


class SourceTargetDataset(Dataset):
    def __init__(
        self, source_filepath, target_filepath, tokenizer, padding, max_seq_length
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.padding = padding

        self.source_df = pd.read_csv(source_filepath)
        self.target_df = pd.read_csv(target_filepath)
        self.target_filename = ntpath.basename(target_filepath)

    def __getitem__(self, index):
        sentence = self.source_df.iloc[index]["sentence"]
        label_source = self.source_df.iloc[index]["label"]

        encoded_input = self.tokenizer(
            str(sentence),
            max_length=self.max_seq_length,
            truncation=True,
            padding=self.padding,
        )
        source_input_ids = encoded_input["input_ids"]
        source_attention_mask = encoded_input["attention_mask"]

        sentence = self.target_df.iloc[index]["sentence"]
        encoded_input = self.tokenizer(
            str(sentence),
            max_length=self.max_seq_length,
            truncation=True,
            padding=self.padding,
        )
        target_input_ids = encoded_input["input_ids"]
        target_attention_mask = encoded_input["attention_mask"]
        if "unlabelled" not in self.target_filename:
            label_target = self.target_df.iloc[index]["label"]
            data_input = {
                "source_input_ids": torch.tensor(source_input_ids),
                "source_attention_mask": torch.tensor(source_attention_mask),
                "target_input_ids": torch.tensor(target_input_ids),
                "target_attention_mask": torch.tensor(target_attention_mask),
                "label_source": torch.tensor(label_source, dtype=torch.long),
                "label_target": torch.tensor(label_target, dtype=torch.long),
            }
        else:
            data_input = {
                "source_input_ids": torch.tensor(source_input_ids),
                "source_attention_mask": torch.tensor(source_attention_mask),
                "target_input_ids": torch.tensor(target_input_ids),
                "target_attention_mask": torch.tensor(target_attention_mask),
                "label_source": torch.tensor(label_source, dtype=torch.long),
            }

        return data_input

    def __len__(self):
        return min(self.source_df.shape[0], self.target_df.shape[0])
