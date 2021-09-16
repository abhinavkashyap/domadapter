from typing import Optional, Dict, List
from rich.traceback import install
import os
import torch

import pytorch_lightning as pl
import pandas as pd

from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader

install(show_locals=True)


class DataModuleSourceTarget(pl.LightningDataModule):
        def __init__(
        self,
        dataset_cache_dir: str,
        source_target: str,
        tokenizer: PreTrainedTokenizer,
        pad_to_max_length: bool = True,
        max_seq_length: int = None,
        batch_size: int = 32):

            """
            Use the torch Datasets to load MNLI datasets
            Parameters
            ----------
            dataset_cache_dir: str
                Folder name stores the GLUE dataset downloaded from the web.
                If downloaded already, use it from the directory
                If `overwrite_cache` is True, then download every time and ignore
                any downloaded versions
            source_target: str
                Indicates the source and target domain of dataset. 
                Should be of the form {source domain}_{target domain}
            tokenizer: PreTrainedTokenizer
                A pretrained tokenizer from the transformer library
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

            super(DataModuleSourceTarget, self).__init__()

            self.dataset_cache_dir = dataset_cache_dir
            self.source_target = source_target
            self.tokenizer = tokenizer
            self.pad_to_max_length = pad_to_max_length
            self.max_seq_length = max_seq_length

            self.train_dataset = None
            self.val_dataset = None
            self.batch_size = batch_size

        def prepare_data(self):
            SourceTargetDataset(
                source_filepath=os.path.join(os.environ["DATASET_CACHE_DIR"], "mnli", self.source_target, "train_source.csv"),
                target_filepath=os.path.join(os.environ["DATASET_CACHE_DIR"], "mnli", self.source_target, "target_unlabelled.csv"),
                tokenizer=self.tokenizer,
                pad_to_max_length = self.pad_to_max_length,
                max_seq_length = self.max_seq_length
            )
            SourceTargetDataset(
                source_filepath=os.path.join(os.environ["DATASET_CACHE_DIR"], "mnli", self.source_target, "dev_source.csv"),
                target_filepath=os.path.join(os.environ["DATASET_CACHE_DIR"], "mnli", self.source_target, "dev_target.csv"),
                tokenizer=self.tokenizer,
                pad_to_max_length = self.pad_to_max_length,
                max_seq_length = self.max_seq_length
            )
        

        def setup(self, stage: Optional[str] = None):
            train_dataset = SourceTargetDataset(
                                source_filepath=os.path.join(os.environ["DATASET_CACHE_DIR"], "mnli", self.source_target, "train_source.csv"),
                                target_filepath=os.path.join(os.environ["DATASET_CACHE_DIR"], "mnli", self.source_target, "target_unlabelled.csv"),
                                tokenizer=self.tokenizer,
                                pad_to_max_length = self.pad_to_max_length,
                                max_seq_length = self.max_seq_length
            )
            val_dataset = SourceTargetDataset(
                                source_filepath=os.path.join(os.environ["DATASET_CACHE_DIR"], "mnli", self.source_target, "dev_source.csv"),
                                target_filepath=os.path.join(os.environ["DATASET_CACHE_DIR"], "mnli", self.source_target, "dev_target.csv"),
                                tokenizer=self.tokenizer,
                                pad_to_max_length = self.pad_to_max_length,
                                max_seq_length = self.max_seq_length
            )
            
            if stage == "fit":
                self.train_dataset = train_dataset
                self.val_dataset = val_dataset 


        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=self.batch_size)


        def val_dataloader(self):
            return DataLoader(self.val_dataset, batch_size=self.batch_size)


    

class SourceTargetDataset(Dataset):
    def __init__(self, source_filepath, target_filepath, tokenizer, pad_to_max_length, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_to_max_length = pad_to_max_length

        self.source_df = pd.read_csv(source_filepath)
        self.target_df = pd.read_csv(target_filepath)

    def __getitem__(self, index):
        premise = self.source_df.iloc[index]["premise"]
        hypothesis = self.source_df.iloc[index]["hypothesis"]
        label = self.source_df.iloc[index]["label"]

        encoded_input = self.tokenizer(
                str(premise),
                str(hypothesis),
                max_length= self.max_seq_length,
                truncation=True,
                pad_to_max_length=self.pad_to_max_length
            )
        source_input_ids = encoded_input["input_ids"]
        source_attention_mask = encoded_input["attention_mask"]

        premise = self.target_df.iloc[index]["premise"]
        hypothesis = self.target_df.iloc[index]["hypothesis"]
        encoded_input = self.tokenizer(
                str(premise),
                str(hypothesis),
                max_length= self.max_seq_length,
                truncation=True,
                pad_to_max_length=self.pad_to_max_length
            )
        target_input_ids = encoded_input["input_ids"]
        target_attention_mask = encoded_input["attention_mask"]


        data_input = {
            "source_input_ids": torch.tensor(source_input_ids),
            "source_attention_mask": torch.tensor(source_attention_mask),
            "target_input_ids": torch.tensor(target_input_ids),
            "target_attention_mask": torch.tensor(target_attention_mask),
            "label": torch.tensor(label, dtype=torch.long),
        }

        return data_input


    def __len__(self):
        return min(self.source_df.shape[0], self.target_df.shape[0])


if __name__ == '__main__':
    from transformers import AutoTokenizer

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_module = DataModuleSourceTarget(
        dataset_cache_dir=os.environ["DATASET_CACHE_DIR"],
        source_target="slate_travel",
        tokenizer=bert_tokenizer,
        pad_to_max_length= True,
        max_seq_length=128
    )
    data_module.prepare_data()
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    print(next(iter(train_loader)))