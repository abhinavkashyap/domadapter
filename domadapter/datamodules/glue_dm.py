import pytorch_lightning as pl
from typing import Optional, Dict, List
from rich.traceback import install
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator
import multiprocessing

# Pretty prints traceback in the console using `rich`
install(show_locals=True)


class GlueDM(pl.LightningDataModule):
    def __init__(
        self,
        task_name: str,
        dataset_cache_dir: str,
        tokenizer: PreTrainedTokenizer,
        overwrite_cache: bool = False,
        pad_to_max_length: bool = True,
        max_seq_length: int = None,
        batch_size: int = 32,
        num_workers: int = 8,
        multinli_genre: Optional[str] = None,
    ):
        """Use the transformer datasets library to download
        GLUE tasks. We should use this later if we decide to do experiments
        on 10% data as used by the paper https://aclanthology.org/2021.acl-long.172/

        Parameters
        ----------
        task_name: str
            GLUE task name. One of
            ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

        dataset_cache_dir: str
            Folder name stores the GLUE dataset downloaded from the web.
            If downloaded already, use it from the directory
            If `overwrite_cache` is True, then download every time and ignore
            any downloaded versions

        tokenizer: PreTrainedTokenizer
            A pretrained tokenizer from the transformer library

        overwrite_cache: bool
            Download the dataset ignoring any downloaded versions?

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

        num_workers: int
            Number of workers to use for dataloaders

        multinli_genre: Optional[str]
            If given then we will use examples
            only from that genre for multinli
        """
        super(GlueDM, self).__init__()
        self.task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),
        }

        self.task_name = task_name
        assert self.task_name in list(
            self.task_to_keys.keys()
        ), f"task_name should be one of {list(self.task_to_keys)}"

        self.dataset_cache_dir = dataset_cache_dir
        self.tokenizer = tokenizer
        self.overwrite_cache = overwrite_cache
        self.pad_to_max_length = pad_to_max_length

        if pad_to_max_length is True:
            self.data_collator = default_data_collator
            self.padding = "max_length"
            self.max_seq_length = max_seq_length
        else:
            self.data_collator = None
            self.padding = None
            self.max_seq_length = None

        self.datasets = None  # Store the dataset
        self.labels: List[str] = None  # Store the labels used
        self.num_labels: int = None  # Stores the number of labels
        self.label2id: Dict[str, int] = None  # Map from label to id
        self.id2label: Dict[int, str] = None  # Map from id to label
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.multinli_genre = multinli_genre

    def prepare_data(self):
        """Download the dataset for the task and store it in the
        dataset cache dir. We are not using the return value of
        the `load_dataset` function because:
        1. LightningDataModule does not recommend setting any properties inside this function
        2. We want to just download in this function and use it when we call `setup`.

        Returns
        -------
        None
        """
        if self.task_name != "mnli":
            load_dataset("glue", self.task_name, cache_dir=self.dataset_cache_dir)
        else:
            # https://huggingface.co/datasets/multi_nli
            # using this instead of load_dataset("glue", "mnli") because we need access to the
            # genre names
            load_dataset("multi_nli")

    def setup(self, stage: Optional[str] = None):

        if self.task_name != "mnli":
            self.datasets = load_dataset("glue", self.task_name, cache_dir=self.dataset_cache_dir)
        else:
            self.datasets = load_dataset("multi_nli")

        if self.task_name == "mnli" and self.multinli_genre is not None:
            self.datasets = self.datasets.remove_columns(
                [
                    "hypothesis_binary_parse",
                    "hypothesis_parse",
                    "premise_parse",
                    "premise_binary_parse",
                    "promptID",
                ]
            )
            # Filter and obtain examples for a given genre only
            self.datasets = self._filter_mnli_dataset(self.datasets, self.multinli_genre)

        # Setup the labels
        is_regression = self.task_name == "stsb"

        if is_regression:
            self.num_labels = 1

        else:
            # The features is a dictionary containing label key.
            # The names property gives the names of the labels

            self.labels = self.datasets["train"].features["label"].names
            self.num_labels = len(self.labels)
            label2id = dict(
                [
                    (
                        label.lower(),
                        self.datasets["train"].features["label"].str2int(label),
                    )
                    for label in self.labels
                ]
            )
            id2label = {v: k for k, v in label2id.items()}
            self.id2label = id2label
            self.label2id = label2id

        # Tokenize the dataset
        self.datasets = self.datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not self.overwrite_cache,
        )

        # Return the dataset

        if stage == "fit":
            self.train_dataset = self.datasets["train"]
            self.train_dataset.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
            )
            self.val_dataset = (
                self.datasets["validation_matched"]
                if self.task_name == "mnli"
                else self.datasets["validation"]
            )
            self.val_dataset.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
            )

        elif stage == "test":
            self.test_dataset = (
                self.datasets["validation_matched"]
                if self.task_name == "mnli"
                else self.datasets["validation"]
            )
            self.test_dataset.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
            )
        else:
            raise ValueError("stage can be on of [fit, val, test]")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
        )

    def preprocess_function(self, examples):
        sentence1_key, sentence2_key = self.task_to_keys[self.task_name]
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )

        result = self.tokenizer(
            *args, padding=self.padding, max_length=self.max_seq_length, truncation=True
        )

        result["labels"] = examples["label"]

        return result

    def get_num_labels(self) -> int:
        """Return the number of labels

        Returns
        -------
        int
            Number of labels in the task

        """
        return self.num_labels

    def _filter_mnli_dataset(self, dataset, genre: str):
        new_dataset = dataset.filter(
            lambda example: example["genre"] == genre, num_proc=self.num_workers
        )
        return new_dataset
