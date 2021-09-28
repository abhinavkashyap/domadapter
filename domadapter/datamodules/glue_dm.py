import pytorch_lightning as pl
from typing import Optional, Dict, List, Any
from rich.traceback import install
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator
import multiprocessing
import datasets

# Pretty prints traceback in the console using `rich`
install(show_locals=True)


class GlueDM(pl.LightningDataModule):
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

            tokenizer_name: str
                A pretrained tokenizer name from the HF library

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

            sample_proportion: Optional[float]
                If propvided should be a number between 0.0 and 1.0
                Sample a proportion of the train dataset
                Useful for simulating low resource scenarios.
                We do not sample dev and test datasets.
                Note: This is not used to run small sample training
                Set the appropriate flags and proportions in the pl.Trainer

            sample_seed: Optional[int]
                Seed used for sampling dataset
            """
    def __init__(
        self,
        hparams: Dict[str, Any]
    ):
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

        self.task_name = hparams["task_name"]
        assert self.task_name in list(
            self.task_to_keys.keys()
        ), f"task_name should be one of {list(self.task_to_keys)}"

        self.dataset_cache_dir = hparams["dataset_cache_dir"]
        self.tokenizer_name = hparams["tokenizer_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.overwrite_cache = hparams["overwrite_cache"]
        self.pad_to_max_length = hparams["pad_to_max_length"]

        if self.pad_to_max_length is True:
            self.data_collator = default_data_collator
            self.padding = "max_length"
            self.max_seq_length = hparams["max_seq_length"]
        else:
            self.data_collator = None
            self.padding = None
            self.max_seq_length = None

        self.batch_size = hparams["batch_size"]
        self.num_workers = hparams["num_processes"]

        self.multinli_genre = hparams.get("multinli_genre", None)
        self.sample_proportion = hparams.get("sample_proportion", 1.0)
        self.sample_seed = hparams.get("sample_seed", 1729)

        assert (
                0.0 < self.sample_proportion <= 1.0
        ), f"Sample Proportion should be between 0.0 and 1.0"

        self.datasets = None  # Store the dataset
        self.labels: List[str] = None  # Store the labels used
        self.num_labels: int = None  # Stores the number of labels
        self.label2id: Dict[str, int] = None  # Map from label to id
        self.id2label: Dict[int, str] = None  # Map from id to label
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

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
        _ = self._load_dataset()

    def setup(self, stage: Optional[str] = None):

        self.datasets = self._load_dataset()

        self.datasets = self._filter_dataset(self.datasets)

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
            self.train_dataset = self._sample_dataset(self.train_dataset)

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

    def _filter_dataset(self, dataset):
        # Select examples if multinli_genre is mentioned
        if self.task_name == "mnli" and self.multinli_genre is not None:
            datasets_ = self.datasets.remove_columns(
                [
                    "hypothesis_binary_parse",
                    "hypothesis_parse",
                    "premise_parse",
                    "premise_binary_parse",
                    "promptID",
                ]
            )

            new_dataset = datasets_.filter(
                lambda example: example["genre"] == self.multinli_genre, num_proc=self.num_workers
            )
            return new_dataset
        else:
            return dataset

    def _sample_dataset(self, dataset):
        """

        Parameters
        ----------
        dataset: datasets.Dataset
            Sample a HF Dataset according to sample proportion

        Returns
        -------
        datasets.Dataset
            A sampled dataset

        """
        if self.sample_proportion == 1.0:
            return dataset
        else:
            dataset_dict = dataset.train_test_split(
                train_size=self.sample_proportion,
                test_size=1 - self.sample_proportion,
                seed=self.sample_seed,
            )
            # Return only the train proportion
            # Ignore the "test" proportion
            return dataset_dict["train"]

    def _load_dataset(self) -> datasets.Dataset:
        """

        Returns
        -------
        datasets.Dataset
            Loads dataset from HF hub and returns it
        """
        if self.task_name != "mnli":
            dataset = load_dataset("glue", self.task_name, cache_dir=self.dataset_cache_dir)
        else:
            # https://huggingface.co/datasets/multi_nli
            # using this instead of load_dataset("glue", "mnli") because we need access to the
            # genre names
            dataset = load_dataset("multi_nli")

        return dataset
