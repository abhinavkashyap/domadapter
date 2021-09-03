from typing import Any, Union, List, Optional, Dict
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pathlib
from collections import defaultdict
from transformers import (
    BertTokenizerFast,
    RobertaTokenizerFast,
    DistilBertTokenizerFast,
)
from domadapter.datamodules.data import InputExample, InputFeatures
from loguru import logger
from tqdm import tqdm
from rich.console import Console
import torch
import srsly
from torch.utils.data import TensorDataset, Dataset
import multiprocessing
from domadapter.datamodules.concat_datasets import ConcatDatasets


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        hparams: Dict[str, Any],
    ):
        """

        hparams: Dict[str, Any]
        hyper-parameters for the classification data module
        the following keys shold be present in the dictionary
        src_train_file: pathlib.Path
            Path where the train files are stored with one example per line
        src_dev_file:  pathlib.Path
            Path where the train files are stored with one example per line
        src_test_file: pathlib.Path
            Path where the train files are stored with one example per line
        cache_dir: pathlib.Path
            Cache directory where features are stored
        labels_filename: pathlib.Path
            The path where the label files are stored
        infer_filename: pathlib.Path
            Same as train, dev and test filenames but this is optional
            and is used only during inference time
        tokenizer_type: str
            The tokenizer that will be used for this dataset
        batch_size: int
            The batch size for the data loader
        """
        super(ClassificationDataModule, self).__init__()
        self.src_dom_train_filename = pathlib.Path(hparams["src_train_file"])
        self.src_dom_dev_filename = pathlib.Path(hparams["src_dev_file"])
        self.src_dom_test_filename = pathlib.Path(hparams["src_test_file"])
        self.trg_dom_train_filename = pathlib.Path(hparams["trg_train_file"])
        self.trg_dom_dev_filename = pathlib.Path(hparams["trg_dev_file"])
        self.trg_dom_test_filename = pathlib.Path(hparams["trg_test_file"])
        self.tokenizer_type = hparams["tokenizer_type"]
        self.infer_filename = hparams.get("infer_filename")
        self.cache_dir = pathlib.Path(hparams["cache_dir"])
        self.labels_filename = pathlib.Path(hparams["label_file"])
        self.batch_size = hparams.get("bsz", 32)
        self.max_seq_length = hparams.get("max_seq_length", 128)
        self.msg_printer = Console()

        self.tokenizer_type_cls_mapping = {
            "bert": BertTokenizerFast,
            "roberta": RobertaTokenizerFast,
            "distilbert": DistilBertTokenizerFast,
        }

        self.tokenizer_type_name_mapping = {
            "bert": "bert-base-uncased",
            "roberta": "roberta-base",
            "distilbert": "distilbert-base-uncased",
        }

        if not self.cache_dir.is_dir():
            self.cache_dir.mkdir(parents=True)
        self.dataset_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Get the class for tokenizer
        tokenizer_cls = self.tokenizer_type_cls_mapping[self.tokenizer_type]

        # get the tokenizer using the tokenizer_name that is required for transformers
        self.tokenizer = tokenizer_cls.from_pretrained(
            self.tokenizer_type_name_mapping[self.tokenizer_type]
        )

        # Get information from the tokenizer
        self.cls_token = self.tokenizer.cls_token
        self.cls_token_segment_id = 2 if self.tokenizer_type == "XLNet" else 0
        self.sep_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.pad_token_segment_id = 0
        self.pad_token_label_id = -100
        self.sequence_a_segment_id = 0
        self.mask_padding_with_zero = True
        self.src_dom_train_features: List[InputFeatures] = None
        self.src_dom_dev_features: List[InputFeatures] = None
        self.src_dom_test_features: List[InputFeatures] = None
        self.trg_dom_train_features: List[InputFeatures] = None
        self.trg_dom_dev_features: List[InputFeatures] = None
        self.trg_dom_test_features: List[InputFeatures] = None
        self.infer_features: List[InputFeatures] = None
        self.src_dom_train_dataset: Dataset = None
        self.src_dom_dev_dataset: Dataset = None
        self.src_dom_test_dataset: Dataset = None
        self.trg_dom_train_dataset: Dataset = None
        self.trg_dom_dev_dataset: Dataset = None
        self.trg_dom_test_dataset: Dataset = None
        self.train_dataset: Dataset = None
        self.dev_dataset: Dataset = None
        self.test_dataset: Dataset = None
        self.infer_dataset: Dataset = None

        self.labels_map: Dict[str, int] = self._get_labels_map()
        self.num_labels = len(self.labels_map)

    def _get_labels_map(self) -> Dict[str, int]:
        labels = []
        with open(str(self.labels_filename)) as fp:
            for line in fp:
                line_ = line.strip()
                labels.append(line_)

        labels_map = dict([(label, idx) for idx, label in enumerate(labels)])
        return labels_map

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.src_dom_train_features = self.extract_features(mode="src-train")
            self.src_dom_dev_features = self.extract_features(mode="src-dev")
            self.trg_dom_train_features = self.extract_features(mode="trg-train")
            self.trg_dom_dev_features = self.extract_features(mode="trg-dev")

            self.src_dom_train_dataset = self.get_dataset(self.src_dom_train_features)
            self.src_dom_dev_dataset = self.get_dataset(self.src_dom_dev_features)
            self.trg_dom_train_dataset = self.get_dataset(self.src_dom_train_features)
            self.trg_dom_dev_dataset = self.get_dataset(self.trg_dom_dev_features)

            self.train_dataset = ConcatDatasets(
                [self.src_dom_train_dataset, self.trg_dom_train_dataset]
            )
            self.dev_dataset = ConcatDatasets(
                [self.src_dom_dev_dataset, self.trg_dom_dev_dataset]
            )
        elif stage == "test":
            self.src_dom_test_features = self.extract_features(mode="src-test")
            self.src_dom_test_dataset = self.get_dataset(self.src_dom_test_features)
            self.trg_dom_test_features = self.extract_features(mode="trg-test")
            self.trg_dom_test_dataset = self.get_dataset(self.trg_dom_test_features)
            self.test_dataset = ConcatDatasets(
                [self.src_dom_test_dataset, self.trg_dom_test_dataset]
            )
        else:
            raise ValueError(f"stage has to be fit or test")

    def setup_infer(self):
        self.infer_features = self.extract_features(mode="infer")
        self.infer_dataset = self.get_dataset(self.infer_features)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
        return loader

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        loader = DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        return loader

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        return loader

    def infer_dataloader(self):
        loader = DataLoader(
            self.infer_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
            shuffle=False,
        )
        return loader

    @staticmethod
    def read_examples_from_file(filename: pathlib.Path, mode: str):
        """

        Parameters
        ----------
        filename: pathlib.Path
            The CONLL file where the examples are stored
            The filename should of the format [line1###label\nline2###label]
            Every line should contain a line in one column and label in the second column
            The columns are demarcated by a ### character
            The line should have words separated by space
        mode: str
            It is either of train, dev or test

        Returns
        -------
        List[InputExample]
            Constructs the examples from the file

        """
        examples = []
        with open(str(filename), "r") as fp:
            for idx, line in enumerate(fp):
                line_, label_ = line.split("###")
                line_ = line_.strip()
                # remove the first and the last `""`
                line_ = line_[1:-1]
                label_ = label_.strip()
                words_ = line_.split()
                label_ = [label_]
                example = InputExample(
                    guid=f"{mode}-{idx}", words=words_, labels=label_
                )
                examples.append(example)

                if idx % 100 == 0:
                    print(f"Input example {example.words}")

        return examples

    def _convert_examples_to_features(
        self,
        examples: List[InputExample],
        max_seq_length: int,
        tokenizer,
        cls_token_at_end: bool = False,
        cls_token: str = "[CLS]",
        cls_token_segment_id: int = 1,
        sep_token: str = "[SEP]",
        sep_token_extra: bool = False,
        pad_on_left: int = False,
        pad_token_id: int = 0,
        pad_token_segment_id: int = 0,
        pad_token_label_id: int = -100,
        sequence_a_segment_id: int = 0,
        mask_padding_with_zero: int = True,
    ) -> List[InputFeatures]:
        """
        Taken from https://github.com/srush/transformers/blob/master/examples/utils_ner.py
        examples: List[InputExample]
            A list of examples that has been created

        labels: List[str]
            The list of labels associated with the examples
            Use a consistent list for every dataset

        max_seq_len: List[str]
            The maximum sequence length in the list of examples

        tokenizer:
            A transformers Tokenizer

        cls_token_at_end: bool
            define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        cls_token_segment_id: int
            define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """

        features = []
        for (ex_index, example) in tqdm(
            enumerate(examples),
            total=len(examples),
            desc="Converting to tokens and ids",
        ):
            if ex_index % 10000 == 0:
                logger.info(f"Writing example {ex_index} of {len(examples)}")

            tokens = []
            label_ids = []

            tokens_example = tokenizer.tokenize(" ".join(example.words))
            labels_example = example.labels

            # map the labels to indices
            label_ids_ = [self.labels_map[label_] for label_ in labels_example]
            label_ids.extend(label_ids_)
            tokens.extend(tokens_example)

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.

            # add a sep token at the end
            tokens += [sep_token]

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [sequence_a_segment_id] * len(tokens)

            # add a cls token at the end
            if cls_token_at_end:
                tokens += [cls_token]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token_id] * padding_length) + input_ids
                input_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids += [pad_token_id] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if ex_index < 5:
                tokens_str = " ".join([str(x) for x in tokens])
                input_ids_str = " ".join([str(x) for x in input_ids])
                input_mask_str = " ".join([str(x) for x in input_mask])
                segment_ids_str = " ".join([str(x) for x in segment_ids])
                label_ids_str = " ".join([str(x) for x in label_ids])
                logger.info("*** Example ***")
                logger.info(f"guid: {example.guid}")
                logger.info(f"tokens: {tokens_str}")
                logger.info(
                    f"input_ids: {input_ids_str}",
                )
                logger.info(
                    f"input_mask: {input_mask_str}",
                )
                logger.info(f"segment_ids: {segment_ids_str}")
                logger.info(f"label_ids: {label_ids_str}")

            # make features and append
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_ids=label_ids,
                )
            )
        self.msg_printer.print("[green] Finished Tokenisation")
        return features

    def extract_features(self, mode: str):
        # get features and store them
        if mode == "src-train":
            data_file = self.src_dom_train_filename
        elif mode == "src-dev":
            data_file = self.src_dom_dev_filename
        elif mode == "src-test":
            data_file = self.src_dom_test_filename
        elif mode == "trg-train":
            data_file = self.trg_dom_train_filename
        elif mode == "trg-dev":
            data_file = self.trg_dom_dev_filename
        elif mode == "trg-test":
            data_file = self.trg_dom_test_filename
        elif mode == "test":
            data_file = self.src_dom_test_filename
        elif mode == "infer":
            data_file = self.infer_filename
        else:
            raise ValueError(f"The mode has to be either train, dev or test")

        cache_path = self.cache_dir.joinpath(f"{self.tokenizer_type}-{data_file.name}")
        stats_path = self.cache_dir.joinpath(
            f"{self.tokenizer_type}-{data_file.name}-stats.json"
        )
        if not cache_path.is_file():
            examples = self.read_examples_from_file(filename=data_file, mode=mode)
            num_examples = len(examples)
            max_seq_length = max([len(example.words) for example in examples])
            max_seq_length = (
                self.max_seq_length
                if max_seq_length > self.max_seq_length
                else max_seq_length
            )

            # lets store the stats for the dataset
            self.dataset_stats[mode]["num_examples"] = num_examples
            self.dataset_stats[mode]["max_seq_length"] = max_seq_length

            features = self._convert_examples_to_features(
                examples=examples,
                max_seq_length=max_seq_length,
                tokenizer=self.tokenizer,
                cls_token_at_end=False,
                cls_token=self.cls_token,
                cls_token_segment_id=self.cls_token_segment_id,
                sep_token=self.sep_token,
                sep_token_extra=False,
                pad_on_left=False,
                pad_token_id=self.pad_token_id,
                pad_token_segment_id=self.pad_token_segment_id,
                pad_token_label_id=self.pad_token_label_id,
                sequence_a_segment_id=self.sequence_a_segment_id,
                mask_padding_with_zero=self.mask_padding_with_zero,
            )
            torch.save(features, str(cache_path))
            srsly.write_json(path=stats_path, data=self.dataset_stats[mode])
            self.msg_printer.print(f"[green] Saved the features at {str(cache_path)}")
            return features

        else:
            with self.msg_printer.status(
                f"Loading features and stats from {str(cache_path)}"
            ):
                features = torch.load(cache_path)
            self.msg_printer.print(
                f"[green] Finished Loading Features and stats from {cache_path}"
            )
            return features

    def get_dataset(self, features: List[InputFeatures]):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        if features[0].segment_ids is not None:
            all_token_type_ids = torch.tensor(
                [f.segment_ids for f in features], dtype=torch.long
            )
        else:
            all_token_type_ids = torch.tensor([0 for f in features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        # noinspection PyTypeChecker
        assert torch.all(all_label_ids < self.num_labels).item(), AssertionError(
            f"label ids error. You have an invalid label id, {all_label_ids}"
        )

        tensor_dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids
        )
        return tensor_dataset

    def get_num_labels(self):
        return self.num_labels
