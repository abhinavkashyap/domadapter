from dataclasses import dataclass, field
from domadapter.datamodules.glue_dm import GlueDM
from transformers import AutoTokenizer
from transformers import HfArgumentParser
from transformers import MultiLingAdapterArguments
from typing import Optional
import os



@dataclass
class DataTrainingArguments:
    """
    Arguments related to training and validation data.

    `HFArgumentParser` turns this class into argparse arguments: allows usage in command
    line
    """

    task_name: Optional[str] = field(default=None, metadata={"help": f"Tasks available {['sst2']}"})

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": f"Max sequence length after tokenization. Sequenes longer "
            f"will be truncated and sequences shorter will be padded."
        },
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "HF stores the datasets in a caché (directory). If this is true, "
            "ignores caché and downloads data."
        },
    )

    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "If True, all sequences are padded to max_seq_length. If False, all sequences "
            "in a batch are padded to the max sequence length within the batch "
        },
    )

    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or json file containing training data"}
    )

    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or json file containing validation data"}
    )

    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "A cache directory to store the datasets downloaded from HF datasets"},
    )

    batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "Batch size of data"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


@dataclass
class ModelArguments:
    """
    Arguments related to model and tokenizer
    `HFArgumentParser` turns this class into argparse arguments: allows usage in command
    line
    """

    model_name: str = field(
        metadata={"help": "Name or the path of a pretrained model. Refer to huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model name"},
    )

    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model name"},
    )

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to store the pretrained models downloaded from s3"}
    )

    use_fast_tokenizer: bool = field(
        default=True, metadata={"help": "Whether to use the Fast version of the tokenizer"}
    )


@dataclass
class TrainerArguments:
    """ Hyperparameters related to training the model
    """


def main():
    # MultiLingAdapterArguments extends from AdapterArguments.
    # THe parameters are a union of both classes
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MultiLingAdapterArguments)
    )
    model_args, data_args, adapter_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    dm = GlueDM(
        task_name=data_args.task_name,
        dataset_cache_dir=os.environ["DATASET_CACHE_DIR"],
        tokenizer=tokenizer,
        overwrite_cache=data_args.overwrite_cache,
        pad_to_max_length=data_args.pad_to_max_length,
        max_seq_length=data_args.max_seq_length,
        batch_size=data_args.batch_size
    )


if __name__ == "__main__":
    main()
