from dataclasses import dataclass, field
from typing import Optional


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

    batch_size: Optional[int] = field(default=32, metadata={"help": "Batch size of data"})

    num_processes: Optional[int] = field(
        default=8, metadata={"help": "Num of workers for Dataloader"}
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
        default=None,
        metadata={"help": "Path to store the pretrained models downloaded from s3"},
    )

    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use the Fast version of the tokenizer"},
    )

    adapter_name: str = field(
        default=None, metadata={"help": "Give your adapter a name of your choice"}
    )


@dataclass
class TrainerArguments:
    """Hyperparameters related to training the model"""

    exp_name: str = field(metadata={"help": "Give a name to your experiment"})

    wandb_proj_name: str = field(
        metadata={"help": "Weights and Biases Project Name. We do not yet support other loggers"}
    )
    seed: int = field(metadata={"help": "Seed number useful to reproduce experiments"})
    train_data_proportion: float = field(
        metadata={
            "help": "A number in [0, 1] indicates the portion of training data used during training"
        }
    )

    validation_data_proportion: float = field(
        metadata={
            "help": "A number in [0, 1] indicates the portion of training data used during training"
        }
    )

    test_data_proportion: float = field(
        metadata={
            "help": "A number in [0, 1] indicates the portion of training data used during training"
        }
    )

    gradient_clip_val: float = field(
        metadata={
            "help": "Gradient clip value. If the gradient norm is beyond this value, it will be "
            "clipped to this value"
        }
    )

    num_epochs: int = field(metadata={"help": "Total number of epochs for training"})

    adam_beta1: float = field(metadata={"help": "Beta1 value for Adam Optimizer"})

    adam_beta2: float = field(metadata={"help": "Beta2 value for Adam Optimizer"})

    adam_epsilon: float = field(metadata={"help": "Adam Epsilon value for Adam Optimizer"})

    learning_rate: float = field(metadata={"help": "Learning rate for optimization"})

    gpus: str = field(metadata={"help": "GPU number to train on. Pass this as a string"})

    monitor_metric: str = field(
        metadata={"help": "This metric will be monitored for storing the best model"}
    )
