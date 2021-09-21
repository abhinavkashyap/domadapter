from dataclasses import dataclass, field, asdict
from domadapter.datamodules.glue_dm import GlueDM
from transformers import AutoTokenizer
from transformers import HfArgumentParser
from transformers import MultiLingAdapterArguments
import pytorch_lightning as pl
from typing import Optional
import os
from pathlib import Path
from rich.prompt import Confirm
import shutil
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from domadapter.models.task_adapter import TaskAdapterModel
from pytorch_lightning.callbacks import ModelCheckpoint


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
        default=None, metadata={"help": "Path to store the pretrained models downloaded from s3"}
    )

    use_fast_tokenizer: bool = field(
        default=False, metadata={"help": "Whether to use the Fast version of the tokenizer"}
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


def main():
    # MultiLingAdapterArguments extends from AdapterArguments.
    # THe parameters are a union of both classes
    # The Argument Parse parses the arguments into instances of the data classes
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MultiLingAdapterArguments, TrainerArguments)
    )
    model_args, data_args, adapter_args, trainer_args = parser.parse_args_into_dataclasses()
    model_args_dict = asdict(model_args)
    data_args_dict = asdict(data_args)
    adapter_args_dict = asdict(adapter_args)
    trainer_args_dict = asdict(trainer_args)

    # Merge all the dictionaries
    # Note: All the dataclasses should have unique keys

    hparams = {**model_args_dict, **data_args_dict, **adapter_args_dict, **trainer_args_dict}
    experiments_dir = Path(os.environ["OUTPUT_DIR"])
    current_exp_dir = experiments_dir.joinpath(trainer_args.exp_name)

    if current_exp_dir.is_dir():
        is_delete = Confirm.ask(f"{current_exp_dir} exists. Delete?")
        if is_delete:
            shutil.rmtree(str(current_exp_dir))
    else:
        current_exp_dir.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    seed_everything(trainer_args.seed)

    dm = GlueDM(
        task_name=data_args.task_name,
        dataset_cache_dir=os.environ["DATASET_CACHE_DIR"],
        tokenizer=tokenizer,
        overwrite_cache=data_args.overwrite_cache,
        pad_to_max_length=data_args.pad_to_max_length,
        max_seq_length=data_args.max_seq_length,
        batch_size=data_args.batch_size,
        num_workers=data_args.num_processes,
    )
    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    dm.setup("test")
    test_loader = dm.test_dataloader()

    model = TaskAdapterModel(
        adapter_name=model_args.adapter_name,
        model_name=model_args.model_name,
        task_name=data_args.task_name,
        num_labels=dm.get_num_labels(),
        cache_dir=model_args.cache_dir,
        tokenizer=tokenizer,
        id2label=dm.id2label,
        adapter_config_name=adapter_args.adapter_config,
        adapter_non_linearity=adapter_args.adapter_non_linearity,
        adapter_reduction_factor=adapter_args.adapter_reduction_factor,
        adam_beta1=trainer_args.adam_beta1,
        adam_beta2=trainer_args.adam_beta2,
        adam_epsilon=trainer_args.adam_epsilon,
        learning_rate=trainer_args.learning_rate,
    )

    logger = WandbLogger(
        name=str(trainer_args.exp_name),
        save_dir=str(current_exp_dir),
        project=trainer_args.wandb_proj_name,
    )

    logger.watch(model, log="gradients", log_freq=10)
    logger.log_hyperparams(hparams)

    callbacks = []

    checkpoints_dir = experiments_dir.joinpath("checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        save_top_k=1,
        mode="max",
        monitor=f"dev/{trainer_args.monitor_metric}",
    )
    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        limit_train_batches=trainer_args.train_data_proportion,
        limit_val_batches=trainer_args.validation_data_proportion,
        limit_test_batches=trainer_args.test_data_proportion,
        terminate_on_nan=True,
        gradient_clip_val=trainer_args.gradient_clip_val,
        max_epochs=trainer_args.num_epochs,
        gpus=trainer_args.gpus,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(dataloaders=test_loader)


if __name__ == "__main__":
    main()
