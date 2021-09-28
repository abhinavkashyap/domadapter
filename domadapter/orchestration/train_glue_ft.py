from dataclasses import field, asdict, dataclass
from domadapter.datamodules.glue_dm import GlueDM
from domadapter.utils.arguments import ModelArguments, DataTrainingArguments, TrainerArguments
from transformers import AutoTokenizer
from transformers import HfArgumentParser
import pytorch_lightning as pl
from typing import Optional
import os
from pathlib import Path
from rich.prompt import Confirm
import shutil
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from domadapter.models.glue_ft import GlueFT
from pytorch_lightning.callbacks import ModelCheckpoint


@dataclass
class TrainGlueFTDataArguments(DataTrainingArguments):
    mnli_genre: Optional[str] = field(
        default="fiction",
        metadata={"help": "A MNLI Genre to finetune the model on"}
    )
    sample_proportion: Optional[float] = field(
        default=1.0,
        metadata={"help": "Provide a number between 0.0 and 1.0 which indicates "
                          "the proportion sampled from training dataset. Useful "
                          "for simulating low resource scenarios"}
    )


def main():
    parser = HfArgumentParser(
        (
            ModelArguments,
            TrainGlueFTDataArguments,
            TrainerArguments,
        )
    )
    model_args, data_args, trainer_args = parser.parse_args_into_dataclasses()

    model_args_dict = asdict(model_args)
    data_args_dict = asdict(data_args)
    trainer_args_dict = asdict(trainer_args)

    # Merge all the dictionaries
    # Note: All the dataclasses should have unique keys

    hparams = {
        **model_args_dict,
        **data_args_dict,
        **trainer_args_dict,
    }

    experiments_dir = Path(os.environ["OUTPUT_DIR"]).joinpath("mnli_ft", f"{data_args.mnli_genre}")
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
        multinli_genre=data_args.mnli_genre,
        sample_proportion=data_args.sample_proportion
    )

    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    dm.setup("test")
    test_loader = dm.test_dataloader()

    model = GlueFT(
        model_name=model_args.model_name,
        task_name=data_args.task_name,
        num_labels=dm.get_num_labels(),
        cache_dir=model_args.cache_dir,
        tokenizer=tokenizer,
        id2label=dm.id2label,
        adam_beta1=trainer_args.adam_beta1,
        adam_beta2=trainer_args.adam_beta2,
        adam_epsilon=trainer_args.adam_epsilon,
        learning_rate=trainer_args.learning_rate,
    )

    logger = WandbLogger(
        name=str(trainer_args.exp_name),
        save_dir=str(current_exp_dir),
        project=trainer_args.wandb_proj_name,
        job_type=f"{data_args.mnli_genre}",
        group="fine-tune"
    )

    logger.watch(model, log="gradients", log_freq=10)
    logger.log_hyperparams(hparams)

    callbacks = []

    checkpoints_dir = current_exp_dir.joinpath("checkpoints")
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
