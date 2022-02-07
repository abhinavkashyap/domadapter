from dataclasses import field, asdict, dataclass
from domadapter.datamodules.glue_dm import GlueDM
from domadapter.utils.arguments import ModelArguments, DataTrainingArguments, TrainerArguments
import json
from transformers import HfArgumentParser
import pytorch_lightning as pl
from typing import Optional
import os
from pathlib import Path
from rich.prompt import Confirm
import shutil
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from domadapter.models.ft.glue_ft import GlueFT
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from domadapter.console import console


@dataclass
class TrainGlueFTDataArguments(DataTrainingArguments):
    multinli_genre: Optional[str] = field(
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

    experiments_dir = Path(os.environ["OUTPUT_DIR"]).joinpath("mnli_ft", f"{data_args.multinli_genre}")
    exp_dir = experiments_dir.joinpath(trainer_args.exp_name)

    if not exp_dir.is_dir():
        exp_dir.mkdir(parents=True)

    wandb_dir = exp_dir.joinpath("wandb")
    wandb_dir.mkdir(parents=True)

    seed_everything(trainer_args.seed)

    dm = GlueDM(hparams)

    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    dm.setup("test")
    test_loader = dm.test_dataloader()
    hparams["num_labels"] = dm.get_num_labels()
    hparams["id2label"] = dm.id2label

    model = GlueFT(hparams)

    logger = WandbLogger(
        name=str(trainer_args.exp_name),
        save_dir=str(wandb_dir),
        project=trainer_args.wandb_proj_name,
        job_type=f"{data_args.multinli_genre}",
        group="fine-tune"
    )

    logger.watch(model, log="gradients", log_freq=10)
    logger.log_hyperparams(hparams)

    callbacks = []

    # I don't think run_id thing is required here as
    # exp_name contains lr, and seed for the training.
    # Adding run_id will increase the folders to traverse

    # print(f"run id {logger.experiment.id}")
    # run_id = logger.experiment.id
    # exp_dir = exp_dir.joinpath(run_id)
    checkpoints_dir = exp_dir.joinpath("checkpoints")
    # Ask to delete if experiment exists
    if checkpoints_dir.is_dir():
        is_delete = Confirm.ask(f"{checkpoints_dir} already exists. Do you want to delete it?")
        if is_delete:
            shutil.rmtree(str(checkpoints_dir))
            console.print(f"[red] Deleted {checkpoints_dir}")
            checkpoints_dir.mkdir(parents=True)
    else:
        checkpoints_dir.mkdir(parents=True)

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

    hparams_file = exp_dir.joinpath("hparams.json")

    with open(hparams_file, "w") as fp:
        json.dump(hparams, fp)


if __name__ == "__main__":
    main()
