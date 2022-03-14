import click
import pathlib
import gc
import os
from domadapter.datamodules.mnli_dm import DataModuleSourceTarget
from domadapter.models.ablations.domain_task_adapter import DomainTaskAdapter
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
import json
from domadapter.console import console
from rich.prompt import Confirm
import shutil
import wandb


@click.command()
@click.option("--dataset-cache-dir", type=str, help="Cache directory for dataset.")
@click.option(
    "--source-target", type=str, help="Source and target domain in source_target format"
)
@click.option("--pretrained-model-name", type=str, help="PLM to be used from HF")
@click.option(
    "--padding", type=str, help="Add padding while tokenizing upto max length"
)
@click.option("--max-seq-length", type=str, help="seq length for tokenizer")
@click.option("--domain-adapter-id", type=str, help="wandb id of domain adapter (same domain adapter will be used)")
@click.option("--num-classes", type=int, help="Number of classes for task adapter classification head")
@click.option("--bsz", type=int, help="batch size")
@click.option("--divergence", type=str, help="divergence on which trained domain adapter is to be loaded")
@click.option("--train-proportion", type=float, help="Train on small proportion")
@click.option("--dev-proportion", type=float, help="Validate on small proportion")
@click.option("--test-proportion", type=float, help="Test on small proportion")
@click.option("--reduction-factor", help="Factor by which the hidden size is reduced")
@click.option("--skip-layers", help="Layers to be skipped while adding adapters")
@click.option("--mode", type=str, help="Train task adapter or train domain and task adapter")
@click.option("--exp-dir", type=str, help="Experiment directory to store artefacts")
@click.option("--seed", type=str, help="Seed for reproducibility")
@click.option("--lr", type=float, help="Learning rate for the entire model")
@click.option("--epochs", type=int, help="Number of epochs to run the training")
@click.option("--gpu", type=int, default=None, help="GPU to run the program on")
@click.option("--log-freq", type=int, help="Log wandb after how many steps")
def train_domain_adapter(
    bsz,
    dataset_cache_dir,
    pretrained_model_name,
    divergence,
    train_proportion,
    dev_proportion,
    test_proportion,
    reduction_factor,
    skip_layers,
    domain_adapter_id,
    mode,
    num_classes,
    max_seq_length,
    padding,
    source_target,
    exp_dir,
    seed,
    log_freq,
    lr,
    epochs,
    gpu,
):
    dataset_cache_dir = pathlib.Path(dataset_cache_dir)
    exp_dir = pathlib.Path(exp_dir)
    domain_adapter_dir = exp_dir

    if mode == 'task':
        exp_dir = exp_dir.joinpath(source_target, "ablations_task_adapter_only")
    else:
        exp_dir = exp_dir.joinpath(source_target, f"ablations_task_adapter_{divergence}")

    if not exp_dir.is_dir():
        exp_dir.mkdir(parents=True)

    domain_adapter_dir = domain_adapter_dir.joinpath(source_target, "domain_adapter", str(domain_adapter_id), "checkpoints")

    seed_everything(seed)

    hyperparams = {
        "bsz": bsz,
        "train_proportion": train_proportion,
        "dev_proportion": dev_proportion,
        "test_proportion": test_proportion,
        "source_target": source_target,
        "num_classes": int(num_classes),
        "dataset_cache_dir": str(dataset_cache_dir),
        "domain_adapter_dir": str(domain_adapter_dir),
        "reduction_factor": reduction_factor,
        "leave_out": skip_layers,
        "exp_dir": str(exp_dir),
        "loss": str(divergence),
        "mode": str(mode),
        "seed": seed,
        "learning_rate": lr,
        "epochs": int(epochs),
        "gpu": gpu,
        "pretrained_model_name": str(pretrained_model_name),
        "max_seq_length": int(max_seq_length),
        "padding": str(padding),
    }

    ###########################################################################
    # Setup the dataset
    ###########################################################################
    dm = DataModuleSourceTarget(hyperparams)
    dm.prepare_data()

    model = DomainTaskAdapter(hyperparams)

    ###########################################################################
    # SETUP THE LOGGERS and Checkpointers
    ###########################################################################
    run_id = wandb.util.generate_id()
    exp_dir = exp_dir.joinpath(run_id)

    if reduction_factor != "None" and skip_layers != "None":
        job_type = f"domain adapter {reduction_factor} {skip_layers}"
    elif reduction_factor != "None":
        job_type = f"domain adapter {reduction_factor}"
    elif skip_layers != "None":
        job_type = f"domain adapter {skip_layers}"

    print(job_type)

    if mode == 'task':
        logger = WandbLogger(
        save_dir=exp_dir,
        id = run_id,
        project=f"MNLI_{pretrained_model_name}",
        job_type=job_type,
        group=source_target,
    )
    else:
        logger = WandbLogger(
        save_dir=exp_dir,
        id = run_id,
        project=f"MNLI_{pretrained_model_name}",
        job_type="domain "+job_type,
        group=source_target,
    )

    checkpoints_dir = exp_dir.joinpath("checkpoints")
    checkpoints_dir.mkdir(parents=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        save_top_k=1,
        mode="min",
        monitor="source_val/loss",
    )
    early_stop_callback = EarlyStopping(
        monitor="source_val/loss", patience=2, verbose=False, mode="min"
    )

    callbacks = [checkpoint_callback, early_stop_callback]

    trainer = Trainer(
        limit_train_batches=train_proportion,
        limit_val_batches=dev_proportion,
        limit_test_batches=test_proportion,
        callbacks=callbacks,
        terminate_on_nan=True,
        log_every_n_steps=log_freq,
        gpus=str(gpu),
        max_epochs=epochs,
        logger=logger,
    )

    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    trainer.fit(model, train_loader, val_loader)

    dm.setup("test")
    test_loader = dm.test_dataloader()
    trainer.test(model, test_loader)

    best_ckpt_path = checkpoint_callback.best_model_path
    model = DomainTaskAdapter.load_from_checkpoint(best_ckpt_path)

    model.save_adapter(
        str(checkpoints_dir), f"task_adapter_{source_target}"
    )  # save adapter after loading model
    os.remove(best_ckpt_path)  # remove saved model

    hparams_file = exp_dir.joinpath("hparams.json")

    with open(hparams_file, "w") as fp:
        json.dump(hyperparams, fp)

    del model
    gc.collect()


if __name__ == "__main__":
    train_domain_adapter()
