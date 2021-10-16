import click
import pathlib
from domadapter.datamodules.clf_datamodule import ClassificationDataModule
from domadapter.models.uda.dan import DAN
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
import json
from rich.prompt import Confirm
import shutil


@click.command()
@click.option("--src-train-file", type=str, help="Src train filename")
@click.option("--src-dev-file", type=str, help="Src dev filename")
@click.option("--src-test-file", type=str, help="Src test filename")
@click.option("--trg-train-file", type=str, help="Trg train filename")
@click.option("--trg-dev-file", type=str, help="Trg dev filename")
@click.option("--trg-test-file", type=str, help="Trg test filename")
@click.option("--label-file", type=str, help="Provide the label file")
@click.option("--tokenizer-type", type=str, help="The tokenizer to be used")
@click.option("--bsz", type=int, default=32, help="Batch size for dataset")
@click.option("--dataset-cache-dir", type=str, help="Cache directory for dataset.")
@click.option(
    "--train-bert-at-layer", type=int, help="BERT will be trained at this layer"
)
@click.option("--num-clf-layers", type=int, help="Number of classification layers")
@click.option(
    "--clf-hidden-size", type=int, help="Hidden dimension of classification layers"
)
@click.option(
    "--freeze-upto", type=int, help="Freeze upto certain layer in the transformer"
)
@click.option("--train-proportion", type=float, help="Train on small proportion")
@click.option("--dev-proportion", type=float, help="Validate on small proportion")
@click.option("--test-proportion", type=float, help="Test on small proportion")
@click.option("--exp-name", type=str, help="Experiment name")
@click.option("--exp-dir", type=str, help="Experiment directory to store artefacts")
@click.option("--seed", type=str, help="Seed for reproducibility")
@click.option("--lr", type=float, help="Learning rate for the entire model")
@click.option("--epochs", type=int, help="Number of epochs to run the training")
@click.option("--gpu", type=int, help="GPU to run the program on")
@click.option("--grad-clip-norm", type=float, help="Gradient Clip Norm value to clip")
@click.option(
    "--is-divergence-reduced", is_flag=True, help="Reduces div between src and trg clf"
)
@click.option("--div-reg-param", type=float, help="divergence regularization parameter")
@click.option("--divergence-reduced", type=str, help="Divergence reduced")
@click.option("--wandb-proj-name", type=str, help="Weights and Biases Project Name")
def train_danformer(
    src_train_file,
    src_dev_file,
    src_test_file,
    trg_train_file,
    trg_dev_file,
    trg_test_file,
    label_file,
    tokenizer_type,
    bsz,
    dataset_cache_dir,
    train_bert_at_layer,
    num_clf_layers,
    clf_hidden_size,
    freeze_upto,
    train_proportion,
    dev_proportion,
    test_proportion,
    exp_name,
    exp_dir,
    seed,
    lr,
    epochs,
    gpu,
    grad_clip_norm,
    is_divergence_reduced,
    div_reg_param,
    divergence_reduced,
    wandb_proj_name,
):
    src_train_file = pathlib.Path(src_train_file)
    src_dev_file = pathlib.Path(src_dev_file)
    src_test_file = pathlib.Path(src_test_file)
    trg_train_file = pathlib.Path(trg_train_file)
    trg_dev_file = pathlib.Path(trg_dev_file)
    trg_test_file = pathlib.Path(trg_test_file)
    label_file = pathlib.Path(label_file)
    dataset_cache_dir = pathlib.Path(dataset_cache_dir)
    exp_dir = pathlib.Path(exp_dir)
    exp_dir = exp_dir.joinpath(exp_name)

    # Ask to delete if experiment exists
    if exp_dir.is_dir():
        is_delete = Confirm.ask(f"{exp_dir} already exists... Delete?")
        if is_delete:
            shutil.rmtree(str(exp_dir))
        exp_dir.mkdir(parents=True)
    else:
        exp_dir.mkdir(parents=True)

    seed_everything(seed)

    hyperparams = {
        "src_train_file": str(src_train_file),
        "src_dev_file": str(src_dev_file),
        "src_test_file": str(src_test_file),
        "trg_train_file": str(trg_train_file),
        "trg_dev_file": str(trg_dev_file),
        "trg_test_file": str(trg_test_file),
        "label_file": str(label_file),
        "tokenizer_type": tokenizer_type,
        "bsz": bsz,
        "cache_dir": str(dataset_cache_dir),
        "train_bert_at_layer": train_bert_at_layer,
        "num_clf_layers": num_clf_layers,
        "clf_hidden_size": clf_hidden_size,
        "train_proportion": train_proportion,
        "dev_proportion": dev_proportion,
        "test_proportion": test_proportion,
        "exp_name": exp_name,
        "exp_dir": str(exp_dir),
        "seed": seed,
        "learning_rate": lr,
        "epochs": epochs,
        "gpu": gpu,
        "grad_clip_norm": grad_clip_norm,
        "freeze_upto": freeze_upto,
        "pretrained_model_name": "bert-base-uncased",
        "is_divergence_reduced": is_divergence_reduced,
        "div_reg_param": div_reg_param,
        "divergence_reduced": divergence_reduced,
        "wandb_proj_name": wandb_proj_name,
    }

    ###########################################################################
    # Setup the dataset
    ###########################################################################
    dm = ClassificationDataModule(hyperparams)
    dm.prepare_data()

    num_labels = dm.num_labels

    hyperparams["num_classes"] = num_labels

    model = DAN(hyperparams)

    ###########################################################################
    # SETUP THE LOGGERS and Checkpointers
    ###########################################################################
    logger = WandbLogger(
        name=exp_name,
        save_dir=str(exp_dir),
        project=wandb_proj_name,
    )

    logger.watch(model, log="gradients", log_freq=10)

    callbacks = []

    checkpoints_dir = exp_dir.joinpath("checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        save_top_k=1,
        mode="max",
        monitor="dev/src_acc",
    )
    callbacks.append(checkpoint_callback)

    trainer = Trainer(
        limit_train_batches=train_proportion,
        limit_val_batches=dev_proportion,
        limit_test_batches=test_proportion,
        callbacks=callbacks,
        terminate_on_nan=True,
        gradient_clip_val=grad_clip_norm,
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

    hparams_file = exp_dir.joinpath("hparams.json")

    with open(hparams_file, "w") as fp:
        json.dump(hyperparams, fp)


if __name__ == "__main__":
    train_danformer()
