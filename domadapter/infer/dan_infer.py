import pathlib
from domadapter.models.dan import DAN
from domadapter.datamodules.clf_datamodule import ClassificationDataModule
import torch
import json
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torchmetrics
import click
from domadapter.console import console


class DANInfer:
    def __init__(self, checkpoints_dir: pathlib.Path, hparams_file: pathlib.Path):
        self.checkpoints_dir = checkpoints_dir
        self.hparams_file = hparams_file

        with open(str(self.hparams_file)) as fp:
            self.hparams = json.load(fp)

        self.batch_size = self.hparams["bsz"]
        self.softmax = nn.Softmax(dim=1)
        self.infer_accuracy = torchmetrics.Accuracy()
        self.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model = self._load_model()

    def _load_model(self):
        filenames = list(self.checkpoints_dir.iterdir())
        assert (
            len(filenames) == 1
        ), f"Make sure only the best model is stored in the checkpoints directory"
        best_filename = self.checkpoints_dir.joinpath(filenames[-1])

        model = DAN.load_from_checkpoint(
            str(best_filename),
        )
        if torch.cuda.is_available():
            model.to(self.device)

        console.print(f"[green] Finished Loading Model from {best_filename}")
        return model

    def predict(self, infer_filename: pathlib.Path, domain="src"):
        dm = ClassificationDataModule(self.hparams)
        dm.infer_filename = infer_filename
        dm.prepare_data()
        dm.setup_infer()
        infer_data_loader = dm.infer_dataloader()

        total_batches = np.ceil(len(dm.infer_dataset) // self.batch_size)
        all_preds = []
        accs = []
        for batch in tqdm(
            infer_data_loader,
            total=total_batches,
            desc="Infer dataset",
            leave=False,
        ):
            input_ids, attention_mask, token_type_ids, label_ids = batch

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)

            if domain == "src":
                logits, _ = self.model.forward_src(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

            elif domain == "trg":
                logits, _ = self.model.forward_trg(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
            else:
                raise ValueError(f"domain has to be one of [src, trg]")

            probs = self.softmax(logits)
            probs = probs.cpu()
            label_ids = label_ids.cpu()

            acc = self.infer_accuracy(probs, label_ids.view(-1))
            accs.append(acc.item())

        return np.mean(accs)


@click.command()
@click.option(
    "--experiment-dir",
    type=str,
    help="Experiment Directory where all the information is stored",
)
@click.option(
    "--infer-filename",
    type=str,
    help="File where examples from the target domain are stored",
)
@click.option("--use-infer-branch", type=str, help="Which branch to use for inference")
def infer_danformer(experiment_dir, infer_filename, use_infer_branch):

    # we will asssume hparms.json and checkpoints directory to be available
    # inside the experiment_dir
    experiment_dir = pathlib.Path(experiment_dir)
    infer_filename = pathlib.Path(infer_filename)
    json_file = experiment_dir.joinpath("hparams.json")
    checkpoints_dir = experiment_dir.joinpath("checkpoints")

    infer = DANInfer(checkpoints_dir=checkpoints_dir, hparams_file=json_file)

    acc = infer.predict(infer_filename=infer_filename, domain=use_infer_branch)
    print(f"Accuracy: {infer_filename}: {acc}")
    return acc


if __name__ == "__main__":
    infer_danformer()
