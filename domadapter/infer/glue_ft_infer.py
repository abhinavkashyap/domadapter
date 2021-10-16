import pathlib
import json
import torch
from domadapter.console import console
from domadapter.models.ft.glue_ft import GlueFT
from domadapter.datamodules.glue_dm import GlueDM
import numpy as np
from tqdm import tqdm
from datasets import load_metric
import datasets
from transformers import AutoTokenizer


class GlueFTInfer:
    def __init__(self, experiments_dir: pathlib.Path):
        self.experiments_dir = experiments_dir
        self.checkpoints_dir = self.experiments_dir.joinpath("checkpoints")
        self.hparams_file = self.experiments_dir.joinpath("hparams.json")

        if not self.checkpoints_dir.is_dir():
            console.print(f"[red] checkpoints directory not found")
        if not self.hparams_file.is_file():
            console.print(f"[red] hparams.json file not found")

        with open(str(self.hparams_file)) as fp:
            self.hparams = json.load(fp)

        self.batch_size = self.hparams["batch_size"]
        self.tokenizer_name = self.hparams["tokenizer_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.task_name = self.hparams["task_name"]
        self.test_metric = self._load_metric()

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

        model = GlueFT.load_from_checkpoint(
            str(best_filename),
        )
        if torch.cuda.is_available():
            model.to(self.device)

        console.print(f"[green] Finished Loading Model from {best_filename}")
        return model

    def get_test_results(self):
        dm = GlueDM(self.hparams)
        dm.prepare_data()
        dm.setup("test")
        test_loader = dm.test_dataloader()
        total_batches = np.ceil(len(dm.test_dataset) // self.batch_size)

        for idx, batch in tqdm(
            enumerate(test_loader),
            total=total_batches,
            desc="Infer dataset",
            leave=False,
        ):
            new_batch = {}
            for input_name in batch:
                new_batch[input_name] = batch[input_name].to(self.device)

            output_dict = self.model.test_step(new_batch, idx)
            predictions = output_dict["predictions"]
            labels = output_dict["labels"]
            self.test_metric.add_batch(predictions=predictions, references=labels)

        test_metric = self.test_metric.compute()

        # The last part of the experiment folder can be considered unique
        exp_name = self.experiments_dir.name
        test_metric["exp_name"] = exp_name
        return test_metric

    def _load_metric(self) -> datasets.Metric:
        """Return a metric for the GLUE task

        Returns
        -------
        datasets.Metric
            A metric object of the HF datasets library

        """
        return load_metric("glue", self.task_name)
