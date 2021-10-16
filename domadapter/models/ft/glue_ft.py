import pytorch_lightning as pl
from domadapter.console import console
from transformers import AutoConfig
from transformers import PreTrainedTokenizer
from transformers import PretrainedConfig
from transformers import AutoModelForSequenceClassification
from transformers import PreTrainedModel
import torch.optim as optim
from typing import Dict, Optional, List, Any
from datasets import load_metric
import datasets
import numpy as np


class GlueFT(pl.LightningModule):
    """Finetunes a pretrained language model on glue task

        Parameters
        ----------
        model_name: str
            Name of Huggingface pretrained model
            If not a pretrained model then the path of the trained model

        task_name: str
            The glue task name

        num_labels: int

        cache_dir: str
            Directory sotres the pretrained language models downloaded from
            Huggingface

        id2label: Dict[int, str]
            A mapping from label id to string

        adam_beta1: float
            Adam optimizer beta1 parameter

        adam_beta2: float
            Adam optimizer beta2 parameter

        adam_epsilon: float
            Adam optimizer epsilon parameter

        weight_decay: Optional[float]
            Weight decay for Adam Optimizer if we apply any.

    """
    def __init__(
        self,
        hparams: Dict[str, Any]
    ):
        super(GlueFT, self).__init__()
        self.save_hyperparameters(hparams)
        self.model_name = hparams["model_name"]
        self.task_name = hparams["task_name"]
        self.num_labels = hparams["num_labels"]
        self.pt_cache_dir = hparams["cache_dir"]
        self.id2label = hparams["id2label"]
        self.adam_beta1 = hparams["adam_beta1"]
        self.adam_beta2 = hparams["adam_beta2"]
        self.adam_epsilon = hparams["adam_epsilon"]
        self.learning_rate = hparams["learning_rate"]
        self.weight_decay = hparams.get("weight_decay", 0.0)

        self.is_regression = self.task_name == "stsb"

        self.pt_config = self._load_pt_config()
        self.model = self._load_pt_model()
        self.train_metric = self._load_metric()
        self.validation_metric = self._load_metric()
        self.test_metric = self._load_metric()

        if self.task_name == "mnli":
            self.train_f1 = self._load_f1_metric()
            self.validation_f1 = self._load_f1_metric()
            self.test_f1 = self._load_f1_metric()

    def forward(self, batch):
        outputs = self.model(**batch)
        return outputs

    def training_step(self, batch, batch_idx):
        """Perform training on a single batch of inputs

        Parameters
        ----------
        batch
            Output from HF PretrainedTokenizer; "labels"
            for supervised learning.
            Look at datamodule/glue_dm.py or similar
            data modules for more information. The batch
            contains

        batch_idx: int
            The index of the batch

        Returns
        -------
        Dict[str, Any]
            loss - The loss for the batch
            predictions - The predictions for the batch
            labels - The labels of the batch

        """
        outputs = self.forward(batch)
        loss = self._compute_loss(outputs)
        predictions = outputs["logits"]
        labels = batch["labels"]
        predictions = (
            predictions.argmax(dim=-1)
            if not self.is_regression
            else predictions.squeeze()
        )
        self.log("train/loss", loss.item())
        return {"loss": loss, "predictions": predictions, "labels": labels}

    def training_epoch_end(self, outputs):
        """Calculate metrics and do book-keeping at the end of epoch

        Parameters
        ----------
        outputs: List[Dict[str, Any]]
            Outputs from the corresponding training/validation/test
            step.
            Use the outputs to calculate metrics for the entire
            epoch and log them

        Returns
        -------
        None

        """

        losses: List[float] = []
        for output in outputs:
            predictions = output["predictions"]
            labels = output["labels"]
            loss = output["loss"].cpu().item()
            self.train_metric.add_batch(predictions=predictions, references=labels)
            if self.task_name == "mnli":
                self.train_f1.add_batch(predictions=predictions, references=labels)
            losses.append(loss)

        train_metric = self.train_metric.compute()
        if self.task_name == "mnli":
            train_f1 = self.train_f1.compute(average="macro")
            train_metric = {**train_metric, **train_f1}
        self.log("train/loss", np.mean(losses))
        for key in train_metric:
            self.log(f"train/{key}", train_metric[key])

    def validation_step(self, batch, batch_idx):
        """Perform validation on a single batch of inputs

         Parameters
        ----------
        batch
            Output from HF PretrainedTokenizer; "labels"
            for supervised learning.
            Look at datamodule/glue_dm.py or similar
            data modules for more information. The batch
            contains

        batch_idx: int
            The index of the batch

        Returns
        -------
        Dict[str, Any]
            loss - The loss for the batch
            predictions - The predictions for the batch
            labels - The labels of the batch
        """
        outputs = self.forward(batch)
        loss = self._compute_loss(outputs)
        predictions = outputs["logits"]
        labels = batch["labels"]
        predictions = (
            predictions.argmax(dim=-1)
            if not self.is_regression
            else predictions.squeeze()
        )
        self.log("dev/loss", loss.item())
        return {"loss": loss, "predictions": predictions, "labels": labels}

    def validation_epoch_end(self, outputs):
        """Calculate metrics and do book-keeping at the end of epoch

        Parameters
        ----------
        outputs: List[Dict[str, Any]]
            Outputs from the corresponding training/validation/test
            step.
            Use the outputs to calculate metrics for the entire
            epoch and log them

        Returns
        -------
        None

        """
        losses: List[float] = []
        for output in outputs:
            predictions = output["predictions"]
            labels = output["labels"]
            loss = output["loss"].cpu().item()
            self.validation_metric.add_batch(predictions=predictions, references=labels)
            if self.task_name == "mnli":
                self.validation_f1.add_batch(predictions=predictions, references=labels)
            losses.append(loss)

        validation_metric = self.validation_metric.compute()
        if self.task_name == "mnli":
            validation_f1 = self.validation_f1.compute(average="macro")
            validation_metric = {**validation_metric, **validation_f1}

        self.log("dev/loss", np.mean(losses))
        for key in validation_metric:
            self.log(f"dev/{key}", validation_metric[key])

    def test_step(self, batch, batch_idx):
        """Perform test on a single batch of inputs

        Parameters
        ----------
        batch
            Output from HF PretrainedTokenizer; "labels"
            for supervised learning.
            Look at datamodule/glue_dm.py or similar
            data modules for more information. The batch
            contains

        batch_idx: int
            The index of the batch

        Returns
        -------
        Dict[str, Any]
            loss - The loss for the batch
            predictions - The predictions for the batch
            labels - The labels of the batch


        """

        outputs = self.forward(batch)
        loss = self._compute_loss(outputs)
        predictions = outputs["logits"]
        labels = batch["labels"]
        predictions = (
            predictions.argmax(dim=-1)
            if not self.is_regression
            else predictions.squeeze()
        )
        return {"loss": loss, "predictions": predictions, "labels": labels}

    def test_epoch_end(self, outputs):
        """Calculate metrics and do book-keeping at the end of epoch

        Parameters
        ----------
        outputs: List[Dict[str, Any]]
            Outputs from the corresponding training/validation/test
            step.
            Use the outputs to calculate metrics for the entire
            epoch and log them

        Returns
        -------
        None

        """
        losses: List[float] = []
        for output in outputs:
            predictions = output["predictions"]
            labels = output["labels"]
            loss = output["loss"].cpu().item()
            self.test_metric.add_batch(predictions=predictions, references=labels)
            if self.task_name == "mnli":
                self.test_f1.add_batch(predictions=predictions, references=labels)
            losses.append(loss)

        test_metric = self.test_metric.compute()
        if self.task_name == "mnli":
            test_f1 = self.test_f1.compute(average="macro")
            test_metric = {**test_metric, **test_f1}

        self.log("test/loss", np.mean(losses))
        for key in test_metric:
            self.log(f"test/{key}", test_metric[key])

    def configure_optimizers(self):
        """

        Returns
        -------
        Dict[str, Any]
            {
            "optimizer" optimizer
            }


        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        return {
            "optimizer": optimizer
        }

    @staticmethod
    def _compute_loss(model_outputs):
        """Compute the loss from HF model outputs

        Parameters
        ----------
        model_outputs
            Hugging face model outputs

        Returns
        -------
        torch.Tensor
            loss

        """
        loss = (
            model_outputs["loss"]
            if isinstance(model_outputs, dict)
            else model_outputs[0]
        )
        return loss

    def _load_metric(self) -> datasets.Metric:
        """Return a metric for the GLUE task

        Returns
        -------
        datasets.Metric
            A metric object of the HF datasets library

        """
        return load_metric("glue", self.task_name)

    def _load_pt_model(self) -> PreTrainedModel:
        """Return a pretrained model. AutoModelWithHeads
        is a class added by AdapterHub. Not available in the
        main branch of Huggingface itself.

        Returns
        -------
        PretrainedModel
            A huggingface pretrained model
        """
        with console.status("Loading PT model"):
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                from_tf=bool(".ckpt" in self.model_name),
                config=self.pt_config,
                cache_dir=self.pt_cache_dir,
            )

        # \u2713 is the unicode for ✓
        console.print("[green] Loaded PT model \u2713")
        return model

    def _load_pt_config(self) -> PretrainedConfig:
        """Load pretrained model config

        Returns
        -------
        PretrainedConfig
            A config for a pretrained model

        """

        with console.status("Loading PT model config"):
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                cache_dir=self.pt_cache_dir,
            )

        # \u2713 is the unicode for ✓
        console.print("[green] Loaded PT model config \u2713")
        return config

    def _load_f1_metric(self) -> datasets.Metric:
        """ Returns F1 metric from hugging face datasets.
        This is used by us when the dataset classes might be imbalanced.
        We sample MNLI datasets and the sampling can create imbalance
        and it is best to report F1 than `accuracy` which is the usual metric.

        Returns
        -------
        datasets.Metric

        """
        return load_metric("f1")