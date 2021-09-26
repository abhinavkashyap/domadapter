import pytorch_lightning as pl
from domadapter.console import console
from transformers import AutoConfig
from transformers import PreTrainedTokenizer
from transformers import PretrainedConfig
from transformers import AutoModelForSequenceClassification
from transformers import PreTrainedModel
import torch.optim as optim
from typing import Dict, Optional, List
from datasets import load_metric
import datasets
import numpy as np


class GlueFT(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        task_name: str,
        num_labels: int,
        cache_dir: str,
        tokenizer: PreTrainedTokenizer,
        id2label: Dict[int, str],
        adam_beta1: float,
        adam_beta2: float,
        adam_epsilon: float,
        learning_rate: float,
        weight_decay: Optional[float] = 0.0,
    ):
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

        tokenizer: PreTrainedTokenizer
            A pretrained tokenizer from the transformer library

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
        super(GlueFT, self).__init__()
        self.model_name = model_name
        self.task_name = task_name
        self.num_labels = num_labels
        self.pt_cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.is_regression = self.task_name == "stsb"

        self.pt_config = self._load_pt_config()
        self.model = self._load_pt_model()
        self.train_metric = self._load_metric()
        self.validation_metric = self._load_metric()
        self.test_metric = self._load_metric()

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
            losses.append(loss)

        train_metric = self.train_metric.compute()
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
            losses.append(loss)

        validation_metric = self.validation_metric.compute()
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
            losses.append(loss)

        test_metric = self.test_metric.compute()
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
