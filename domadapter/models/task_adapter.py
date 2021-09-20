import pytorch_lightning as pl
from domadapter.console import console
from transformers import AutoConfig
from transformers import PreTrainedTokenizer
from transformers import PretrainedConfig
from transformers import AutoModelWithHeads
from transformers import PreTrainedModel
from transformers import AdapterConfig
import torch.optim as optim
from typing import Dict, Optional, List
import torch.nn as nn
from datasets import load_metric
import datasets
import numpy as np


class TaskAdapterModel(pl.LightningModule):
    def __init__(
        self,
        adapter_name: str,
        model_name: str,
        task_name: str,
        num_labels: int,
        cache_dir: str,
        tokenizer: PreTrainedTokenizer,
        id2label: Dict[int, str],
        adapter_config_name: str,
        adapter_non_linearity: str,
        adapter_reduction_factor: int,
        adam_beta1: float,
        adam_beta2: float,
        adam_epsilon: float,
        learning_rate: float,
        weight_decay: Optional[float] = 0.0,
    ):
        """Trains a adapter on a given Glue task.
        THe code has been adaopted from the Huggingface example
        on training glue models

        Parameters
        ----------
        adapter_name: str
            Unique id representing the adapter

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

        adapter_config_name: str
            The architecture name of the adapter
            pfeiffer, houlsby are supported

        adapter_non_linearity: str
            relu is the popular one

        adapter_reduction_factor: int
            Adapter down projects the hidden dimension and then up projects it
            This specifies the reduction factor of the down projection with
            respect to the original pretrained model

        adam_beta1: float
            Adam optimizer beta1 parameter

        adam_beta2: float
            Adam optimizer beta2 parameter

        adam_epsilon: float
            Adam optimizer epsilon parameter

        weight_decay: Optional[float]
            Weight decay for Adam Optimizer if we apply any.

        """
        super(TaskAdapterModel, self).__init__()
        self.adapter_name = adapter_name
        self.model_name = model_name
        self.task_name = task_name
        self.num_labels = num_labels
        self.pt_cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.adapter_config_name = adapter_config_name
        self.adapter_non_linearity = adapter_non_linearity
        self.adapter_reduction_factor = adapter_reduction_factor
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.is_regression = self.task_name == "stsb"

        self.pt_config = self._load_pt_config()
        self.model = self._load_pt_model()
        self.adapter_config = self._load_adapter_config()
        self._load_adapter()
        self.model.train_adapter([self.adapter_name])
        self.model.set_active_adapters(self.adapter_name)
        self.train_metric = self._load_metric()
        self.validation_metric = self._load_metric()
        self.test_metric = self._load_metric()

    def _load_adapter_config(self):
        with console.status("Loading Adapter config"):
            config = AdapterConfig.load(
                self.adapter_config_name,
                non_linarity=self.adapter_non_linearity,
                reduction_factor=self.adapter_reduction_factor,
            )

        # \u2713 is the unicode for ✓
        console.print("[green] Loaded Adapter config \u2713")
        return config

    def _load_adapter(self):
        with console.status(f"Adding the adapter to the model"):
            self.model.add_adapter(self.adapter_name, config=self.adapter_config)

        # \u2713 is the unicode for ✓
        console.print("[green] Loaded Adapter \u2713")

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
            model = AutoModelWithHeads.from_pretrained(
                self.model_name,
                from_tf=bool(".ckpt" in self.model_name),
                config=self.pt_config,
                cache_dir=self.pt_cache_dir,
            )
            model.add_classification_head(
                head_name=self.adapter_name,
                num_labels=self.num_labels,
                id2label=self.id2label,
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
                finetuning_task=self.adapter_name,
                cache_dir=self.pt_cache_dir,
            )

        # \u2713 is the unicode for ✓
        console.print("[green] Loaded PT model config \u2713")
        return config

    def forward(self, batch):
        outputs = self.model(**batch)
        return outputs

    def training_step(self, batch, batch_idx):
        """ Perform training on a single batch of inputs

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
            predictions.argmax(dim=-1) if not self.is_regression else predictions.squeeze()
        )
        self.log("train/loss", loss.item())
        return {"loss": loss, "predictions": predictions, "labels": labels}

    def training_epoch_end(self, outputs):
        """ Calculate metrics and do book-keeping at the end of epoch

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
        """ Perform validation on a single batch of inputs

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
            predictions.argmax(dim=-1) if not self.is_regression else predictions.squeeze()
        )
        self.log("dev/loss", loss.item())
        return {"loss": loss, "predictions": predictions, "labels": labels}

    def validation_epoch_end(self, outputs):
        """ Calculate metrics and do book-keeping at the end of epoch

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
        """ Perform test on a single batch of inputs

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
            predictions.argmax(dim=-1) if not self.is_regression else predictions.squeeze()
        )
        self.log("test/loss", loss.item())
        return {"loss": loss, "predictions": predictions, "labels": labels}

    def test_epoch_end(self, outputs):
        """ Calculate metrics and do book-keeping at the end of epoch

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
        decay_parameters = self.get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if hasattr(self.model, "config") and hasattr(self.model.config, "adapter_fusion_models"):
            no_decay = [
                f"adapter_fusion_layer.{n}.value" for n in self.model.config.adapter_fusion_models
            ]
            decay_parameters = [name for name in decay_parameters if name not in no_decay]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = optim.AdamW
        optimizer_kwargs = {
            "betas": (self.adam_beta1, self.adam_beta2),
            "eps": self.adam_epsilon,
            "lr": self.learning_rate,
        }
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return {"optimizer": optimizer}

    def get_parameter_names(self, model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in self.get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result

    @staticmethod
    def _compute_loss(model_outputs):
        """ Compute the loss from HF model outputs

        Parameters
        ----------
        model_outputs
            Hugging face model outputs

        Returns
        -------
        torch.Tensor
            loss

        """
        loss = model_outputs["loss"] if isinstance(model_outputs, dict) else model_outputs[0]
        return loss

    def _load_metric(self) -> datasets.Metric:
        """Return a metric for the GLUE task

        Returns
        -------
        datasets.Metric
            A metric object of the HF datasets library

        """
        return load_metric("glue", self.task_name)
