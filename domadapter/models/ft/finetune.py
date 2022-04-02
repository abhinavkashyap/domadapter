import torch
import os
import pytorch_lightning as pl
from typing import Any, Optional, Dict, List, Union
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from domadapter.console import console
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers.adapters.composition import Stack
import numpy as np

import torchmetrics


class FT(pl.LightningModule):
    def __init__(self, hparams):
        """FT LightningModule

        Args:
            hparams (Optional[Dict[str, Any]], optional): [description]. Defaults to None.
        """
        super(FT, self).__init__()

        self.save_hyperparameters(hparams)

        # config
        self.config = AutoConfig.from_pretrained(self.hparams["pretrained_model_name"], num_labels=hparams["num_classes"])

        # load the model weights
        with console.status(
            f"Loading {self.hparams['pretrained_model_name']} Model", spinner="monkey"
        ):
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.hparams["pretrained_model_name"], config=self.config
            )
        console.print(f"[green] Loaded {self.hparams['pretrained_model_name']} base model")

        self.criterion = CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()  # accuracy
        self.f1 = torchmetrics.F1(num_classes=hparams["num_classes"], average="macro")  # F1

        self.softmax = nn.Softmax(dim=1)

        #######################################################################
        # OPTIMIZER RELATED VARIABLES
        #######################################################################
        self.learning_rate = self.hparams.get("learning_rate")
        self.scheduler_factor = self.hparams.get("scheduler_factor", 0.1)
        self.scheduler_patience = self.hparams.get("scheduler_patience", 2)
        self.scheduler_threshold = self.hparams.get("scheduler_threshold", 0.0001)
        self.scheduler_cooldown = self.hparams.get("scheduler_cooldown", 0)
        self.scheduler_eps = self.hparams.get("scheduler_eps", 1e-8)

    def forward(self, input_ids, attention_mask):
        """forward function of FT

        Args:
            input_ids (Tensor): input ids tensor
            attention_mask (Tensor): attention mask tensor
        """
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits

    def configure_optimizers(self):
        # This was giving a warning:
        # RuntimeWarning: Found unsupported keys in the lr scheduler dict: ['reduce_lr_on_plateau']
        # rank_zero_warn(f"Found unsupported keys in the lr scheduler dict: {extra_keys}", RuntimeWarning)
        # They were reduce_lr_on_plateau on global steps instead of epochs (link given below)
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/673#issuecomment-572606187
        learning_rate = self.learning_rate
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        lr_scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            threshold=self.scheduler_threshold,
            threshold_mode="rel",
            cooldown=self.scheduler_cooldown,
            eps=self.scheduler_eps,
            verbose=True,
        )
        return (
            [optimizer],
            [
                {
                    "scheduler": lr_scheduler,
                    "reduce_lr_on_plateau": True,
                    "monitor": "val/loss",
                    "interval": "epoch",
                }
            ],
        )

    def training_step(self, batch, batch_idx):
        """training step of FT"""
        # get the input ids and attention mask
        input_ids, attention_mask = batch["source_input_ids"], batch["source_attention_mask"]
        # get the logits
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        # get the labels
        labels = batch["label_source"]
        # get the loss
        loss = self.criterion(logits, labels)
        accuracy = self.accuracy(labels, torch.argmax(self.softmax(logits), dim=1))
        f1 = self.f1(labels, torch.argmax(self.softmax(logits), dim=1))

        self.log(name="train/accuracy", value=accuracy)
        self.log(name="train/f1", value=f1)
        self.log(name="train/loss", value=loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """validation step of FT"""
        # get the input ids and attention mask for source data
        input_ids, attention_mask = batch["source_input_ids"], batch["source_attention_mask"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        labels = batch["label_source"]
        loss = self.criterion(logits, labels)
        accuracy = self.accuracy(labels, torch.argmax(self.softmax(logits), dim=1))
        f1 = self.f1(labels, torch.argmax(self.softmax(logits), dim=1))

        # this will log the mean div value across epoch
        self.log(name="val/loss", value=loss)
        self.log(name="val/accuracy", value=accuracy)
        self.log(name="val/f1", value=f1)

        return {
            "val/loss": loss,
            "val/accuracy": accuracy,
            "val/f1": f1,
        }

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x["val/loss"] for x in outputs]).mean()
        mean_accuracy = torch.stack([x["val/accuracy"] for x in outputs]).mean()
        mean_f1 = torch.stack([x["val/f1"] for x in outputs]).mean()

        # this will log the mean div value across epoch
        self.log(name="val/loss", value=mean_loss)
        self.log(name="val/accuracy", value=mean_accuracy)
        self.log(name="val/f1", value=mean_f1)

    def test_step(self, batch, batch_idx):
        """validation step of FT"""
        # get the input ids and attention mask for source data
        input_ids, attention_mask = batch["source_input_ids"], batch["source_attention_mask"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        labels = batch["label_source"]
        loss = self.criterion(logits, labels)
        accuracy = self.accuracy(labels, torch.argmax(self.softmax(logits), dim=1))
        f1 = self.f1(labels, torch.argmax(self.softmax(logits), dim=1))

        # this will log the mean div value across epoch
        self.log(name="test/loss", value=loss)
        self.log(name="test/accuracy", value=accuracy)
        self.log(name="test/f1", value=f1)

        # need not to log here (or we can do it but let's log at the end of each epoch)
        return {
            "test/loss": loss,
            "test/accuracy": accuracy,
            "test/f1": f1,
        }

    def test_epoch_end(self, outputs):
        mean_loss = torch.stack([x["test/loss"] for x in outputs]).mean()
        mean_accuracy = torch.stack([x["test/accuracy"] for x in outputs]).mean()
        mean_f1 = torch.stack([x["test/f1"] for x in outputs]).mean()

        # this will log the mean div value across epoch
        self.log(name="test/loss", value=mean_loss)
        self.log(name="test/accuracy", value=mean_accuracy)
        self.log(name="test/f1", value=mean_f1)
