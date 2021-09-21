import torch
import os
import pytorch_lightning as pl
from typing import Any, Optional, Dict, List, Union
from transformers import AutoModelWithHeads
from transformers import AutoConfig
from domadapter.console import console
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers.adapters.composition import Stack
import numpy as np

import torchmetrics

class DomainTaskAdaptor(pl.LightningModule):

    def __init__(self, hparams):
        """DomainTaskAdaptor LightningModule to task adapter after domain adapter training.

        Args:
            hparams (Optional[Dict[str, Any]], optional): [description]. Defaults to None.
        """
        super(DomainTaskAdaptor, self).__init__()

        self.save_hyperparameters(hparams)

        # config
        self.config = AutoConfig.from_pretrained(self.hparams['pretrained_model_name'])
        # to get the layer wise pre-trained model outputs
        self.config.output_hidden_states = True

        # load the model weights
        with console.status(f"Loading {self.hparams['pretrained_model_name']} Model", spinner="monkey"):
            self.model = AutoModelWithHeads.from_pretrained(
                self.hparams['pretrained_model_name'], config=self.config
            )
        console.print(
            f"[green] Loaded {self.hparams['pretrained_model_name']} base model"
        )

        with console.status(f"Loading {self.hparams['source_target']} domain adapter", spinner="monkey"):
            # load domain adapter to PLM
            self.model.load_adapter(os.path.join(hparams['exp_dir'], "domain_adapter"))
        # add task adapter to PLM
        self.model.add_adapter(f"task_adapter_{self.hparams['source_target']}")
        # add classification head to task adapter
        self.model.add_classification_head(f"task_adapter_{self.hparams['source_target']}", num_labels=self.hparams['num_classes'])
        # stack adapters
        self.model.active_adapters = Stack(f"domain_adapter_{self.hparams['source_target']}", f"task_adapter_{self.hparams['source_target']}")
        # Freeze all parameters and train only task adapter
        self.model.train_adapter([f"task_adapter_{self.hparams['source_target']}"])

        self.criterion = CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()  # accuracy
        self.f1 = torchmetrics.F1(num_classes=hparams['num_classes'], average='macro')  # F1

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
        """forward function of DomainTaskAdaptor

        Args:
            input_ids (Tensor): input ids tensor
            attention_mask (Tensor): attention mask tensor
        """
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits

    def save_adapter(self, location, adapter_name):
        """Module to save adapter.
        Args:
            location str: Location where to save adapter.
            adapter_name: Name of adapter to be saved.
        """
        self.model.save_adapter(location, adapter_name)

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
                    "monitor": "source_val/loss",
                    "interval": "epoch",
                }
            ],
        )

    def training_step(self, batch, batch_idx):
        """training step of DomainTaskAdaptor"""
        # get the input ids and attention mask
        input_ids, attention_mask = batch["source_input_ids"], batch["source_attention_mask"]
        # get the logits
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        # get the labels
        labels = batch['label_source']
        # get the loss
        loss = self.criterion(logits, labels)
        accuracy = self.accuracy(labels, torch.argmax(self.softmax(logits),dim=1))
        f1 = self.f1(labels, torch.argmax(self.softmax(logits),dim=1))

        self.log(
            "train/accuracy",
            accuracy,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/f1",
            f1,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        return loss


    def validation_step(self, batch, batch_idx):
        """validation step of DomainTaskAdaptor"""
        # get the input ids and attention mask for source data
        input_ids, attention_mask = batch["source_input_ids"], batch["source_attention_mask"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        labels = batch['label_source']
        source_loss = self.criterion(logits, labels)
        source_accuracy = self.accuracy(labels, torch.argmax(self.softmax(logits),dim=1))
        source_f1 = self.f1(labels, torch.argmax(self.softmax(logits),dim=1))

        # get the input ids and attention mask for target data
        input_ids, attention_mask = batch["target_input_ids"], batch["target_attention_mask"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        labels = batch['label_target']
        target_loss = self.criterion(logits, labels)
        target_accuracy = self.accuracy(labels, torch.argmax(self.softmax(logits),dim=1))
        target_f1 = self.f1(labels, torch.argmax(self.softmax(logits),dim=1))

        # this will log the mean div value across epoch
        self.log(
            "source_val/loss",
            value=source_loss,
            prog_bar=False,
            on_step=True,
            logger=True,
            on_epoch=False,
        )
        self.log(
            "source_val/accuracy",
            value=source_accuracy,
            prog_bar=False,
            on_step=True,
            logger=True,
            on_epoch=False,
        )
        self.log(
            "target_val/loss",
            value=target_loss,
            prog_bar=False,
            on_step=True,
            logger=True,
            on_epoch=False,
        )
        self.log(
            "target_val/accuracy",
            value=target_accuracy,
            prog_bar=False,
            on_step=True,
            logger=True,
            on_epoch=False,
        )
        self.log(
            "target_val/f1",
            value=target_f1,
            prog_bar=False,
            on_step=True,
            logger=True,
            on_epoch=False,
        )
        self.log(
            "source_val/f1",
            value=source_f1,
            prog_bar=False,
            on_step=True,
            logger=True,
            on_epoch=False,
        )

        return {"source_val/loss":source_loss,
                "source_val/accuracy":source_accuracy,
                "source_val/f1":source_f1,
                "target_val/loss":target_loss,
                "target_val/accuracy":target_accuracy,
                "target_val/f1":target_f1,
        }

    def validation_epoch_end(self, outputs):
        mean_source_loss = torch.stack([x['source_val/loss'] for x in outputs]).mean()
        mean_source_accuracy = torch.stack([x['source_val/accuracy'] for x in outputs]).mean()
        mean_source_f1 = torch.stack([x['source_val/f1'] for x in outputs]).mean()

        mean_target_loss = torch.stack([x['target_val/loss'] for x in outputs]).mean()
        mean_target_accuracy = torch.stack([x['target_val/accuracy'] for x in outputs]).mean()
        mean_target_f1 = torch.stack([x['target_val/f1'] for x in outputs]).mean()

        # this will log the mean div value across epoch
        self.log(
            "source_val/loss",
            value=mean_source_loss,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "source_val/accuracy",
            value=mean_source_accuracy,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "target_val/loss",
            value=mean_target_loss,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "target_val/accuracy",
            value=mean_target_accuracy,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "target_val/f1",
            value=mean_target_f1,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "source_val/f1",
            value=mean_source_f1,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx):
        """validation step of DomainTaskAdaptor"""
                # get the input ids and attention mask for source data
        input_ids, attention_mask = batch["source_input_ids"], batch["source_attention_mask"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        labels = batch['label_source']
        source_loss = self.criterion(logits, labels)
        source_accuracy = self.accuracy(labels, torch.argmax(self.softmax(logits),dim=1))
        source_f1 = self.f1(labels, torch.argmax(self.softmax(logits),dim=1))

        # get the input ids and attention mask for target data
        input_ids, attention_mask = batch["target_input_ids"], batch["target_attention_mask"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        labels = batch['label_target']
        target_loss = self.criterion(logits, labels)
        target_accuracy = self.accuracy(labels, torch.argmax(self.softmax(logits),dim=1))
        target_f1 = self.f1(labels, torch.argmax(self.softmax(logits),dim=1))

        # this will log the mean div value across epoch
        self.log(
            "source_test/loss",
            value=source_loss,
            prog_bar=False,
            on_step=True,
            logger=True,
            on_epoch=False,
        )
        self.log(
            "source_test/accuracy",
            value=source_accuracy,
            prog_bar=False,
            on_step=True,
            logger=True,
            on_epoch=False,
        )
        self.log(
            "target_test/loss",
            value=target_loss,
            prog_bar=False,
            on_step=True,
            logger=True,
            on_epoch=False,
        )
        self.log(
            "target_test/accuracy",
            value=target_accuracy,
            prog_bar=False,
            on_step=True,
            logger=True,
            on_epoch=False,
        )
        self.log(
            "target_test/f1",
            value=target_f1,
            prog_bar=False,
            on_step=True,
            logger=True,
            on_epoch=False,
        )
        self.log(
            "source_test/f1",
            value=source_f1,
            prog_bar=False,
            on_step=True,
            logger=True,
            on_epoch=False,
        )

        # need not to log here (or we can do it but let's log at the end of each epoch)
        return {"source_test/loss":source_loss,
                "source_test/accuracy":source_accuracy,
                "source_test/f1":source_f1,
                "target_test/loss":target_loss,
                "target_test/accuracy":target_accuracy,
                "target_test/f1":target_f1,
        }

    def test_epoch_end(self, outputs):
        mean_source_loss = torch.stack([x['source_test/loss'] for x in outputs]).mean()
        mean_source_accuracy = torch.stack([x['source_test/accuracy'] for x in outputs]).mean()
        mean_source_f1 = torch.stack([x['source_test/f1'] for x in outputs]).mean()

        mean_target_loss = torch.stack([x['target_test/loss'] for x in outputs]).mean()
        mean_target_accuracy = torch.stack([x['target_test/accuracy'] for x in outputs]).mean()
        mean_target_f1 = torch.stack([x['target_test/f1'] for x in outputs]).mean()

        # this will log the mean div value across epoch
        self.log(
            "source_test/loss",
            value=mean_source_loss,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "source_test/accuracy",
            value=mean_source_accuracy,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "target_test/loss",
            value=mean_target_loss,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "target_test/accuracy",
            value=mean_target_accuracy,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "target_test/f1",
            value=mean_target_f1,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "source_test/f1",
            value=mean_source_f1,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )





