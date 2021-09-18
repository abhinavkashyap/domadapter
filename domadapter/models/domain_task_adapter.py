import torch
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

from domadapter.models.domain_adapter import DomainAdapter
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
        self.config = AutoConfig.from_pretrained(self.hprams['pretrained_model_name'])
        # to get the layer wise pre-trained model outputs
        self.config.output_hidden_states = True

        # load the model weights
        with console.status(f"Loading {self.hprams['pretrained_model_name']} Model", spinner="monkey"):
            self.model = AutoModelWithHeads.from_pretrained(
                self.hprams['pretrained_model_name'], config=self.config
            )
        console.print(
            f"[green] Loaded {self.hprams['pretrained_model_name']} base model"
        )

        # load domain adapter to PLM
        self.model.load_adapter(self.hprams['domain_adapter_name'])
        # add task adapter to PLM
        self.model.add_adapter(self.hprams['task_adapter_name'])
        # add classification head to task adapter
        self.model.add_classification_head(self.hprams['task_adapter_name'], num_labels=self.hprams['task_adapter_num_labels'])
        # stack adapters
        self.model.active_adapters = Stack(self.hprams['domain_adapter_name'], self.hprams['task_adapter_name'])
        # Freeze all parameters and train only task adapter
        self.model.train_adapter([self.hprams['task_adapter_name']])

        self.criterion = CrossEntropyLoss()
        # accuracy
        self.accuracy = torchmetrics.Accuracy()


    def forward(self, input_ids, attention_mask):
        """forward function of DomainTaskAdaptor

        Args:
            input_ids (Tensor): input ids tensor
            attention_mask (Tensor): attention mask tensor
        """
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return logits.pooler_output

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
        # compute accuracy
        accuracy = self.accuracy(labels, logits.softmax(dim=-1).view(-1))

        self.log(
            "train/accuracy",
            accuracy,
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
        source_accuracy = self.accuracy(labels, logits.softmax(dim=-1).view(-1))

        # get the input ids and attention mask for target data
        input_ids, attention_mask = batch["target_input_ids"], batch["target_attention_mask"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        labels = batch['label_target']
        target_loss = self.criterion(logits, labels)
        target_accuracy = self.accuracy(labels, logits.softmax(dim=-1).view(-1))

        # need not to log here (or we can do it but let's log at the end of each epoch)
        return {"source_val/loss":source_loss,
                "source_val/accuracy":source_accuracy,
                "target_val/loss":target_loss,
                "target_val/accuracy":target_accuracy
        }

    def validation_epoch_end(self, outputs):
        mean_source_loss = torch.stack([x['source_val/loss'] for x in outputs]).mean()
        mean_source_accuracy = torch.stack([x['source_val/accuracy'] for x in outputs]).mean()

        mean_target_loss = torch.stack([x['target_val/loss'] for x in outputs]).mean()
        mean_target_accuracy = torch.stack([x['target_val/accuracy'] for x in outputs]).mean()

        # this will log the mean div value across epoch
        self.log(
            "source_val/loss",
            value=mean_source_loss,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        # this will log the mean div value across epoch
        self.log(
            "source_val/loss",
            value=mean_source_accuracy,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
                # this will log the mean div value across epoch
        self.log(
            "target_val/loss",
            value=mean_target_loss,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        # this will log the mean div value across epoch
        self.log(
            "target_val/accuracy",
            value=mean_target_accuracy,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        # need not to return

    '''
    not sure about testing part. should we do this post complete training?
    hence commenting. to ask @abhinav

    def test_step(self, batch, batch_idx):
        """test step of stage-2 task adapter"""
        # get the input ids and attention mask
        input_ids, attention_mask = batch["source_input_ids"], batch["source_attention_mask"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        labels = batch['label']
        loss = self.criterion(logits, labels)
        # compute accuracy
        accuracy = self.accuracy(labels, logits.softmax(dim=-1).view(-1))
        # need not to log here (or we can do it but let's log at the end of each epoch)
        return {"test/loss":loss, "test/accuracy":accuracy}


    def test_epoch_end(self, outputs):

        mean_loss = torch.stack([x['test/loss'] for x in outputs]).mean()
        mean_accuracy = torch.stack([x['test/accuracy'] for x in outputs]).mean()

        # this will log the mean loss across epoch
        self.log(
            "test/loss",
            value=mean_loss,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        # this will log the mean accuracy value across epoch
        self.log(
            "test/accuracy",
            value=mean_accuracy,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        # need not to return

    '''


    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.hparams['learning_rate'],
            betas=self.hparams['betas'],
            eps=self.hparams['eps'],
            weight_decay=self.hparams['weight_decay'],
            # amsgrad=self.hparams['amsgrad'], not using this hparam
        )
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
                    "monitor": "train/divergence",
                    "interval": "epoch",
                }
            ],
        )




