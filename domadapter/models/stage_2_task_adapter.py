import torch
import pytorch_lightning as pl
from typing import Any, Optional, Dict, List, Union
from transformers import BertModel
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertPooler
from domadapter.console import console
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from domadapter.models.linear_clf import LinearClassifier
import numpy as np
from domadapter.divergences.gaussian_mkmmd_divergence import GaussianMKMMD
from domadapter.divergences.rbf_mkmmd_divergence import RBFMKMMD
from domadapter.divergences.cmd_divergence import CMD

from domadapter.models.linear_clf import LinearClassifier
from domadapter.models.domain_adapter import DomainAdapter
import torchmetrics 

class Stage2TaskAdaptor(pl.LightningModule):

    def __init__(self, hparams, config):

        """Stage2TaskAdaptor LightningModule to task adapter after domain adapter training.

        Args:
            hparams (Optional[Dict[str, Any]], optional): [description]. Defaults to None.
        """
        super(Stage2TaskAdaptor, self).__init__()

        self.hparams = hparams
        self.config = config

        # create domain adapter model
        self.domain_adapter = DomainAdapter(hparams, config)

        # load the weights for trained domain adapter
        self.domain_adapter.load_state_dict(torch.load(self.config.domain_adapter_path))

        # add adapter a new task adapter 
        self.domain_adapter.add_adapter(self.config['task_adapter_name'])

        # activate the task adapter
        self.domain_adapter.train_adapter(self.config['task_adapter_name'])

        # add a classification head over it
        self.classification = LinearClassifier(
            num_hidden_layers=self.config['num_hidden_layers'],
            input_siz=self.config['hidden_size'],
            hidden_size=self.config['hidden_size'],
            output_size=self.config['num_classes'],
            return_hiddens=False,
        )

        # put it into pytorch sequential model
        self.model = nn.Sequential(*[self.domain_adapter, self.classification])

        # cross entropy loss function
        self.criterion = CrossEntropyLoss()

        # accuracy 
        self.accuracy = torchmetrics.Accuracy()

    
    def forward(self, input_ids, attention_mask):
        """forward function of stage-2 task adapter

        Args:
            input_ids (Tensor): input ids tensor
            attention_mask (Tensor): attention mask tensor
        """
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return logits

    def training_step(self, batch, batch_idx):
        """training step of stage-2 task adapter"""
        # get the input ids and attention mask
        input_ids, attention_mask = batch["source_input_ids"], batch["source_attention_mask"]
        # get the logits
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        # get the labels
        labels = batch['label']
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
        """validation step of stage-2 task adapter"""
        # get the input ids and attention mask
        input_ids, attention_mask = batch["source_input_ids"], batch["source_attention_mask"]
        # get the logits
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        # get the labels
        labels = batch['label']
        # get the loss
        loss = self.criterion(logits, labels)

        # compute accuracy
        accuracy = self.accuracy(labels, logits.softmax(dim=-1).view(-1))

        # need not to log here (or we can do it but let's log at the end of each epoch)
    
        return {"val/loss":loss, "val/accuracy":accuracy}

    def validation_epoch_end(self, outputs):

        mean_loss = torch.stack([x['val/loss'] for x in outputs]).mean()
        mean_accuracy = torch.stack([x['val/accuracy'] for x in outputs]).mean()

        # this will log the mean div value across epoch 
        self.log(
            "val/loss",
            value=mean_loss,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        # this will log the mean div value across epoch 
        self.log(
            "val/accuracy",
            value=mean_accuracy,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        # need not to return


    def test_step(self, batch, batch_idx):
        """validation step of stage-2 task adapter"""
        # get the input ids and attention mask
        input_ids, attention_mask = batch["source_input_ids"], batch["source_attention_mask"]
        # get the logits
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        # get the labels
        labels = batch['label']
        # get the loss
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




