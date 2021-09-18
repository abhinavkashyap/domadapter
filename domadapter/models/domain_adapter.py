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


class DomainAdapter(pl.LightningModule):

    def __init__(self, config, hparams:Optional[Dict[str, Any]] = None):
        """Domain Adapter LightningModule to train domain adapter using cmd as divernnce

        Args:
            hparams (Optional[Dict[str, Any]], optional): [description]. Defaults to None.
        """
        super(DomainAdapter, self).__init__()

        self.hparams = hparams
        self.config = config

        # bert config 
        self.bert_config = (
            BertConfig()
            if self.hparams.get("bert_config") is None
            else self.hparams.get("bert_config")
        )
        # to get the layer wise pre-trained bert model outputs
        self.bert_config.output_hidden_states = True

        # load the bert model weights
        with console.status("Loading BERT Model", spinner="monkey"):
            self.bert = BertModel.from_pretrained(
                self.pretrained_model_name, config=self.bert_config
            )
        console.print(
            f"[green] Loaded BERT model from {self.hprams['pretrained_model_name']}"
        )

        # add adapter a new adapter 
        self.bert.add_adapter(self.config['domain_adapter_name'])

        # activate the adapter
        self.bert.train_adapter(self.config['domain_adapter_name'])


        # object to compute the divergence
        self.criterion = CMD()



    def forward(self, input_ids, attention_mask=None):
        """Forward pass of the model"""
        # get the bert output
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # get the last layer pooler output
        pooler_output = bert_output.pooler_output

        return pooler_output
    

    # def configure_optimizers(self):
    #     return torch.optim.AdamW(
    #         params=self.model.parameters(), 
    #         lr=self.hparams['learning_rate'], 
    #         betas=self.hparams['betas'], 
    #         eps=self.hparams['eps'], 
    #         weight_decay=self.hparams['weight_decay'],
    #         # amsgrad=self.hparams['amsgrad'], not using this hparam
    #     )
        
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

    def training_step(self, batch, batch_idx):

        BATCH_SIZE = batch['input_ids'].shape[0]

        # concat the source and target data and pass it to the bert model
        input_ids = torch.cat((batch["source_input_ids"], batch["target_input_ids"]), axis=0)
        attention_mask = torch.cat((batch["source_attention_mask"], batch["target_attention_mask"]), axis=0)

        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        src_feature, trg_feature = torch.split(tensor=outputs, split_size_or_sections=BATCH_SIZE, dim=0)
        divergence = self.criterion.calculate(src_hidden=src_feature, trg_hidden=trg_feature)

        self.log(
            "train/divergence",
            divergence,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return {"train/divergence": divergence}

    def validation_step(self, batch, batch_idx):

        BATCH_SIZE = batch['input_ids'].shape[0]

        # concat the source and target data and pass it to the bert model
        input_ids = torch.cat((batch["source_input_ids"], batch["target_input_ids"]), axis=0)
        attention_mask = torch.cat((batch["source_attention_mask"], batch["target_attention_mask"]), axis=0)

        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        src_feature, trg_feature = torch.split(tensor=outputs, split_size_or_sections=BATCH_SIZE, dim=0)
        divergence = self.criterion.calculate(src_hidden=src_feature, trg_hidden=trg_feature)

        # we can comment the logging here
        self.log(
            "val/divergence",
            value=divergence,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return {"val/divergence": divergence}

    def validation_epoch_end(self, outputs):

        mean_divergenence = torch.stack([x['val/divergence'] for x in outputs]).mean()

        # this will show the mean div value across epoch 
        self.log(
            "val/divergence",
            value=mean_divergenence,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        # need not to return
        # return {"val/divergence": mean_divergenence}

    
    def test_step(self, batch, batch_idx):

        BATCH_SIZE = batch['input_ids'].shape[0]

        # concat the source and target data and pass it to the bert model
        input_ids = torch.cat((batch["source_input_ids"], batch["target_input_ids"]), axis=0)
        attention_mask = torch.cat((batch["source_attention_mask"], batch["target_attention_mask"]), axis=0)

        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        src_feature, trg_feature = torch.split(tensor=outputs, split_size_or_sections=BATCH_SIZE, dim=0)
        divergence = self.criterion.calculate(src_hidden=src_feature, trg_hidden=trg_feature)

    
        return {"test/divergence": divergence}

    
    def test_epoch_end(self, outputs):
        # compute the mean of divergence across epoch and return it
        mean_divergence = torch.stack([x['test/divergence'] for x in outputs]).mean()

        # log it
        self.log(
            "test/divergence",
            value=mean_divergence,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        return {"test/divergence": mean_divergence}
        






