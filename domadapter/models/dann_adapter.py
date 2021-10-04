from domadapter.models.gradient_reversal import GradientReversal

import torch
import pytorch_lightning as pl
import numpy as np
from typing import Any, Optional, Dict
from transformers import AutoModelWithHeads
from transformers import AutoConfig
from domadapter.console import console
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchmetrics


class DANNAdapter(pl.LightningModule):
    def __init__(self, hparams: Optional[Dict[str, Any]] = None):
        """DANNAdapter LightningModule to train adapter like DANN.
        Args:
            hparams (Optional[Dict[str, Any]], optional): [description]. Defaults to None.
        """
        super(DANNAdapter, self).__init__()

        self.save_hyperparameters(hparams)
        # config
        self.config = AutoConfig.from_pretrained(self.hparams["pretrained_model_name"])
        # to get the layer wise pre-trained model outputs
        self.config.output_hidden_states = True

        # load the model weights
        with console.status(
            f"Loading {self.hparams['pretrained_model_name']} Model", spinner="monkey"
        ):
            self.model = AutoModelWithHeads.from_pretrained(
                self.hparams["pretrained_model_name"], config=self.config
            )
        console.print(f"[green] Loaded {self.hparams['pretrained_model_name']} model")
        with console.status(
                f"Adding {self.hparams['source_target']} task adapter", spinner="monkey"
            ):
                # add task adapter to PLM
                self.model.add_adapter(f"DANN_adapter_{self.hparams['source_target']}")
                # add classification head to task adapter
                self.model.add_classification_head(
                    f"DANN_adapter_{self.hparams['source_target']}",
                    num_labels=self.hparams["num_classes"],
                )
                # Freeze all parameters and train only task adapter
                self.model.train_adapter(f"DANN_adapter_{self.hparams['source_target']}")
                console.print(f"[green] Added {self.hparams['source_target']} DANN adapter")

        self.domain_classifier = nn.Sequential(
            nn.Linear(
                in_features=self.config.hidden_size,
                out_features=self.hparams["hidden_size"],
            ),
            nn.ReLU(),
            nn.Linear(in_features=self.hparams["hidden_size"], out_features=2),
        )

        self.criterion_domain = CrossEntropyLoss()
        self.criterion_task = CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()  # accuracy
        self.f1 = torchmetrics.F1(
            num_classes=hparams["num_classes"], average="macro"
        )  # F1

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

    def forward(self, input_ids, attention_mask=None):
        """Forward pass of the model"""
        # get the model output
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output

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
                    "monitor": "val/divergence",
                    "interval": "epoch",
                }
            ],
        )

    def training_step(self, batch, batch_idx):
        """training step of DANNAdapter"""
        # Classification loss
        input_ids, attention_mask = batch["source_input_ids"], batch["source_attention_mask"]
        # get the logits
        logits = self(input_ids=input_ids, attention_mask=attention_mask).logits
        # get the labels
        labels = batch["label_source"]
        # get the loss
        class_loss = self.criterion(logits, labels)
        accuracy = self.accuracy(labels, torch.argmax(self.softmax(logits), dim=1))
        f1 = self.f1(labels, torch.argmax(self.softmax(logits), dim=1))

        # Domain loss
        start_steps = self.current_epoch * batch["source_input_ids"].shape[0]
        total_steps = self.hparams["epochs"] * batch["source_input_ids"].shape[0]

        p = float(batch_idx + start_steps) / total_steps
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        # concat the source and target data and pass it to the model
        input_ids = torch.cat(
            [batch["source_input_ids"], batch["target_input_ids"]], dim=0
        )
        attention_mask = torch.cat(
            [batch["source_attention_mask"], batch["target_attention_mask"]], dim=0
        )

        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[1 : len(outputs.hidden_states)]

        domain_source_labels = torch.zeros(batch["source_input_ids"].shape[0]).type(
            torch.LongTensor
        )
        domain_target_labels = torch.ones(batch["source_input_ids"].shape[0]).type(
            torch.LongTensor
        )
        domain_combined_label = torch.cat(
            [domain_source_labels, domain_target_labels], dim=0
        )

        domain_loss = 0
        for num in range(len(hidden_states)):
            src_feature, trg_feature = torch.split(
                tensor=hidden_states[num],
                split_size_or_sections=input_ids.shape[0] // 2,
                dim=0,
            )
            # hidden_states[num] shape: [batch_size, seq_length, hidden_dim]
            # change their shape to [batch_size, hidden_dim]
            combined_feature = torch.mean(src_feature, dim=1)
            trg_feature = torch.mean(trg_feature, dim=1)
            combined_feature = GradientReversal.apply(combined_feature.pooler_output, alpha)  # Use GradientReversal
            domain_pred = self.domain_classifier(combined_feature)
            divergence += self.criterion.calculate(
                source_sample=src_feature, target_sample=trg_feature
            )

        self.log(name="train/loss", value=divergence)
        return divergence

    def validation_step(self, batch, batch_idx):
        # concat the source and target data and pass it to the model
        input_ids = torch.cat(
            (batch["source_input_ids"], batch["target_input_ids"]), dim=0
        )
        attention_mask = torch.cat(
            (batch["source_attention_mask"], batch["target_attention_mask"]), dim=0
        )

        outputs = self(input_ids=input_ids, attention_mask=attention_mask)

        divergence = 0
        for num in range(len(outputs)):
            src_feature, trg_feature = torch.split(
                tensor=outputs[num],
                split_size_or_sections=input_ids.shape[0] // 2,
                dim=0,
            )
            # src_feature shape: [batch_size, seq_length, hidden_dim]
            # trg_feature shape: [batch_size, seq_length, hidden_dim]
            # change their shape to [batch_size, hidden_dim]
            src_feature = torch.mean(src_feature, dim=1)
            trg_feature = torch.mean(trg_feature, dim=1)
            divergence += self.criterion.calculate(
                source_sample=src_feature, target_sample=trg_feature
            )

        self.log(name="val/divergence", value=divergence)
        return {"loss": divergence}

    def validation_epoch_end(self, outputs):
        mean_divergenence = torch.stack([x["loss"] for x in outputs]).mean()
        # this will show the mean div value across epoch
        self.log(name="val/divergence", value=mean_divergenence)
