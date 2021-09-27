from utils.tensor_utils import GradientReversal

import torch
import pytorch_lightning as pl
from typing import Any, Optional, Dict, List, Union
from transformers import AutoModel
from transformers import AutoConfig
from domadapter.console import console
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

import torchmetrics


class DANN(pl.LightningModule):
    def __init__(self, hparams):
        """DANN LightningModule to task adapter after domain adapter training.

        Args:
            hparams (Optional[Dict[str, Any]], optional): [description]. Defaults to None.
        """
        super(DANN, self).__init__()

        self.save_hyperparameters(hparams)

        # config
        self.config = AutoConfig.from_pretrained(self.hparams["pretrained_model_name"])
        # to get the layer wise pre-trained model outputs
        self.config.output_hidden_states = True

        # load the model weights
        with console.status(
            f"Loading {self.hparams['pretrained_model_name']} Model", spinner="monkey"
        ):
            self.model = AutoModel.from_pretrained(
                self.hparams["pretrained_model_name"], config=self.config
            )
        console.print(f"[green] Loaded {self.hparams['pretrained_model_name']} base model")

        self.task_classifier = nn.Sequential(
            nn.Linear(in_features=self.config.hidden_size, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=hparams["num_classes"])
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features=self.config.hidden_size, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2)
        )

        self.criterion_domain = CrossEntropyLoss()
        self.criterion_task = CrossEntropyLoss()
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
        """forward function of DANN

        Args:
            input_ids (Tensor): input ids tensor
            attention_mask (Tensor): attention mask tensor
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    def configure_optimizers(self):
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
        """training step of DANN"""
        # Classification loss
        source_logits = self(input_ids=batch["source_input_ids"], attention_mask=batch["source_attention_mask"])
        # get output from task classifier
        task_output = self.task_classifier(source_logits.pooler_output)
        # get the labels
        labels = batch["label_source"]
        # get the loss
        class_loss = self.criterion_task(task_output, labels)
        accuracy = self.accuracy(labels, torch.argmax(task_output, dim=1))
        f1 = self.f1(labels, torch.argmax(task_output, dim=1))

        # Domain loss
        start_steps = self.current_epoch * batch["source_input_ids"].shape[0]
        total_steps = self.hparams["epochs"] * batch["source_input_ids"].shape[0]

        p = float(batch_idx + start_steps) / total_steps
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        input_ids = torch.cat([batch["source_input_ids"], batch["target_input_ids"]], dim=0)
        attention_mask = torch.cat(
            [batch["source_attention_mask"], batch["target_attention_mask"]], dim=0
        )
        combined_feature = self(input_ids=input_ids, attention_mask=attention_mask)
        domain_pred = GradientReversal(combined_feature, alpha)  # TODO: hidden states of 12 layers vs last layer

        domain_source_labels = torch.zeros(batch["source_input_ids"].shape[0]).type(torch.LongTensor)
        domain_target_labels = torch.ones(batch["source_input_ids"].shape[0]).type(torch.LongTensor)
        domain_combined_label = torch.cat([domain_source_labels, domain_target_labels], dim=0)
        domain_loss = self.criterion_domain(domain_pred, domain_combined_label)
        accuracy = self.accuracy(labels, torch.argmax(task_output, dim=1))
        f1 = self.f1(labels, torch.argmax(task_output, dim=1))

        final_loss = class_loss + domain_loss

        self.log(name="train/accuracy", value=accuracy)
        self.log(name="train/f1", value=f1)
        self.log(name="train/class_loss", value=class_loss)
        self.log(name="train/domain_loss", value=domain_loss)
        self.log(name="train/final_loss", value=final_loss)

        return final_loss

    def validation_step(self, batch, batch_idx):
        """validation step of DANN"""
        # Source classification loss
        source_logits = self(input_ids=batch["source_input_ids"], attention_mask=batch["source_attention_mask"])
        # get output from task classifier
        task_output = self.task_classifier(source_logits.pooler_output)
        labels = batch["label_source"]
        source_class_loss = self.criterion_task(task_output, labels)
        source_accuracy = self.accuracy(labels, torch.argmax(task_output, dim=1))
        source_f1 = self.f1(labels, torch.argmax(task_output, dim=1))

        # Target classification loss
        target_logits = self(input_ids=batch["source_input_ids"], attention_mask=batch["source_attention_mask"])
        # get output from task classifier
        task_output = self.task_classifier(target_logits.pooler_output)
        labels = batch["label_target"]
        target_class_loss = self.criterion_task(task_output, labels)
        target_accuracy = self.accuracy(labels, torch.argmax(task_output, dim=1))
        target_f1 = self.f1(labels, torch.argmax(task_output, dim=1))

        # Domain loss
        start_steps = self.current_epoch * batch["source_input_ids"].shape[0]
        total_steps = self.hparams["epochs"] * batch["source_input_ids"].shape[0]

        p = float(batch_idx + start_steps) / total_steps
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        input_ids = torch.cat([batch["source_input_ids"], batch["target_input_ids"]], dim=0)
        attention_mask = torch.cat(
            [batch["source_attention_mask"], batch["target_attention_mask"]], dim=0
        )
        combined_feature = self(input_ids=input_ids, attention_mask=attention_mask)
        domain_pred = GradientReversal(combined_feature, alpha)  # TODO: hidden states of 12 layers vs last layer

        domain_source_labels = torch.zeros(batch["source_input_ids"].shape[0]).type(torch.LongTensor)
        domain_target_labels = torch.ones(batch["source_input_ids"].shape[0]).type(torch.LongTensor)
        domain_combined_label = torch.cat([domain_source_labels, domain_target_labels], dim=0)
        domain_loss = self.criterion_domain(domain_pred, domain_combined_label)
        domain_accuracy = self.accuracy(labels, torch.argmax(task_output, dim=1))
        domain_f1 = self.f1(labels, torch.argmax(task_output, dim=1))

        self.log(name="source_val/class_loss", value=source_class_loss)
        self.log(name="source_val/loss", value=source_class_loss+domain_loss)
        self.log(name="source_val/accuracy", value=source_accuracy)
        self.log(name="source_val/f1", value=source_f1)
        self.log(name="target_val/class_loss", value=target_class_loss)
        self.log(name="target_val/loss", value=target_class_loss+domain_loss)
        self.log(name="target_val/accuracy", value=target_accuracy)
        self.log(name="target_val/f1", value=target_f1)
        self.log(name="val/domain_loss", value=domain_loss)
        self.log(name="val/domain_accuracy", value=domain_accuracy)
        self.log(name="val/domain_f1", value=domain_f1)


        return {
            "source_val/class_loss": source_class_loss,
            "source_val/loss": source_class_loss+domain_loss,
            "source_val/accuracy": source_accuracy,
            "source_val/f1": source_f1,
            "target_val/class_loss": target_class_loss,
            "target_val/loss": target_class_loss+domain_loss,
            "target_val/accuracy": target_accuracy,
            "target_val/f1": target_f1,
            "val/domain_loss": domain_loss,
            "val/domain_accuracy": domain_accuracy,
            "val/domain_f1": domain_f1,
        }

    def validation_epoch_end(self, outputs):
        mean_source_loss = torch.stack([x["source_val/loss"] for x in outputs]).mean()
        mean_source_class_loss = torch.stack([x["source_val/class_loss"] for x in outputs]).mean()
        mean_source_accuracy = torch.stack([x["source_val/accuracy"] for x in outputs]).mean()
        mean_source_f1 = torch.stack([x["source_val/f1"] for x in outputs]).mean()

        mean_target_loss = torch.stack([x["target_val/loss"] for x in outputs]).mean()
        mean_target_class_loss = torch.stack([x["target_val/class_loss"] for x in outputs]).mean()
        mean_target_accuracy = torch.stack([x["target_val/accuracy"] for x in outputs]).mean()
        mean_target_f1 = torch.stack([x["target_val/f1"] for x in outputs]).mean()

        mean_domain_loss = torch.stack([x["val/domain_loss"] for x in outputs]).mean()
        mean_domain_accuracy = torch.stack([x["val/domain_accuracy"] for x in outputs]).mean()
        mean_domain_f1 = torch.stack([x["val/domain_f1"] for x in outputs]).mean()

        # this will log the mean div value across epoch
        self.log(name="source_val/loss", value=mean_source_loss)
        self.log(name="source_val/class_loss", value=mean_source_class_loss)
        self.log(name="source_val/accuracy", value=mean_source_accuracy)
        self.log(name="source_val/f1", value=mean_source_f1)
        self.log(name="target_val/class_loss", value=mean_target_class_loss)
        self.log(name="target_val/loss", value=mean_target_loss)
        self.log(name="target_val/accuracy", value=mean_target_accuracy)
        self.log(name="target_val/f1", value=mean_target_f1)
        self.log(name="val/domain_loss", value=mean_domain_loss)
        self.log(name="val/domain_accuracy", value=mean_domain_accuracy)
        self.log(name="val/domain_f1", value=mean_domain_f1)

    def test_step(self, batch, batch_idx):
        """Test step of DANN"""
        # Source classification loss
        source_logits = self(input_ids=batch["source_input_ids"], attention_mask=batch["source_attention_mask"])
        # get output from task classifier
        task_output = self.task_classifier(source_logits.pooler_output)
        labels = batch["label_source"]
        source_class_loss = self.criterion_task(task_output, labels)
        source_accuracy = self.accuracy(labels, torch.argmax(task_output, dim=1))
        source_f1 = self.f1(labels, torch.argmax(task_output, dim=1))

        # Target classification loss
        target_logits = self(input_ids=batch["source_input_ids"], attention_mask=batch["source_attention_mask"])
        # get output from task classifier
        task_output = self.task_classifier(target_logits.pooler_output)
        labels = batch["label_target"]
        target_class_loss = self.criterion_task(task_output, labels)
        target_accuracy = self.accuracy(labels, torch.argmax(task_output, dim=1))
        target_f1 = self.f1(labels, torch.argmax(task_output, dim=1))

        # Domain loss
        start_steps = self.current_epoch * batch["source_input_ids"].shape[0]
        total_steps = self.hparams["epochs"] * batch["source_input_ids"].shape[0]

        p = float(batch_idx + start_steps) / total_steps
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        input_ids = torch.cat([batch["source_input_ids"], batch["target_input_ids"]], dim=0)
        attention_mask = torch.cat(
            [batch["source_attention_mask"], batch["target_attention_mask"]], dim=0
        )
        combined_feature = self(input_ids=input_ids, attention_mask=attention_mask)
        domain_pred = GradientReversal(combined_feature, alpha)  # TODO: hidden states of 12 layers vs last layer

        domain_source_labels = torch.zeros(batch["source_input_ids"].shape[0]).type(torch.LongTensor)
        domain_target_labels = torch.ones(batch["source_input_ids"].shape[0]).type(torch.LongTensor)
        domain_combined_label = torch.cat([domain_source_labels, domain_target_labels], dim=0)
        domain_loss = self.criterion_domain(domain_pred, domain_combined_label)
        domain_accuracy = self.accuracy(labels, torch.argmax(task_output, dim=1))
        domain_f1 = self.f1(labels, torch.argmax(task_output, dim=1))

        # this will log the mean div value across epoch
        self.log(name="source_test/class_loss", value=source_class_loss)
        self.log(name="source_test/loss", value=source_class_loss+domain_loss)
        self.log(name="source_test/accuracy", value=source_accuracy)
        self.log(name="source_test/f1", value=source_f1)
        self.log(name="target_test/class_loss", value=target_class_loss)
        self.log(name="target_test/loss", value=target_class_loss+domain_loss)
        self.log(name="target_test/accuracy", value=target_accuracy)
        self.log(name="target_test/f1", value=target_f1)
        self.log(name="test/domain_loss", value=domain_loss)
        self.log(name="test/domain_accuracy", value=domain_accuracy)
        self.log(name="test/domain_f1", value=domain_f1)


        return {
            "source_test/class_loss": source_class_loss,
            "source_test/loss": source_class_loss+domain_loss,
            "source_test/accuracy": source_accuracy,
            "source_test/f1": source_f1,
            "target_test/class_loss": target_class_loss,
            "target_test/loss": target_class_loss+domain_loss,
            "target_test/accuracy": target_accuracy,
            "target_test/f1": target_f1,
            "test/domain_loss": domain_loss,
            "test/domain_accuracy": domain_accuracy,
            "test/domain_f1": domain_f1,
        }

    def test_epoch_end(self, outputs):
        mean_source_loss = torch.stack([x["source_test/loss"] for x in outputs]).mean()
        mean_source_class_loss = torch.stack([x["source_test/class_loss"] for x in outputs]).mean()
        mean_source_accuracy = torch.stack([x["source_test/accuracy"] for x in outputs]).mean()
        mean_source_f1 = torch.stack([x["source_test/f1"] for x in outputs]).mean()

        mean_target_loss = torch.stack([x["target_test/loss"] for x in outputs]).mean()
        mean_target_class_loss = torch.stack([x["target_test/class_loss"] for x in outputs]).mean()
        mean_target_accuracy = torch.stack([x["target_test/accuracy"] for x in outputs]).mean()
        mean_target_f1 = torch.stack([x["target_test/f1"] for x in outputs]).mean()

        mean_domain_loss = torch.stack([x["test/domain_loss"] for x in outputs]).mean()
        mean_domain_accuracy = torch.stack([x["test/domain_accuracy"] for x in outputs]).mean()
        mean_domain_f1 = torch.stack([x["test/domain_f1"] for x in outputs]).mean()

        # this will log the mean div value across epoch
        self.log(name="source_test/loss", value=mean_source_loss)
        self.log(name="source_test/class_loss", value=mean_source_class_loss)
        self.log(name="source_test/accuracy", value=mean_source_accuracy)
        self.log(name="source_test/f1", value=mean_source_f1)
        self.log(name="target_test/class_loss", value=mean_target_class_loss)
        self.log(name="target_test/loss", value=mean_target_loss)
        self.log(name="target_test/accuracy", value=mean_target_accuracy)
        self.log(name="target_test/f1", value=mean_target_f1)
        self.log(name="test/domain_loss", value=mean_domain_loss)
        self.log(name="test/domain_accuracy", value=mean_domain_accuracy)
        self.log(name="test/domain_f1", value=mean_domain_f1)
