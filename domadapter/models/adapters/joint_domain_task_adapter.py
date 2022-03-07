import torch
import os
import pytorch_lightning as pl
from transformers import AutoModelWithHeads
from transformers import AutoConfig
from domadapter.console import console
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers.adapters.composition import Stack
import numpy as np
from collections import defaultdict

from domadapter.divergences.cmd_divergence import CMD
from domadapter.divergences.coral_divergence import Coral
from domadapter.divergences.mkmmd_divergence import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel


import torchmetrics


class JointDomainTaskAdapter(pl.LightningModule):
    def __init__(self, hparams):
        """DomainTaskAdapter LightningModule to task adapter after domain adapter training.

        Args:
            hparams (Optional[Dict[str, Any]], optional): [description]. Defaults to None.
        """
        super(JointDomainTaskAdapter, self).__init__()

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
        console.print(f"[green] Loaded {self.hparams['pretrained_model_name']} base model")

        with console.status(
            f"Adding {self.hparams['source_target']} adapter", spinner="monkey"
        ):
            # add task adapter to PLM
            self.model.add_adapter(f"adapter_{self.hparams['source_target']}")
            # add classification head to task adapter
            self.model.add_classification_head(
                f"adapter_{self.hparams['source_target']}",
                num_labels=self.hparams["num_classes"],
            )
            # Freeze all parameters and train only task adapter
            self.model.train_adapter(f"adapter_{self.hparams['source_target']}")
            console.print(f"[green] Added {self.hparams['source_target']} adapter")

        self.criterion = CrossEntropyLoss()
        if self.hparams["divergence"] == 'cmd':
            self.divergence = CMD()
        elif self.hparams["divergence"] == 'coral':
            self.divergence = Coral()
        elif self.hparams["divergence"] == 'mkmmd':
            kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
            self.divergence = MultipleKernelMaximumMeanDiscrepancy(kernels)

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
        """forward function of DomainTaskAdapter

        Args:
            input_ids (Tensor): input ids tensor
            attention_mask (Tensor): attention mask tensor
        """
        # get the model output
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = output.hidden_states[9 : len(output.hidden_states)]
        return hidden_states, output.logits

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
        # concat the source and target data and pass it to the model
        input_ids = torch.cat(
            [batch["source_input_ids"], batch["target_input_ids"]], dim=0
        )
        attention_mask = torch.cat(
            [batch["source_attention_mask"], batch["target_attention_mask"]], dim=0
        )
        # get the labels
        labels = batch["label_source"]

        hidden_states, logits = self(input_ids=input_ids, attention_mask=attention_mask)

        divergence = 0
        for num in range(len(hidden_states)):
            src_feature, trg_feature = torch.split(
                tensor=hidden_states[num],
                split_size_or_sections=input_ids.shape[0] // 2,
                dim=0,
            )
            # src_feature shape: [batch_size, seq_length, hidden_dim]
            # trg_feature shape: [batch_size, seq_length, hidden_dim]
            # change their shape to [batch_size, hidden_dim]
            src_feature = torch.mean(src_feature, dim=1)
            trg_feature = torch.mean(trg_feature, dim=1)
            divergence += self.divergence.calculate(
                source_sample=src_feature, target_sample=trg_feature
            )

        # get the loss
        logits, _ = torch.split(
            tensor=logits,
            split_size_or_sections=input_ids.shape[0] // 2,
            dim=0,
        )
        task_loss = self.criterion(logits, labels)
        loss = task_loss + divergence
        accuracy = self.accuracy(labels, torch.argmax(self.softmax(logits), dim=1))
        f1 = self.f1(labels, torch.argmax(self.softmax(logits), dim=1))

        metrics = {
            "train/accuracy": accuracy,
            "train/f1": f1,
            "train/taskclf_loss": task_loss,
            "train/loss": loss,
            "train/domain_loss": divergence
        }

        for key, val in metrics.items():
            self.log(name=key, value=val)

        return loss


    def training_epoch_end(self, outputs):
        self._log_metrics(outputs)

    def validation_step(self, batch, batch_idx):
        # concat the source and target data and pass it to the model
        input_ids = torch.cat(
            [batch["source_input_ids"], batch["target_input_ids"]], dim=0
        )
        attention_mask = torch.cat(
            [batch["source_attention_mask"], batch["target_attention_mask"]], dim=0
        )

        hidden_states, logits = self(input_ids=input_ids, attention_mask=attention_mask)

        divergence = 0
        for num in range(len(hidden_states)):
            src_feature, trg_feature = torch.split(
                tensor=hidden_states[num],
                split_size_or_sections=input_ids.shape[0] // 2,
                dim=0,
            )
            # src_feature shape: [batch_size, seq_length, hidden_dim]
            # trg_feature shape: [batch_size, seq_length, hidden_dim]
            # change their shape to [batch_size, hidden_dim]
            src_feature = torch.mean(src_feature, dim=1)
            trg_feature = torch.mean(trg_feature, dim=1)
            divergence += self.divergence.calculate(
                source_sample=src_feature, target_sample=trg_feature
            )

        # get the loss for source
        logits_source, logits_target = torch.split(
            tensor=logits,
            split_size_or_sections=input_ids.shape[0] // 2,
            dim=0,
        )
        source_taskclf_loss = self.criterion(logits_source, batch["label_source"])
        source_loss = source_taskclf_loss + divergence

        target_taskclf_loss = self.criterion(logits_target, batch["label_target"])
        target_loss = target_taskclf_loss + divergence

        loss = source_taskclf_loss + target_taskclf_loss + divergence

        source_accuracy = self.accuracy(batch["label_source"], torch.argmax(self.softmax(logits_source), dim=1))
        source_f1 = self.f1(batch["label_source"], torch.argmax(self.softmax(logits_source), dim=1))

        target_accuracy = self.accuracy(batch["label_target"], torch.argmax(self.softmax(logits_target), dim=1))
        target_f1 = self.f1(batch["label_target"], torch.argmax(self.softmax(logits_target), dim=1))

        metrics = {
            "source_val/loss": source_loss,
            "source_val/taskclf_loss": source_taskclf_loss,
            "val/domain_loss": divergence,
            "val/loss": loss,
            "source_val/accuracy": source_accuracy,
            "source_val/f1": source_f1,
            "target_val/loss": target_loss,
            "target_val/taskclf_loss": target_taskclf_loss,
            "target_val/accuracy": target_accuracy,
            "target_val/f1": target_f1,
        }

        for key, val in metrics.items():
            self.log(name=key, value=val)

        return metrics

    def validation_epoch_end(self, outputs):
        self._log_metrics(outputs)

    def test_step(self, batch, batch_idx):
                # concat the source and target data and pass it to the model
        input_ids = torch.cat(
            [batch["source_input_ids"], batch["target_input_ids"]], dim=0
        )
        attention_mask = torch.cat(
            [batch["source_attention_mask"], batch["target_attention_mask"]], dim=0
        )

        hidden_states, logits = self(input_ids=input_ids, attention_mask=attention_mask)

        divergence = 0
        for num in range(len(hidden_states)):
            src_feature, trg_feature = torch.split(
                tensor=hidden_states[num],
                split_size_or_sections=input_ids.shape[0] // 2,
                dim=0,
            )
            # src_feature shape: [batch_size, seq_length, hidden_dim]
            # trg_feature shape: [batch_size, seq_length, hidden_dim]
            # change their shape to [batch_size, hidden_dim]
            src_feature = torch.mean(src_feature, dim=1)
            trg_feature = torch.mean(trg_feature, dim=1)
            divergence += self.divergence.calculate(
                source_sample=src_feature, target_sample=trg_feature
            )

        # get the loss for source
        logits_source, logits_target = torch.split(
            tensor=logits,
            split_size_or_sections=input_ids.shape[0] // 2,
            dim=0,
        )
        source_taskclf_loss = self.criterion(logits_source, batch["label_source"])
        source_loss = source_taskclf_loss + divergence

        target_taskclf_loss = self.criterion(logits_target, batch["label_target"])
        target_loss = target_taskclf_loss + divergence

        loss = source_taskclf_loss + target_taskclf_loss + divergence

        source_accuracy = self.accuracy(batch["label_source"], torch.argmax(self.softmax(logits_source), dim=1))
        source_f1 = self.f1(batch["label_source"], torch.argmax(self.softmax(logits_source), dim=1))

        target_accuracy = self.accuracy(batch["label_target"], torch.argmax(self.softmax(logits_target), dim=1))
        target_f1 = self.f1(batch["label_target"], torch.argmax(self.softmax(logits_target), dim=1))

        metrics = {
            "source_test/loss": source_loss,
            "source_test/taskclf_loss": source_taskclf_loss,
            "test/domain_loss": divergence,
            "test/loss": loss,
            "source_test/accuracy": source_accuracy,
            "source_test/f1": source_f1,
            "target_test/loss": target_loss,
            "target_test/taskclf_loss": target_taskclf_loss,
            "target_test/accuracy": target_accuracy,
            "target_test/f1": target_f1,
        }

        for key, val in metrics.items():
            self.log(name=key, value=val)

        return metrics

    def test_epoch_end(self, outputs):
        self._log_metrics(outputs)


    def _log_metrics(self, outputs):
        metrics = list(outputs[0].keys())
        metrics_dict = defaultdict(list)

        for output in outputs:
            for metric in metrics:
                metrics_dict[metric].append(output[metric].cpu().item())

        for metric in metrics:
            self.log(f"{metric}", np.mean(metrics_dict[metric]))
