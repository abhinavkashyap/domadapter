from domadapter.models.gradient_reversal import GradientReversal

from collections import defaultdict
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
            self.feature_extractor = AutoModelWithHeads.from_pretrained(
                self.hparams["pretrained_model_name"], config=self.config
            )
        console.print(f"[green] Loaded {self.hparams['pretrained_model_name']} model")
        with console.status(
                f"Adding {self.hparams['source_target']} task adapter", spinner="monkey"
            ):
                # add task adapter to PLM
                self.feature_extractor.add_adapter(f"DANN_adapter_{self.hparams['source_target']}")
                # Freeze all parameters and train only task adapter
                self.feature_extractor.train_adapter(f"DANN_adapter_{self.hparams['source_target']}")
                console.print(f"[green] Added {self.hparams['source_target']} DANN adapter")

        self.task_classifier = nn.Sequential(
            nn.Linear(
                in_features=self.config.hidden_size,
                out_features=self.hparams["hidden_size"],
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.hparams["hidden_size"],
                out_features=self.hparams["hidden_size"],
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.hparams["hidden_size"],
                out_features=hparams["num_classes"],
            ),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(
                in_features=self.config.hidden_size,
                out_features=self.hparams["hidden_size"],
            ),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.hparams["hidden_size"], out_features=2),
        )

        self.domain_clf_loss = CrossEntropyLoss()
        self.task_clf_loss = CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()  # accuracy
        self.src_dev_acc = torchmetrics.Accuracy()
        self.src_test_acc = torchmetrics.Accuracy()
        self.trg_dev_acc = torchmetrics.Accuracy()
        self.trg_test_acc = torchmetrics.Accuracy()
        self.val_dom_clf_acc = torchmetrics.Accuracy()
        self.test_dom_clf_acc = torchmetrics.Accuracy()

        self.train_f1 = torchmetrics.F1(num_classes=hparams["num_classes"], average="macro")  # F1
        self.src_dev_f1 = torchmetrics.F1(num_classes=hparams["num_classes"], average="macro")
        self.src_test_f1 = torchmetrics.F1(num_classes=hparams["num_classes"], average="macro")
        self.trg_dev_f1 = torchmetrics.F1(num_classes=hparams["num_classes"], average="macro")
        self.trg_test_f1 = torchmetrics.F1(num_classes=hparams["num_classes"], average="macro")
        self.val_dom_clf_f1 = torchmetrics.F1(num_classes=2, average="macro")
        self.test_dom_clf_f1 = torchmetrics.F1(num_classes=2, average="macro")

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
        self.switch_off_adv_train = self.hparams.get("switch_off_adv_train", False)
        self.is_dynamic_dann_alpha = self.hparams.get("is_dynamic_dann_alpha", False)
        self.dann_alpha = self.hparams.get("dann_alpha")

    def forward(self, src_inp_ids, src_attn_mask, trg_inp_ids, trg_attn_mask, alpha):
        """

        Parameters
        ----------
        src_inp_ids: torch.Tensor
            Size: B * L
            B = Batch Size
            L - Sequence Length
        src_attn_mask: torch.Tensor
            Size: B * L
            B = Batch Size
            L - Sequence Length
        trg_inp_ids: torch.Tensor
            Size: B * L
            B = Batch Size
            L - Sequence Length
        trg_attn_mask: torch.Tensor
            Size: B * L
            B = Batch Size
            L - Sequence Length
        alpha: float
        Returns
        -------
        src_taskclf_logits, trg_taskclf_logits, src_domclf_logits, trg_domclf_logits
        """

        bsz = src_inp_ids.size(0)
        inp_ids = torch.cat([src_inp_ids, trg_inp_ids], dim=0)
        attn_mask = torch.cat([src_attn_mask, trg_attn_mask], dim=0)

        outputs = self.feature_extractor(input_ids=inp_ids, attention_mask=attn_mask)
        pooler_output = outputs.pooler_output
        # shape of pooler_output = (B,768)
        src_features, trg_features = torch.split(pooler_output, [bsz, bsz], dim=0)

        hidden_states = outputs.hidden_states[1 : len(outputs.hidden_states)]
        hidden_states  = torch.stack(list(hidden_states), dim=0)  # shape of hidden_states = 12 (tuple)
        hidden_states = torch.mean(hidden_states, dim=2)  # hidden_states shape = [12, 2*B, L, 768]
        hidden_states = torch.mean(hidden_states, dim=2)  # hidden_states shape = [12, 2*B, 768]

        # B * number_classes
        src_taskclf_logits = self.task_classifier(src_features)
        trg_taskclf_logits = self.task_classifier(trg_features)

        src_features, trg_features = torch.split(hidden_states, [bsz, bsz], dim=1)  # src_features shape = [12, B, 768]
        src_features = src_features.reshape(src_features.size(0)*src_features.size(1), src_features.size(2))    # src_features shape = [12*B, 768]
        trg_features = trg_features.reshape(trg_features.size(0)*trg_features.size(1), trg_features.size(2))

        if self.switch_off_adv_train is False:
            src_grl_features = GradientReversal.apply(src_features, alpha)
            trg_grl_features = GradientReversal.apply(trg_features, alpha)

            src_domclf_logits = self.domain_classifier(src_grl_features)
            trg_domclf_logits = self.domain_classifier(trg_grl_features)
        else:
            src_domclf_logits = None
            trg_domclf_logits = None

        return src_taskclf_logits, trg_taskclf_logits, src_domclf_logits, trg_domclf_logits

    # def save_adapter(self, location, adapter_name):
    #     """Module to save adapter.
    #     Args:
    #         location str: Location where to save adapter.
    #         adapter_name: Name of adapter to be saved.
    #     """
    #     self.model.save_adapter(location, adapter_name)

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
        src_inp_ids = batch["source_input_ids"]
        src_attn_mask = batch["source_attention_mask"]
        trg_inp_ids = batch["target_input_ids"]
        trg_attn_mask = batch["target_attention_mask"]
        bsz = src_inp_ids.size(0)

        # get the labels
        labels = batch["label_source"]

        start_steps = self.current_epoch * src_inp_ids.shape[0]
        total_steps = self.hparams["epochs"] * src_inp_ids.shape[0]

        if self.is_dynamic_dann_alpha:
            p = float(batch_idx + start_steps) / total_steps
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        else:
            alpha = self.dann_alpha
            assert alpha is not None, f"Set dynamic_dann_alpha to True or pass dann_alpha"

        src_taskclf_logits, trg_taskclf_logits, src_domclf_logits, trg_domclf_logits = self(
            src_inp_ids=src_inp_ids,
            src_attn_mask=src_attn_mask,
            trg_inp_ids=trg_inp_ids,
            trg_attn_mask=trg_attn_mask,
            alpha=alpha,
        )
        # get the loss
        class_loss = self.task_clf_loss(src_taskclf_logits, labels)

        # Domain loss
        if self.switch_off_adv_train is False:
            domain_src_labels = torch.zeros(src_domclf_logits.size(0)).type(torch.LongTensor).to(self.device)  # multiply by 12 for all layers
            domain_trg_labels = torch.ones(trg_domclf_logits.size(0)).type(torch.LongTensor).to(self.device)  # multiply by 12 for all layers

            src_dom_loss = self.domain_clf_loss(src_domclf_logits, domain_src_labels)
            trg_dom_loss = self.domain_clf_loss(trg_domclf_logits, domain_trg_labels)
            dom_loss = src_dom_loss + trg_dom_loss
        else:
            dom_loss = torch.FloatTensor([0.0]).to(self.device)

        accuracy = self.train_acc(preds=src_taskclf_logits, target=labels)
        f1 = self.train_f1(preds=src_taskclf_logits, target=labels)

        loss = class_loss + dom_loss

        metrics = {
            "train/accuracy": accuracy,
            "train/f1": f1,
            "train/taskclf_loss": class_loss,
            "train/domain_loss": dom_loss,
            "train/loss": loss,
        }

        for key, val in metrics.items():
            self.log(name=key, value=val)

        return metrics

    def training_epoch_end(self, outputs):
        self._log_metrics(outputs)

    def validation_step(self, batch, batch_idx):
        """validation step of DANNAdapter"""
        # Source classification loss
        # Classification loss
        src_inp_ids = batch["source_input_ids"]
        src_attn_mask = batch["source_attention_mask"]
        trg_inp_ids = batch["target_input_ids"]
        trg_attn_mask = batch["target_attention_mask"]
        bsz = src_inp_ids.size(0)

        # get the labels
        src_labels = batch["label_source"]
        trg_labels = batch["label_target"]

        start_steps = self.current_epoch * src_inp_ids.shape[0]
        total_steps = self.hparams["epochs"] * src_inp_ids.shape[0]

        if self.is_dynamic_dann_alpha:
            p = float(batch_idx + start_steps) / total_steps
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        else:
            alpha = self.dann_alpha
            assert alpha is not None, f"Set dynamic_dann_alpha to True or pass dann_alpha"

        src_taskclf_logits, trg_taskclf_logits, src_domclf_logits, trg_domclf_logits = self(
            src_inp_ids=src_inp_ids,
            src_attn_mask=src_attn_mask,
            trg_inp_ids=trg_inp_ids,
            trg_attn_mask=trg_attn_mask,
            alpha=alpha,
        )
        # get the loss
        src_task_loss = self.task_clf_loss(src_taskclf_logits, src_labels)
        trg_task_loss = self.task_clf_loss(trg_taskclf_logits, trg_labels)

        if self.switch_off_adv_train is False:
            # Domain loss
            src_dom_labels = torch.zeros(src_domclf_logits.size(0)).type(torch.LongTensor).to(self.device)  # multiply by 12 for all layers
            trg_dom_labels = torch.ones(trg_domclf_logits.size(0)).type(torch.LongTensor).to(self.device)  # multiply by 12 for all layers

            src_dom_loss = self.domain_clf_loss(src_domclf_logits, src_dom_labels)
            trg_dom_loss = self.domain_clf_loss(trg_domclf_logits, trg_dom_labels)
        else:
            src_dom_loss = torch.FloatTensor([0.0]).to(self.device)
            trg_dom_loss = torch.FloatTensor([0.0]).to(self.device)

        dom_loss = src_dom_loss + trg_dom_loss

        src_loss = src_task_loss + src_dom_loss
        trg_loss = trg_task_loss + trg_dom_loss

        src_acc = self.src_dev_acc(preds=src_taskclf_logits, target=src_labels)
        src_f1 = self.src_dev_f1(preds=src_taskclf_logits, target=src_labels)

        trg_acc = self.trg_dev_acc(preds=trg_taskclf_logits, target=trg_labels)
        trg_f1 = self.trg_dev_f1(preds=trg_taskclf_logits, target=trg_labels)

        if self.switch_off_adv_train is False:
            src_dom_clf_acc = self.val_dom_clf_acc(
                preds=src_domclf_logits, target=src_dom_labels
            )
            trg_dom_clf_acc = self.val_dom_clf_acc(
                preds=trg_domclf_logits, target=trg_dom_labels
            )

            src_dom_clf_f1 = self.val_dom_clf_f1(preds=src_domclf_logits, target=src_dom_labels)
            trg_dom_clf_f1 = self.val_dom_clf_f1(preds=trg_domclf_logits, target=trg_dom_labels)

            dom_acc = src_dom_clf_acc + trg_dom_clf_acc
            dom_f1 = src_dom_clf_f1 + trg_dom_clf_f1
        else:
            dom_acc = torch.FloatTensor([0.0]).to(self.device)
            dom_f1 = torch.FloatTensor([0.0]).to(self.device)

        metrics = {
            "source_val/taskclf_loss": src_task_loss,
            "source_val/loss": src_loss,
            "source_val/accuracy": src_acc,
            "source_val/f1": src_f1,
            "target_val/taskclf_loss": trg_task_loss,
            "target_val/loss": trg_loss,
            "target_val/accuracy": trg_acc,
            "target_val/f1": trg_f1,
            "val/domain_loss": dom_loss,
            "val/domain_acc": dom_acc,
            "val/domain_f1": dom_f1,
        }

        for key, val in metrics.items():
            self.log(name=key, value=val)

        return metrics

    def validation_epoch_end(self, outputs):
        self._log_metrics(outputs)

    def test_step(self, batch, batch_idx):
        """validation step of DANNAdapter"""
        # Source classification loss
        # Classification loss
        src_inp_ids = batch["source_input_ids"]
        src_attn_mask = batch["source_attention_mask"]
        trg_inp_ids = batch["target_input_ids"]
        trg_attn_mask = batch["target_attention_mask"]
        bsz = src_inp_ids.size(0)

        # get the labels
        src_labels = batch["label_source"]
        trg_labels = batch["label_target"]

        start_steps = self.current_epoch * src_inp_ids.shape[0]
        total_steps = self.hparams["epochs"] * src_inp_ids.shape[0]

        if self.is_dynamic_dann_alpha:
            p = float(batch_idx + start_steps) / total_steps
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        else:
            alpha = self.dann_alpha
            assert alpha is not None, f"Set dynamic_dann_alpha to True or pass dann_alpha"

        src_taskclf_logits, trg_taskclf_logits, src_domclf_logits, trg_domclf_logits = self(
            src_inp_ids=src_inp_ids,
            src_attn_mask=src_attn_mask,
            trg_inp_ids=trg_inp_ids,
            trg_attn_mask=trg_attn_mask,
            alpha=alpha,
        )
        # get the loss
        src_task_loss = self.task_clf_loss(src_taskclf_logits, src_labels)
        trg_task_loss = self.task_clf_loss(trg_taskclf_logits, trg_labels)

        if self.switch_off_adv_train is False:
            # Domain loss
            src_dom_labels = torch.zeros(src_domclf_logits.size(0)).type(torch.LongTensor).to(self.device)  # multiply by 12 for all layers
            trg_dom_labels = torch.ones(trg_domclf_logits.size(0)).type(torch.LongTensor).to(self.device)  # multiply by 12 for all layers

            src_dom_loss = self.domain_clf_loss(src_domclf_logits, src_dom_labels)
            trg_dom_loss = self.domain_clf_loss(trg_domclf_logits, trg_dom_labels)
        else:
            src_dom_loss = torch.FloatTensor([0.0]).to(self.device)
            trg_dom_loss = torch.FloatTensor([0.0]).to(self.device)

        dom_loss = src_dom_loss + trg_dom_loss

        src_loss = src_task_loss + src_dom_loss
        trg_loss = trg_task_loss + trg_dom_loss

        src_acc = self.src_test_acc(preds=src_taskclf_logits, target=src_labels)
        src_f1 = self.src_test_f1(preds=src_taskclf_logits, target=src_labels)

        trg_acc = self.trg_test_acc(preds=trg_taskclf_logits, target=trg_labels)
        trg_f1 = self.trg_test_f1(preds=trg_taskclf_logits, target=trg_labels)

        if self.switch_off_adv_train is False:
            src_dom_clf_acc = self.test_dom_clf_acc(
                preds=src_domclf_logits, target=src_dom_labels
            )
            trg_dom_clf_acc = self.test_dom_clf_acc(
                preds=trg_domclf_logits, target=trg_dom_labels
            )

            src_dom_clf_f1 = self.test_dom_clf_f1(preds=src_domclf_logits, target=src_dom_labels)
            trg_dom_clf_f1 = self.test_dom_clf_f1(preds=trg_domclf_logits, target=trg_dom_labels)
            dom_acc = src_dom_clf_acc + trg_dom_clf_acc
            dom_f1 = src_dom_clf_f1 + trg_dom_clf_f1
        else:
            dom_acc = torch.FloatTensor([0.0]).to(self.device)
            dom_f1 = torch.FloatTensor([0.0]).to(self.device)

        metrics = {
            "source_test/taskclf_loss": src_task_loss,
            "source_test/loss": src_loss,
            "source_test/accuracy": src_acc,
            "source_test/f1": src_f1,
            "target_test/taskclf_loss": trg_task_loss,
            "target_test/loss": trg_loss,
            "target_test/accuracy": trg_acc,
            "target_test/f1": trg_f1,
            "test/domain_loss": dom_loss,
            "test/domain_accuracy": dom_acc,
            "test/domain_f1": dom_f1,
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
