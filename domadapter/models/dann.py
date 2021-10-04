from domadapter.models.gradient_reversal import GradientReversal
import torch
import pytorch_lightning as pl
from transformers import AutoModel
from transformers import AutoConfig
from domadapter.console import console
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from collections import defaultdict

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
            self.feature_extractor = AutoModel.from_pretrained(
                self.hparams["pretrained_model_name"], config=self.config
            )
        console.print(f"[green] Loaded {self.hparams['pretrained_model_name']} base model")

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
        pooler_outputs = outputs.pooler_output
        src_features, trg_features = torch.split(pooler_outputs, [bsz, bsz], dim=0)

        # B * number_classes
        src_taskclf_logits = self.task_classifier(src_features)
        trg_taskclf_logits = self.task_classifier(trg_features)

        if self.switch_off_adv_train is False:
            src_grl_features = GradientReversal.apply(src_features, alpha)
            trg_grl_features = GradientReversal.apply(trg_features, alpha)

            src_domclf_logits = self.domain_classifier(src_grl_features)
            trg_domclf_logits = self.domain_classifier(trg_grl_features)
        else:
            src_domclf_logits = None
            trg_domclf_logits = None

        return src_taskclf_logits, trg_taskclf_logits, src_domclf_logits, trg_domclf_logits

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
                    "monitor": "val/src_f1",
                    "interval": "epoch",
                }
            ],
        )

    def training_step(self, batch, batch_idx):
        """training step of DANN"""
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
            domain_src_labels = torch.zeros(bsz).type(torch.LongTensor).to(self.device)
            domain_trg_labels = torch.ones(bsz).type(torch.LongTensor).to(self.device)

            src_dom_loss = self.domain_clf_loss(src_domclf_logits, domain_src_labels)
            trg_dom_loss = self.domain_clf_loss(trg_domclf_logits, domain_trg_labels)
            dom_loss = src_dom_loss + trg_dom_loss
        else:
            dom_loss = torch.FloatTensor([0.0]).to(self.device)

        accuracy = self.train_acc(preds=src_taskclf_logits, target=labels)
        f1 = self.train_f1(preds=src_taskclf_logits, target=labels)

        loss = class_loss + dom_loss

        metrics = {
            "train/src_accuracy": accuracy,
            "train/src_f1": f1,
            "train/src_taskclf_loss": class_loss,
            "train/domain_loss": dom_loss,
            "train/loss": loss,
            "loss": loss,
        }

        for key, val in metrics.items():
            self.log(name=key, value=val)

        return metrics

    def training_epoch_end(self, outputs):
        self._log_metrics(outputs)

    def validation_step(self, batch, batch_idx):
        """validation step of DANN"""
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
            src_dom_labels = torch.zeros(bsz).type(torch.LongTensor).to(self.device)
            trg_dom_labels = torch.ones(bsz).type(torch.LongTensor).to(self.device)

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
            "val/src_task_loss": src_task_loss,
            "val/src_loss": src_loss,
            "val/src_acc": src_acc,
            "val/src_f1": src_f1,
            "val/trg_task_loss": trg_task_loss,
            "val/trg_loss": trg_loss,
            "val/trg_acc": trg_acc,
            "val/trg_f1": trg_f1,
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
        """validation step of DANN"""
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
            src_dom_labels = torch.zeros(bsz).type(torch.LongTensor).to(self.device)
            trg_dom_labels = torch.ones(bsz).type(torch.LongTensor).to(self.device)

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
            "test/src_task_loss": src_task_loss,
            "test/src_loss": src_loss,
            "test/src_acc": src_acc,
            "test/src_f1": src_f1,
            "test/trg_task_loss": trg_task_loss,
            "test/trg_loss": trg_loss,
            "test/trg_acc": trg_acc,
            "test/trg_f1": trg_f1,
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
