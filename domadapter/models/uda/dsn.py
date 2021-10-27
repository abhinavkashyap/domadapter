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

from domadapter.models.modules.dsn_losses import DiffLoss, MSE
from domadapter.divergences.cmd_divergence import CMD

import torchmetrics


class DSN(pl.LightningModule):
    def __init__(self, hparams):
        """DSN LightningModule

        Args:
            hparams (Optional[Dict[str, Any]], optional): [description]. Defaults to None.
        """
        super(DSN, self).__init__()

        self.save_hyperparameters(hparams)

        # config
        self.config = AutoConfig.from_pretrained(self.hparams["pretrained_model_name"])

        # load the model weights
        with console.status(
            f"Loading {self.hparams['pretrained_model_name']} Model", spinner="monkey"
        ):
            self.feature_extractor = AutoModel.from_pretrained(
                self.hparams["pretrained_model_name"], config=self.config
            )
        console.print(
            f"[green] Loaded {self.hparams['pretrained_model_name']} base model"
        )

        self.normalize = nn.Sequential(
            nn.Linear(
                in_features=self.config.hidden_size,
                out_features=hparams["hidden_size"],
            ),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hparams["hidden_size"]),
        )

        self.task_classifier = nn.Sequential(
            nn.Linear(
                in_features=hparams["hidden_size"],
                out_features=hparams["hidden_size"],
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=hparams["hidden_size"],
                out_features=hparams["num_classes"],
            ),
        )

        self.shared_encoder = nn.Sequential(
            nn.Linear(
                in_features=hparams["hidden_size"],
                out_features=hparams["hidden_size"],
            ),
            nn.Sigmoid(),
        )

        self.private_source_encoder = nn.Sequential(
            nn.Linear(
                in_features=hparams["hidden_size"],
                out_features=hparams["hidden_size"],
            ),
            nn.Sigmoid(),
        )

        self.private_target_encoder = nn.Sequential(
            nn.Linear(
                in_features=hparams["hidden_size"],
                out_features=hparams["hidden_size"],
            ),
            nn.Sigmoid(),
        )

        self.shared_decoder = nn.Sequential(
            nn.Linear(
                in_features=hparams["hidden_size"],
                out_features=hparams["hidden_size"],
            )
        )

        self.task_clf_loss = CrossEntropyLoss()
        self.similarity_loss = CMD()
        self.diff_loss = DiffLoss()
        self.recon_loss = MSE()

        self.train_acc = torchmetrics.Accuracy()  # accuracy
        self.src_dev_acc = torchmetrics.Accuracy()
        self.src_test_acc = torchmetrics.Accuracy()
        self.trg_dev_acc = torchmetrics.Accuracy()
        self.trg_test_acc = torchmetrics.Accuracy()

        self.train_f1 = torchmetrics.F1(
            num_classes=hparams["num_classes"], average="macro"
        )  # F1
        self.src_dev_f1 = torchmetrics.F1(
            num_classes=hparams["num_classes"], average="macro"
        )
        self.src_test_f1 = torchmetrics.F1(
            num_classes=hparams["num_classes"], average="macro"
        )
        self.trg_dev_f1 = torchmetrics.F1(
            num_classes=hparams["num_classes"], average="macro"
        )
        self.trg_test_f1 = torchmetrics.F1(
            num_classes=hparams["num_classes"], average="macro"
        )

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
        self.diff_weight = self.hparams.get("diff_weight", 0.3)
        self.sim_weight = self.hparams.get("sim_weight", 1.0)
        self.recon_weight = self.hparams.get("recon_weight", 1.0)

    def forward(self, src_inp_ids, src_attn_mask, trg_inp_ids, trg_attn_mask):
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
        Returns
        -------
        xs, xt, hcs, hct, hps, hpt, src_taskclf_logits, trg_taskclf_logits, src_decoder_output, trg_decoder_output

        """

        bsz = src_inp_ids.size(0)
        inp_ids = torch.cat([src_inp_ids, trg_inp_ids], dim=0)
        attn_mask = torch.cat([src_attn_mask, trg_attn_mask], dim=0)

        outputs = self.feature_extractor(input_ids=inp_ids, attention_mask=attn_mask)
        pooler_outputs_normalized = self.normalize(outputs.pooler_output)
        xs, xt = torch.split(pooler_outputs_normalized, [bsz, bsz], dim=0)

        # common encoder
        hct = self.shared_encoder(xs)
        hcs = self.shared_encoder(xt)

        # private source encoder
        hps = self.private_source_encoder(xs)
        # private target encoder
        hpt = self.private_target_encoder(xt)
        # task classifier (source and target)
        src_taskclf_logits = self.task_classifier(hcs)
        trg_taskclf_logits = self.task_classifier(hct)

        # shared decoder
        shared_decoder_input_source = hcs + hps
        shared_decoder_input_target = hct + hpt

        src_decoder_output = self.shared_decoder(shared_decoder_input_source)
        trg_decoder_output = self.shared_decoder(shared_decoder_input_target)

        # Name of vars are exactly how it's mentioned in paper
        # https://proceedings.neurips.cc/paper/2016/file/45fbc6d3e05ebd93369ce542e8f2322d-Paper.pdf

        return (
            xs,
            xt,
            hcs,
            hct,
            hps,
            hpt,
            src_taskclf_logits,
            trg_taskclf_logits,
            src_decoder_output,
            trg_decoder_output,
        )

    def configure_optimizers(self):
        learning_rate = self.learning_rate
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        lr_scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
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
                    "monitor": "source_val/f1",
                    "interval": "epoch",
                }
            ],
        )

    def training_step(self, batch, batch_idx):
        """training step of DSN"""
        # Classification loss
        src_inp_ids = batch["source_input_ids"]
        src_attn_mask = batch["source_attention_mask"]
        trg_inp_ids = batch["target_input_ids"]
        trg_attn_mask = batch["target_attention_mask"]
        bsz = src_inp_ids.size(0)

        # get the labels
        labels = batch["label_source"]

        (
            xs,
            xt,
            hcs,
            hct,
            hps,
            hpt,
            src_taskclf_logits,
            trg_taskclf_logits,
            src_decoder_output,
            trg_decoder_output,
        ) = self(
            src_inp_ids=src_inp_ids,
            src_attn_mask=src_attn_mask,
            trg_inp_ids=trg_inp_ids,
            trg_attn_mask=trg_attn_mask,
        )
        # get the loss
        class_loss = self.task_clf_loss(src_taskclf_logits, labels)

        diff_loss_source = self.diff_loss(hcs, hps)
        diff_loss_target = self.diff_loss(hct, hpt)
        final_diff_loss = diff_loss_source + diff_loss_target

        recon_loss_source = self.recon_loss(xs, src_decoder_output)
        recon_loss_target = self.recon_loss(xt, trg_decoder_output)
        final_recon_loss = recon_loss_source + recon_loss_target

        similarity_loss = self.similarity_loss.calculate(
            source_sample=hcs, target_sample=hct
        )

        accuracy = self.train_acc(preds=src_taskclf_logits, target=labels)
        f1 = self.train_f1(preds=src_taskclf_logits, target=labels)

        loss = (
            class_loss
            + self.diff_weight * final_diff_loss
            + self.sim_weight * similarity_loss
            + self.recon_weight * final_recon_loss
        )

        metrics = {
            "train/accuracy": accuracy,
            "train/f1": f1,
            "train/taskclf_loss": class_loss,
            "train/difference_loss": final_diff_loss,
            "train/reconstruction_loss": final_recon_loss,
            "train/similarity_loss": similarity_loss,
            "train/loss": loss,
        }

        for key, val in metrics.items():
            self.log(name=key, value=val)

        return loss

    def training_epoch_end(self, outputs):
        self._log_metrics(outputs)

    def validation_step(self, batch, batch_idx):
        """validation step of DSN"""
        src_inp_ids = batch["source_input_ids"]
        src_attn_mask = batch["source_attention_mask"]
        trg_inp_ids = batch["target_input_ids"]
        trg_attn_mask = batch["target_attention_mask"]
        bsz = src_inp_ids.size(0)

        # get the labels
        src_labels = batch["label_source"]
        trg_labels = batch["label_target"]

        (
            xs,
            xt,
            hcs,
            hct,
            hps,
            hpt,
            src_taskclf_logits,
            trg_taskclf_logits,
            src_decoder_output,
            trg_decoder_output,
        ) = self(
            src_inp_ids=src_inp_ids,
            src_attn_mask=src_attn_mask,
            trg_inp_ids=trg_inp_ids,
            trg_attn_mask=trg_attn_mask,
        )
        # get the loss
        src_task_loss = self.task_clf_loss(src_taskclf_logits, src_labels)
        trg_task_loss = self.task_clf_loss(trg_taskclf_logits, trg_labels)

        diff_loss_source = self.diff_loss(hcs, hps)
        diff_loss_target = self.diff_loss(hct, hpt)
        final_diff_loss = diff_loss_source + diff_loss_target

        recon_loss_source = self.recon_loss(xs, src_decoder_output)
        recon_loss_target = self.recon_loss(xt, trg_decoder_output)
        final_recon_loss = recon_loss_source + recon_loss_target

        similarity_loss = self.similarity_loss.calculate(
            source_sample=hcs, target_sample=hct
        )

        src_acc = self.src_dev_acc(preds=src_taskclf_logits, target=src_labels)
        src_f1 = self.src_dev_f1(preds=src_taskclf_logits, target=src_labels)

        trg_acc = self.trg_dev_acc(preds=trg_taskclf_logits, target=trg_labels)
        trg_f1 = self.trg_dev_f1(preds=trg_taskclf_logits, target=trg_labels)

        loss = (
            src_task_loss
            + self.diff_weight * final_diff_loss
            + self.sim_weight * similarity_loss
            + self.recon_weight * final_recon_loss
        )

        metrics = {
            "source_val/taskclf_loss": src_task_loss,
            "source_val/accuracy": src_acc,
            "source_val/f1": src_f1,
            "target_val/taskclf_loss": trg_task_loss,
            "target_val/accuracy": trg_acc,
            "target_val/f1": trg_f1,
            "val/loss": loss,
            "val/reconstruction_loss": final_recon_loss,
            "val/difference_loss": final_diff_loss,
            "val/similarity_loss": similarity_loss,
        }

        for key, val in metrics.items():
            self.log(name=key, value=val)

        return metrics

    def validation_epoch_end(self, outputs):
        self._log_metrics(outputs)

    def test_step(self, batch, batch_idx):
        """test step of DSN"""
        src_inp_ids = batch["source_input_ids"]
        src_attn_mask = batch["source_attention_mask"]
        trg_inp_ids = batch["target_input_ids"]
        trg_attn_mask = batch["target_attention_mask"]
        bsz = src_inp_ids.size(0)

        # get the labels
        src_labels = batch["label_source"]
        trg_labels = batch["label_target"]

        (
            xs,
            xt,
            hcs,
            hct,
            hps,
            hpt,
            src_taskclf_logits,
            trg_taskclf_logits,
            src_decoder_output,
            trg_decoder_output,
        ) = self(
            src_inp_ids=src_inp_ids,
            src_attn_mask=src_attn_mask,
            trg_inp_ids=trg_inp_ids,
            trg_attn_mask=trg_attn_mask,
        )
        # get the loss
        src_task_loss = self.task_clf_loss(src_taskclf_logits, src_labels)
        trg_task_loss = self.task_clf_loss(trg_taskclf_logits, trg_labels)

        diff_loss_source = self.diff_loss(hcs, hps)
        diff_loss_target = self.diff_loss(hct, hpt)
        final_diff_loss = diff_loss_source + diff_loss_target

        recon_loss_source = self.recon_loss(xs, src_decoder_output)
        recon_loss_target = self.recon_loss(xt, trg_decoder_output)
        final_recon_loss = recon_loss_source + recon_loss_target

        similarity_loss = self.similarity_loss.calculate(
            source_sample=hcs, target_sample=hct
        )

        src_acc = self.src_dev_acc(preds=src_taskclf_logits, target=src_labels)
        src_f1 = self.src_dev_f1(preds=src_taskclf_logits, target=src_labels)

        trg_acc = self.trg_dev_acc(preds=trg_taskclf_logits, target=trg_labels)
        trg_f1 = self.trg_dev_f1(preds=trg_taskclf_logits, target=trg_labels)

        loss = (
            src_task_loss
            + self.diff_weight * final_diff_loss
            + self.sim_weight * similarity_loss
            + self.recon_weight * final_recon_loss
        )

        metrics = {
            "source_test/taskclf_loss": src_task_loss,
            "source_test/accuracy": src_acc,
            "source_test/f1": src_f1,
            "target_test/taskclf_loss": trg_task_loss,
            "target_test/accuracy": trg_acc,
            "target_test/f1": trg_f1,
            "test/loss": loss,
            "test/reconstruction_loss": final_recon_loss,
            "test/difference_loss": final_diff_loss,
            "test/similarity_loss": similarity_loss,
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
