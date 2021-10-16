import torch
import pytorch_lightning as pl
from typing import Any, Optional, Dict, List
from transformers import BertModel
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertPooler
from domadapter.console import console
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from domadapter.models.modules.linear_clf import LinearClassifier
import numpy as np
from domadapter.divergences.gaussian_mkmmd_divergence import GaussianMKMMD
from domadapter.divergences.rbf_mkmmd_divergence import RBFMKMMD
import torchmetrics


class DAN(pl.LightningModule):
    def __init__(
        self,
        hparams: Optional[Dict[str, Any]] = None,
    ):
        """The classifier will be trained @ layer n. All the other layers above it are frozen
        It will not be trained

        n: int
            The layer at which the classifier will be trained
            n \in [0, num_layers) where num_layers are the number of layers in the bert model specified
        num_classes: int
            The number of classes in classification
        freeze_upto: int
            Freeze upto certain layer in the transformer
        bert_config
            Optional[BertConfig]
        param: hparams
            Dict[str, Any]
            The configuration of hyperparameters. Every model has a config file stored in
            a json file. Refer to robsesame/configs folder for the configs

        """
        super(DAN, self).__init__()

        # Save the hyper-parameters to the lightning module
        # available as self.hparams hereafter.
        self.save_hyperparameters(hparams)

        self.layer = self.hparams["train_bert_at_layer"]
        self.num_classes = self.hparams["num_classes"]
        self.freeze_upto = self.hparams["freeze_upto"]
        self.pretrained_model_name = self.hparams["pretrained_model_name"]
        self.bert_config = (
            BertConfig()
            if self.hparams.get("bert_config") is None
            else self.hparams.get("bert_config")
        )
        self.bert_config.output_hidden_states = True

        with console.status("Loading BERT Model", spinner="monkey"):
            self.bert = BertModel.from_pretrained(
                self.pretrained_model_name, config=self.bert_config
            )
        console.print(f"[green] Loaded BERT model from {self.pretrained_model_name}")

        self.num_layers = len(self.bert.encoder.layer)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.num_clf_layers = self.hparams["num_clf_layers"]
        self.clf_hidden_size = self.hparams["clf_hidden_size"]
        self.is_divergence_reduced = self.hparams["is_divergence_reduced"]
        self.divergence_reduced = self.hparams["divergence_reduced"]
        self.div_reg_param = self.hparams.get("div_reg_param")

        self.loss = CrossEntropyLoss()
        # Define the pooler
        self.pooler = BertPooler(self.bert_config)
        # Define the linear classifier
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        # This is the classifier htat is trained @ layer n of the bert model that you have chosen
        self.src_clf = LinearClassifier(
            num_hidden_layers=self.num_clf_layers,
            input_size=self.bert_hidden_size,
            hidden_size=self.clf_hidden_size,
            output_size=self.num_classes,
        )

        self.trg_clf = LinearClassifier(
            num_hidden_layers=self.num_clf_layers,
            input_size=self.bert_hidden_size,
            hidden_size=self.clf_hidden_size,
            output_size=self.num_classes,
        )

        assert (
            0 <= self.layer < self.num_layers
        ), f"The model that you defined has {self.num_layers}. The classification layer that you specified is {self.layer}"

        # freeze-all-layers above a the specified layer
        self._freeze_layers()

        #######################################################################
        # OPTIMIZER RELATED VARIABLES
        #######################################################################
        self.learning_rate = self.hparams.get("learning_rate")
        self.scheduler_factor = self.hparams.get("scheduler_factor", 0.1)
        self.scheduler_patience = self.hparams.get("scheduler_patience", 2)
        self.scheduler_threshold = self.hparams.get("scheduler_threshold", 0.0001)
        self.scheduler_cooldown = self.hparams.get("scheduler_cooldown", 0)
        self.scheduler_eps = self.hparams.get("scheduler_eps", 1e-8)

        self.training_accuracy = torchmetrics.Accuracy()
        self.validation_accuracy = torchmetrics.Accuracy()
        self.testing_accuracy = torchmetrics.Accuracy()
        self.softmax = nn.Softmax(dim=1)

        self.mk_mmd_gaussian = GaussianMKMMD()
        self.mk_mmd_rbf = RBFMKMMD()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=None,
        )

        hidden_states = outputs.hidden_states

        # get the hidden state at the layer specified
        # +1 because 0th hidden state is the hidden state of the embedding layer
        # batch_size, sequence_length, hidden_size
        hidden_state = hidden_states[self.layer + 1]

        # send it to the pooler

        # batch_size, hidden_size
        pooled_output = self.pooler(hidden_state)

        bsz = int(pooled_output.size(0))

        # split the pooled output before passing it to the classifiers
        # batch_size/2, hidden_size,
        # batch_size/2, hidden_size
        src_output, trg_output = torch.split(
            pooled_output, dim=0, split_size_or_sections=[bsz // 2, bsz // 2]
        )

        # List[torch.tensor]
        # batch_size, num_classes
        src_clf_hiddens, src_logits = self.src_clf(src_output)
        trg_clf_hiddens, trg_logits = self.trg_clf(trg_output)

        return src_logits, trg_logits, src_clf_hiddens, trg_clf_hiddens

    def forward_src(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=None,
        )

        hidden_states = outputs.hidden_states

        # get the hidden state at the layer specified
        # +1 because 0th hidden state is the hidden state of the embedding layer
        # batch_size, sequence_length, hidden_size
        hidden_state = hidden_states[self.layer + 1]

        # send it to the pooler
        # batch_size, hidden_size
        pooled_output = self.pooler(hidden_state)

        # batch_size, num_classes
        src_clf_hiddens, src_logits = self.src_clf(pooled_output)

        return src_logits, src_clf_hiddens

    def forward_trg(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=None,
        )

        hidden_states = outputs.hidden_states

        # get the hidden state at the layer specified
        # +1 because 0th hidden state is the hidden state of the embedding layer
        # batch_size, sequence_length, hidden_size
        hidden_state = hidden_states[self.layer + 1]

        # send it to the pooler
        # batch_size, hidden_size
        pooled_output = self.pooler(hidden_state)

        # batch_size, num_classes
        trg_clf_hiddens, trg_logits = self.trg_clf(pooled_output)

        return trg_logits, trg_clf_hiddens

    def training_step(self, batch, batch_idx):
        src_batch, trg_batch = batch
        (
            src_input_ids,
            src_attention_mask,
            src_token_type_ids,
            src_label_ids,
        ) = src_batch
        (
            trg_input_ids,
            trg_attention_mask,
            trg_token_type_ids,
            trg_label_ids,
        ) = trg_batch

        input_ids = torch.cat([src_input_ids, trg_input_ids], dim=0)
        attention_mask = torch.cat([src_attention_mask, trg_attention_mask], dim=0)
        token_type_ids = torch.cat([src_token_type_ids, trg_token_type_ids], dim=0)

        src_logits, trg_logits, src_clf_hiddens, trg_clf_hiddens = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        if self.is_divergence_reduced:
            divergences = []
            for src_hidden, trg_hidden in zip(src_clf_hiddens, trg_clf_hiddens):
                if self.divergence_reduced == "gaussian":
                    divergence = self.mk_mmd_gaussian.calculate(src_hidden, trg_hidden)
                elif self.divergence_reduced == "rbf":
                    divergence = self.mk_mmd_rbf.calculate(src_hidden, trg_hidden)
                else:
                    raise ValueError(f"Divergence reduce should be in [gaussian, rbf]")
                divergences.append(divergence)

            divergences = torch.FloatTensor(divergences)
            divergence = torch.mean(divergences)
            divergence = self.div_reg_param * divergence
        else:
            divergence = 0

        loss = (
            self.loss(src_logits.view(-1, self.num_classes), src_label_ids.view(-1))
            + divergence
        )

        probs = self.softmax(src_logits)
        acc = self.training_accuracy(probs, src_label_ids.view(-1))

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/src_acc",
            acc,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        if self.is_divergence_reduced:
            self.log(
                "train/divergence",
                divergence,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        src_batch, trg_batch = batch
        src_input_ids, src_attention_mask, src_token_type_ids, src_label_ids = src_batch
        trg_input_ids, trg_attention_mask, trg_token_type_ids, trg_label_ids = trg_batch

        input_ids = torch.cat([src_input_ids, trg_input_ids], dim=0)
        attention_mask = torch.cat([src_attention_mask, trg_attention_mask], dim=0)
        token_type_ids = torch.cat([src_token_type_ids, trg_token_type_ids], dim=0)

        src_logits, trg_logits, _, _ = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        loss = self.loss(src_logits.view(-1, self.num_classes), src_label_ids.view(-1))

        probs = self.softmax(src_logits)
        acc = self.validation_accuracy(probs, src_label_ids.view(-1))

        return {"src_loss": loss, "src_acc": acc}

    def validation_epoch_end(self, outputs: List[Any]):
        src_losses = []
        src_accs = []
        for out in outputs:
            loss = out["src_loss"]
            acc = out["src_acc"]
            src_losses.append(loss.item())
            src_accs.append(acc.item())

        mean_src_loss = np.mean(src_losses)
        mean_src_acc = np.mean(src_accs)

        self.log(
            name="dev/src_loss",
            value=mean_src_loss,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )

        self.log(
            name="dev/src_acc",
            value=mean_src_acc,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx):
        src_batch, trg_batch = batch
        src_input_ids, src_attention_mask, src_token_type_ids, src_label_ids = src_batch
        trg_input_ids, trg_attention_mask, trg_token_type_ids, trg_label_ids = trg_batch

        input_ids = torch.cat([src_input_ids, trg_input_ids], dim=0)
        attention_mask = torch.cat([src_attention_mask, trg_attention_mask], dim=0)
        token_type_ids = torch.cat([src_token_type_ids, trg_token_type_ids], dim=0)

        src_logits, trg_logits, _, _ = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        loss = self.loss(src_logits.view(-1, self.num_classes), src_label_ids.view(-1))

        probs = self.softmax(src_logits)
        acc = self.testing_accuracy(probs, src_label_ids.view(-1))

        return {"loss": loss, "acc": acc}

    def test_epoch_end(self, outputs: List[Any]):
        src_losses = []
        src_accs = []
        for out in outputs:
            loss = out["loss"]
            acc = out["acc"]
            src_losses.append(loss.item())
            src_accs.append(acc.item())

        mean_src_loss = np.mean(src_losses)
        mean_src_acc = np.mean(src_accs)

        self.log(
            name="test/src_loss",
            value=mean_src_loss,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )

        self.log(
            name="test/src_acc",
            value=mean_src_acc,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )

    def _freeze_layers(self):

        if self.freeze_upto == 0:
            return

        # starts from 1 because 0th layer is the embedding layer
        for layer_idx in range(1, self.freeze_upto + 1):
            for param in list(self.bert.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False

        console.print(
            f"[green] Froze layers {list(range(self.layer+1, self.num_layers))}"
        )

    def configure_optimizers(self):
        learning_rate = self.learning_rate
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
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
                    "monitor": "dev/src_loss",
                    "interval": "epoch",
                }
            ],
        )
