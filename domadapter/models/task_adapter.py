import pytorch_lightning as pl
from domadapter.console import console
from transformers import AutoConfig
from transformers import PreTrainedTokenizer
from transformers import PretrainedConfig
from transformers import AutoModelWithHeads
from transformers import PreTrainedModel
from transformers import AdapterConfig
from typing import Dict


class GlueTaskAdapterModel(pl.LightningModule):
    def __init__(
        self,
        adapter_name: str,
        model_name: str,
        num_labels: int,
        cache_dir: str,
        tokenizer: PreTrainedTokenizer,
        id2label: Dict[int, str],
        adapter_config_name: str,
        adapter_non_linearity: str,
        adapter_reduction_factor: int,
    ):
        """

        Parameters
        ----------
        adapter_name: str
            Unique id representing the adapter

        model_name: str
            Name of Huggingface pretrained model
            If not a pretrained model then the path of the trained model

        num_labels: int

        cache_dir: str
            Directory sotres the pretrained language models downloaded from
            Huggingface

        tokenizer: PreTrainedTokenizer
            A pretrained tokenizer from the transformer library

        id2label: Dict[int, str]
            A mapping from label id to string

        adapter_config_name: str
            The architecture name of the adapter
            pfeiffer, houlsby are supported

        adapter_non_linearity: str
            relu is the popular one

        adapter_reduction_factor: int
            Adapter down projects the hidden dimension and then up projects it
            This specifies the reduction factor of the down projection with
            respect to the original pretrained model

        """
        super(GlueTaskAdapterModel, self).__init__()
        self.adapter_name = adapter_name
        self.model_name = model_name
        self.num_labels = num_labels
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.adapter_config_name = adapter_config_name
        self.adapter_non_linearity = adapter_non_linearity
        self.adapter_reduction_factor = adapter_reduction_factor

        self.pt_config = self._load_pt_config()
        self.model = self._load_pt_model()
        self.adapter_config = self._load_adapter_config()
        self._load_adapter()
        self.model.train_adapter([self.adapter_name])
        self.model.set_active_adapters(self.adapter_name)

    def _load_adapter_config(self):
        with console.status("Loading Adapter config"):
            config = AdapterConfig.load(
                self.adapter_config_name,
                non_linarity=self.adapter_non_linearity,
                reduction_factor=self.adapter_reduction_factor,
            )

        # \u2713 is the unicode for ✓
        console.print("[green] Loaded Adapter config \u2713")
        return config

    def _load_adapter(self):
        with console.status(f"Adding the adapter to the model"):
            self.model.add_adapter(self.adapter_name, config=self.adapter_config)

        # \u2713 is the unicode for ✓
        console.print("[green] Loaded Adapter \u2713")

    def _load_pt_model(self) -> PreTrainedModel:
        """Returns a pretrained model. AutoModelWithHeads
        is a class added by AdapterHub. Not available in the
        main branch of Huggingface itself.

        Returns
        -------

        """
        with console.status("Loading PT model"):
            model = AutoModelWithHeads.from_pretrained(
                self.model_name,
                from_tf=bool(".ckpt" in self.model_name),
                config=self.pt_config,
                cache_dir=self.cache_dir,
            )
            model.add_classification_head(
                head_name=self.adapter_name,
                num_labels=self.num_labels,
                id2label=self.id2label,
            )

        # \u2713 is the unicode for ✓
        console.print("[green] Loaded PT model \u2713")
        return model

    def _load_pt_config(self) -> PretrainedConfig:
        """Load pretrained model config

        Returns
        -------
        PretrainedConfig
            A config for a pretrained model

        """

        with console.status("Loading PT model config"):
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                finetuning_task=self.adapter_name,
                cache_dir=self.cache_dir,
            )

        # \u2713 is the unicode for ✓
        console.print("[green] Loaded PT model config \u2713")
        return config

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass