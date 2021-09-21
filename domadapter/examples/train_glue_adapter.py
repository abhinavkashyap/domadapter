import os


from dataclasses import dataclass, field
from datasets import load_dataset, load_metric
import logging
from rich.traceback import install
import sys
from transformers import HfArgumentParser
from transformers import TrainingArguments, MultiLingAdapterArguments
from transformers import AdapterConfig
from transformers.trainer_utils import is_main_process
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelWithHeads
from transformers import PretrainedConfig
import transformers.adapters.composition as ac
from transformers import set_seed
from transformers import EvalPrediction
from transformers import Trainer
from transformers import default_data_collator
from typing import Optional
import random
import numpy as np

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
# Pretty prints traceback in the console using rich
install(show_locals=False)

# setup logger
logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments related to training and validation data.

    `HFArgumentParser` turns this class into argparse arguments: allows usage in command
    line
    """

    task_name: Optional[str] = field(
        default=None, metadata={"help": f"Tasks available {['sst2']}"}
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": f"Max sequence length after tokenization. Sequenes longer "
            f"will be truncated and sequences shorter will be padded."
        },
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "HF stores the datasets in a caché (directory). If this is true, "
            "ignores caché and downloads data."
        },
    )

    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "If True, all sequences are padded to max_seq_length. If False, all sequences "
            "in a batch are padded to the max sequence length within the batch "
        },
    )

    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or json file containing training data"}
    )

    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or json file containing validation data"}
    )

    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "A cache directory to store the datasets downloaded from HF datasets"
        },
    )

    def __post_init__(self):
        if self.task_name:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    f"Unknown task. You should pick one from {list(task_to_keys.keys())}"
                )

        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                f"Need either a GLUE task name or a training/validation name"
            )

        else:
            extension = self.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], f"Training file should be a csv or json file"
            extension = self.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], f"Validatio file should be a csv or json file"


@dataclass
class ModelArguments:
    """
    Arguments related to model and tokenizer
    `HFArgumentParser` turns this class into argparse arguments: allows usage in command
    line
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Name or the path of a pretrained model. Refer to huggingface.co/models"
        }
    )

    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model name"
        },
    )

    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model name"
        },
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to store the pretrained models downloaded from s3"},
    )

    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use the Fast version of the tokenizer"},
    )


def main():

    # The parser takes dataclass types as a tuple
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
            MultiLingAdapterArguments,
        )
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        """
        We can pass a json file as the only argument
        The json file contains all the arguments
        """
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        # Parse command-line args into instances of the specified dataclass types.
        (
            model_args,
            data_args,
            training_args,
            adapter_args,
        ) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if is_main_process(training_args.local_rank)
        else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # setting seed for reproducibility.
    set_seed(training_args.seed)

    # Get the datasets
    # If the task name is mentioned then download the dataset from HF datasets hub
    # If a local csv or json filename is mentioned, then load the dataset from the file
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if data_args.task_name:
        datasets = load_dataset(
            "glue", data_args.task_name, cache_dir=data_args.dataset_cache_dir
        )

    elif data_args.train_file.endswith(".csv"):
        datasets = load_dataset(
            "csv",
            data_files={
                "train": data_args.train_file,
                "validation": data_args.validation_file,
            },
            cache_dir=data_args.dataset_cache_dir,
        )

    else:
        datasets = load_dataset(
            "csv",
            data_files={
                "json": data_args.train_file,
                "validation": data_args.validation_file,
            },
            cache_dir=data_args.dataset_cache_dir,
        )

    # get the labels
    label_list = None
    if data_args.task_name:
        is_regression = data_args.task_name == "stsb"
        if is_regression:
            num_labels = 1
        else:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
    else:
        is_regression = datasets["train"].features["label"].dtype in [
            "float32",
            "float64",
        ]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()
            num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.config_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    # AutoModelWithHeads is only defined for adapters
    model = AutoModelWithHeads.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # NOTE: The head name can be anything
    model.add_classification_head(
        head_name=data_args.task_name or "glue",
        num_labels=num_labels,
        id2label={i: v for i, v in enumerate(label_list)} if num_labels > 0 else None,
    )

    # Setting up adapters
    task_name: str = None
    if adapter_args.train_adapter:
        task_name = data_args.task_name or "glue"

        if task_name not in model.config.adapters:
            adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                non_linearity=adapter_args.adapter_nonlinearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
            )

            # Loading a pretrained adapter from the hub
            if adapter_args.load_adapter:
                model.load_adapter(
                    adapter_args.load_adapter, config=adapter_config, load_as=task_name
                )

            # Otherwise add a fresh adapter
            else:
                model.add_adapter(task_name, config=adapter_config)

        # Optionally load a pre-trained language adapter
        # You can only load a pretrained language adapter in this case
        # The loaded language adapter will be stacked with the task adapter
        if adapter_args.load_lang_adapter:
            lang_adapter_config = AdapterConfig.load(
                adapter_args.lang_adapter_config,
                non_linearity=adapter_args.lang_adapter_non_linearity,
                reduction_factor=adapter_args.lang_adapter_reduction_factor,
            )

            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adater,
                config=lang_adapter_config,
                load_as=adapter_args.language,
            )

        else:
            lang_adapter_name = None

        # Freeze the model except for those of the task adapter
        model.train_adapter([task_name])

        # Set the adapters to be used in forward pass
        if lang_adapter_name:
            model.set_active_adapters(ac.Stack(lang_adapter_name, task_name))

        else:
            model.set_active_adapters(task_name)

    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(f"Use --train_adapter for training.")

    ###################################
    # Processing the Datasets
    ###################################
    if data_args.task_name:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    # Not sure for which GLUE datasets this happens
    else:
        non_label_column_names = [
            name for name in datasets["train"].column_names if name != "label"
        ]
        if (
            "sentence1" in non_label_column_names
            and "sentence2" in non_label_column_names
        ):
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    # if specifiec Pad all sequences to maximum length
    # else pad every batch to the max length in that batch
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length

    else:
        padding = False
        max_length = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: label_name_to_id[label_list[i]] for i in range(num_labels)
            }
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )

        result = tokenizer(
            *args, padding=padding, max_length=max_length, truncation=True
        )

        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]

        return result

    datasets = datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    train_dataset = datasets["train"]
    eval_dataset = (
        datasets["validation_matched"]
        if data_args.task_name == "mnli"
        else datasets["validation"]
    )

    if data_args.task_name is not None:
        test_dataset = (
            datasets["test_matched"]
            if data_args.task_name == "mnli"
            else datasets["test"]
        )

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training dataset: {train_dataset[index]}.")

    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
        do_save_full_model=not adapter_args.train_adapter,
        do_save_adapters=adapter_args.train_adapter,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{task}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results {task} *****")
                    for key, value in eval_result.items():
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

            eval_results.update(eval_result)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = (
                np.squeeze(predictions)
                if is_regression
                else np.argmax(predictions, axis=1)
            )

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{task}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")
    return eval_results


if __name__ == "__main__":
    main()
