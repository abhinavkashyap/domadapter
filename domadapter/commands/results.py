import click
import torch.nn as nn
import torch
from transformers import AutoTokenizer
from transformers import BertModelWithHeads
from tqdm import tqdm
import pandas as pd
import numpy as np
from torchmetrics import F1
from datasets import load_dataset
from pathlib import Path
from rich.console import Console
import os


console = Console()
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 8


@click.group()
def results():
    pass


@results.command()
@click.option(
    "--source",
    type=click.Choice(["fiction", "slate", "government", "telephone", "travel"]),
    help="Mention the source domain",
)
@click.option(
    "--target",
    type=click.Choice(["fiction", "slate", "government", "telephone", "travel"]),
    help="Mention the source domain",
)
def mnli_joint_dt(source, target):
    dataset_cache = os.environ["DATASET_CACHE_DIR"]
    dataset_cache = Path(dataset_cache)

    # This path is written after running domadapter download mnli 
    test_file = dataset_cache.joinpath(f"mnli/{source}_{target}/test_target.csv")
    
    def return_output(dataset, model):
        batch_text = []
        batch_labels = []

        global_predictions = []
        global_labels = []
        global_sentences = []

        for i in tqdm(range(len(dataset))):
            if (
                i % BATCH_SIZE == 0 and i != 0
            ):  # batch size can be increased depending on your RAM
                batch_text.append((dataset["premise"][i], dataset["hypothesis"][i]))
                batch_labels.append(dataset["label"][i])

                encoding = tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                )
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)

                outputs = model(input_ids, attention_mask).logits
                # get the softmax and argmax of the logits for outputs
                outputs_labels = softmax(outputs).argmax(dim=1).detach().cpu().numpy()

                # convert batch_labels to numpy array
                batch_labels = np.array(batch_labels)

                global_labels += batch_labels.tolist()
                global_predictions += outputs_labels.tolist()
                global_sentences += batch_text

                batch_text = []
                batch_labels = []

            else:
                batch_text.append((dataset["premise"][i], dataset["hypothesis"][i]))
                batch_labels.append(dataset["label"][i])

        if len(batch_labels) > 0:
            encoding = tokenizer(
                batch_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask).logits
            # get the softmax and argmax of the logits for outputs
            outputs_labels = softmax(outputs).argmax(dim=1).detach().cpu().numpy()

            # convert batch_labels to numpy array
            batch_labels = np.array(batch_labels)

            global_labels += batch_labels.tolist()
            global_predictions += outputs_labels.tolist()
            global_sentences += batch_text

        return global_sentences, global_predictions, global_labels

    idx_label_mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
    label_idx_mapping = {v: k for k, v in idx_label_mapping.items()}

    softmax = nn.Softmax(dim=1)
    device = torch.device("cpu")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model = BertModelWithHeads.from_pretrained("bert-base-uncased")
    adapter_name = model.load_adapter(
        f"domadapter/joint_dt_{source}_{target}", source="hf"
    )
    model.set_active_adapters([adapter_name])
    model.to(device)

    dataset = load_dataset("csv", data_files=[str(test_file)])["train"]
    sentences, output, gold_label = return_output(dataset, model)

    output = [idx_label_mapping[i] for i in output]
    gold_label = [idx_label_mapping[i] for i in gold_label]

    # convert lists to dataframe and save as CSV
    df = pd.DataFrame(
        {"sentence": sentences, "prediction": output, "label": gold_label}
    )

    predictions = df["prediction"]
    predictions = predictions.tolist()
    predictions = [label_idx_mapping[prediction] for prediction in predictions]
    predictions = torch.IntTensor(predictions)

    labels = df["label"]
    labels = labels.tolist()
    labels = [label_idx_mapping[label] for label in labels]
    labels = torch.IntTensor(labels)

    f1 = F1(num_classes=3)
    f1_score = f1(predictions, labels)
    print(f"F1 Score: {f1_score}")


if __name__ == "__main__":
    results()
