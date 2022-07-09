import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import json

from sklearn.metrics import f1_score
from domadapter.console import console
from transformers import AutoModelWithHeads, AutoConfig, AdapterConfig, AutoTokenizer
from transformers.adapters.composition import Stack

from datasets import load_dataset

# global variables
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64

parser = argparse.ArgumentParser(description='Arguments for adapter outputs')
parser.add_argument('--domain-adapter', type=str,
                    help='directory containing checkpoint of domain adapter')
parser.add_argument('--task-adapter', type=str,
                    help='directory containing checkpoint of task adapter')
parser.add_argument('--data-module', type=str,
                    help='mnli/sa')
parser.add_argument('--dataset', type=str,
                    help='CSV file whose data points are to be predicted')

args = parser.parse_args()

softmax = nn.Softmax(dim=1)
'''
python test_domadapter.py \
--domain-adapter "/home/bhavitvya/domadapter/experiments/camera_photo_baby/domain_adapter/16nlp83v/checkpoints" \
--task-adapter "/home/bhavitvya/domadapter/experiments/camera_photo_baby/task_adapter_mkmmd/2y0oancq/checkpoints" \
--data-module "sa" \
--dataset "/home/bhavitvya/domadapter/data/sa/camera_photo_baby/test_target.csv"
'''

# check if torch GPU available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# load tokenizer
config = AutoConfig.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", usefast=True)
model = AutoModelWithHeads.from_pretrained("bert-base-uncased", config=config)

# load domain adapter checkpoints and activate it
with console.status(
        f"Loading domain and task adapter", spinner="monkey"
    ):
        # load domain adapter to PLM
        domain_adapter = model.load_adapter(args.domain_adapter)
        # load task adapter to PLM
        task_adapter = model.load_adapter(args.task_adapter)

console.print(f"🤗 Loaded {domain_adapter} and {task_adapter}")

# stack adapters
model.active_adapters = Stack(
        domain_adapter,
        task_adapter,
)
model.train_adapter([task_adapter])
# put model on device
model.to(device)
model.eval()

model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
console.print(f"🤗 trainable params are {model_total_params}")

# load csv dataset
dataset = load_dataset("csv", data_files=[args.dataset])["train"]

def return_output(dataset, model):
    batch_text = []
    batch_labels = []

    global_predictions = []
    global_labels = []
    global_sentences = []

    for i in tqdm(range(len(dataset))):
        # if(i<=6):
            if i % BATCH_SIZE == 0 and i!=0:  #batch size can be increased depending on your RAM
                if args.data_module == "mnli":
                    batch_text.append((dataset["premise"][i], dataset["hypothesis"][i]))
                else:
                    batch_text.append(dataset["sentence"][i])

                batch_labels.append(dataset["label"][i])

                encoding = tokenizer(batch_text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_SEQ_LENGTH)
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

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
                gc.collect()

            else:
                if args.data_module == "mnli":
                    batch_text.append((dataset["premise"][i], dataset["hypothesis"][i]))
                else:
                    batch_text.append(dataset["sentence"][i])
                batch_labels.append(dataset["label"][i])

    if len(batch_labels) > 0:
        encoding = tokenizer(batch_text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_SEQ_LENGTH)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask).logits
        # get the softmax and argmax of the logits for outputs
        outputs_labels = softmax(outputs).argmax(dim=1).detach().cpu().numpy()

        # convert batch_labels to numpy array
        batch_labels = np.array(batch_labels)

        global_labels += batch_labels.tolist()
        global_predictions += outputs_labels.tolist()
        global_sentences += batch_text

    return global_sentences, global_predictions, global_labels


print("running inference")
sentences, output, gold_label = return_output(dataset, model)
f1_score = f1_score(gold_label, output, average='macro')

output = {
    "domain_adapter": domain_adapter,
    "task_adapter": task_adapter,
    "f1": f1_score
}

with open(f"results_{args.data_module}.txt", 'a') as convert_file:
     convert_file.write(json.dumps(output))
     convert_file.write("\n")

del model
gc.collect()