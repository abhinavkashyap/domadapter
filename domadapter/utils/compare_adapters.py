import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import gc
import json

from transformers import AutoTokenizer
from transformers import AutoModelWithHeads, AutoConfig

from datasets import load_dataset

# global variables
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 8

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--adapter1', type=str,
                    help='directory containing checkpoint of adapter 1')
parser.add_argument('--adapter2', type=str,
                    help='directory containing checkpoint of adapter 2')
parser.add_argument('--dataset', type=str,
                    help='CSV file whose data points are to be predicted')

args = parser.parse_args()

'''
python compare_adapter_outputs.py
--adapter1 "adapter1" \
--adapter2 "adapter2" \
--dataset "/Users/bhavitvyamalik/Desktop/work/domadapter/data/mnli/fiction_government/test_target.csv"
'''

softmax = nn.Softmax(dim=1)

# check if torch GPU available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# load tokenizer
config = AutoConfig.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# load csv dataset
dataset = load_dataset("csv", data_files=[args.dataset])["train"]


def return_output(dataset, model):
    batch_text = []
    batch_labels = []

    correct_predictions = []
    incorrect_predictions = []

    for i in tqdm(range(len(dataset))):
        # if(i<=6):
        if i % BATCH_SIZE == 0 and i != 0:  # batch size can be increased depending on your RAM
            batch_text.append((dataset["premise"][i], dataset["hypothesis"][i]))
            batch_labels.append(dataset["label"][i])

            encoding = tokenizer(batch_text, return_tensors='pt', padding='max_length', truncation=True,
                                 max_length=MAX_SEQ_LENGTH)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask).logits
            # get the softmax and argmax of the logits for outputs
            outputs_labels = softmax(outputs).argmax(dim=1).detach().cpu().numpy()

            # convert batch_labels to numpy array
            batch_labels = np.array(batch_labels)

            correct_index, incorrect_index = np.where(outputs_labels == batch_labels), np.where(
                outputs_labels != batch_labels)

            assert len(correct_index[0]) + len(incorrect_index[0]) == len(batch_labels)

            # get samples from batch_labels from correct_index
            correct_predictions += [batch_text[i] for i in correct_index[0]]
            incorrect_predictions += [batch_text[i] for i in incorrect_index[0]]

            batch_text = []
            batch_labels = []
            gc.collect()

        else:
            batch_text.append((dataset["premise"][i], dataset["hypothesis"][i]))
            batch_labels.append(dataset["label"][i])

    return correct_predictions, incorrect_predictions


# load model1
model1 = AutoModelWithHeads.from_pretrained("bert-base-uncased", config=config)

# load adapter1 checkpoints and activate it
adapter_name_1 = model1.load_adapter(args.adapter1)
model1.train_adapter([adapter_name_1])
# put model on device
model1.to(device)

print("running for adapter1")
correct1, incorrect1 = return_output(dataset, model1)

del model1
gc.collect()

# load model2
model2 = AutoModelWithHeads.from_pretrained("bert-base-uncased", config=config)

# load adapter2 checkpoints and activate it
adapter_name_2 = model2.load_adapter(args.adapter2)
model2.train_adapter([adapter_name_2])
# put model on device
model2.to(device)

print("running for adapter2")
correct2, incorrect2 = return_output(dataset, model2)

dict_first = {
    "correct": correct1,
    "incorrect": incorrect1
}
dict_second = {
    "correct": correct2,
    "incorrect": incorrect2
}

with open("adapter1.json", "w") as outfile:
    json.dump(dict_first, outfile, indent=4)
with open("adapter2.json", "w") as outfile:
    json.dump(dict_second, outfile, indent=4)