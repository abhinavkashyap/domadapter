import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

from transformers import AutoTokenizer
from transformers import AutoModelWithHeads, AutoConfig

from datasets import load_dataset

# global variables
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 8

parser = argparse.ArgumentParser(description='Arguments for adapter outputs')
parser.add_argument('--adapter', type=str,
                    help='directory containing checkpoint of adapter')
parser.add_argument('--output', type=str,
                    help='name of output CSV file')
parser.add_argument('--dataset', type=str,
                    help='CSV file whose data points are to be predicted')

args = parser.parse_args()

'''
python compare_adapter.py
--adapter "adapter" \
--output "output_adapter.csv" \
--dataset "/Users/bhavitvyamalik/Desktop/work/domadapter/data/mnli/fiction_government/test_target.csv"
'''

id_str = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

softmax = nn.Softmax(dim=1)

# check if torch GPU available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# load tokenizer
config = AutoConfig.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelWithHeads.from_pretrained("bert-base-uncased", config=config)

# load adapter checkpoints and activate it
adapter_name = model.load_adapter(args.adapter)
model.set_active_adapters([adapter_name])
# put model on device
model.to(device)

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
                batch_text.append((dataset["premise"][i], dataset["hypothesis"][i]))
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
                batch_text.append((dataset["premise"][i], dataset["hypothesis"][i]))
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
output = [id_str[i] for i in output]
gold_label = [id_str[i] for i in gold_label]

# convert lists to dataframe and save as CSV
df = pd.DataFrame({"sentence": sentences, "prediction": output, "label": gold_label})
df.to_csv(args.output, index=False)

del model
gc.collect()