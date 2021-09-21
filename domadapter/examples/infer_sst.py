import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if __name__ == "__main__":
    """
    1. Loads a Pretrained Adapter from the repository
    2. Activates the adapter
    3. Predicts the sentiment of the sentence.
    Refer to https://docs.adapterhub.ml/quickstart.html for the example
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

    # load the adapter
    adapter_name = model.load_adapter("sst-2@ukp", config="pfeiffer")

    # uses the adapter in every forward pass of the model
    model.set_active_adapters(adapter_name)

    # prepare the inputs
    sentence = "It is also great fun"
    tokenized = torch.tensor([tokenizer.encode(sentence)])
    outputs = model(tokenized)
    print(outputs.logits)
