import torch.nn as nn

class DomainClassifierModule(nn.Module):
    def __init__(self, hparams):
        super(DomainClassifierModule, self).__init__()
        self.linear_hidden = nn.ModuleList([nn.Linear(in_features=768, out_features=hparams["hidden_size"]) for i in range(3)])  # 768 had to be hardcoded here ;-;
        self.prediction_head = nn.ModuleList([nn.Linear(in_features=hparams["hidden_size"], out_features=2) for i in range(3)])

    def forward(self, x, layer):
        # ModuleList can act as an iterable, or be indexed using ints
        x = nn.ReLU()(self.linear_hidden[layer](x))
        x = self.prediction_head[layer](x)
        return x