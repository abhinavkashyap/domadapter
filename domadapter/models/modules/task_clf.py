import torch.nn as nn


class TaskClassifierModule(nn.Module):
    def __init__(self, hparams):
        super(TaskClassifierModule, self).__init__()
        self.linear_hidden = nn.ModuleList(
            [
                nn.Linear(in_features=768, out_features=hparams["hidden_size"])
                for i in range(12)
            ]
        )  # 768 had to be hardcoded here ;-;
        self.linear_hidden_same = nn.ModuleList(
            [
                nn.Linear(
                    in_features=hparams["hidden_size"],
                    out_features=hparams["hidden_size"],
                )
                for i in range(12)
            ]
        )
        self.prediction_head = nn.ModuleList(
            [
                nn.Linear(
                    in_features=hparams["hidden_size"],
                    out_features=hparams["num_classes"],
                )
                for i in range(12)
            ]
        )

    def forward(self, x, layer):
        # ModuleList can act as an iterable, or be indexed using ints
        x = nn.ReLU()(self.linear_hidden[layer](x))
        x = nn.ReLU()(self.linear_hidden_same[layer](x))
        x = self.prediction_head[layer](x)
        return x
