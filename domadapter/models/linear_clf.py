import torch.nn as nn
import torch


class LinearClassifier(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        return_hiddens=True,
    ):
        """ A MLP that can be added as a classification layer.

        Parameters
        ----------
        num_hidden_layers: int
            Number of hidden layers in the linear classifier
        input_size: int
            The input size to the linear classifier
        hidden_size: int
            The hiden size of the linear classifier
        output_size: int
            The output size of the hidden classifier
        return_hiddens: bool
            If true, the hidden representations of all the layers of the classifier
            will be returned.
        """
        super(LinearClassifier, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.return_hiddens = return_hiddens
        self.init_range = 0.1

        layers = []
        inp_size = self.input_size

        for i in range(self.num_hidden_layers):
            linear_layer = nn.Linear(inp_size, self.hidden_size)
            inp_size = self.hidden_size
            layers.append(linear_layer)

            # we do not need a relu on the last activation layer
            if i < (self.num_hidden_layers - 1):
                layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_size, self.output_size))

        self.classifier = nn.Sequential(*layers)

        self.__init_weights()

    def forward(self, x: torch.Tensor):
        """

        Parameters
        ----------
        x: torch.Tensor
            Shape: B * D
            B - Batch size
            D - Embedding/hidden dimension

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
        if self.return_hiddens is True
            Return all hidden activations
        else
            Return the last actication

        """
        activations = []
        activation_ = None

        input_ = x
        for layer in self.classifier:
            activation_ = layer(input_)
            activations.append(activation_)
            input_ = activation_
        if self.return_hiddens:
            return activations, activation_
        else:
            return None, activation_

    def __init_weights(self):
        """ Initialize weights form uniform distribution

        Returns
        -------
        None
        """
        for module in self.classifier:
            try:
                module.weight.data.uniform_(-self.init_range, self.init_range)
                module.bias.data.fill_(0)
            except:
                # This is mostly parameter less module that is added
                pass
