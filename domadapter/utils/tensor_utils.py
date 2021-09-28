from typing import ForwardRef
import torch


def pairwise_distance(x: torch.Tensor, y: torch.Tensor):
    """Calculates the pairwise distance between two tensor

    :param x:  torch.Tensor
        Shape N, D
        N - Number of tensors
        D - The dimensions of the tensors
    :param y: torch.Tensor
        shape M, D
        N - Number of tensors
        D - The dimensions of the tensor
    :return:
        Pairwise Distances
        N, M
    """

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError("Both inputs should be matrices.")

    if x.shape[1] != y.shape[1]:
        raise ValueError("The number of features should be the same.")

    # Preparing for broadcasting
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)

    # N, D, M
    output = (x - y) ** 2

    # N, M
    output = torch.sum(output, 1)
    return output
