import torch
from domadapter.divergences.base_divergence import BaseDivergence


def coral(source: torch.Tensor, target: torch.Tensor):
    """
    :param source: torch.Tensor
        The source domain torch tensors (features)
    :param target: torch.Tensor
        The target domain torch tensors (features)
    :return
        float
            The second order correlational measures between the two domains
    """
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)

    return loss

class Coral(BaseDivergence):
    def __init__(
        self,
    ):
        pass

    def calculate(
        self,
        source_sample: torch.Tensor,
        target_sample: torch.Tensor,
    ):
        """

        :param source_sample: torch.Tensor
            batch_size, embedding_dimension
        :param target_sample: torch.Tensor
            batch_size, embedding_dimension

        :return: List[float]
        The divergence between the samples

        """
        assert source_sample.size() == target_sample.size()

        measure = coral(source_sample, target_sample)

        return measure

    def __call__(self, source_sample: torch.Tensor, target_sample: torch.Tensor):
        return self.calculate(source_sample, target_sample)
