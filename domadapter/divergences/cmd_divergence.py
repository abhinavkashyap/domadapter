import torch
from domadapter.divergences.base_divergence import BaseDivergence


def cmd(x1, x2, n_moments=5):
    """Central Moment Discrepancy
    The code is taken from
    https://github.com/wzell/mann/blob/master/models/central_moment_discrepancy.py
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)
    - Zellinger, Werner et al. "Robust unsupervised domain adaptation
    for neural networks via moment alignment," arXiv preprint arXiv:1711.06114,
    2017.
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    mx1 = torch.mean(x1, 0)
    mx2 = torch.mean(x2, 0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1, mx2)
    scms = dm
    for i in range(n_moments - 1):
        # moment diff of centralized samples
        scms += moment_diff(sx1, sx2, i + 2)
    return scms


def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    power = torch.pow(x1-x2,2)
    summed = torch.sum(power)
    sqrt = summed**(0.5)
    return sqrt


def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = torch.mean(torch.pow(sx1, k), 0)
    ss2 = torch.mean(torch.pow(sx2, k), 0)
    return l2diff(ss1, ss2)

class CMD(BaseDivergence):
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

        mmd_measure = cmd(source_sample, target_sample)

        return mmd_measure

    def __call__(self, source_sample: torch.Tensor, target_sample: torch.Tensor):
        return self.calculate(source_sample, target_sample)