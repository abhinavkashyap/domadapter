from domadapter.divergences.base_divergence import BaseDivergence
import torch


def rbf_mkmmd(x, y, kernel, device):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.
    Implementation from https://www.kaggle.com/onurtunali/maximum-mean-discrepancy
    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
        device: the device on which to place the tensors
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2.0 * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2.0 * zz  # Used for C in (1)

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2.0 * XY)


class RBFMKMMD(BaseDivergence):
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
        device_number = source_sample.get_device()
        device_string = f"cuda:{device_number}" if device_number >= 0 else "cpu"
        device = torch.device(device_string)

        mmd_measure = rbf_mkmmd(source_sample, target_sample, "rbf", device)

        return mmd_measure

    def __call__(self, source_sample: torch.Tensor, target_sample: torch.Tensor):
        return self.calculate(source_sample, target_sample)
