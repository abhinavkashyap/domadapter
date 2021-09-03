from domadapter.divergences.base_divergence import BaseDivergence
import torch
from domadapter.utils.tensor_utils import pairwise_distance


def mixture_gaussian_kernel(
    pairwise_distance_matrix: torch.Tensor, gamma_values: torch.Tensor
):
    """calculates the gaussian kernel matrix

    Parameters
    ----------
    pairwise_distance_matrix: torch.Tensor
        A pairwise distance matrix of shape N * M
    gamma_values: torch.Tensor
        Shape K
        the K gamma values for K different kernels

    Returns
    -------
    M * N tensor
    """
    assert pairwise_distance_matrix.ndim == 2
    assert gamma_values.ndim == 1
    # make sure that gamma values are always greater than 0
    # otherwise the reciprocal will produce NaNs
    # noinspection PyTypeChecker
    assert torch.all(gamma_values > 0).item() is True

    M, N = pairwise_distance_matrix.shape

    # K, 1
    gamma_values = gamma_values.view(-1, 1)

    # take the reciprocal of gamma_values
    # K, 1
    gamma_values = 1 / gamma_values

    # 1, M*N
    distances = pairwise_distance_matrix.view(1, -1)

    # K, MN
    # distance value for every gamma
    values = torch.matmul(gamma_values, distances)

    # take the negative exponetial of everything
    values = torch.exp(-values)

    # take mean along the K values to normalize
    # MN
    values = torch.mean(values, 0)

    values = values.view(M, N)

    return values


def gaussian_mk_mmd(
    first_tensor: torch.Tensor,
    second_tensor: torch.Tensor,
    start: int = 2 ** (-8),
    end: int = 2 ** 8,
    multiplication_factor: int = 2 ** (1 / 2),
):
    """
    Defines a multikernal mmd with gaussian kernel
    phi(x,y) = exp(frac{-||x-y||_2^2}{gamma})
    The initial width parameter(gamma) of a gaussian kernel is set to be
    the median of the pairwise distances between the samples. Then
    multiple kernels are instantiated with width parameters from start*gamma
    to end*gamma in multiplication_factors steps

    Parameters
    ----------
    first_tensor: torch.Tensor
        Shape M * D
    second_tensor: torch.Tensor
        Shape N * D
    start: int
        The starting multiplciation factor for gamma
    end: int
        The ending multiplication factor for gamma
    multiplication_factor: int
        The multiplication factor to multiply by.

    Returns
    --------
    MMD Value for the source and the target tensors
    float
    """
    # generate the different lambdas
    progression = [start]
    current_val = start

    first_first_pairwise = pairwise_distance(first_tensor, first_tensor)
    second_second_pairwise = pairwise_distance(second_tensor, second_tensor)
    first_second_pairwise = pairwise_distance(first_tensor, second_tensor)

    # The median approximation of the initial value of gamma
    gamma = torch.median(first_second_pairwise).item()

    try:
        assert gamma > 0
    except AssertionError:
        print(
            f"We tried to calculate the gamma value for your matrices. "
            f"But it turns out that the median distance is zero. We are going to set the gamma value to 1"
        )
        gamma = 1

    while current_val <= end:
        ele = current_val * multiplication_factor
        progression.append(ele)
        current_val = ele

    progression_values = gamma * torch.Tensor(progression)

    # convert progression values also to cuda
    # if cuda, the other tensors would be tensors before calling this function
    if torch.cuda.is_available():
        device_number = first_tensor.get_device()
        device = torch.device(f"cuda:{device_number}")
        progression_values = progression_values.to(device)

    corr_a = mixture_gaussian_kernel(first_first_pairwise, progression_values)
    corr_b = mixture_gaussian_kernel(second_second_pairwise, progression_values)
    corr_ab = mixture_gaussian_kernel(first_second_pairwise, progression_values)

    corr_a = torch.mean(corr_a)
    corr_b = torch.mean(corr_b)
    corr_ab = torch.mean(corr_ab)

    mmd = corr_a + corr_b - (2 * corr_ab)

    return mmd


class GaussianMKMMD(BaseDivergence):
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

        mmd_measure = gaussian_mk_mmd(source_sample, target_sample)

        return mmd_measure

    def __call__(self, source_sample: torch.Tensor, target_sample: torch.Tensor):
        return self.calculate(source_sample, target_sample)
