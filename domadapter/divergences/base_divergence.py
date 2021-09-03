from abc import ABCMeta, abstractmethod
import torch


class BaseDivergence(metaclass=ABCMeta):
    @abstractmethod
    def calculate(self, source_sample: torch.Tensor, target_sample: torch.Tensor):
        pass
