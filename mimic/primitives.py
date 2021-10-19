from abc import ABC, abstractmethod
import torch

class AbstractEncoder(ABC):
    size_input: tuple
    n_output: int
    def __init__(self, size_input, n_output): 
        self.size_input = size_input
        self.n_output = n_output

    @abstractmethod
    def __call__(self, image: torch.Tensor) -> torch.Tensor: ...
