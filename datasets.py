from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset


class PixelRegressionDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, size: int, mask: Tensor) -> None:
        super().__init__()
        self.size = size
        gi, gj = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        grid = torch.stack([gi, gj], dim=2)
        self.samples = grid[mask]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index]
        target = 2 * sample / (self.size - 1) - 1
        img = torch.zeros((3, self.size, self.size))
        img[:, sample[0], sample[1]] = 1.0
        return img, target
