import torch
from torch import Tensor
from torch.utils.data import Dataset


class PixelRegressionDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def __len__(self) -> int:
        return self.size**2

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        i = index // self.size
        j = index % self.size
        target = 2 * torch.tensor([i, j]) / (self.size - 1) - 1
        img = torch.zeros((3, self.size, self.size))
        img[:, i, j] = 1.0
        return img, target
