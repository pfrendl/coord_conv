import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def make_pos_grid(height: int, width: int, device: torch.device) -> Tensor:
    gi, gj = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij",
    )
    grid = torch.stack([gi, gj], dim=0)
    return grid


class AddCoords(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        pos_grid = make_pos_grid(height=x.shape[2], width=x.shape[3], device=x.device)
        pos_grid = pos_grid.expand((x.shape[0], -1, -1, -1))
        return torch.cat([x, pos_grid], dim=1)


class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        gain = math.sqrt(2)
        return gain * x.relu()


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        gain = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(gain * torch.randn((out_features, in_features)))
        self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(input=x, weight=self.weight, bias=self.bias)


class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0
    ) -> None:
        super().__init__()
        gain = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = nn.Parameter(gain * torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros((out_channels,)))
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(input=x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding)


class Regressor0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=2, kernel_size=1),
            nn.Flatten(start_dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class Regressor1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            AddCoords(),
            Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=2, kernel_size=1),
            nn.Flatten(start_dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class Regressor2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            AddCoords(),
            Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            AddCoords(),
            Conv2d(in_channels=34, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            AddCoords(),
            Conv2d(in_channels=34, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            AddCoords(),
            Conv2d(in_channels=34, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            AddCoords(),
            Conv2d(in_channels=34, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            AddCoords(),
            Conv2d(in_channels=34, out_channels=2, kernel_size=1),
            nn.Flatten(start_dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class Regressor3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_features = 32
        self.seq = nn.Sequential(
            AddCoords(),
            Conv2d(in_channels=5, out_channels=self.num_features, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=self.num_features, out_channels=2, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.seq(x)
        return x.mean(dim=(2, 3))
