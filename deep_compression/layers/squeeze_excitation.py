import torch
import torch.nn as nn
from torch import Tensor


class SEBlock(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.estimator = nn.AdaptiveAvgPool2d(1)
        self.controller = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        n, c, _, _ = x.shape
        estimate = self.estimator(x).view(n, c)
        control = self.controller(estimate).view(n, c, 1, 1)
        return x * control
