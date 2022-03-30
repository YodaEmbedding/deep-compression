import torch
import torch.nn as nn
from torch import Tensor


class ResidualPredBlock(nn.Module):
    def __init__(self, num_channels: int, pred_proportion: float = 0.5):
        super().__init__()

        c_p = int(num_channels * pred_proportion)
        self.c_p = c_p

        self.pred_net = nn.Sequential(
            nn.Conv2d(c_p, c_p, 1, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_p, c_p, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_p, c_p, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Split into two tensors.
        # x1 is used to predict x2 (and is thus passed through).
        x1 = x[:, : self.c_p]
        x2 = x[:, self.c_p :]

        y1 = x1  # pass-through
        y2 = x2 - self.pred_net(x1)  # residual

        return torch.cat([y1, y2], dim=1)
