import torch
import torch.nn as nn
from torch import Tensor


class ResidualPredBlock(nn.Module):
    def __init__(self, num_channels: int, pred_proportion: float = 0.5):
        super().__init__()

        c_x1 = int(num_channels * pred_proportion)
        c_x2 = num_channels - c_x1
        self.c_x1 = c_x1
        self.c_x2 = c_x2

        self.pred_net = nn.Sequential(
            nn.Conv2d(c_x1, c_x1, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_x1, c_x2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_x2, c_x2, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Split into two tensors.
        # x1 is used to predict x2 (and is thus passed through).
        x1 = x[:, : self.c_x1]
        x2 = x[:, self.c_x1 :]

        y1 = x1  # pass-through
        y2 = x2 - self.pred_net(x1)  # residual

        return torch.cat([y1, y2], dim=1)
