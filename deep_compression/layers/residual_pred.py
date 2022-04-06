import torch
import torch.nn as nn
from torch import Tensor


class ResidualPredBlock(nn.Module):
    def __init__(
        self,
        c_x1: int,
        c_x2: int,
        pred_net: nn.Module,
        mode: str = "encoder",
    ):
        super().__init__()
        self.c_x1 = c_x1
        self.c_x2 = c_x2
        self.mode = mode
        self.unregistered_modules = {}
        self.unregistered_modules["pred_net"] = pred_net

    def forward(self, x: Tensor) -> Tensor:
        pred_net = self.unregistered_modules["pred_net"]
        sgn = -1 if self.mode == "encoder" else 1

        # Split into two tensors.
        # x1 is used to predict x2 (and is thus passed through).
        x1 = x[:, : self.c_x1]
        x2 = x[:, self.c_x1 :]

        y1 = x1  # pass-through
        y2 = x2 + sgn * pred_net(x1)  # residual

        return torch.cat([y1, y2], dim=1)


def create_pred_net(c_x1: int, c_x2: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(c_x1, c_x1, 1, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_x1, c_x2, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_x2, c_x2, 3, padding=1),
    )
