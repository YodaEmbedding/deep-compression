import math
from typing import Optional

import torch
import torch.nn as nn


BPP_REL_TOL = 0.01
LMBDA_GAIN = 10 ** (1 / 10000)
MIN_BATCHES = 20000


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    lmbda: torch.Tensor
    num_batches: torch.Tensor

    def __init__(self, lmbda=1e-2, target_bpp: Optional[float] = None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.target_bpp = target_bpp
        self.register_buffer("lmbda", torch.Tensor([lmbda]))
        num_batches = torch.tensor([0], dtype=torch.int64)
        self.register_buffer("num_batches", num_batches)

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        self._compute_lmbda(actual_bpp=out["bpp_loss"].item())
        lmbda = self.lmbda[0]
        out["loss"] = lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        self.num_batches += 1

        return out

    def _compute_lmbda(self, actual_bpp):
        if self.target_bpp is None:
            return
        if self.num_batches.item() < MIN_BATCHES:
            return
        abs_diff = actual_bpp - self.target_bpp
        rel_diff = abs_diff / self.target_bpp
        if abs(rel_diff) < BPP_REL_TOL:
            return
        sign = 1 if abs_diff < 0 else -1
        gain = LMBDA_GAIN ** sign
        self.lmbda *= gain
