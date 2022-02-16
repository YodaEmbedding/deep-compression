import math

import torch
import torch.nn as nn

from ..layers.utils import channel_covariance


class BatchChannelDecorrelationLoss(nn.Module):
    def __init__(self, lmbda=1e-2, lmbda_corr=1e-4):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.lmbda_corr = lmbda_corr

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        y = output["y"]
        cov_y = channel_covariance(y)
        eye = torch.eye(y.shape[1], dtype=cov_y.dtype, device=cov_y.device)
        off_diagonal_mask = 1 - eye
        cov_off_y = cov_y * off_diagonal_mask
        out["corr_loss"] = (cov_off_y ** 2).sum()

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = (
            self.lmbda * 255 ** 2 * out["mse_loss"]
            + out["bpp_loss"]
            + self.lmbda_corr * out["corr_loss"]
        )

        return out
