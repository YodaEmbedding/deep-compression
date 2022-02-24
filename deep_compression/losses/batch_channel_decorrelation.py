import math

import numpy as np
import torch
import torch.nn as nn

from ..layers.utils import channel_covariance


class BatchChannelDecorrelationLoss(nn.Module):
    def __init__(self, lmbda=1e-2, lmbda_corr=1e-4, top_k_corr=None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.lmbda_corr = lmbda_corr
        self.top_k_corr = top_k_corr

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        # Compute decorrelating loss term for top-k channels by rate.
        y = output["y"]
        rates = channel_rates(y.detach().cpu().numpy())
        idx = rates.argsort()[::-1].copy()
        y_top_k = y[idx[:self.top_k_corr]]
        out["corr_loss"] = channel_correlation_loss(y_top_k)

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


def channel_rates(y: np.ndarray, method="range") -> np.ndarray:
    c, *_ = y.shape
    y = y.reshape(c, -1)
    y = y.round()
    y = y.astype(np.int32)
    if method == "range":
        return y.max(axis=1) - y.min(axis=1)
    if method == "entropy":
        return np.array([entropy(x) for x in y])
    raise ValueError(f"Unknown method {method}.")


def entropy(labels):
    ps = np.bincount(labels) / len(labels)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def channel_correlation_loss(y):
    cov_y = channel_covariance(y)
    eye = torch.eye(y.shape[1], dtype=cov_y.dtype, device=cov_y.device)
    off_diagonal_mask = 1 - eye
    cov_off_y = cov_y * off_diagonal_mask
    return (cov_off_y ** 2).sum()
