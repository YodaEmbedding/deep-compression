import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..layers.utils import channel_covariance
from .rate_distortion import RateDistortionLoss

BPP_REL_TOL = 0.01
LMBDA_GAIN = 10 ** (1 / 10000)
MIN_BATCHES = 20000


class BatchChannelDecorrelationLoss(RateDistortionLoss):
    def __init__(
        self,
        lmbda=1e-2,
        lmbda_corr=1e-4,
        top_k_corr=None,
        target_bpp: Optional[float] = None,
    ):
        super().__init__(lmbda=lmbda, target_bpp=target_bpp)
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
        y_top_k = y[:, idx[: self.top_k_corr]]
        out["corr_loss"] = channel_correlation_loss(y_top_k)

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        self._compute_lmbda(actual_bpp=out["bpp_loss"].item())
        lmbda = self.lmbda[0]
        out["loss"] = (
            lmbda * 255 ** 2 * out["mse_loss"]
            + out["bpp_loss"]
            + self.lmbda_corr * out["corr_loss"]
        )

        return out


def channel_rates(y: np.ndarray, method="range") -> np.ndarray:
    rates_batch = np.stack([channel_rates_single(x, method=method) for x in y])
    return rates_batch.sum(axis=0)


def channel_rates_single(y: np.ndarray, method="range") -> np.ndarray:
    c, _, _ = y.shape
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
    _, c, _, _ = y.shape
    cov_y = channel_covariance(y)
    eye = torch.eye(c, dtype=cov_y.dtype, device=cov_y.device)
    off_diagonal_mask = 1 - eye
    cov_off_y = cov_y * off_diagonal_mask
    return (cov_off_y ** 2).sum()
