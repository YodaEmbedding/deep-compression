import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..layers.utils import channel_covariance


BPP_REL_TOL = 0.01
LMBDA_GAIN = 10 ** (1 / 10000)
MIN_BATCHES = 20000


class BatchChannelDecorrelationLoss(nn.Module):
    lmbda: torch.Tensor
    num_batches: torch.Tensor

    def __init__(
        self,
        lmbda=1e-2,
        lmbda_corr=1e-4,
        top_k_corr=None,
        target_bpp: Optional[float] = None,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.target_bpp = target_bpp
        self.register_buffer("lmbda", torch.Tensor([lmbda]))
        num_batches = torch.tensor([0], dtype=torch.int64)
        self.register_buffer("num_batches", num_batches)
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

        self.num_batches += 1

        return out

    def _compute_lmbda(self, actual_bpp):
        if not self.training:
            return
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
