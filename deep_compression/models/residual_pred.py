import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers import GDN
from compressai.models.utils import conv, deconv

from deep_compression.layers import ResidualPredBlock
from deep_compression.models.compressai import FactorizedPrior


class ResidualPredFactorizedPrior(FactorizedPrior):
    def __init__(self, N, M, pred_proportion, **kwargs):
        super().__init__(N, M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
            ResidualPredBlock(M, pred_proportion=pred_proportion),
        )

        self.g_s = nn.Sequential(
            ResidualPredBlock(M, pred_proportion=pred_proportion),
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )
