import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import GDN
from compressai.models.utils import conv, deconv

from deep_compression.layers import ResidualPredBlock, create_pred_net
from deep_compression.models.compressai import FactorizedPrior


class ResidualPredFactorizedPrior(FactorizedPrior):
    def __init__(self, N, M, pred_proportion, **kwargs):
        super().__init__(N, M, **kwargs)

        c_x1 = int(M * pred_proportion)
        c_x2 = M - c_x1

        self.pred_net = create_pred_net(c_x1, c_x2)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
            ResidualPredBlock(c_x1, c_x2, self.pred_net, mode="encoder"),
        )

        self.g_s = nn.Sequential(
            ResidualPredBlock(c_x1, c_x2, self.pred_net, mode="decoder"),
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )
