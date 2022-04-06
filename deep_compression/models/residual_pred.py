# fmt: off

import torch.nn as nn
from compressai.layers import (
    GDN,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.models.utils import conv, deconv

from deep_compression.layers import ResidualPredBlock, create_pred_net
from deep_compression.models.compressai import (
    Cheng2020Anchor,
    FactorizedPrior,
    JointAutoregressiveHierarchicalPriors,
)
from deep_compression.utils import register_model


@register_model("bmshj2018-factorized-residual-pred")
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


@register_model("mbt2018-residual-pred")
class ResidualPredJointAutoregressiveHierarchicalPriors(
    JointAutoregressiveHierarchicalPriors
):
    def __init__(self, N=192, M=192, pred_proportion=0.5, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        c_x1 = int(M * pred_proportion)
        c_x2 = M - c_x1

        self.pred_net = create_pred_net(c_x1, c_x2)

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
            ResidualPredBlock(c_x1, c_x2, self.pred_net, mode="encoder"),
        )

        self.g_s = nn.Sequential(
            ResidualPredBlock(c_x1, c_x2, self.pred_net, mode="decoder"),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )


@register_model("cheng2020-anchor-residual-pred")
class ResidualPredCheng2020Anchor(Cheng2020Anchor):
    def __init__(self, N=192, pred_proportion=0.5, **kwargs):
        super().__init__(N=N, **kwargs)

        M = N
        c_x1 = int(M * pred_proportion)
        c_x2 = M - c_x1

        self.pred_net = create_pred_net(c_x1, c_x2)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            ResidualPredBlock(c_x1, c_x2, self.pred_net, mode="encoder"),
        )

        self.g_s = nn.Sequential(
            ResidualPredBlock(c_x1, c_x2, self.pred_net, mode="decoder"),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )
