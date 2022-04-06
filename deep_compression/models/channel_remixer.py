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

from deep_compression.layers import SEBlock
from deep_compression.models.compressai import (
    Cheng2020Anchor,
    FactorizedPrior,
    JointAutoregressiveHierarchicalPriors,
)
from deep_compression.utils import register_model


@register_model("bmshj2018-factorized-chan-remixer")
class ChannelRemixerFactorizedPrior(FactorizedPrior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)
        self.remixer = SEBlock(num_channels=M)
        self.demixer = SEBlock(num_channels=M)

    def forward(self, x):
        y_pre = self.g_a(x)
        y = self.remixer(y_pre)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        y_hat_post = self.demixer(y_hat)
        x_hat = self.g_s(y_hat_post)

        return {
            "x": x,
            "y_pre": y_pre,
            "y": y,
            "x_hat": x_hat,
            "y_hat": y_hat,
            "y_hat_post": y_hat_post,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    def compress(self, x):
        y_pre = self.g_a(x)
        y = self.remixer(y_pre)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        y_hat_post = self.demixer(y_hat)
        x_hat = self.g_s(y_hat_post).clamp_(0, 1)
        return {"x_hat": x_hat}


@register_model("mbt2018-chan-remixer")
class ChannelRemixerJointAutoregressiveHierarchicalPriors(
    JointAutoregressiveHierarchicalPriors
):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
            SEBlock(num_channels=M),
        )

        self.g_s = nn.Sequential(
            SEBlock(num_channels=M),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )


@register_model("cheng2020-anchor-chan-remixer")
class ChannelRemixerCheng2020Anchor(Cheng2020Anchor):
    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            SEBlock(num_channels=N),
        )

        self.g_s = nn.Sequential(
            SEBlock(num_channels=N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )
