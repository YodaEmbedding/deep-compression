import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_compression.layers import SEBlock
from deep_compression.models.compressai import FactorizedPrior
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
