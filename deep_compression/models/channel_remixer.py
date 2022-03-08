import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_compression.layers import SEBlock
from deep_compression.models.compressai import FactorizedPrior


class ChannelRemixerFactorizedPrior(FactorizedPrior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)
        self.remixer = SEBlock(num_channels=M)
        self.demixer = SEBlock(num_channels=M)

    def forward(self, x):
        y = self.g_a(x)
        y = self.remixer(y)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        y_hat = self.demixer(y_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x": x,
            "y": y,
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    def compress(self, x):
        y = self.g_a(x)
        y = self.remixer(y)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        y_hat = self.demixer(y_hat)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
