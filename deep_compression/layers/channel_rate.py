from typing import Optional

import torch
import torch.nn as nn


class ChannelRate(nn.Module):
    rates: torch.Tensor

    def __init__(
        self,
        num_channels: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        kw = {"device": device, "dtype": dtype}
        self.rates = nn.Parameter(torch.ones(num_channels, **kw))

    def forward(self, x):
        return x / self.rates.reshape(-1, 1, 1)


class ChannelRateInverse(nn.Module):
    rates: torch.Tensor

    def __init__(
        self,
        rates: torch.Tensor,
    ):
        super().__init__()

        self.rates = rates

    def forward(self, x: torch.Tensor, rates: Optional[torch.Tensor] = None):
        if rates is None:
            rates = self.rates
        return x * rates.reshape(-1, 1, 1)


def create_pair(
    num_channels: int,
    device=None,
    dtype=None,
) -> tuple[ChannelRate, ChannelRateInverse]:
    channel_rate = ChannelRate(
        num_channels=num_channels,
        device=device,
        dtype=dtype,
    )

    channel_rate_inv = ChannelRateInverse(
        rates=channel_rate.rates,
    )

    return channel_rate, channel_rate_inv
