import torch
import torch.nn as nn


class ChannelRate(nn.Module):
    rates: torch.Tensor

    def __init__(
        self,
        num_features: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        kw = {"device": device, "dtype": dtype}
        self.rates = nn.Parameter(torch.zeros(num_features, **kw))

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

    def forward(self, x: torch.Tensor):
        return x * self.rates.reshape(-1, 1, 1)


def create_pair(
    num_features: int,
    device=None,
    dtype=None,
) -> tuple[ChannelRate, ChannelRateInverse]:
    channel_rate = ChannelRate(
        num_features=num_features,
        device=device,
        dtype=dtype,
    )

    channel_rate_inv = ChannelRateInverse(
        rates=channel_rate.rates,
    )

    return channel_rate, channel_rate_inv
