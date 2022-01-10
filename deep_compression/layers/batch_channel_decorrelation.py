import torch
import torch.nn as nn

from .utils import channel_covariance


class BatchChannelDecorrelation(nn.Module):
    def __init__(
        self,
        num_features: int,
        momentum_k: float = 0.0,
        momentum_u: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.momentum_k = momentum_k
        self.momentum_u = momentum_u

        kw = {"device": device, "dtype": dtype}
        self.register_buffer("running_k", torch.eye(num_features, **kw))
        self.register_buffer("running_u", torch.eye(num_features, **kw))
        # self.register_buffer("num_batches_tracked", ...)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self._update(x)

        # x = u.T @ x
        # x = torch.einsum("ji, nj... -> ni...", self.running_u, x)
        x = torch.einsum("ji, njyx -> niyx", self.running_u, x)

        return x

    def _update(self, x):
        k_new = channel_covariance(x)

        self.running_k = (
            self.momentum_k * k_new + (1 - self.momentum_k) * self.running_k
        )

        # K = U V U^T
        _, u_new = torch.linalg.eigh(self.running_k)

        self.running_u = (
            self.momentum_u * u_new + (1 - self.momentum_u) * self.running_u
        )


class BatchChannelDecorrelationInverse(nn.Module):
    def forward(self, x, u):
        # x = u @ x
        # x = torch.einsum("ij, nj... -> ni...", u, x)
        x = torch.einsum("ij, njyx -> niyx", u, x)

        return x
