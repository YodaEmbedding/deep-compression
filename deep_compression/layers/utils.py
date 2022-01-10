import torch


def channel_covariance(x: torch.Tensor) -> torch.Tensor:
    r"""Computes covariance of by-channel dot products over many samples.

    ```
    cov[X_i, X_j]
      = E[(X_i - E[X_i]) \cdot (X_j - E[X_j])]
      = E[B_i \cdot B_j]
      = (1/N) \sum_k (b_i^k \cdot b_j^k)

    B_i
      = X_i - E[X_i]
      = X_i - (1/N) \sum_k x_i^k
    ```

    where:

    - `x_i^k` is a vector representing the ith channel of the kth sample of x.
    - `X_i` is a random vector over the sample set `{x_i^1, ..., x_i^N}`.
    - The resulting covariance matrix is of shape CxC.
    """
    n, *_ = x.shape
    mu_x = torch.mean(x, dim=0, keepdim=True)
    x_centered = x - mu_x
    cov = torch.einsum("kiyx, kjyx -> ij", x_centered, x_centered) / n
    return cov
