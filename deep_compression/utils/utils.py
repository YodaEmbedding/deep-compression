from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from numpy import linalg as LA


class ActivationHook:
    def __init__(self):
        self._activation = {}

    def __getitem__(self, key):
        return self._activation[key]

    def of(self, name):
        def hook(model, input, output):
            self._activation[name] = output.detach()

        return hook

    def for_model(self, model, module_names):
        for name in module_names:
            getattr(model, name).register_forward_hook(self.of(name))


def preprocess_img(img):
    x = (img.transpose(2, 0, 1) / 256).astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0)
    return x


# TODO * 255 or * 256?
def postprocess_img(img):
    x = (img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return x


def compute_psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return -10 * np.log10(mse / 255 ** 2)


def normalize_tensor(x, axis=(-1, -2)):
    mu = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    z = (x - mu) / std
    return z


def normal_clip(x, sigma=4):
    std = x.std(axis=(1, 2))[:, np.newaxis, np.newaxis]
    z = x.clip(-sigma * std, sigma * std)
    return z


def correlate_all(frame: np.ndarray) -> np.ndarray:
    r"""Computes matrix of Pearson correlation coefficients between channels.

    Each channel x^i represents a sample
    `\{ x^i_{km} : 0 < k < H, 0 < m < W \}`.
    We compute the sample Pearson correlation coefficient between
    x^i and x^j by writing the ordered elements of these as pairs,
    normalizing them w.r.t. their respective sample means and variances,
    taking their pairwise products, and then summing them.
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample

    Concretely:

    ```
    corr[X_i, X_j] = Z_i \cdot Z_j

    X_i = (x^i_{11}, ..., x^i_{HW})

    (Z_i)_k = ((X_i)_k - \mu_i) / \sigma_i

    \mu_i = \frac{1}{HW} \sum_k (X_i)_k

    \sigma_i = \sqrt{ \frac{1}{HW} \sum_k ((X_i)_k - \mu_i)^2 }
    ```
    """
    # Reshape each channel into vector.
    c, *_ = frame.shape
    frame = frame.reshape(c, -1)
    _, channel_size = frame.shape

    # We will be comparing all channels against each other.
    x = frame

    # Compute mean and std.
    mu_x = x.mean(axis=1)
    sig_x = x.std(axis=1)

    # Measure similarity between channels.
    # We do this by computing the Pearson correlation coefficients.
    # Resulting `corr` shape is (c, c).
    x_centered = x - mu_x[:, np.newaxis]
    num = np.einsum("ik, jk -> ij", x_centered, x_centered)
    den = np.einsum("i, j -> ij", sig_x, sig_x)
    corr = (num / den) / channel_size

    # Mask out NaN values, which may happen if a channel is uniformly blank.
    corr[np.isnan(corr)] = 0

    # Warnings.
    corr_non_inf = corr[~np.isinf(corr)]
    if (np.abs(corr_non_inf) > 1.01).any():
        corr_interval = f"[{corr_non_inf.min()}, {corr_non_inf.max()}]"
        print(f">>> WARNING: corr > 1.  corr range: {corr_interval}")

    return corr


def channel_covariance(x: np.ndarray) -> np.ndarray:
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
    mu_x = x.mean(axis=0, keepdims=True)
    x_centered = x - mu_x
    cov = np.einsum("kiyx, kjyx -> ij", x_centered, x_centered) / n
    return cov


def channel_covariance_torch(x: torch.Tensor) -> torch.Tensor:
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
    mu_x = x.mean(dim=0, keepdim=True)
    x_centered = x - mu_x
    cov = torch.einsum("kiyx, kjyx -> ij", x_centered, x_centered) / n
    return cov


def decorrelation_matrix(
    cov: np.ndarray, mode: str = "simple", ignore: Optional[np.ndarray] = None
) -> np.ndarray:
    if mode == "simple":
        _, u = LA.eig(cov)
        return u

    if mode == "ignore":
        if ignore is None:
            return decorrelation_matrix(cov)

        c = cov.shape[0]
        keep = np.ones(c, dtype=bool)
        keep[ignore] = False
        keep = np.arange(c)[keep]

        cov_ = cov[np.ix_(keep, keep)]
        _, u_ = LA.eig(cov_)

        u = np.eye(c, dtype=u_.dtype)
        u[np.ix_(keep, keep)] = u_

        return u

    raise ValueError(f"Unknown mode {mode}.")


@torch.no_grad()
def inference_single_image_uint8(
    model, img: np.ndarray
) -> tuple[np.ndarray, list[bytes]]:
    x = preprocess_img(img)

    enc_dict = model.compress(x)
    encoded = [x[0] for x in enc_dict["strings"]]
    dec_dict = model.decompress(**enc_dict)

    x_hat = dec_dict["x_hat"].numpy()[0]
    img_rec = postprocess_img(x_hat)

    return img_rec, encoded


@torch.no_grad()
def inference(model, x: torch.Tensor, skip_decompress=True):
    """Run compression model on image batch."""
    n, _, h, w = x.shape
    pad, unpad = _get_pad(h, w)

    x_padded = F.pad(x, pad, mode="constant", value=0)
    out_enc = model.compress(x_padded)
    out_dec = (
        model(x_padded)
        if skip_decompress
        else model.decompress(out_enc["strings"], out_enc["shape"])
    )
    out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)

    num_pixels = n * h * w
    num_bits = sum(len(s[0]) for s in out_enc["strings"]) * 8.0
    bpp = num_bits / num_pixels

    return {
        "out_enc": out_enc,
        "out_dec": out_dec,
        "bpp": bpp,
    }


def _get_pad(h, w):
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    pad = (padding_left, padding_right, padding_top, padding_bottom)
    unpad = (-padding_left, -padding_right, -padding_top, -padding_bottom)
    return pad, unpad
