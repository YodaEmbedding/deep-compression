import compressai.zoo.image as cai_zoo_img


def create_model_factory(model_name, base_model_name):
    def model_factory(*args, **kwargs):
        return models[base_model_name](model_name, *args, **kwargs)

    return model_factory


def bmshj2018_factorized(
    architecture,
    quality,
    metric="mse",
    pretrained=False,
    progress=True,
    **kwargs,
):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(
            f'Invalid quality "{quality}", should be between (1, 8)'
        )

    return cai_zoo_img._load_model(
        architecture, metric, quality, pretrained, progress, **kwargs
    )


def bmshj2018_hyperprior(
    architecture,
    quality,
    metric="mse",
    pretrained=False,
    progress=True,
    **kwargs,
):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(
            f'Invalid quality "{quality}", should be between (1, 8)'
        )

    return cai_zoo_img._load_model(
        architecture, metric, quality, pretrained, progress, **kwargs
    )


def mbt2018_mean(
    architecture,
    quality,
    metric="mse",
    pretrained=False,
    progress=True,
    **kwargs,
):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(
            f'Invalid quality "{quality}", should be between (1, 8)'
        )

    return cai_zoo_img._load_model(
        architecture, metric, quality, pretrained, progress, **kwargs
    )


def mbt2018(
    architecture,
    quality,
    metric="mse",
    pretrained=False,
    progress=True,
    **kwargs,
):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(
            f'Invalid quality "{quality}", should be between (1, 8)'
        )

    return cai_zoo_img._load_model(
        architecture, metric, quality, pretrained, progress, **kwargs
    )


def cheng2020_anchor(
    architecture,
    quality,
    metric="mse",
    pretrained=False,
    progress=True,
    **kwargs,
):
    r"""Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        quality (int): Quality levels (1: lowest, highest: 6)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 6:
        raise ValueError(
            f'Invalid quality "{quality}", should be between (1, 6)'
        )

    return cai_zoo_img._load_model(
        architecture, metric, quality, pretrained, progress, **kwargs
    )


def cheng2020_attn(
    architecture,
    quality,
    metric="mse",
    pretrained=False,
    progress=True,
    **kwargs,
):
    r"""Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        quality (int): Quality levels (1: lowest, highest: 6)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 6:
        raise ValueError(
            f'Invalid quality "{quality}", should be between (1, 6)'
        )

    return cai_zoo_img._load_model(
        architecture, metric, quality, pretrained, progress, **kwargs
    )


models = {
    "bmshj2018-factorized": bmshj2018_factorized,
    "bmshj2018-hyperprior": bmshj2018_hyperprior,
    "mbt2018-mean": mbt2018_mean,
    "mbt2018": mbt2018,
    "cheng2020-anchor": cheng2020_anchor,
    "cheng2020-attn": cheng2020_attn,
}
