import os
from typing import Any

import torch.optim as optim
import yaml
from aim.sdk.utils import generate_run_hash

from deep_compression.losses import (
    BatchChannelDecorrelationLoss,
    RateDistortionLoss,
)


def create_criterion(conf):
    if conf.name == "RateDistortionLoss":
        return RateDistortionLoss(
            lmbda=conf.lambda_,
            target_bpp=conf.get("target_bpp", None),
        )
    if conf.name == "BatchChannelDecorrelationLoss":
        return BatchChannelDecorrelationLoss(
            lmbda=conf.lambda_,
            lmbda_corr=conf.lambda_corr,
            top_k_corr=conf.top_k_corr,
        )
    raise ValueError("Unknown criterion.")


def configure_optimizers(net, conf):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=conf.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=conf.aux_learning_rate,
    )

    return {"net": optimizer, "aux": aux_optimizer}


def configure_logs(logdir: str) -> dict[str, Any]:
    filename = os.path.join(logdir, "info.yaml")
    try:
        with open(filename) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}
        config["run_hash"] = generate_run_hash()
        os.makedirs(logdir, exist_ok=True)
        with open(filename, "w") as f:
            yaml.safe_dump(config, f)
    return config
