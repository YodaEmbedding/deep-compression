import argparse
import os
import sys
from typing import Any

import aim
import catalyst
import catalyst.utils
import torch
import torch.optim as optim
import yaml
from aim.sdk.utils import generate_run_hash
from catalyst import dl
from omegaconf import OmegaConf

import deep_compression
from deep_compression.datasets.utils import (
    get_data_transforms,
    get_dataloaders,
    get_datasets,
)
from deep_compression.losses import (
    BatchChannelDecorrelationLoss,
    RateDistortionLoss,
)
from deep_compression.runners import CustomRunner
from deep_compression.utils.catalyst import AimLogger
from deep_compression.zoo import model_architectures


def create_criterion(conf):
    if conf.name == "RateDistortionLoss":
        return RateDistortionLoss(lmbda=conf.lambda_)
    if conf.name == "BatchChannelDecorrelationLoss":
        return BatchChannelDecorrelationLoss(
            lmbda=conf.lambda_,
            lmbda_corr=conf.lambda_corr,
            top_k_corr=conf.top_k_corr,
        )
    raise ValueError("Unknown criterion.")


def configure_optimizers(net, args):
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
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
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


def build_args(parser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs",
        help="Path to logdir",
    )
    parser.add_argument(
        "--aim_repo",
        type=str,
        default=".",
        help="Path to directory containing .aim repo",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a checkpoint",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training",
    )
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example training script.")
    build_args(parser)
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)
    conf = OmegaConf.load(args.config)
    logdir = args.logdir
    seed = conf.hp.experiment.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"

    catalyst.utils.set_global_seed(seed)
    catalyst.utils.prepare_cudnn(benchmark=True)

    data_transforms = get_data_transforms(conf.hp.data)
    datasets = get_datasets(conf, data_transforms)
    loaders = get_dataloaders(conf.hp.data, device, datasets)

    model = model_architectures[conf.hp.model](**conf.hp.model_params)
    model = model.to(device)

    criterion = create_criterion(conf.hp.criterion)
    optimizer = configure_optimizers(model, conf.hp.optimizer)

    # if args.checkpoint is not None:
    #     checkpoint = catalyst.utils.load_checkpoint(path=args.checkpoint)
    #     catalyst.utils.unpack_checkpoint(
    #         checkpoint=checkpoint,
    #         model=model,
    #         optimizer=optimizer,
    #         criterion=criterion,
    #     )

    resume = (
        None
        if not args.resume
        else args.checkpoint
        if args.checkpoint is not None
        else os.path.join(logdir, "checkpoints", "runner.last.pth")
    )

    log_config = configure_logs(logdir)

    scheduler = {
        "net": optim.lr_scheduler.ReduceLROnPlateau(optimizer["net"], "min"),
    }

    runner = CustomRunner(
        config_path=args.config,
    )

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        hparams=OmegaConf.to_container(conf.hp),
        num_epochs=conf.hp.experiment.epochs,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
        callbacks=[
            dl.SchedulerCallback(
                scheduler_key="net",
                mode="epoch",
                loader_key="valid",
                metric_key="loss",
            ),
            dl.CheckpointCallback(
                logdir=os.path.join(logdir, "checkpoints"),
                loader_key="valid",
                metric_key="loss",
                minimize=True,
                mode="runner",
                resume_runner=resume,
                topk=10000,
            ),
            dl.EarlyStoppingCallback(
                patience=conf.hp.experiment.patience,
                loader_key="valid",
                metric_key="loss",
                minimize=True,
            ),
        ],
        loggers={
            "tensorboard": dl.TensorboardLogger(
                logdir=os.path.join(logdir, "tensorboard"),
            ),
            "mlflow": dl.MLflowLogger(
                experiment=conf.experiment,
                run=conf.run,
            ),
            "aim": AimLogger(
                experiment=conf.experiment,
                run_hash=log_config["run_hash"],
                repo=aim.Repo(
                    args.aim_repo,
                    init=not aim.Repo.exists(args.aim_repo),
                ),
            ),
        },
        check=conf.hp.experiment.get("check", False),
    )


if __name__ == "__main__":
    main()
