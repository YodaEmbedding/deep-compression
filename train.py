import argparse
import os
import sys

import aim
import catalyst
import catalyst.utils
import torch
import torch.optim as optim
from catalyst import dl
from omegaconf import OmegaConf

import deep_compression
from deep_compression.config import (
    configure_logs,
    configure_optimizers,
    create_criterion,
)
from deep_compression.datasets.utils import (
    get_data_transforms,
    get_dataloaders,
    get_datasets,
)
from deep_compression.runners import CustomRunner
from deep_compression.utils.catalyst import AimLogger
from deep_compression.zoo import model_architectures


def setup(conf: OmegaConf):
    catalyst.utils.set_global_seed(conf.hp.experiment.seed)
    catalyst.utils.prepare_cudnn(benchmark=True)

    data_transforms = get_data_transforms(conf.hp.data)
    datasets = get_datasets(conf, data_transforms)
    loaders = get_dataloaders(conf.hp.data, conf.device, datasets)

    model = model_architectures[conf.hp.model](**conf.hp.model_hp)
    model = model.to(conf.device)

    criterion = create_criterion(conf.hp.criterion)
    optimizer = configure_optimizers(model, conf.hp.optimizer)
    scheduler = {
        "net": optim.lr_scheduler.ReduceLROnPlateau(optimizer["net"], "min"),
    }

    checkpoint_path = (
        None
        if not conf.resume
        else conf.checkpoint
        if conf.checkpoint is not None
        else os.path.join(conf.logdir, "checkpoints", "runner.last.pth")
    )

    return (
        model,
        loaders,
        criterion,
        optimizer,
        scheduler,
        checkpoint_path,
    )


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
    conf = OmegaConf.merge(conf, vars(args))
    conf.device = "cuda" if torch.cuda.is_available() else "cpu"

    (
        model,
        loaders,
        criterion,
        optimizer,
        scheduler,
        checkpoint_path,
    ) = setup(conf)

    log_config = configure_logs(conf.logdir)

    runner = CustomRunner(
        config_path=conf.config,
    )

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=conf.logdir,
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
                logdir=os.path.join(conf.logdir, "checkpoints"),
                loader_key="valid",
                metric_key="loss",
                minimize=True,
                mode="runner",
                resume_runner=checkpoint_path,
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
                logdir=os.path.join(conf.logdir, "tensorboard"),
            ),
            "aim": AimLogger(
                experiment=conf.experiment,
                run_hash=log_config["run_hash"],
                repo=aim.Repo(
                    conf.aim_repo,
                    init=not aim.Repo.exists(conf.aim_repo),
                ),
            ),
        },
        check=conf.hp.experiment.get("check", False),
    )


if __name__ == "__main__":
    main()
