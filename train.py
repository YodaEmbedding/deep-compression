import argparse
import os
import sys

import catalyst
import catalyst.utils
import torch
import torch.optim as optim
from catalyst import dl, metrics
from catalyst.typing import Criterion, Optimizer
from compressai.datasets import ImageFolder
from compressai.models.google import CompressionModel
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms

import deep_compression
from deep_compression.losses import (
    BatchChannelDecorrelationLoss,
    RateDistortionLoss,
)
from deep_compression.utils.catalyst import EveryCheckpointCallback
from deep_compression.utils.metrics import compute_metrics
from deep_compression.utils.utils import inference
from deep_compression.zoo import model_architectures


class CustomRunner(dl.Runner):
    criterion: Criterion
    model: CompressionModel
    optimizer: dict[str, Optimizer]
    metrics: dict[str, metrics.IMetric]

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        keys = ["loss", "aux_loss", "bpp_loss", "mse_loss"]
        if self.is_infer_loader:
            keys += ["psnr", "ms-ssim"]
            keys += ["bpp"]
            self.model.update()
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False) for key in keys
        }

    def handle_batch(self, batch):
        if self.is_infer_loader:
            return self.predict_batch(batch)

        x = batch

        out_net = self.model(x)
        out_criterion = self.criterion(out_net, x)

        loss = out_criterion["loss"]

        if self.is_train_loader:
            loss.backward()
            if self.hparams["clip_max_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.hparams["clip_max_norm"]
                )
            self.optimizer["net"].step()

        aux_loss = self.model.aux_loss()

        if self.is_train_loader:
            aux_loss.backward()
            self.optimizer["aux"].step()

        if self.is_train_loader:
            self.optimizer["net"].zero_grad()
            self.optimizer["aux"].zero_grad()

        d = {"loss": loss, "aux_loss": aux_loss, **out_criterion}
        self.batch_metrics.update(d)

        for key in self.meters.keys():
            self.meters[key].update(
                _coerce_item(self.batch_metrics[key]),
                self.batch_size,
            )

    def predict_batch(self, batch):
        x = batch.to(self.engine.device)

        out_infer = inference(self.model, x)
        out_net = out_infer["out_net"]
        out_criterion = self.criterion(out_net, x)
        out_metrics = compute_metrics(x, out_net["x_hat"], ["psnr", "ms-ssim"])

        loss = out_criterion["loss"]
        aux_loss = self.model.aux_loss()

        d = {
            "loss": loss,
            "aux_loss": aux_loss,
            **out_criterion,
            **out_metrics,
            "bpp": out_infer["bpp"],
        }

        self.batch_metrics.update(d)

        for key in self.meters.keys():
            self.meters[key].update(
                _coerce_item(self.batch_metrics[key]),
                self.batch_size,
            )

    def on_loader_end(self, runner):
        for key in self.meters.keys():
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)


def _coerce_item(x):
    return x.item() if hasattr(x, "item") else x


def create_criterion(conf):
    if conf.name == "RateDistortionLoss":
        return RateDistortionLoss(lmbda=conf.lambda_)
    if conf.name == "BatchChannelDecorrelationLoss":
        return BatchChannelDecorrelationLoss(
            lmbda=conf.lambda_, lmbda_corr=conf.lambda_corr
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
    seed = conf.hparams.experiment.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"

    catalyst.utils.set_global_seed(seed)
    catalyst.utils.prepare_cudnn(benchmark=True)

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomCrop(conf.data.patch_size),
                transforms.ToTensor(),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.CenterCrop(conf.data.patch_size),
                transforms.ToTensor(),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    }

    datasets = {
        "train": ImageFolder(
            conf.dataset, split="train", transform=data_transforms["train"]
        ),
        "valid": ImageFolder(
            conf.dataset, split="valid", transform=data_transforms["valid"]
        ),
        "test": ImageFolder(
            conf.dataset, split="test", transform=data_transforms["test"]
        ),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=conf.data.batch_size,
            num_workers=conf.data.num_workers,
            shuffle=True,
            pin_memory=(device == "cuda"),
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=conf.data.test_batch_size,
            num_workers=conf.data.num_workers,
            shuffle=False,
            pin_memory=(device == "cuda"),
        ),
        "infer": DataLoader(
            datasets["test"],
            batch_size=1,
            num_workers=conf.data.num_workers,
            shuffle=False,
            pin_memory=(device == "cuda"),
        ),
    }

    model = model_architectures[conf.model](**conf.hparams.model)
    model = model.to(device)

    criterion = create_criterion(conf.hparams.criterion)
    optimizer = configure_optimizers(model, conf.hparams.optimizer)

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
        else "last_full"
    )

    scheduler = {
        "net": optim.lr_scheduler.ReduceLROnPlateau(optimizer["net"], "min"),
    }

    runner = CustomRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        hparams=dict(conf.hparams.training),
        num_epochs=conf.hparams.experiment.epochs,
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
            EveryCheckpointCallback(
                logdir=os.path.join(logdir, "checkpoints_all"),
                loader_key="valid",
                metric_key="loss",
                minimize=True,
                mode="full",
                resume=resume,
                # save_n_best=1,
            ),
            dl.CheckpointCallback(
                logdir=os.path.join(logdir, "checkpoints"),
                loader_key="valid",
                metric_key="loss",
                minimize=True,
                mode="full",
                resume=resume,
                # save_n_best=1,
            ),
            dl.EarlyStoppingCallback(
                patience=7,
                loader_key="valid",
                metric_key="loss",
                minimize=True,
            ),
        ],
        loggers={
            "tensorboard": dl.TensorboardLogger(
                logdir=os.path.join(logdir, "tensorboard"),
            ),
        },
        # check=True,
    )


if __name__ == "__main__":
    main()
