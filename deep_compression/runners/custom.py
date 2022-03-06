import torch
from catalyst import dl, metrics
from catalyst.typing import TorchCriterion, TorchOptimizer
from compressai.models.google import CompressionModel

from deep_compression.utils import git
from deep_compression.utils.metrics import compute_metrics
from deep_compression.utils.utils import inference


class CustomRunner(dl.Runner):
    criterion: TorchCriterion
    model: CompressionModel
    optimizer: dict[str, TorchOptimizer]
    metrics: dict[str, metrics.IMetric]

    def __init__(self, *args, config_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._config_path = config_path

    def on_experiment_start(self, runner):
        super().on_experiment_start(runner)
        self._log_git_hparams()
        self._log_git_diff()
        self.log_artifact("config", path_to_artifact=self._config_path)

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
            clip_max_norm = self.hparams["training"]["clip_max_norm"]
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), clip_max_norm
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

        out_infer = inference(self.model, x, skip_decompress=True)
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

    def _log_git_diff(self):
        git_diff_path = "diff.patch"
        with open(git_diff_path, "w") as f:
            f.write(git.diff())
        self.log_artifact("git_diff", path_to_artifact=git_diff_path)

    def _log_git_hparams(self):
        hparams = {}
        hparams["git"] = {
            "hash": git.commit_hash()[:7],
            "main_hash": git.common_ancestor_hash()[:7],
            "branch": git.branch_name(),
        }
        self.log_hparams(hparams)


def _coerce_item(x):
    return x.item() if hasattr(x, "item") else x
