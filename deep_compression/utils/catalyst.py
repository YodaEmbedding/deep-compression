import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Union

import catalyst
from catalyst.callbacks.checkpoint import (
    CheckpointCallback,
    ICheckpointCallback,
    _default_states,
    _get_required_files,
    _load_checkpoint,
    _load_runner,
    _load_states_from_file_map,
    _save_checkpoint,
)
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.extras.metric_handler import MetricHandler
from catalyst.utils.config import save_config


class EveryCheckpointCallback(CheckpointCallback):
    """CheckpointCallback, but also saves every checkpoint.

    Main changes:
     - `suffix = f"{runner.global_epoch_step:04}..."`
     - `_truncate_checkpoints`
    """

    def _save_checkpoint(
        self, runner: IRunner, checkpoint: Dict, is_best: bool, is_last: bool
    ) -> str:
        """
        Saves checkpoints: full with model/criterion/optimizer/scheduler
        and truncated with model only.

        Args:
            runner: current runner.
            checkpoint: data to save.
            is_best: if ``True`` then also will be generated best checkpoint file.
            is_last: if ``True`` then also will be generated last checkpoint file.

        Returns:
            path to saved checkpoint
        """
        logdir = Path(f"{self.logdir}/")
        suffix = f"{runner.global_epoch_step:04}.{runner.stage_key}.{runner.stage_epoch_step}"
        checkpoint_path = None

        if self.mode in ("all", "full"):
            checkpoint_path = _save_checkpoint(
                runner=runner,
                logdir=logdir,
                checkpoint=checkpoint,
                suffix=f"{suffix}_full",
                is_best=is_best,
                is_last=is_last,
                extra_suffix="_full",
            )
        if self.mode in ("all", "model"):
            exclude = ["criterion", "optimizer", "scheduler"]
            checkpoint_path = _save_checkpoint(
                runner=runner,
                checkpoint={
                    key: value
                    for key, value in checkpoint.items()
                    if all(z not in key for z in exclude)
                },
                logdir=logdir,
                suffix=suffix,
                is_best=is_best,
                is_last=is_last,
            )
        return checkpoint_path

    def _truncate_checkpoints(self) -> None:
        pass
