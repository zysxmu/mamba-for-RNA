"""Callback to monitor the speed of each step and each epoch.

https://github.com/HazyResearch/transformers/blob/master/src/callbacks/speed_monitor.py
Adapted from:
    https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/gpu_stats_monitor.html#GPUStatsMonitor
"""

import json
import time
from typing import Any

import torch
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.types import STEP_OUTPUT


class Timer(Callback):
    """Monitor the speed of each step and each epoch.
    """
    def __init__(
        self,
        step: bool = True,
        inter_step: bool = True,
        epoch: bool = True,
        val: bool = True,
    ):
        super().__init__()
        self._log_stats = AttributeDict( {
            'step_time': step,
            'inter_step_time': inter_step,
            'epoch_time': epoch,
            'val_time': val,
        })

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._snap_epoch_time = None
        self._step_time_total = 0.0
        self._tokens_total = 0
        self._timed_steps = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._snap_step_time = None
        self._snap_inter_step_time = None
        self._snap_epoch_time = time.time()

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._log_stats.step_time:
            self._snap_step_time = time.time()

        if not self._should_log(trainer):
            return

        logs = {}
        if self._log_stats.inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            logs["timer/inter_step"] = (time.time() - self._snap_inter_step_time) # * 1000

        if trainer.logger: trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._log_stats.inter_step_time:
            self._snap_inter_step_time = time.time()

        logs = {}
        if self._log_stats.step_time and self._snap_step_time:
            elapsed = time.time() - self._snap_step_time
            self._step_time_total += elapsed
            self._timed_steps += 1
            logs["timer/step"] = elapsed
            if isinstance(batch, (tuple, list)) and torch.is_tensor(batch[0]):
                processed_tokens = batch[0].numel() * max(1, trainer.world_size)
                self._tokens_total += processed_tokens
                logs["timer/tokens_per_second"] = processed_tokens / max(elapsed, 1e-12)
            if torch.cuda.is_available():
                logs["timer/peak_memory_gib"] = (
                    torch.cuda.max_memory_allocated() / (1024 ** 3)
                )

        if not self._should_log(trainer):
            return
        if trainer.logger: trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule,) -> None:
        logs = {}
        if self._log_stats.epoch_time and self._snap_epoch_time:
            logs["timer/epoch"] = time.time() - self._snap_epoch_time
        if trainer.logger: trainer.logger.log_metrics(logs, step=trainer.global_step)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._snap_val_time = time.time()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule,) -> None:
        logs = {}
        if self._log_stats.val_time and self._snap_val_time:
            logs["timer/validation"] = time.time() - self._snap_val_time
        if trainer.logger: trainer.logger.log_metrics(logs) # , step=trainer.global_step)

    @rank_zero_only
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        elapsed = max(self._step_time_total, 1e-12)
        metrics = {
            "timed_steps": self._timed_steps,
            "mean_step_time_seconds": (
                self._step_time_total / max(1, self._timed_steps)
            ),
            "processed_tokens": self._tokens_total,
            "mean_tokens_per_second": self._tokens_total / elapsed,
            "peak_memory_gib": (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
                if torch.cuda.is_available()
                else 0.0
            ),
        }
        with open("runtime_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    @staticmethod
    def _should_log(trainer) -> bool:
        return (trainer.global_step + 1) % trainer.log_every_n_steps == 0 or trainer.should_stop
