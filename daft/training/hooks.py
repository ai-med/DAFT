# This file is part of Dynamic Affine Feature Map Transform (DAFT).
#
# DAFT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DAFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DAFT. If not, see <https://www.gnu.org/licenses/>.
import numbers
from pathlib import Path
from typing import Dict, Optional, Sequence
from urllib.parse import quote

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.tensorboard.writer import SummaryWriter

from .metrics import Metric


class Hook:
    """Base class for hooks called by :class:`daft.training.train_and_eval.ModelRunner`"""

    def on_begin_epoch(self) -> None:
        """Called before each iteration over the data."""

    def on_end_epoch(self) -> None:
        """Called after the data has been fully consumed."""

    def before_step(self, inputs: Dict[str, Tensor]) -> None:
        """Called before the model is evaluated on a batch.

        Args:
          inputs:
            A Dict with the batch's data obtained from the DataLoader and passed to the model.
        """

    def after_step(self, outputs: Dict[str, Tensor]) -> None:
        """Called fater the model is evaluated on a batch.

        Args:
          outputs:
            A Dict with the outputs returned by the model for the current batch.
        """


class CheckpointSaver(Hook):
    """Saves checkpoints every N epochs.

    Args:
      model (Module):
        The model which state should be saved.
      checkpoint_dir (str):
        Base directory for the checkpoint files.
      metrics (list of Metric|None):
        Instances of metrics to compute. Used for keeping track of best performing model.
      save_every_n_epochs (int):
        Optional; Save every N steps.
      max_keep (int):
        Optional; Keep the latest N checkpoints, or all, if None.
    """

    def __init__(
        self,
        model: Module,
        checkpoint_dir: str,
        save_every_n_epochs: int = 1,
        max_keep: Optional[int] = None,
        metrics: Optional[Sequence[Metric]] = None,
    ) -> None:
        self._model = model
        self._checkpoint_dir = Path(checkpoint_dir)
        self._save_every_n_epochs = save_every_n_epochs
        self._max_keep = max_keep
        self._epoch = 0
        self._ckpkt_remove = []
        self._metrics = metrics
        self._best_metrics = {}

    def _forward(self, fn_name, *args):
        for m in self._metrics:
            fn = getattr(m, fn_name)
            fn(*args)

    def before_step(self, inputs: Dict[str, Tensor]) -> None:
        self._inputs = inputs

    def after_step(self, outputs: Dict[str, Tensor]) -> None:
        if self._metrics is not None:
            self._forward("update", self._inputs, outputs)
        self._inputs = None

    def on_begin_epoch(self) -> None:
        if self._metrics is not None:
            self._forward("reset")

    def on_end_epoch(self) -> None:
        self._epoch += 1
        if self._epoch % self._save_every_n_epochs == 0:
            ckpt_path = self._save()
            if self._max_keep is not None:
                self._remove()
                self._ckpkt_remove.append(ckpt_path)
        if self._metrics is not None:
            self._save_best_models()

    def _save(self):
        path = self._checkpoint_dir / "discriminator_{:04d}.pth".format(self._epoch)
        torch.save(
            self._model.state_dict(), path,
        )

        return path

    def _save_best_models(self):
        for m in self._metrics:
            stats = m.values()
            for name, value in stats.items():
                previous = self._best_metrics.get(name, float("-inf"))
                if (value > previous and not m.lower_is_better) or (value < abs(previous) and m.lower_is_better):
                    self._best_metrics[name] = value
                    safe_name = quote(name, safe="")  # remove slashes and other bad stuff
                    safe_name = safe_name.replace("%2F", "_")
                    path = self._checkpoint_dir / "best_discriminator_{}.pth".format(safe_name)
                    torch.save(
                        self._model.state_dict(), path,
                    )

    def _remove(self):
        if len(self._ckpkt_remove) == self._max_keep:
            self._ckpkt_remove[0].unlink()
            self._ckpkt_remove = self._ckpkt_remove[1:]


class TensorBoardLogger(Hook):
    """Logs metrics after every epoch for visualization in TensorBoard.

    Args:
      log_dir (str):
        The path of the directory where to save the log files to be parsed by TensorBoard.
      metrics (list of Metric):
        Instances of metrics to compute and log.
    """

    def __init__(self, log_dir: str, metrics: Sequence[Metric]) -> None:
        self._writer = SummaryWriter(log_dir)
        self._metrics = metrics
        self._epoch = 0

    def _forward(self, fn_name, *args):
        for m in self._metrics:
            fn = getattr(m, fn_name)
            fn(*args)

    def on_begin_epoch(self) -> None:
        self._forward("reset")

    def on_end_epoch(self) -> None:
        self._epoch += 1
        self._write_all()

    def before_step(self, inputs: Dict[str, Tensor]) -> None:
        self._inputs = inputs

    def after_step(self, outputs: Dict[str, Tensor]) -> None:
        self._forward("update", self._inputs, outputs)
        self._inputs = None

    def _write_all(self):
        for m in self._metrics:
            for name, value in m.values().items():
                self._write(name, value)

    def _write(self, name: str, value):
        if isinstance(value, numbers.Number):
            self._writer.add_scalar(name, value, global_step=self._epoch)
        else:
            self._writer.add_histogram(name, value, global_step=self._epoch)
