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
from typing import Optional, Sequence

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from ..models.base import BaseModel


class LossWrapper(BaseModel):
    """Wraps an existing torch Module by given inputs and outputs names.

    Args:
      loss (Module):
        Instance of module to wrap.
      input_names (list of str):
        Names of inputs in the order expected by `loss.forward`.
        Names do not need to match argument names, just their position.
      output_names (list of str):
        Names of outputs returned by `loss.forward`.
    """

    def __init__(
        self,
        loss: Module,
        input_names: Sequence[str],
        output_names: Optional[Sequence[str]] = None,
        binary: bool = False,
    ) -> None:
        self._input_names = tuple(input_names)
        if output_names is None:
            self._output_names = ("loss",)
        else:
            self._output_names = tuple(output_names)
        super().__init__()
        self._loss = loss
        self._binary = binary

    @property
    def input_names(self) -> Sequence[str]:
        return self._input_names

    @property
    def output_names(self) -> Sequence[str]:
        return self._output_names

    def forward(self, *inputs):
        if self._binary:
            inputs = (torch.squeeze(inputs[0]), inputs[1].type(torch.cuda.FloatTensor))
        outputs = self._loss(*inputs)
        # if self._binary:
        #     outputs = torch.unsqueeze(outputs,dim=-1)
        if not isinstance(outputs, (list, tuple)):
            outputs = (outputs,)
        assert len(outputs) == len(self.output_names)
        return dict(zip(self.output_names, outputs))


class DataLoaderWrapper:
    """Wraps a DataLoader by given its outputs names.

    Args:
      dataloader (DataLoader):
        Instance of DataLoader to wrap.
      output_names (list of str):
        Names of outputs returned by `dataloader`.
    """

    def __init__(self, dataloader: DataLoader, output_names: Sequence[str]) -> None:
        self._dataloader = dataloader
        self._output_names = output_names

    @property
    def output_names(self) -> Sequence[str]:
        return self._output_names

    def __iter__(self):
        return iter(self._dataloader)

    def __len__(self):
        return len(self._dataloader)


class NamedDataLoader(DataLoader):
    """A data loader where outputs have names.

    Args:
      dataset (Dataset):
        The dataset from which to load the data.
      output_names (list of str):
        Names of outputs returned by `dataset`.
      **kwargs:
        Additional arguments passed to :class:`torch.utils.data.DataLoader`.
    """

    def __init__(self, dataset: Dataset, *, output_names: Sequence[str], **kwargs) -> None:
        super().__init__(dataset=dataset, **kwargs)
        self._output_names = output_names

    @property
    def output_names(self) -> Sequence[str]:
        return self._output_names
