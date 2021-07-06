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
from abc import ABCMeta, abstractmethod
from typing import Any, Sequence

from torch.nn import Module


def check_is_unique(values: Sequence[Any]) -> bool:
    if len(values) != len(set(values)):
        raise ValueError("values of list must be unique")


class BaseModel(Module, metaclass=ABCMeta):
    """Abstract base class for models that can be executed by
    :class:`daft.training.train_and_eval.ModelRunner`.
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        check_is_unique(self.input_names)
        check_is_unique(self.output_names)
        check_is_unique(list(self.input_names) + list(self.output_names))

    @property
    @abstractmethod
    def input_names(self) -> Sequence[str]:
        """Names of parameters passed to self.forward"""

    @property
    @abstractmethod
    def output_names(self) -> Sequence[str]:
        """Names of tensors returned by self.forward"""
