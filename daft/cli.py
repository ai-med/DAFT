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
import argparse
import inspect
import json
import warnings
from abc import ABCMeta, abstractmethod
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from .data_utils import adni_hdf
from .data_utils.surv_data import cox_collate_fn
from .models.base import BaseModel
from .models.losses import CoxphLoss
from .networks import vol_networks
from .training.metrics import Accuracy, BalancedAccuracy, ConcordanceIndex, Mean, Metric
from .training.wrappers import LossWrapper, NamedDataLoader


def create_parser():
    parser = argparse.ArgumentParser("Shape Continuum")
    parser.set_defaults(shape="vol_with_bg")

    g = parser.add_argument_group("General")
    g.add_argument("--task", choices=["clf", "surv"], required=True, help="classification or survival analysis")
    g.add_argument("--workers", type=int, default=4, help="number of data loading workers. Default: %(default)s")

    g = parser.add_argument_group("Training")
    g.add_argument("--epoch", type=int, default=200, help="number of epochs for training. Default: %(default)s")
    g.add_argument("--pretrain", type=Path, help="Path to a model checkpoint")
    g.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate for training. Default: %(default)s"
    )
    g.add_argument("--decay_rate", type=float, default=1e-4, help="weight decay. Default: %(default)s")
    g.add_argument(
        "--optimizer", choices=["Adam", "SGD", "AdamW"], default="AdamW", help="type of optimizer. Default: %(default)s"
    )

    g = parser.add_argument_group("Architecture")
    g.add_argument(
        "--discriminator_net",
        choices=["resnet", "concat1fc", "concat2fc", "mlpcatmlp", "duanmu", "film", "daft"],
        default="daft",
        help="which architecture to use. Default: %(default)s",
    )
    g.add_argument("--n_basefilters", type=int, default=4, help="Number of base filters. Default: %(default)s")

    g = parser.add_argument_group("FiLM Block Architecture")
    g.add_argument(
        "--film_location",
        type=int,
        default=0,
        help="location of FiLM when a Film-based model is trained. Default: %(default)s",
    )
    g.add_argument(
        "--bottleneck_factor",
        type=float,
        default=7.0,
        help="Reduction factor in a Film-based model. Default: %(default)s",
    )
    g.add_argument(
        "--scale", choices=["enabled", "disabled"], default="enabled", help="scaling in film. Default: %(default)s"
    )
    g.add_argument(
        "--shift", choices=["enabled", "disabled"], default="enabled", help="shifting in film. Default: %(default)s"
    )
    g.add_argument(
        "--activation",
        choices=["linear", "tanh", "sigmoid"],
        default="linear",
        help="activation in film. Default: %(default)s",
    )

    g = parser.add_argument_group("Data")
    g.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="The number of output units of the network. For binary classification, 1, "
        "for multi-class classification, the number of classes. Default: %(default)s",
    )
    g.add_argument("--batchsize", type=int, default=20, help="input batch size. Default: %(default)s")
    g.add_argument(
        "--num_points", type=int, default=1500, help="number of points each point cloud has. Default: %(default)s"
    )
    g.add_argument("--train_data", type=Path, required=True, help="path to training dataset")
    g.add_argument("--val_data", type=Path, required=True, help="path to validation dataset")
    g.add_argument("--test_data", type=Path, required=True, help="path to testing dataset")
    g.add_argument(
        "--dataset",
        choices=["longitudinal", "baseline"],
        default="longitudinal",
        help="Whether training data contains longitudinal data (multiple visits per patient), "
        "or only baseline data (one visit per patient). Default: %(default)s",
    )
    g.add_argument(
        "--drop_missing",
        action="store_true",
        default=False,
        help="wether to drop missing values in tabular data. Default: %(default)s",
    )

    g = parser.add_argument_group("Logging")
    g.add_argument(
        "--experiment_name",
        nargs="?",
        const=True,  # present, but not followed by a command-line argument
        default=False,  # not present
        help="Whether to give the experiment a particular name (default: current date and time).",
    )
    g.add_argument(
        "--tb_comment", action="store_true", default=False, help="any comment for storing on tensorboard",
    )
    g.add_argument(
        "--tensorboard", action="store_true", default=False, help="visualize training progress on tensorboard",
    )

    # normalization
    g = parser.add_argument_group("Data Normalization")
    g.add_argument(
        "--normalize_image",
        choices=["rescale", "standardize", "minmax"],
        default="rescale",
        help="Normalization function to apply to image data. Default rescale. Options: "
        "rescale -> divide by maximum value of datatype; "
        "standardize -> mean and stddev of whole dataset; "
        "minmax -> normalize to [0..1] with minimum and maximum per sample. Default: %(default)s",
    )
    g.add_argument(
        "--normalize_tabular",
        action="store_true",
        default=False,
        help="Normalize tabular data with mean and variance of whole dataset. Default: %(default)s",
    )

    return parser


def get_number_of_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class BaseModelFactory(metaclass=ABCMeta):
    """Abstract base class for creating models and data loaders.

    Args:
      arguments:
        Command line arguments.
    """

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.args = arguments
        if arguments.task == "clf":
            if arguments.num_classes == 1:
                warnings.warn(
                    "Data for classification tasks should consist of more than one label:"
                    f" num_classes = {arguments.num_classes}."
                )
            if arguments.num_classes > 2:
                self._task = adni_hdf.Task.MULTI_CLASSIFICATION
            else:
                self._task = adni_hdf.Task.BINARY_CLASSIFICATION
        elif arguments.task == "surv":
            assert arguments.num_classes == 1
            self._task = adni_hdf.Task.SURVIVAL_ANALYSIS
        else:
            raise ValueError("task={!r} is not supported".format(arguments.task))

    @property
    def task(self) -> adni_hdf.Task:
        return self._task

    @property
    def checkpoints_dir(self):
        return self._checkpoints_dir

    def make_directories(self) -> Tuple[str, str, str]:
        """ "Create directories to hold logs and checkpoints.

        Returns:
          experiment_dir (Path):
            Path to base directory.
          checkpoints_dir (Path):
            Path to directory where checkpoints should be saved to.
          tb_dir (Path):
            Path to directory where TensorBoard log should be written to.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        args = self.args
        base_dir = Path(f"experiments_{args.task}")
        if args.experiment_name:
            if isinstance(args.experiment_name, str):
                experiment = args.experiment_name
            else:
                experiment = input("Enter a name for your experiment: ")
        else:
            experiment = f"shape_{args.shape}_network_{args.discriminator_net}"

        experiment_dir = base_dir / experiment / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=False)

        checkpoints_dir = experiment_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        self._checkpoints_dir = checkpoints_dir

        tb_dir = experiment_dir / "tb_log"
        return experiment_dir, checkpoints_dir, tb_dir

    def get_optimizer(self, params: Sequence[torch.Tensor]) -> Optimizer:
        """Create an optimizer.

        Args:
          params (list of of torch.Tensor):
            List of parameters to optimize.

        Returns:
          optim (Optimizer):
            Instance of the selected optimizer.
        """
        args = self.args
        if args.optimizer == "SGD":
            optimizerD = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9)
        elif args.optimizer == "Adam":
            optimizerD = torch.optim.Adam(
                params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate,
            )
        elif args.optimizer == "AdamW":
            optimizerD = torch.optim.AdamW(
                params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate,
            )
        else:
            raise ValueError(f"unknown optimizer {args.optimizer}")
        return optimizerD

    def get_args(self) -> Dict[str, Any]:
        """Return arguments as dictionary."""
        args = vars(self.args)
        for k, v in args.items():
            if isinstance(v, Path):
                args[k] = str(v.resolve())
        return args

    def write_args(self, filename: str) -> None:
        """Write command line arguments to JSON file.

        Args:
          filename (str):
            Path to JSON file.
        """
        args = self.get_args()

        with open(filename, "w") as f:
            json.dump(args, f, indent=2)

    def _init_model(self, model: BaseModel) -> None:
        """Initialize the model.

        If path to checkpoint is provided, initializes weights from checkpoint.

        Args:
          model (BaseModel):
            Model to initialize.
        """
        args = self.args
        model_name = model.__class__.__name__
        if args.pretrain is not None:
            print(f"Load {model_name} model from {args.pretrain}")
            model.load_state_dict(torch.load(args.pretrain))
        else:
            print(f"Training {model_name} from scratch")

        n_params = get_number_of_parameters(model)
        print(f"Number of parameters: {n_params:,}")

    def get_loss(self) -> LossWrapper:
        """Return the loss to optimize."""
        if self._task == adni_hdf.Task.SURVIVAL_ANALYSIS:
            loss = LossWrapper(
                CoxphLoss(), input_names=["logits", "event", "riskset"], output_names=["partial_log_lik"]
            )
        else:
            if self.args.num_classes > 2:
                loss = LossWrapper(
                    torch.nn.CrossEntropyLoss(), input_names=["logits", "target"], output_names=["cross_entropy"]
                )
            else:
                loss = LossWrapper(
                    torch.nn.BCEWithLogitsLoss(),
                    input_names=["logits", "target"],
                    output_names=["cross_entropy"],
                    binary=True,
                )
        return loss

    def get_metrics(self) -> Sequence[Metric]:
        """Returns a list of metrics to compute."""
        if self._task == adni_hdf.Task.SURVIVAL_ANALYSIS:
            metrics = [Mean("partial_log_lik")]
        else:
            metrics = [Mean("cross_entropy")]
        metrics.extend(self.get_test_metrics())
        return metrics

    def get_test_metrics(self) -> Sequence[Metric]:
        if self._task == adni_hdf.Task.SURVIVAL_ANALYSIS:
            metrics = [ConcordanceIndex("logits", "event", "time")]
        else:
            metrics = [
                Accuracy("logits", "target"),
                BalancedAccuracy(self.args.num_classes, "logits", "target"),
            ]
        return metrics

    def get_and_init_model(self) -> BaseModel:
        """Create and initialize a model."""
        model = self.get_model()
        self._init_model(model)
        return model

    @property
    def data_loader_target_names(self):
        if self._task == adni_hdf.Task.SURVIVAL_ANALYSIS:
            target_names = ["event", "time", "riskset"]
        else:
            target_names = ["target"]
        return target_names

    def _make_named_data_loader(
        self, dataset: Dataset, model_data_names: Sequence[str], is_training: bool = False
    ) -> NamedDataLoader:
        """Create a NamedDataLoader for the given dataset.

        Args:
          dataset (Dataset):
            The dataset to wrap.
          model_data_names (list of str):
            Should correspond to the names of the first `len(model_data_names)` outputs
            of the dataset and that are fed to model in a forward pass. The names
            of the targets used to compute the loss will be retrieved from :meth:`data_loader_target_names`.
          is_training (bool):
            Whether to enable training mode or not.
        """
        batch_size = self.args.batchsize
        if len(dataset) < batch_size:
            if is_training:
                raise RuntimeError(
                    "batch size ({:d}) cannot exceed dataset size ({:d})".format(batch_size, len(dataset))
                )

            batch_size = len(dataset)

        collate_fn = default_collate
        if self._task == adni_hdf.Task.SURVIVAL_ANALYSIS:
            collate_fn = partial(cox_collate_fn, data_collate=collate_fn)

        kwargs = {"batch_size": batch_size, "collate_fn": collate_fn, "shuffle": is_training, "drop_last": is_training}

        output_names = list(model_data_names) + self.data_loader_target_names
        loader = NamedDataLoader(dataset, output_names=output_names, **kwargs)
        return loader

    @abstractmethod
    def get_model(self) -> BaseModel:
        """Returns a model instance."""

    @abstractmethod
    def get_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Returns a data loader instance for training, evaluation, and testing, respectively."""


class HeterogeneousModelFactory(BaseModelFactory):
    """Factory for models taking 3D image volumes and tabular data."""

    def get_data(self):
        args = self.args
        rescale = args.normalize_image == "rescale"
        standardize = args.normalize_image == "standardize"
        minmax = args.normalize_image == "minmax"
        train_data, transform_kwargs, transform_tabular_kwargs = adni_hdf.get_heterogeneous_dataset_for_train(
            args.train_data,
            self._task,
            args.shape,
            rescale=rescale,
            standardize=standardize,
            minmax=minmax,
            transform_age=False,
            transform_education=False,
            normalize_tabular=args.normalize_tabular,
            dataset=args.dataset,
            drop_missing=args.drop_missing,
        )
        self.tabular_size = len(train_data.meta["tabular"]["columns"])
        trainDataLoader = self._make_named_data_loader(train_data, ["image", "tabular"], is_training=True)

        eval_data = adni_hdf.get_heterogeneous_dataset_for_eval(
            args.val_data,
            self._task,
            transform_kwargs,
            args.shape,
            transform_tabular_kwargs,
            drop_missing=args.drop_missing,
        )
        valDataLoader = self._make_named_data_loader(eval_data, ["image", "tabular"])

        test_data = adni_hdf.get_heterogeneous_dataset_for_eval(
            args.test_data,
            self._task,
            transform_kwargs,
            args.shape,
            transform_tabular_kwargs,
            drop_missing=args.drop_missing,
        )
        testDataLoader = self._make_named_data_loader(test_data, ["image", "tabular"])
        return trainDataLoader, valDataLoader, testDataLoader

    def get_model(self):
        args = self.args
        class_dict = {
            "resnet": vol_networks.HeterogeneousResNet,
            "concat1fc": vol_networks.ConcatHNN1FC,
            "concat2fc": vol_networks.ConcatHNN2FC,
            "mlpcatmlp": vol_networks.ConcatHNNMCM,
            "duanmu": vol_networks.InteractiveHNN,
            "film": vol_networks.FilmHNN,
            "daft": vol_networks.DAFT,
        }
        if args.discriminator_net not in class_dict:
            raise ValueError("network {!r} is unsupported".format(args.discriminator_net))
        # common args of all models
        model_args = {
            "in_channels": 1,
            "n_outputs": (args.num_classes if args.num_classes > 2 else 1),
            "n_basefilters": args.n_basefilters,
        }

        # calc bottleneck dimension of complex models
        bdim = int((4 * model_args["n_basefilters"] + self.tabular_size) / args.bottleneck_factor)

        cls = class_dict[args.discriminator_net]
        # set model-specific args
        class_params = set(inspect.signature(cls).parameters.keys())
        if "bottleneck_dim" in class_params:
            model_args["bottleneck_dim"] = bdim
        if "ndim_non_img" in class_params:
            model_args["ndim_non_img"] = self.tabular_size
        elif "filmblock_args" in class_params:
            model_args["filmblock_args"] = {
                "ndim_non_img": self.tabular_size,
                "location": args.film_location,
                "bottleneck_dim": bdim,
                "scale": args.scale == "enabled",
                "shift": args.shift == "enabled",
                "activation": args.activation,
            }
        return cls(**model_args)


def get_factory(args: argparse.Namespace) -> BaseModelFactory:
    factory = HeterogeneousModelFactory(args)
    return factory
