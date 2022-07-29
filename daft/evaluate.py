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
from pathlib import Path
from typing import Any, Dict, List, Union

import h5py
import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from torch.nn import Module
from torch.utils.data import DataLoader

from .cli import HeterogeneousModelFactory, get_number_of_parameters
from .testing.test_and_save import evaluate_model, load_best_model


def compute_metrics(factory, model, data_loader, device, progressbar=True):
    metrics, preds = evaluate_model(
        metrics=factory.get_test_metrics(), model=model, data=data_loader, device=device, progressbar=progressbar,
    )

    if "event" in preds:
        y_true = preds["event"].numpy()
    else:
        y_true = preds["target"].numpy()
    metrics["n_samples"] = y_true.shape[0]
    metrics["n_parameters"] = get_number_of_parameters(model)

    return metrics


def add_prefix(adict, prefix):
    out = {}
    for k, v in adict.items():
        out[f"{prefix}{k}"] = v
    return out


def config_to_args(saved_args: Dict[str, Any], defaults: Dict[str, Any]) -> List[str]:
    cmd = []
    for key, value in saved_args.items():
        if key not in defaults and value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))

    return cmd


def load_model_and_evaluate(
    args: argparse.Namespace, ckpt_dir: Path, device, progressbar=True
) -> Dict[str, Union[float, str]]:
    saved_args = vars(args)

    factory = HeterogeneousModelFactory(args)

    this_args = factory.get_args()
    for k, v in saved_args.items():
        assert v == this_args[k], f"{k}: {v} != {this_args[k]}"
    assert len(set(this_args.keys()).symmetric_difference(set(saved_args.keys()))) == 0

    _, valid_loader, test_loader = factory.get_data()

    model = load_best_model(factory, ckpt_dir, device=device)

    valid_metrics = add_prefix(compute_metrics(factory, model, valid_loader, device, progressbar), "valid_")
    test_metrics = add_prefix(compute_metrics(factory, model, test_loader, device, progressbar), "test_")

    result = {"checkpoint": str(ckpt_dir)}
    result.update(valid_metrics)
    result.update(test_metrics)
    return result


#########################################################################
# Survival Evaluation
#########################################################################


def load_event_time(h5_path: str) -> np.ndarray:
    data = []
    with h5py.File(h5_path, mode="r") as fin:
        for name, grp in fin.items():
            if name != "stats":
                data.append((grp.attrs["event"] == "yes", grp.attrs["time"]))
    return np.array(data, dtype=[("event", bool), ("time", float)])


def get_data_with_logits(model: Module, loader: DataLoader) -> Dict[str, np.ndarray]:
    data = {
        "event": [],
        "time": [],
        "logits": [],
    }

    dev = torch.device("cuda")

    model = model.to(dev).eval()

    with torch.no_grad():
        for image, tabular, event, time, _riskset in loader:
            data["event"].append(event.squeeze(1))
            data["time"].append(time)

            outputs = model(image.to(dev), tabular.to(dev))
            logits = outputs["logits"].detach().cpu()
            data["logits"].append(logits.squeeze(1))

    for k, v in data.items():
        data[k] = torch.cat(v).numpy()

    data["y"] = np.fromiter(zip(data.pop("event"), data.pop("time")), dtype=[("event", bool), ("time", float)])

    return data


def get_metrics(model: Module, data_loader: DataLoader, train_y: np.ndarray) -> pd.Series:
    test_data = get_data_with_logits(model, data_loader)
    test_y = test_data["y"]
    logits = test_data["logits"]

    y_all = np.concatenate((test_y, train_y))
    q = np.linspace(10, 76, 66)
    times = np.quantile(y_all["time"][y_all["event"]], q / 100.0)

    auc_t, auc = cumulative_dynamic_auc(train_y, test_y, logits, times=times)
    data = pd.Series(dict(zip(map(lambda x: f"AUC({int(x)})", q), auc_t)))  # noqa: C417
    data.loc["iAUC"] = auc

    cindex_ipcw = concordance_index_ipcw(train_y, test_y, logits)
    data.loc["ipcwCI"] = cindex_ipcw[0]

    cindex_harrell = concordance_index_censored(test_y["event"], test_y["time"], logits)
    data.loc["concordance/cindex"] = cindex_harrell[0]

    data.loc["n_samples"] = test_y.shape[0]
    data.loc["n_parameters"] = get_number_of_parameters(model)

    return data


def load_model_and_evaluate_survival(
    args: argparse.Namespace, ckpt_dir: Path, device, progressbar=True
) -> Dict[str, Union[float, str]]:
    saved_args = vars(args)

    factory = HeterogeneousModelFactory(args)

    this_args = factory.get_args()
    for k, v in saved_args.items():
        assert v == this_args[k], f"{k}: {v} != {this_args[k]}"
    assert len(set(this_args.keys()).symmetric_difference(set(saved_args.keys()))) == 0

    _, valid_loader, test_loader = factory.get_data()

    model = load_best_model(factory, ckpt_dir, device=device)

    train_y = load_event_time(args.train_data)

    valid_metrics = get_metrics(model, valid_loader, train_y).add_prefix("valid_")
    test_metrics = get_metrics(model, test_loader, train_y).add_prefix("test_")

    result = {"checkpoint": str(ckpt_dir)}
    result.update(valid_metrics)
    result.update(test_metrics)
    return result
