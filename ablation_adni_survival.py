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
import os
import socket
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import ray
import torch

from daft.evaluate import load_model_and_evaluate_survival
from train import main as main_train


@ray.remote(num_gpus=1)
def run_experiment(fold, cmd):
    gpu_ids = ray.get_gpu_ids()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    factory = main_train(cmd)

    results = load_model_and_evaluate_survival(
        factory.args, factory.checkpoints_dir, torch.device("cuda"), progressbar=False
    )
    results["fold"] = fold
    results["hostname"] = socket.gethostname()

    return results


def get_args(data_dir: Path, fold: int, loc: str, scale: str, shift: str, activation: str) -> Dict[str, Any]:
    cfg = {
        "task": "surv",
        "epoch": "80",
        "batchsize": "256",
        "optimizer": "AdamW",
        "train_data": str((data_dir / f"{fold}-train.h5").resolve()),
        "val_data": str((data_dir / f"{fold}-valid.h5").resolve()),
        "test_data": str((data_dir / f"{fold}-test.h5").resolve()),
        "discriminator_net": "daft",
        "learning_rate": "0.0055",
        "decay_rate": "0.01",
        "experiment_name": f"split-{fold}/daft_loc{loc}_scale{scale}_shift{shift}_act{activation}",
        "num_classes": "1",
        "n_basefilters": "4",
        "film_location": loc,
        "bottleneck_factor": "7",
        "normalize_image": "minmax",
        "dataset": "longitudinal",
        "scale": scale,
        "shift": shift,
        "activation": activation,
    }
    cmd = []
    for k, v in cfg.items():
        cmd.append(f"--{k}")
        cmd.append(v)
    cmd.append("--normalize_tabular")

    return cmd


def iter_experiments(data_dir: Path, fold: int) -> Tuple[int, Dict[str, Any]]:
    for loc in map(str, filter(lambda x: x != 2, range(5))):
        yield fold, get_args(data_dir, fold, loc, "enabled", "enabled", "linear")

    s_s_tuples = [
        ("enabled", "disabled", "linear"),
        ("disabled", "enabled", "linear"),
    ]
    for activation in (
        "tanh",
        "sigmoid",
    ):
        s_s_tuples.append(("enabled", "enabled", activation))

    for scale, shift, activation in s_s_tuples:
        yield fold, get_args(data_dir, fold, "2", scale, shift, activation)


def save_as_csv(results, outfile):
    df = pd.DataFrame(results)
    df.to_csv(outfile)


def run_fold(data_dir: Path, fold: int) -> None:
    filename = f"results_adni_survival_ablation_split-{fold}"

    remaining_ids = [run_experiment.remote(f, c) for f, c in iter_experiments(data_dir, fold)]
    results = []
    prev_file = None
    while len(remaining_ids) > 0:
        done_id, remaining_ids = ray.wait(remaining_ids)
        results.append(ray.get(done_id[0]))

        with tempfile.NamedTemporaryFile(suffix=f"{filename}_partial.csv", dir=".", delete=False, mode="w") as fout:
            save_as_csv(results, fout)
            new_file = Path(fout.name)
        if prev_file is not None:
            prev_file.unlink()
        prev_file = new_file

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    outfile = prev_file.with_name(f"{filename}-{timestamp}.csv")
    print(f"Saving results to {outfile}")
    prev_file.rename(outfile)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ray_address", required=True, help="The address of the Ray cluster to connect to.")
    parser.add_argument("--data_dir", required=True, type=Path, help="Path to directory containing HDF5 files.")

    args = parser.parse_args()

    ray.init(address=args.ray_address)

    for fold in range(5):
        run_fold(args.data_dir, fold)


if __name__ == "__main__":
    main()
