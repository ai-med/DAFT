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
import torch
from torch.optim.lr_scheduler import LambdaLR

from daft.cli import HeterogeneousModelFactory, create_parser
from daft.training.hooks import CheckpointSaver, TensorBoardLogger
from daft.training.train_and_eval import train_and_evaluate


def main(args=None):
    args = create_parser().parse_args(args=args)

    torch.manual_seed(20210129)
    factory = HeterogeneousModelFactory(args)

    experiment_dir, checkpoints_dir, tb_log_dir = factory.make_directories()

    factory.write_args(experiment_dir / "experiment_args.json")

    train_loader, valid_loader, _ = factory.get_data()
    discriminator = factory.get_and_init_model()
    optimizerD = factory.get_optimizer(filter(lambda p: p.requires_grad, discriminator.parameters()))
    loss = factory.get_loss()

    tb_log_dir = experiment_dir / "tensorboard"
    checkpoints_dir = experiment_dir / "checkpoints"
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_metrics = factory.get_metrics()
    train_hooks = [TensorBoardLogger(str(tb_log_dir / "train"), train_metrics)]

    eval_metrics_tb = factory.get_metrics()
    eval_hooks = [TensorBoardLogger(str(tb_log_dir / "eval"), eval_metrics_tb)]
    eval_metrics_cp = factory.get_metrics()
    eval_hooks.append(
        CheckpointSaver(discriminator, checkpoints_dir, save_every_n_epochs=1, max_keep=5, metrics=eval_metrics_cp)
    )

    def lr_factor(epoch):
        if epoch <= int(0.6 * args.epoch):
            return 1
        if epoch <= int(0.9 * args.epoch):
            return 0.1
        return 0.05

    scheduler = LambdaLR(optimizerD, lr_lambda=lr_factor)

    dev = torch.device("cuda")
    train_and_evaluate(
        model=discriminator,
        loss=loss,
        train_data=train_loader,
        optimizer=optimizerD,
        scheduler=scheduler,
        num_epochs=args.epoch,
        eval_data=valid_loader,
        train_hooks=train_hooks,
        eval_hooks=eval_hooks,
        device=dev,
        progressbar=False,
    )

    return factory


if __name__ == "__main__":
    main()
