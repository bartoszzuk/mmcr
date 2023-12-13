import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from typing_extensions import Self

from data.dataset import CIFAR10DataModule
from model import models
from model.modules import SelfSupervisedModel

torch.set_float32_matmul_precision('medium')


@dataclass(frozen=True)
class TrainArguments:
    dataset: Path | str = 'cifar10'
    batch_size: int = 1024
    num_views: int = 16
    max_epochs: int = 500
    learning_rate: float = 1e-3
    projection_dim: int = 128
    num_neighbours: int = 20
    dev: bool = False

    @classmethod
    def from_command_line(cls, arguments: argparse.Namespace) -> Self:
        return cls(**vars(arguments))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default='cifar10')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-views', type=int, default=16)
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--learning-rate', type=int, default=1e-3)
    parser.add_argument('--projection-dim', type=int, default=128)
    parser.add_argument('--num-neighbours', type=int, default=200)
    parser.add_argument('--dev', action='store_true')

    return parser.parse_args()


def silence_deprecation_warnings() -> None:
    for line in [110, 111, 117, 118]:
        warnings.filterwarnings('ignore', category=UserWarning, module='torch.overrides', lineno=line)


def train():
    silence_deprecation_warnings()

    args = TrainArguments.from_command_line(parse_arguments())
    data = CIFAR10DataModule('cifar10', args.num_views, args.batch_size, args.dev)

    knn = models.KNearestNeighbours(args.num_neighbours)

    model = models.ResNet(args.projection_dim)
    model = torch.compile(model)
    model = SelfSupervisedModel(model, args.max_epochs, args.learning_rate, args.num_views, knn=knn)

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            monitor='Valid|Top1 Accuracy',
            save_top_k=1,
            save_last=True,
            mode='max',
            verbose=True,
            filename='{epoch}-{Valid|Top1 Accuracy:.2f}',
        ),
    ]

    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        max_epochs=args.max_epochs,
        logger=TensorBoardLogger(save_dir='logs', name=''),
        callbacks=callbacks,
        deterministic=True
    )

    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    train()
