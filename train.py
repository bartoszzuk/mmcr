import os
import warnings
from dataclasses import dataclass

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from data.dataset import CIFAR10DataModule
from model import models
from model.modules import SelfSupervisedModel

torch.set_float32_matmul_precision('medium')


def silence_deprecation_warnings() -> None:
    for line in [110, 111, 117, 118]:
        warnings.filterwarnings('ignore', category=UserWarning, module='torch.overrides', lineno=line)


@dataclass(frozen=True)
class TrainArguments:
    batch_size: int = 1024
    num_views: int = 16
    num_workers: int = os.cpu_count() - 2
    max_epochs: int = 500
    learning_rate: float = 1e-3
    projection_dim: int = 128
    num_neighbours: int = 20


def train():
    silence_deprecation_warnings()

    args = TrainArguments()
    data = CIFAR10DataModule('cifar10', args.num_views, args.batch_size, dev=True)

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
