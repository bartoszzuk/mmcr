import argparse
import os
import warnings
from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from mmcr.data import PretrainDataModule
from mmcr.config import PretrainConfig
from mmcr import models
from mmcr.modules import PretrainModule

torch.set_float32_matmul_precision('medium')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default='cifar10')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-views', type=int, default=16)
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=os.cpu_count() - 2)
    parser.add_argument('--learning-rate', type=int, default=1e-3)
    parser.add_argument('--projection-dim', type=int, default=128)
    parser.add_argument('--num-neighbours', type=int, default=200)
    parser.add_argument('--warmup-duration', type=float, default=0.1)
    parser.add_argument('--dev', action='store_true')

    return parser.parse_args()


def silence_deprecation_warnings() -> None:
    for line in [110, 111, 117, 118]:
        warnings.filterwarnings('ignore', category=UserWarning, module='torch.overrides', lineno=line)


def pretrain():
    silence_deprecation_warnings()

    config = PretrainConfig.from_command_line(parse_arguments())
    data = PretrainDataModule(config)

    knn = models.KNearestNeighbours(config.num_neighbours)

    model = models.ResNet(config.projection_dim)
    model = torch.compile(model)
    model = PretrainModule(model, knn, config)

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
        max_epochs=config.max_epochs,
        logger=TensorBoardLogger(save_dir='logs', name=''),
        callbacks=callbacks,
        deterministic=True
    )

    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    pretrain()
