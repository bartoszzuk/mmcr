import argparse
import os
from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from mmcr.config import FinetuneConfig
from mmcr.data import FinetuneDataModule
from mmcr.models import ResNetForClassification
from mmcr.modules import FinetuneModule
from pretrain import silence_compilation_warnings

torch.set_float32_matmul_precision('medium')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default='cifar10')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=os.cpu_count() - 2)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-2)
    parser.add_argument('--warmup-duration', type=float, default=0)
    parser.add_argument('--compile', action='store_true')

    return parser.parse_args()


def finetune() -> None:
    config = FinetuneConfig.from_command_line(parse_arguments())
    data = FinetuneDataModule(config)

    model = ResNetForClassification.from_pretrained(config.checkpoint)
    model.freeze_backbone()

    if config.compile:
        model = torch.compile(model)
        silence_compilation_warnings()

    model = FinetuneModule(model, config)

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
    finetune()
