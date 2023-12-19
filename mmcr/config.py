import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Self


@dataclass(frozen=True)
class PretrainConfig:
    dataset: Path | str = 'cifar10'
    batch_size: int = 1024
    num_views: int = 16
    max_epochs: int = 500
    warmup_duration: float = 0.1
    num_workers: int = os.cpu_count() - 2
    learning_rate: float = 1e-3
    projection_dim: int = 128
    num_neighbours: int = 200
    dev: bool = False

    @classmethod
    def from_command_line(cls, arguments: argparse.Namespace) -> Self:
        return cls(**vars(arguments))


@dataclass(frozen=True)
class FinetuneConfig:
    checkpoint: Path | str
    dataset: Path | str = 'cifar10'
    batch_size: int = 512
    max_epochs: int = 50
    num_workers: int = os.cpu_count() - 2
    warmup_duration: float = 0
    learning_rate: float = 1e-3

    @classmethod
    def from_command_line(cls, arguments: argparse.Namespace) -> Self:
        return cls(**vars(arguments))



