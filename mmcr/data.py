import os
from typing import Callable

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from mmcr.config import LinearEvaluateConfig, PretrainConfig

BASIC = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

AUGMENTATION = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
    transforms.RandomSolarize(0.5, p=0.2),
    BASIC
])


class MultiviewDataset(CIFAR10):

    def __init__(self, root: str, train: bool = True, transform: Callable = None, num_views: int = 1) -> None:
        super().__init__(root, train, transform, download=True)
        assert num_views >= 1, 'Number of views must be larger than zero'
        self.num_views = num_views

    def __getitem__(self, index: int) -> Tensor:
        image = self.data[index]
        image = Image.fromarray(image)

        views = [image] * self.num_views

        if self.transform:
            views = [self.transform(view) for view in views]
            views = torch.stack(views)

        return views


class PretrainDataModule(LightningDataModule):

    def __init__(self, config: PretrainConfig) -> None:
        super().__init__()

        self.train_batch_size = config.batch_size
        self.valid_batch_size = config.batch_size * config.num_views
        self.num_workers = config.num_workers

        root = config.dataset

        self.train_dataset = MultiviewDataset(root, transform=AUGMENTATION, train=True, num_views=config.num_views)
        self.valid_source_dataset = CIFAR10(root, transform=BASIC, download=True, train=True)
        self.valid_target_dataset = CIFAR10(root, transform=BASIC, download=True, train=False)

        if config.dev:
            train_size = len(self.train_dataset) // 10
            valid_size = len(self.valid_target_dataset) // 10

            self.train_dataset = Subset(self.train_dataset, range(train_size))
            self.valid_source_dataset = Subset(self.valid_source_dataset, range(train_size))
            self.valid_target_dataset = Subset(self.valid_target_dataset, range(valid_size))

            print(f'[Dataset] Using dev version: train size: {train_size}, valid size: {valid_size}')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, self.train_batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(self.valid_source_dataset, batch_size=self.valid_batch_size, num_workers=self.num_workers),
            DataLoader(self.valid_target_dataset, batch_size=self.valid_batch_size, num_workers=self.num_workers)
        ]

    def predict_dataloader(self) -> list[DataLoader]:
        return self.val_dataloader()


class LinearEvaluateDataModule(LightningDataModule):

    def __init__(self, config: LinearEvaluateConfig) -> None:
        super().__init__()

        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.train_dataset = CIFAR10(config.dataset, download=True, train=True, transform=AUGMENTATION)
        self.valid_dataset = CIFAR10(config.dataset, download=True, train=False, transform=BASIC)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset, self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()
