from typing import Callable

import torch
import torchmetrics
from lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from mmcr.config import FinetuneConfig, PretrainConfig
from mmcr import utils, losses
from mmcr.models import KNearestNeighbours


class FinetuneModule(LightningModule):

    def __init__(self, model: nn.Module | Callable, config: FinetuneConfig) -> None:
        super().__init__()
        self.model = model

        self.max_epochs = config.max_epochs
        self.warmup_epochs = round(config.max_epochs * config.warmup_duration)
        self.learning_rate = config.learning_rate

        self.top1_accuracy_valid = torchmetrics.Accuracy(num_classes=10, task='multiclass', top_k=1)
        self.top5_accuracy_valid = torchmetrics.Accuracy(num_classes=10, task='multiclass', top_k=5)

    def configure_optimizers(self) -> (list[Optimizer], list[LRScheduler]):
        optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=1e-6)
        scheduler = utils.cosine_with_warmup(optimizer, self.warmup_epochs, self.max_epochs)

        return [optimizer], [scheduler]

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        images, labels = batch

        logits = self.model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        self.log('Train|Loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], index: int) -> None:
        images, labels = batch

        logits = self.model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        scores = torch.softmax(logits, dim=1)

        self.top1_accuracy_valid(scores, labels)
        self.top5_accuracy_valid(scores, labels)

        self.log('Valid|Loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('Valid|Top1 Accuracy', self.top1_accuracy_valid, on_step=False, on_epoch=True)
        self.log('Valid|Top5 Accuracy', self.top5_accuracy_valid, on_step=False, on_epoch=True)


class PretrainModule(LightningModule):

    def __init__(self, model: nn.Module | Callable, knn: KNearestNeighbours, config: PretrainConfig) -> None:
        super().__init__()
        self.model = model
        self.knn = knn

        self.max_epochs = config.max_epochs
        self.warmup_epochs = round(config.max_epochs * config.warmup_duration)
        self.learning_rate = config.learning_rate

        self.top1_accuracy_valid = torchmetrics.Accuracy(num_classes=10, task='multiclass', top_k=1)
        self.top5_accuracy_valid = torchmetrics.Accuracy(num_classes=10, task='multiclass', top_k=5)

    def configure_optimizers(self) -> (list[Optimizer], list[LRScheduler]):
        optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=1e-6)
        scheduler = utils.cosine_with_warmup(optimizer, self.warmup_epochs, self.max_epochs)

        return [optimizer], [scheduler]

    def training_step(self, batch: Tensor) -> Tensor:
        views = torch.flatten(batch, end_dim=1)

        projections, _ = self.model(views)
        projections = projections.view(-1, self.num_views, projections.size(-1))

        loss = losses.maximum_manifold_capacity(projections, gamma=0)

        self.log('Train|Loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], index: int, dataloader_idx: int = 0) -> None:
        images, labels = batch

        _, embeddings = self.model(images)

        if dataloader_idx == 0:
            self.knn.add(embeddings, labels)

        if dataloader_idx == 1:
            scores = self.knn.score(embeddings)

            self.top1_accuracy_valid(scores, labels)
            self.top5_accuracy_valid(scores, labels)
            self.log('Valid|Top1 Accuracy', self.top1_accuracy_valid, on_epoch=True, add_dataloader_idx=False)
            self.log('Valid|Top5 Accuracy', self.top5_accuracy_valid, on_epoch=True, add_dataloader_idx=False)

    def on_validation_end(self) -> None:
        self.knn.reset()
