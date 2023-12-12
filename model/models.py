import torch
import torchmetrics
import torchvision
from torch import nn, Tensor


class ResNet(nn.Module):

    def __init__(self, projection_dim: int):
        super().__init__()

        backbone = torchvision.models.resnet18()
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        backbone = [layer for name, layer in backbone.named_children() if name not in {'maxpool', 'fc'}]

        self.backbone = nn.Sequential(*backbone)
        self.projector = nn.Sequential(
            nn.Linear(512, 2 * projection_dim, bias=False),
            nn.BatchNorm1d(2 * projection_dim),
            nn.ReLU(),
            nn.Linear(2 * projection_dim, projection_dim)
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        embeddings = self.backbone(inputs)
        embeddings = torch.flatten(embeddings, start_dim=1)

        projections = self.projector(embeddings)

        return projections, embeddings


class KNearestNeighbours:

    def __init__(self, num_neighbours: int = 200, temperature: float = 0.5):
        self.num_neighbours = num_neighbours
        self.temperature = temperature

        self.train_embeddings = []
        self.train_labels = []

    def add(self, embeddings: Tensor, labels: Tensor) -> None:
        self.train_embeddings.append(embeddings)
        self.train_labels.append(labels)

    def reset(self) -> None:
        self.train_embeddings = []
        self.train_labels = []

    def score(self, embeddings: Tensor) -> Tensor:
        train_embeddings = torch.concat(self.train_embeddings)
        similarity = torchmetrics.functional.pairwise_cosine_similarity(embeddings, train_embeddings)

        weights, indices = similarity.topk(k=self.num_neighbours, dim=-1)
        weights = torch.exp(weights / self.temperature)

        labels = torch.concat(self.train_labels).expand(embeddings.size(0), -1)
        labels = torch.gather(labels, dim=-1, index=indices)
        labels = torch.nn.functional.one_hot(labels)

        scores = torch.sum(labels * weights.unsqueeze(-1), dim=1)

        return scores
