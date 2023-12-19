import torch.nn.functional
from torch import Tensor


def maximum_manifold_capacity(embeddings: Tensor, gamma: float) -> Tensor:
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    embeddings = torch.transpose(embeddings, 1, 2)

    centroids = embeddings.mean(dim=-1)

    loss = -1 * torch.linalg.matrix_norm(centroids, ord='nuc')

    if gamma > 0:
        loss += gamma * torch.linalg.matrix_norm(embeddings, ord='nuc').mean()

    return loss
