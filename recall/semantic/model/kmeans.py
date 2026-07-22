import torch
from tqdm import tqdm
from kernels.quantize import kmeans_quantize
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class Kmeans(nn.Module):
    def __init__(self, k, dim):
        super().__init__()

        self.k = k
        self.dim = dim

        self.register_buffer("centroids", torch.zeros(k, dim))
        self.register_buffer("acc_cluster_emb", torch.zeros(k, dim))
        self.register_buffer("acc_cluster_size", torch.zeros(k))

    @torch.no_grad()
    def init_from_samples(self, x):
        # x: [N, dim]
        assert x.shape[0] >= self.k

        perm = torch.randperm(x.shape[0], device=x.device)[:self.k]
        self.centroids.copy_(x[perm])

        self.acc_cluster_emb.zero_()
        self.acc_cluster_size.zero_()

    @torch.no_grad()
    def update_centroids(self):
        update_mask = self.acc_cluster_size > 0

        new_centroids = self.centroids.clone()

        new_centroids[update_mask] = (
            self.acc_cluster_emb[update_mask]
            / self.acc_cluster_size[update_mask].unsqueeze(1)
        )

        update_norm = torch.norm(self.centroids - new_centroids, dim=1)

        self.centroids.copy_(new_centroids)

        self.acc_cluster_emb.zero_()
        self.acc_cluster_size.zero_()

        return update_norm

    @torch.no_grad()
    def forward(self, x, acc=False):
        indices = kmeans_quantize(x, self.centroids).long()

        if acc:
            self.acc_cluster_emb.scatter_add_(
                0,
                indices.unsqueeze(1).expand(indices.shape[0], self.dim),
                x,
            )

            self.acc_cluster_size.scatter_add_(
                0,
                indices,
                torch.ones(indices.shape[0], device=x.device),
            )

        return self.centroids[indices], indices

class RQKmeans(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        n_layers: int = 3,
    ):
        super().__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            Kmeans(k=codebook_size, dim=dim)
            for _ in range(self.n_layers)
        ])

    @torch.no_grad()
    def init_layer_centroids(self, data: DataLoader, layer_idx: int, layer: Kmeans):
        samples = []
        total = 0

        for batch in data:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = batch.to(layer.centroids.device)

            # 当前层学习的是前面所有层拟合后的 residual
            for pre_layer in self.layers[:layer_idx]:
                emb, _ = pre_layer(batch, acc=False)
                batch = batch - emb

            samples.append(batch)
            total += batch.shape[0]

            if total >= max(layer.k * 10, layer.k):
                break

        samples = torch.cat(samples, dim=0)
        layer.init_from_samples(samples)

    @torch.no_grad()
    def fit_codebooks(self, data: DataLoader, max_iters=10000, tol=1e-6):
        for i, layer in enumerate(self.layers):
            print(f"Training RQKmeans layer {i}")

            self.init_layer_centroids(data, i, layer)

            for iter_idx in tqdm(range(max_iters)):
                for batch in data:
                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]

                    batch = batch.to(layer.centroids.device)

                    for pre_layer in self.layers[:i]:
                        emb, _ = pre_layer(batch, acc=False)
                        batch = batch - emb

                    layer(batch, acc=True)

                update_norm = layer.update_centroids().max()

                if update_norm < tol:
                    print(f"Layer {i} converged at iter {iter_idx}")
                    break

    @torch.no_grad()
    def forward(self, x):
        ids = []

        for layer in self.layers:
            emb, batch_ids = layer(x, acc=False)
            x = x - emb
            ids.append(batch_ids)

        return torch.stack(ids, dim=1)  # [B, n_layers]
