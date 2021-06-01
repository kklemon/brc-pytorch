import random
import torch

from dataclasses import dataclass
from torch.utils.data import Dataset, IterableDataset


@dataclass
class BenchmarkMeta:
    in_dim: int
    out_dim: int
    categorical_output: bool = False
    sequential_output: bool = False


class CopyFirstInputBenchmark(IterableDataset):
    def __init__(self, t: int, dim: int = 1):
        self.meta = BenchmarkMeta(in_dim=dim, out_dim=dim)
        self.dim = dim
        self.t = t

    def sample_data(self):
        return torch.randn(self.t + 1, self.dim)

    def __iter__(self):
        while True:
            x = self.sample_data()
            y = x[0]
            yield x, y


class DenoisingBenchmark(IterableDataset):
    def __init__(self, t: int, n: int = 0, k: int = 5):
        self.meta = BenchmarkMeta(in_dim=2, out_dim=k)

        assert t > 0
        assert 0 <= n <= t - k

        self.t = t
        self.n = n
        self.k = k

    def __iter__(self):
        while True:
            x = torch.Tensor(self.t, 2)

            indices = torch.tensor(sorted(random.sample(range(self.t - self.n), self.k)))
            x[:, 0] = -1
            x[indices, 0] = 0
            x[-1, 0] = 1

            x[:, 1] = torch.randn(self.t)

            y = x[indices, 1]

            yield x, y


class SequentialImageClassification(Dataset):
    def __init__(self, dataset, n_black_pixels: int = 0):
        c, w, h = dataset[0][0].shape

        self.meta = BenchmarkMeta(in_dim=c, out_dim=len(dataset.classes), categorical_output=True)

        self.n_pixels = w * h

        self.dataset = dataset
        self.n_channels = self.in_dim = c
        self.n_black_pixels = n_black_pixels
        self.seq_len = self.n_pixels + self.n_black_pixels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        assert image.shape[0] == self.n_channels
        image = image.view(self.n_channels, -1).T

        x = torch.Tensor(self.seq_len, self.in_dim)
        x[:self.n_pixels] = image
        x[self.n_pixels:] = 0

        return x, label



