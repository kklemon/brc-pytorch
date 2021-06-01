# class SequentialImageClassification(Dataset):
#     def __init__(self, dataset, rel_noise: float = 0.0):
#         c, w, h = dataset[0]
#         self.n_pixels = w * h
#
#         self.dataset = dataset
#         self.n_channels = c
#         self.in_dim = c + 1
#         self.rel_noise = rel_noise
#         self.abs_noise = round(self.rel_noise * self.n_pixels)
#         self.seq_len = self.n_pixels + self.abs_noise
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         image, label = self.dataset[idx]
#         assert image.shape[0] == self.n_channels - 1
#         image = image.view(self.n_channels, -1).T
#
#         noise_indices = torch.tensor(random.sample(range(self.seq_len), self.abs_noise))
#         noise_mask = torch.zeros(self.seq_len, dtype=torch.bool)
#         noise_mask[noise_indices] = True
#
#         x = torch.Tensor(self.seq_len, self.in_dim)
#         x[noise_mask, 0] = -1
#         x[noise_mask, 1:] = torch.randn(self.abs_noise, self.n_channels)
#
#         assert sum(~noise_mask) == image.shape[-1]
#
#         x[~noise_mask, 0] = 1
#         x[~noise_mask, 1:] = image
#         x[-1, 0] = 0
#
#         return x, label