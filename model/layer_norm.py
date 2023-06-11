import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + 1e-5) + self.beta
