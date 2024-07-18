import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import torch.nn as nn
import numpy as np


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=512):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp( torch.cat([self.embed(input, self.basis), input], dim=2))  # B x N x C
        return embed


if __name__ == "__main__":
    model = PointEmbed()
    input = torch.randn(2, 1024, 3)
    output = model(input)
    print(output.shape)
