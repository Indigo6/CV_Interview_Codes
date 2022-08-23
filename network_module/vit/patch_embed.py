import torch

from torch import nn


class Patch_Embedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim,
                              patch_size, patch_size,
                              bias=False)
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.conv(x)
        x = torch.transpose(torch.flatten(x, 2, 3), 1, 2)
        # x: [B, N, C]
        return x