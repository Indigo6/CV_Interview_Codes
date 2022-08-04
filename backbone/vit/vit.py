from turtle import forward
from torch import nn
import torch


class Attention(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        self.scale = embed_dim ** -0.5
        self.softmax = nn.Softmax(-1)
    
    def forward(self, x):
        # x: [B, N, embed_dim]
        q,k,v = torch.chunk(self.qkv(x), 3, -1)
        atten = torch.matmul(q, k.T)
        # atten: [B, N, N]
        atten = self.softmax(atten * self.scale)
        out = torch.matmul(atten, v)

        return out



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


class ViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass