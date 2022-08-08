from turtle import forward
import torch

from torch import nn
from patch_embed import Patch_Embedding
from vit import MLP


class SwinBlock(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()

    def forward(self, x):
        return x


class SwinLayer(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.atten = SwinBlock(embed_dim)
        self.atten_norm = nn.LayerNorm(embed_dim)

        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        h = x
        x = self.atten_norm(x)
        x = self.atten(x)
        x = x + h
        
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x


class SwinTransformer(nn.Module):
    def __init__(self, patch_size, embed_dim) -> None:
        super().__init__()
        self.patch_embed = Patch_Embedding(patch_size, 3, embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)