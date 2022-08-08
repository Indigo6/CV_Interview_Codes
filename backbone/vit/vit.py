import torch

from torch import nn
from attention import MultiHeadAttention


class MLP(nn.Module):
    def __init__(self, in_features, split_ratio=3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features * split_ratio)
        self.fc2 = nn.Linear(in_features * split_ratio, in_features)
        self.activate = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)

        return x


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


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
        self.ffn = MLP(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        self.atten = MultiHeadAttention(embed_dim)
        self.atten_norm = nn.LayerNorm(embed_dim)

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


class ViT(nn.Module):
    def __init__(self, patch_size, embed_dim) -> None:
        super().__init__()
        self.patch_embed = Patch_Embedding(patch_size, 3, embed_dim)
        layer_list = [EncoderLayer(embed_dim) for i in range(5)]
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x):
        x = self.patch_embed(x)
        print("Patch Embedding Shape: ", x.shape)

        for layer in self.layers:
            x = layer(x)

        return x

if __name__ == "__main__":
    image = torch.ones([1, 3, 256, 256])
    vit = ViT(patch_size=32, embed_dim=16)
    out = vit(image)
    print("Output Shape: ", out.shape)