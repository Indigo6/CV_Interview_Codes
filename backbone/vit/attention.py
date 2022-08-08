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
        atten = torch.matmul(q, k.transpose(1, 2))
        # atten: [B, N, N]
        atten = self.softmax(atten * self.scale)
        out = torch.matmul(atten, v)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8) -> None:
        super().__init__()
        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.softmax = nn.Softmax(-1)
    
    def forward(self, x):
        # x: [B, N, embed_dim]
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, self.num_heads, -1) # qkv: [B, N, num_heads, head_dim*3)
        q,k,v = torch.chunk(qkv.transpose(1, 2), 3, -1)
        atten = torch.matmul(q, k.transpose(-2, -1))    # atten: [B, num_heads, N, N]
        atten = self.softmax(atten * self.scale)
        out = torch.matmul(atten, v)    # out: [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).flatten(2, 3) # out: [B, N, embed_dim]

        return out