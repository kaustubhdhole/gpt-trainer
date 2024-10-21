# mathematic trick in self attention

import torch
from torch import nn


# B, T, C = 4, 8, 32  # batch, time, channels
# x = torch.randn(B, T, C)
#
# # Averaging all the previous timesteps channelwise
# # (1) Without matrix multiplication
# xbow = torch.zeros((B, T, C))
# for b in range(B):
#     for t in range(T):
#         xprev = x[b, :t + 1]  # (t, C)
#         xbow[b, t] = torch.mean(xprev, 0)
#
# print(x[0])
# print(xbow[0])
#
# # (2) With batched matrix multiplication
# # a = torch.triu(torch.ones(3, 3))
# a = torch.tril(torch.ones(3, 3))
# a = a / torch.sum(a, 1, keepdim=True)
# b = torch.randint(1, 5, (3, 4), dtype=torch.float)
# c = a @ b  # @ is batched matrix multiply
#
# wei = torch.tril(torch.ones(T, T))
# wei = wei / wei.sum(1, keepdim=True)
# xbow2 = wei @ x  # (T, T) @ (B, T, C) -->(T, T) being multiplied to each batch of size (T, C)
# torch.allclose(xbow, xbow2)
#
# # (3)
# tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T, T))
# wei = wei.masked_fill(tril == 0, float('-inf'))
# wei = torch.nn.functional.softmax(wei, dim=-1)
# xbow3 = wei @ x
# torch.allclose(xbow, xbow3)
#
# # self attention head
# head_size = 16
# key = nn.Linear(C, head_size, bias=False)
# query = nn.Linear(C, head_size, bias=False)
# value = nn.Linear(C, head_size, bias=False)
# k = key(x)  # (B, T, head_size)
# q = query(x)  # (B, T, head_size)
# v = value(x) # (B, T, head_size)
# wei = q @ k.transpose(-2, -1)  # (B, T, 16) @ (B, 16, T) --> (B, T, T)
#
# tril = torch.tril(torch.ones(T, T))
# wei = wei.masked_fill(tril == 0, float('-inf'))
# wei = torch.nn.functional.softmax(wei, dim=-1)
#
# out = wei @ v


class Head(torch.nn.Module):
    def __init__(self, embed_size, head_size=16, block_size=8):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, 16) @ (B, 16, T) --> (B, T, T)
        # scaling is used to control the variance at initialisation
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.nn.functional.softmax(wei, dim=-1)
        out = wei @ v  # (B, T, T) x (B, T, T)
        return out


class MultiHead(nn.Module):
    def __init__(self, embed_size, head_size, num_heads, dropout = 0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(embed_size, head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_emd, dropout = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emd, n_emd*4),
            nn.ReLU(),
            nn.Linear(n_emd*4, n_emd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHead(n_embd, head_size, n_heads)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
