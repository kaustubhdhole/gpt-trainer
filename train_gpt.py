# read file

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

max_iters = 20000
eval_interval = 1000
eval_iters = 200
embed_dim = 32
block_size = 8
batch_size = 4

with open("data/tiny_shakespeare.txt", 'r+') as rfile:
    data = rfile.read()

chars = sorted(list(set(data)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(data), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]

torch.manual_seed(1337)

def sample_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # [3,343,8894,12]
    x = torch.stack([data[i:i + block_size] for i in ix])  #
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x = x.to(device);
    y = y.to(device)
    return x, y


xb, yb = sample_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

from torch import nn
from torch.nn import functional as F
import torch
from torch import nn
from torch.nn import functional as F
import torch

from self_attention import Head, MultiHead, FeedForward, Block


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        head_size = 16
        #self.sa_head = Head(embed_dim, head_size)
        # self.sa_head = MultiHead(embed_dim, head_size, 4)
        # self.ffnet = FeedForward(head_size)
        self.blocks = nn.Sequential(
            Block(embed_dim, n_heads=4),
            Block(embed_dim, n_heads=4),
            Block(embed_dim, n_heads=4),
            nn.LayerNorm(embed_dim)
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos_embeddings = self.position_embedding_table(torch.arange(T, device=device))
        token_embeddings = self.token_embedding_table(idx) # (B, T, C)
        x = token_embeddings + pos_embeddings
        # x = self.sa_head(x) # B, T,
        # x = self.ffnet(x) # applied per token level
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape # (batch size, time steps, channel size (vocab size))
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) # or targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond) #
            # focus on the last time step
            logits = logits[:, -1,:] # becomes (B, C) from (B, T, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample 1
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size, embed_dim, block_size)
model = model.to(device)
logits, loss = model(xb, yb)
print(logits.shape)
print(loss)


# average up loss over multiple batches to get better loss estimate
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = sample_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # because some layers will have different behaviours at training and inference time
    return out


print(decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

batch_size = 128
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:4f}")
    xb, yb = sample_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=10000)[0].tolist()))
