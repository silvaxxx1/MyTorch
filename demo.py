"""
MyTorch Demo
Exercises the core functionality: tensor ops, autograd, training loop,
data loading, and key nn modules.
"""

import numpy as np
import mytorch
import mytorch.functional as F
from mytorch import nn, optim, data, utils
from mytorch.tensor import Tensor, cat, stack

# ─────────────────────────────────────────────
# 1. Tensor ops & autograd
# ─────────────────────────────────────────────
print("=" * 60)
print("1. Tensor ops & autograd")
print("=" * 60)

a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

c = a @ b          # matmul
d = c.sum()        # scalar
d.backward()

print(f"a @ b =\n{c.data}")
print(f"grad of a:\n{a.grad}")
print(f"grad of b:\n{b.grad}")

# ─────────────────────────────────────────────
# 2. Activation functions
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Activation functions")
print("=" * 60)

x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"relu:    {F.relu(x).data}")
print(f"sigmoid: {np.round(F.sigmoid(x).data, 4)}")
print(f"tanh:    {np.round(F.tanh(x).data, 4)}")
print(f"gelu:    {np.round(F.gelu(x).data, 4)}")

# ─────────────────────────────────────────────
# 3. Linear layer + backward
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Linear layer + backward")
print("=" * 60)

layer = nn.Linear(4, 8)
x = Tensor(np.random.randn(16, 4), requires_grad=True)
out = layer(x)
loss = out.sum()
loss.backward()
print(f"input:  {x.shape}  →  output: {out.shape}")
print(f"weight grad shape: {layer._params[0].grad.shape}")

# ─────────────────────────────────────────────
# 4. Full training loop (XOR-like classification)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Training loop — 4-class classification")
print("=" * 60)

mytorch.manual_seed(42)
np.random.seed(42)

N, D, C = 512, 16, 4
X = np.random.randn(N, D).astype(np.float32)
y = np.random.randint(0, C, N)

dataset = data.TensorDataset(Tensor(X), Tensor(y))
loader  = data.DataLoader(dataset, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(D, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, C),
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 6):
    total_loss = 0.0
    model.train()
    for xb, yb in loader:
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.data)

    print(f"  epoch {epoch}/5  loss={total_loss / len(loader):.4f}")

# ─────────────────────────────────────────────
# 5. Normalization layers
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Normalization layers")
print("=" * 60)

x = Tensor(np.random.randn(8, 16).astype(np.float32))

bn = nn.BatchNorm1d(16)
ln = nn.LayerNorm(16)

bn.train()
out_bn = bn(x)
out_ln = ln(x)
print(f"BatchNorm1d output shape: {out_bn.shape}")
print(f"LayerNorm   output shape: {out_ln.shape}")

# ─────────────────────────────────────────────
# 6. Embedding
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. Embedding")
print("=" * 60)

vocab_size, embed_dim = 100, 32
emb = nn.Embedding(vocab_size, embed_dim)
token_ids = Tensor(np.array([0, 5, 42, 7, 3]))
embedded = emb(token_ids)
print(f"token ids shape: {token_ids.shape}  →  embedded: {embedded.shape}")

# ─────────────────────────────────────────────
# 7. Multi-head attention
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. MultiHeadAttention")
print("=" * 60)

B, T, d_model = 2, 10, 64
mha = nn.MultiHeadAttention(embed_dim=d_model, num_heads=4)
q = Tensor(np.random.randn(B, T, d_model).astype(np.float32))
k = Tensor(np.random.randn(B, T, d_model).astype(np.float32))
v = Tensor(np.random.randn(B, T, d_model).astype(np.float32))
attn_out = mha(q, k, v)
print(f"Q/K/V: {q.shape}  →  attention output: {attn_out.shape}")

# ─────────────────────────────────────────────
# 8. Optimizers
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. Optimizers")
print("=" * 60)

def make_model():
    return nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))

def train_one_epoch(model, optimizer):
    x = Tensor(np.random.randn(32, 8).astype(np.float32))
    y = Tensor(np.random.randint(0, 2, 32))
    loss = F.cross_entropy(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.data)

for name, opt_cls, kwargs in [
    ("SGD",   optim.SGD,   {"lr": 0.01}),
    ("Adam",  optim.Adam,  {"lr": 0.001}),
    ("AdamW", optim.AdamW, {"lr": 0.001, "weight_decay": 0.01}),
]:
    m = make_model()
    o = opt_cls(m.parameters(), **kwargs)
    loss = train_one_epoch(m, o)
    print(f"  {name:6s}  loss={loss:.4f}")

# ─────────────────────────────────────────────
# 9. LR Schedulers
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. LR Schedulers")
print("=" * 60)

m = make_model()
base_optimizer = optim.SGD(m.parameters(), lr=0.1)
scheduler = optim.StepLR(base_optimizer, step_size=3, gamma=0.5)
lrs = []
for _ in range(9):
    lrs.append(round(base_optimizer.lr, 5))
    scheduler.step()
print(f"  StepLR (step=3, gamma=0.5): {lrs}")

# ─────────────────────────────────────────────
# 10. Checkpointing
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("10. Checkpointing")
print("=" * 60)

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
ckpt_optimizer = optim.Adam(model.parameters(), lr=1e-3)
utils.save_checkpoint(model, ckpt_optimizer, epoch=1, filepath="demo_checkpoint.pkl")
utils.load_checkpoint("demo_checkpoint.pkl", model=model, optimizer=ckpt_optimizer)
n_params = utils.count_parameters(model)
print(f"  Saved & loaded checkpoint. Parameters: {n_params}")

import os
os.remove("demo_checkpoint.pkl")

# ─────────────────────────────────────────────
# 11. cat / stack
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("11. cat / stack")
print("=" * 60)

a = Tensor(np.ones((3, 4)))
b = Tensor(np.ones((3, 4)) * 2)
print(f"  cat axis=0:  {cat([a, b], axis=0).shape}")
print(f"  cat axis=1:  {cat([a, b], axis=1).shape}")
print(f"  stack axis=0: {stack([a, b], axis=0).shape}")

print("\n" + "=" * 60)
print("All demos completed successfully.")
print("=" * 60)
