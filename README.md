<p align="center">
  <img src="MyTorch.png" alt="MyTorch Logo" width="200"/>
</p>

<h1 align="center">MyTorch</h1>

<p align="center">
  A deep learning framework built from scratch — autograd, CNN, RNN, and Triton-fused kernels coming next.
</p>

---

## Overview

MyTorch is a PyTorch-inspired deep learning framework written from scratch. **v1** is a stable, fully functional core for DNN, CNN, and RNN projects with complete forward and backward passes. **v2** is actively being developed with a new architecture aimed at production-quality training speed.

---

## Roadmap

### v1 — Stable (current)
- Full autograd engine with dynamic computational graph
- DNN, CNN, RNN/GRU/LSTM — all with correct backward passes
- Optimizers: SGD, Adam, AdamW, RMSprop + LR schedulers
- Data utilities: Dataset, TensorDataset, DataLoader
- Training utilities: gradient clipping, early stopping, checkpointing, metrics
- CPU (NumPy) and GPU (CuPy) backends

### v2 — In Development

| Pillar | Description |
|--------|-------------|
| **1 — Array Backend** | Unified `Array` module replacing the scattered `xp` pattern. Single device-agnostic API over NumPy and CuPy. Foundation for everything below. |
| **2 — GPT-2 (manual)** | GPT-2 implemented from scratch with fully manual forward and backward passes — no autograd. Serves as a reference to understand exactly what the engine needs to handle. |
| **3 — Robust Autograd** | Redesigned autograd using a `Function` class pattern (static `forward`/`backward`). More composable, easier to extend, and built on the new Array backend. |
| **4 — Triton Kernels** | Custom CUDA kernels via [Triton](https://github.com/openai/triton) for fused ops: flash attention, fused LayerNorm, fused GELU/matmul, fused cross-entropy, fused optimizer steps. |

---

## Installation

```bash
git clone https://github.com/silvaxxx1/MyTorch.git
cd MyTorch
uv sync
```

Requires Python 3.12+. GPU support requires CUDA 12.x and `cupy-cuda12x`.

---

## Quick Start

```python
import numpy as np
import mytorch.functional as F
from mytorch import nn, optim, data
from mytorch.tensor import Tensor

X = Tensor(np.random.randn(512, 16).astype("float32"))
y = Tensor(np.random.randint(0, 4, 512))
loader = data.DataLoader(data.TensorDataset(X, y), batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(16, 64),
    nn.ReLU(),
    nn.Linear(64, 4),
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for xb, yb in loader:
        loss = F.cross_entropy(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Examples

End-to-end runnable examples covering all three supported architectures:

| File | Architecture | What it covers |
|------|-------------|----------------|
| [`examples/dnn.py`](examples/dnn.py) | DNN | Multi-class classification — BatchNorm, Dropout, Adam + StepLR, early stopping, checkpointing |
| [`examples/cnn.py`](examples/cnn.py) | CNN | Image classification — Conv2d → BatchNorm2d → MaxPool2d → FC head |
| [`examples/rnn.py`](examples/rnn.py) | RNN / GRU / LSTM | Sequence classification — all three recurrent architectures compared side-by-side |

```bash
PYTHONPATH=. python examples/dnn.py
PYTHONPATH=. python examples/cnn.py
PYTHONPATH=. python examples/rnn.py
```

---

## Package Structure (v1)

```
mytorch/
├── tensor.py        # Tensor class + autograd engine
├── functional.py    # Activations and loss functions
├── nn/              # Linear, Conv2d, RNN, GRU, LSTM, BatchNorm, Dropout, Embedding...
├── optim/           # SGD, Adam, AdamW, RMSprop + LR schedulers
├── data/            # Dataset, TensorDataset, DataLoader
└── utils/           # Checkpointing, metrics, gradient clipping, EarlyStopping
examples/
├── dnn.py
├── cnn.py
└── rnn.py
```

---

## Autograd

```python
from mytorch.tensor import Tensor

a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

c = (a @ b).sum()
c.backward()

print(a.grad)  # dc/da
print(b.grad)  # dc/db
```

---

## Running Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

---

## License

MIT
