# MyTorch

A minimal deep learning framework built from scratch, inspired by PyTorch. Supports both CPU (NumPy) and GPU (CuPy) backends through a unified API.

---

## Features

- **Autograd** — automatic differentiation with a dynamic computational graph
- **CPU / GPU** — seamless switching between NumPy and CuPy backends via `.device`
- **Neural network layers** — Linear, Conv2d, RNN, GRU, LSTM, MultiHeadAttention, Transformer, Embedding, Dropout, BatchNorm, LayerNorm, and more
- **Optimizers** — SGD, Adam, AdamW, RMSprop with momentum and weight decay
- **LR Schedulers** — StepLR, CosineAnnealingLR, OneCycleLR
- **Data utilities** — Dataset, TensorDataset, DataLoader with shuffle and batching
- **Training utilities** — gradient clipping, early stopping, checkpointing, metrics

---

## Installation

```bash
git clone https://github.com/silvaxxx1/SilvaXNet
cd SilvaXNet
uv sync
uv pip install -e .
```

Requires Python 3.12+. GPU support requires CUDA 12.x and `cupy-cuda12x`.

---

## Quick Start

```python
import numpy as np
import mytorch.functional as F
from mytorch import nn, optim, data
from mytorch.tensor import Tensor

# Data
X = Tensor(np.random.randn(512, 16).astype("float32"))
y = Tensor(np.random.randint(0, 4, 512))
loader = data.DataLoader(data.TensorDataset(X, y), batch_size=64, shuffle=True)

# Model
model = nn.Sequential(
    nn.Linear(16, 64),
    nn.ReLU(),
    nn.Linear(64, 4),
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    for xb, yb in loader:
        loss = F.cross_entropy(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Package Structure

```
mytorch/
├── tensor.py        # Tensor class + autograd engine
├── functional.py    # Stateless activations and loss functions
├── nn/              # Layer modules
│   ├── linear.py
│   ├── conv.py
│   ├── rnn.py       # RNN, GRU, LSTM (+ cell variants)
│   ├── attention.py # MultiHeadAttention
│   ├── transformer.py
│   ├── normalization.py
│   ├── embedding.py
│   └── dropout.py
├── optim/           # Optimizers + LR schedulers
├── data/            # Dataset, TensorDataset, DataLoader
└── utils/           # Checkpointing, metrics, gradient clipping, EarlyStopping
```

---

## CPU / GPU

```python
from mytorch.tensor import Tensor

# CPU (default)
x = Tensor([1.0, 2.0, 3.0], device="cpu")

# GPU (requires CuPy)
x = Tensor([1.0, 2.0, 3.0], device="gpu")

# Move between devices
x_gpu = x.to("gpu")
x_cpu = x_gpu.to("cpu")

# Move a whole model to GPU
model.to("gpu")
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

## Saving and Loading Checkpoints

```python
from mytorch import utils

# Save
utils.save_checkpoint(model, optimizer, epoch=5, filepath="checkpoint.pkl")

# Load
utils.load_checkpoint("checkpoint.pkl", model=model, optimizer=optimizer)
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

---

## License

MIT
