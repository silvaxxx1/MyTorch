# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MyTorch** is a deep learning framework built from scratch with a PyTorch-inspired API.

- **v1 (stable, tagged `v1.0.0`)** — DNN, CNN, RNN core with full autograd. CPU (NumPy) and GPU (CuPy) backends via a `tensor.xp` abstraction.
- **v2 (in development)** — New architecture built around 4 pillars (see below).

## v2 Development Direction

### Pillar 1 — Unified Array Backend
Replace the scattered `xp = np/cp` pattern with a proper `backend/` module containing an `Array` class that unifies NumPy and CuPy under one device-agnostic API. All of v2 is built on this.

### Pillar 2 — GPT-2 (manual backward, no autograd)
Standalone `gpt2/` directory. GPT-2 architecture implemented with fully manual forward and backward passes. Purpose: understand which ops need fusing before the autograd engine handles them automatically.

### Pillar 3 — Robust Autograd Engine
Redesign autograd using a `Function` class pattern with static `forward(ctx, ...)` and `backward(ctx, grad)` methods. Built on the Array backend. Replaces v1's closure-per-op approach.

### Pillar 4 — Triton Kernel Fusion
Custom CUDA kernels via Triton for fused ops: flash attention, fused LayerNorm, fused GELU/matmul, fused cross-entropy, fused Adam. Each kernel is a `Function` subclass so it integrates naturally with the autograd engine.

**Build order:** Pillar 1 → (Pillar 2 in parallel with Pillar 3) → Pillar 4.

---

## Commands

```bash
# Install dependencies and create .venv (uses uv)
uv sync

# Activate the environment
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows

# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_tensor.py
python -m pytest tests/test_nn.py -v

# Run examples
PYTHONPATH=. python examples/dnn.py
PYTHONPATH=. python examples/cnn.py
PYTHONPATH=. python examples/rnn.py
```

## v1 Architecture

```
mytorch/
├── tensor.py        # Tensor class + autograd engine
├── functional.py    # Stateless activations and loss functions
├── nn/              # Linear, Conv2d, RNN, GRU, LSTM, BatchNorm, Dropout, Embedding
├── optim/           # SGD, Adam, AdamW, RMSprop + LR schedulers
├── data/            # Dataset, TensorDataset, DataLoader
└── utils/           # Checkpointing, metrics, gradient clipping, EarlyStopping
tests/               # All tests live here
```

### Tensor & Autograd (`mytorch/tensor.py`)

`Tensor` wraps a NumPy or CuPy array. Key attributes:
- `.data` — the underlying array
- `.xp` — the backend module (`numpy` or `cupy`)
- `.device` — `"cpu"` or `"gpu"`
- `._prev` — set of parent `Tensor` nodes in the computational graph
- `._grad_fn` — closure that propagates gradients to `._prev`

`backward()` builds a topological ordering of `_prev` nodes and calls each `_grad_fn` in reverse. Gradients accumulate in `.grad` with `+=`.

### CPU/GPU Pattern (v1)

Use `tensor.xp` instead of `np` to write device-agnostic ops:
```python
xp = tensor.xp
out = xp.maximum(0, tensor.data)  # works on both CPU and GPU
```

### Modules (`mytorch/nn/`)

All layers inherit from `Module` (`nn/modules.py`). Key methods:
- `parameters()` — recursively yields all `Tensor` leaves with `requires_grad=True`
- `__call__` → `forward()` — standard forward pass hook
- `train()` / `eval()` — toggles training mode for dropout/batchnorm

### Weight Initialization

He initialization for ReLU networks, Xavier for tanh/sigmoid — applied in each layer's `__init__`.
