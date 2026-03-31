# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MyTorch** is a minimal deep learning framework built from scratch with a PyTorch-inspired API. It supports both CPU (NumPy) and GPU (CuPy) backends via a unified `xp` abstraction.

## Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_tensor.py
uv run pytest tests/test_nn.py -v

# Run a single test
uv run pytest tests/test_tensor.py::test_add_forward
```

## Architecture

```
mytorch/
├── tensor.py        # Tensor class + autograd engine
├── functional.py    # Stateless activations and loss functions
├── nn/              # Layer modules (Linear, Conv2d, RNN, LSTM, attention, etc.)
├── optim/           # Optimizers (SGD, Adam, AdamW, RMSprop) + LR schedulers
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

### CPU/GPU Pattern

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
