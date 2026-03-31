import numpy as np
import pytest
from mytorch.tensor import Tensor
from mytorch import nn
import mytorch.functional as F


def test_linear_forward():
    layer = nn.Linear(4, 8)
    x = Tensor(np.random.randn(16, 4), requires_grad=True)
    out = layer(x)
    assert out.shape == (16, 8)


def test_sequential_forward():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    x = Tensor(np.random.randn(5, 4))
    out = model(x)
    assert out.shape == (5, 2)


def test_relu():
    x = Tensor(np.array([-1.0, 0.0, 1.0]), requires_grad=True)
    out = F.relu(x)
    np.testing.assert_allclose(out.data, [0.0, 0.0, 1.0])


def test_cross_entropy():
    logits = Tensor(np.random.randn(8, 4), requires_grad=True)
    targets = Tensor(np.array([0, 1, 2, 3, 0, 1, 2, 3]))
    loss = F.cross_entropy(logits, targets)
    assert loss.data.ndim == 0 or loss.data.shape == ()
