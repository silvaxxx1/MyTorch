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


def test_cross_entropy_backward():
    logits = Tensor(np.random.randn(4, 3), requires_grad=True)
    targets = Tensor(np.array([0, 1, 2, 1]))
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_linear_backward():
    layer = nn.Linear(4, 2)
    x = Tensor(np.random.randn(8, 4), requires_grad=True)
    loss = layer(x).sum()
    loss.backward()
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


def test_conv2d_forward():
    conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
    x = Tensor(np.random.randn(2, 1, 8, 8), requires_grad=True)
    out = conv(x)
    assert out.shape == (2, 4, 8, 8)


def test_conv2d_backward():
    conv = nn.Conv2d(2, 4, kernel_size=3, padding=1)
    x = Tensor(np.random.randn(2, 2, 6, 6), requires_grad=True)
    out = conv(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert conv.weight.grad is not None and conv.weight.grad.shape == conv.weight.shape
    assert conv.bias.grad is not None


def test_maxpool2d_backward():
    pool = nn.MaxPool2d(2)
    x = Tensor(np.random.randn(2, 4, 8, 8), requires_grad=True)
    out = pool(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


def test_batchnorm1d_backward():
    bn = nn.BatchNorm1d(8)
    x = Tensor(np.random.randn(16, 8), requires_grad=True)
    out = bn(x)
    out.sum().backward()
    assert x.grad is not None
    assert bn.gamma.grad is not None
    assert bn.beta.grad is not None


def test_rnn_backward():
    rnn = nn.RNN(input_size=8, hidden_size=16)
    x = Tensor(np.random.randn(5, 3, 8), requires_grad=True)
    out, h = rnn(x)
    assert out.shape == (5, 3, 16)
    out.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


def test_lstm_backward():
    lstm = nn.LSTM(input_size=8, hidden_size=16)
    x = Tensor(np.random.randn(5, 3, 8), requires_grad=True)
    out, h, c = lstm(x)
    assert out.shape == (5, 3, 16)
    out.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


def test_mse_loss_backward():
    pred = Tensor(np.random.randn(4, 2), requires_grad=True)
    target = Tensor(np.random.randn(4, 2))
    loss = F.mse_loss(pred, target)
    loss.backward()
    assert pred.grad is not None and pred.grad.shape == pred.shape
