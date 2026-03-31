import numpy as np
import pytest
from mytorch.tensor import Tensor


def test_add_forward():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = a + b
    np.testing.assert_allclose(c.data, [5.0, 7.0, 9.0])


def test_mul_backward():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    c = (a * b).sum()
    c.backward()
    np.testing.assert_allclose(a.grad, [4.0, 5.0])
    np.testing.assert_allclose(b.grad, [2.0, 3.0])


def test_matmul_backward():
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(4, 5), requires_grad=True)
    c = (a @ b).sum()
    c.backward()
    assert a.grad.shape == (3, 4)
    assert b.grad.shape == (4, 5)


def test_reshape():
    a = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    b = a.reshape(3, 2)
    b.sum().backward()
    assert a.grad.shape == (2, 3)


def test_to_device_cpu():
    a = Tensor([1.0, 2.0], device="cpu")
    b = a.to("cpu")
    assert b.device == "cpu"
