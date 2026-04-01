import numpy as np
import pytest
from mytorch.tensor import Tensor, cat, stack


# ── Gradient checker ──────────────────────────────────────────────────────────

def gradcheck(fn, inputs, eps=1e-5, rtol=1e-3, atol=1e-4):
    """Compare analytic grads against numerical finite differences."""
    # analytic
    outputs = fn(*inputs)
    outputs.sum().backward()

    for x in inputs:
        if not x.requires_grad:
            continue
        analytic = x.grad.copy()
        numeric = np.zeros_like(x.data)
        for idx in np.ndindex(x.shape):
            x.data[idx] += eps
            fp = fn(*[Tensor(t.data.copy(), requires_grad=t.requires_grad) for t in inputs]).data.sum()
            x.data[idx] -= 2 * eps
            fm = fn(*[Tensor(t.data.copy(), requires_grad=t.requires_grad) for t in inputs]).data.sum()
            x.data[idx] += eps
            numeric[idx] = (fp - fm) / (2 * eps)
        np.testing.assert_allclose(analytic, numeric, rtol=rtol, atol=atol,
                                   err_msg=f"Grad mismatch for input shape {x.shape}")


# ── Forward ops ───────────────────────────────────────────────────────────────

def test_add_forward():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    np.testing.assert_allclose((a + b).data, [5.0, 7.0, 9.0])


def test_sub_forward():
    a = Tensor([5.0, 6.0])
    b = Tensor([1.0, 2.0])
    np.testing.assert_allclose((a - b).data, [4.0, 4.0])


def test_mul_forward():
    a = Tensor([2.0, 3.0])
    b = Tensor([4.0, 5.0])
    np.testing.assert_allclose((a * b).data, [8.0, 15.0])


def test_div_forward():
    a = Tensor([6.0, 9.0])
    b = Tensor([2.0, 3.0])
    np.testing.assert_allclose((a / b).data, [3.0, 3.0])


def test_pow_forward():
    a = Tensor([2.0, 3.0])
    np.testing.assert_allclose((a ** 2).data, [4.0, 9.0])


def test_neg_forward():
    a = Tensor([1.0, -2.0])
    np.testing.assert_allclose((-a).data, [-1.0, 2.0])


# ── Backward: scalar ops ──────────────────────────────────────────────────────

def test_mul_backward():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    (a * b).sum().backward()
    np.testing.assert_allclose(a.grad, [4.0, 5.0])
    np.testing.assert_allclose(b.grad, [2.0, 3.0])


def test_sub_backward():
    a = Tensor([3.0, 4.0], requires_grad=True)
    b = Tensor([1.0, 2.0], requires_grad=True)
    (a - b).sum().backward()
    np.testing.assert_allclose(a.grad, [1.0, 1.0])
    np.testing.assert_allclose(b.grad, [-1.0, -1.0])


def test_div_backward():
    def fn(a, b): return a / b
    a = Tensor(np.array([4.0, 9.0]), requires_grad=True)
    b = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    gradcheck(fn, [a, b])


def test_pow_backward():
    def fn(a): return a ** 3
    a = Tensor(np.array([1.0, 2.0, -1.0]), requires_grad=True)
    gradcheck(fn, [a])


def test_neg_backward():
    a = Tensor([1.0, -2.0, 3.0], requires_grad=True)
    (-a).sum().backward()
    np.testing.assert_allclose(a.grad, [-1.0, -1.0, -1.0])


def test_matmul_backward():
    def fn(a, b): return a @ b
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(4, 5), requires_grad=True)
    gradcheck(fn, [a, b])


# ── Reduce ops ────────────────────────────────────────────────────────────────

def test_sum_backward():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    a.sum().backward()
    np.testing.assert_allclose(a.grad, np.ones((2, 2)))


def test_sum_axis_backward():
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    a.sum(axis=1).sum().backward()
    np.testing.assert_allclose(a.grad, np.ones((3, 4)))


def test_mean_backward():
    a = Tensor(np.array([1.0, 2.0, 3.0, 4.0]), requires_grad=True)
    a.mean().backward()
    np.testing.assert_allclose(a.grad, np.full(4, 0.25))


# ── Shape ops ─────────────────────────────────────────────────────────────────

def test_reshape_backward():
    a = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    a.reshape(3, 2).sum().backward()
    assert a.grad.shape == (2, 3)
    np.testing.assert_allclose(a.grad, np.ones((2, 3)))


def test_transpose_backward():
    def fn(a): return a.transpose(0, 2, 1)
    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    gradcheck(fn, [a])


def test_squeeze_backward():
    a = Tensor(np.ones((3, 1, 4)), requires_grad=True)
    a.squeeze(1).sum().backward()
    assert a.grad.shape == (3, 1, 4)
    np.testing.assert_allclose(a.grad, np.ones((3, 1, 4)))


def test_unsqueeze_backward():
    a = Tensor(np.ones((3, 4)), requires_grad=True)
    a.unsqueeze(0).sum().backward()
    assert a.grad.shape == (3, 4)


# ── Broadcast backward ────────────────────────────────────────────────────────

def test_broadcast_add_backward():
    # (3, 4) + (4,) — bias-style broadcast
    def fn(a, b): return a + b
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(4), requires_grad=True)
    gradcheck(fn, [a, b])


def test_broadcast_mul_backward():
    def fn(a, b): return a * b
    a = Tensor(np.random.randn(2, 3), requires_grad=True)
    b = Tensor(np.random.randn(3), requires_grad=True)
    gradcheck(fn, [a, b])


# ── Indexing ──────────────────────────────────────────────────────────────────

def test_getitem_backward():
    a = Tensor(np.arange(12.0).reshape(3, 4), requires_grad=True)
    a[1].sum().backward()
    expected = np.zeros((3, 4))
    expected[1] = 1.0
    np.testing.assert_allclose(a.grad, expected)


def test_advanced_index_backward():
    # used by cross_entropy internally
    a = Tensor(np.random.randn(4, 5), requires_grad=True)
    idx = (np.array([0, 1, 2, 3]), np.array([1, 2, 0, 4]))
    a[idx].sum().backward()
    expected = np.zeros((4, 5))
    expected[idx] = 1.0
    np.testing.assert_allclose(a.grad, expected)


# ── cat / stack ───────────────────────────────────────────────────────────────

def test_cat_equal_backward():
    a = Tensor(np.ones((3, 4)), requires_grad=True)
    b = Tensor(np.ones((3, 4)) * 2, requires_grad=True)
    cat([a, b], axis=0).sum().backward()
    np.testing.assert_allclose(a.grad, np.ones((3, 4)))
    np.testing.assert_allclose(b.grad, np.ones((3, 4)))


def test_cat_unequal_backward():
    # the bug we fixed — unequal sizes along concat axis
    a = Tensor(np.ones((2, 4)), requires_grad=True)
    b = Tensor(np.ones((5, 4)) * 2, requires_grad=True)
    cat([a, b], axis=0).sum().backward()
    assert a.grad.shape == (2, 4)
    assert b.grad.shape == (5, 4)
    np.testing.assert_allclose(a.grad, np.ones((2, 4)))
    np.testing.assert_allclose(b.grad, np.ones((5, 4)))


def test_stack_backward():
    a = Tensor(np.ones((3, 4)), requires_grad=True)
    b = Tensor(np.ones((3, 4)) * 2, requires_grad=True)
    stack([a, b], axis=0).sum().backward()
    np.testing.assert_allclose(a.grad, np.ones((3, 4)))
    np.testing.assert_allclose(b.grad, np.ones((3, 4)))


# ── Grad bookkeeping ──────────────────────────────────────────────────────────

def test_grad_accumulates():
    a = Tensor([1.0, 2.0], requires_grad=True)
    (a * 2).sum().backward()
    first = a.grad.copy()
    (a * 3).sum().backward()
    np.testing.assert_allclose(a.grad, first + np.array([3.0, 3.0]))


def test_zero_grad():
    a = Tensor([1.0, 2.0], requires_grad=True)
    (a * 2).sum().backward()
    assert a.grad is not None
    a.zero_grad()
    assert a.grad is None


def test_no_grad_leaf():
    a = Tensor([1.0, 2.0], requires_grad=False)
    b = Tensor([3.0, 4.0], requires_grad=True)
    (a + b).sum().backward()
    assert a.grad is None
    assert b.grad is not None


def test_to_device_cpu():
    a = Tensor([1.0, 2.0], device="cpu")
    assert a.to("cpu").device == "cpu"
