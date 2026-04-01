import numpy as np
import pytest
from mytorch.tensor import Tensor
from mytorch import nn
import mytorch.functional as F
from mytorch import optim, data, utils


# ── Helpers ───────────────────────────────────────────────────────────────────

def gradcheck(fn, inputs, eps=1e-4, rtol=1e-2, atol=1e-3):
    """Numerical gradient check — only for small deterministic layers."""
    out = fn(*inputs)
    out.sum().backward()
    for x in inputs:
        if not x.requires_grad:
            continue
        analytic = x.grad.copy()
        numeric = np.zeros_like(x.data)
        for idx in np.ndindex(x.shape):
            orig = x.data[idx]
            x.data[idx] = orig + eps
            fp = fn(*[Tensor(t.data.copy(), requires_grad=t.requires_grad) for t in inputs]).data.sum()
            x.data[idx] = orig - eps
            fm = fn(*[Tensor(t.data.copy(), requires_grad=t.requires_grad) for t in inputs]).data.sum()
            x.data[idx] = orig
            numeric[idx] = (fp - fm) / (2 * eps)
        np.testing.assert_allclose(analytic, numeric, rtol=rtol, atol=atol)


# ── Linear ────────────────────────────────────────────────────────────────────

def test_linear_forward():
    layer = nn.Linear(4, 8)
    x = Tensor(np.random.randn(16, 4), requires_grad=True)
    assert layer(x).shape == (16, 8)


def test_linear_backward():
    layer = nn.Linear(4, 2)
    x = Tensor(np.random.randn(8, 4), requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None and x.grad.shape == (8, 4)
    assert layer.weight.grad is not None and layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad is not None and layer.bias.grad.shape == layer.bias.shape


def test_linear_gradcheck():
    layer = nn.Linear(3, 2)
    x = Tensor(np.random.randn(4, 3), requires_grad=True)
    gradcheck(lambda x: layer(x), [x])


def test_sequential_forward():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    assert model(Tensor(np.random.randn(5, 4))).shape == (5, 2)


# ── Activations ───────────────────────────────────────────────────────────────

def test_relu_values():
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), requires_grad=True)
    out = F.relu(x)
    np.testing.assert_allclose(out.data, [0.0, 0.0, 0.0, 1.0, 2.0])
    out.sum().backward()
    np.testing.assert_allclose(x.grad, [0.0, 0.0, 0.0, 1.0, 1.0])


def test_sigmoid_backward():
    x = Tensor(np.array([0.0]), requires_grad=True)
    F.sigmoid(x).backward()
    np.testing.assert_allclose(x.grad, [0.25], atol=1e-6)  # sig(0)*(1-sig(0)) = 0.25


def test_tanh_backward():
    x = Tensor(np.array([0.0]), requires_grad=True)
    F.tanh(x).backward()
    np.testing.assert_allclose(x.grad, [1.0], atol=1e-6)  # 1 - tanh(0)^2 = 1


def test_gelu_backward():
    def fn(x): return F.gelu(x)
    x = Tensor(np.random.randn(4), requires_grad=True)
    gradcheck(fn, [x])


# ── Loss functions ────────────────────────────────────────────────────────────

def test_cross_entropy_scalar():
    logits = Tensor(np.random.randn(8, 4), requires_grad=True)
    loss = F.cross_entropy(logits, Tensor(np.array([0, 1, 2, 3, 0, 1, 2, 3])))
    assert loss.data.shape == ()


def test_cross_entropy_backward():
    logits = Tensor(np.random.randn(4, 3), requires_grad=True)
    F.cross_entropy(logits, Tensor(np.array([0, 1, 2, 1]))).backward()
    assert logits.grad is not None and logits.grad.shape == (4, 3)


def test_cross_entropy_gradcheck():
    logits = Tensor(np.random.randn(3, 4), requires_grad=True)
    targets = np.array([0, 2, 1])
    gradcheck(lambda x: F.cross_entropy(x, Tensor(targets)), [logits])


def test_mse_loss_backward():
    pred = Tensor(np.random.randn(4, 2), requires_grad=True)
    F.mse_loss(pred, Tensor(np.random.randn(4, 2))).backward()
    assert pred.grad is not None and pred.grad.shape == (4, 2)


def test_mae_loss_backward():
    pred = Tensor(np.random.randn(4, 2), requires_grad=True)
    F.mae_loss(pred, Tensor(np.random.randn(4, 2))).backward()
    assert pred.grad is not None and pred.grad.shape == (4, 2)


def test_bce_loss_backward():
    pred = Tensor(np.random.rand(8).astype(np.float64) * 0.8 + 0.1, requires_grad=True)
    target = Tensor((np.random.rand(8) > 0.5).astype(np.float64))
    F.binary_cross_entropy(pred, target).backward()
    assert pred.grad is not None and pred.grad.shape == (8,)


def test_huber_loss_backward():
    pred = Tensor(np.random.randn(4, 2), requires_grad=True)
    F.huber_loss(pred, Tensor(np.random.randn(4, 2))).backward()
    assert pred.grad is not None and pred.grad.shape == (4, 2)


# ── Conv2d ────────────────────────────────────────────────────────────────────

def test_conv2d_forward():
    conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
    assert conv(Tensor(np.random.randn(2, 1, 8, 8))).shape == (2, 4, 8, 8)


def test_conv2d_no_padding_shape():
    conv = nn.Conv2d(3, 8, kernel_size=3)
    assert conv(Tensor(np.random.randn(2, 3, 10, 10))).shape == (2, 8, 8, 8)


def test_conv2d_backward():
    conv = nn.Conv2d(2, 4, kernel_size=3, padding=1)
    x = Tensor(np.random.randn(2, 2, 6, 6), requires_grad=True)
    conv(x).sum().backward()
    assert x.grad is not None and x.grad.shape == (2, 2, 6, 6)
    assert conv.weight.grad.shape == conv.weight.shape
    assert conv.bias.grad.shape == conv.bias.shape


def test_conv2d_gradcheck():
    conv = nn.Conv2d(1, 2, kernel_size=2)
    x = Tensor(np.random.randn(1, 1, 4, 4), requires_grad=True)
    gradcheck(lambda x: conv(x), [x], eps=1e-4, rtol=1e-2, atol=1e-3)


# ── Pooling ───────────────────────────────────────────────────────────────────

def test_maxpool2d_shape():
    assert nn.MaxPool2d(2)(Tensor(np.random.randn(2, 4, 8, 8))).shape == (2, 4, 4, 4)


def test_maxpool2d_backward():
    pool = nn.MaxPool2d(2)
    x = Tensor(np.random.randn(2, 4, 8, 8), requires_grad=True)
    pool(x).sum().backward()
    assert x.grad is not None and x.grad.shape == (2, 4, 8, 8)


def test_maxpool2d_routes_to_max():
    # gradient must land only on the max element of each window
    x = Tensor(np.array([[[[1.0, 3.0], [2.0, 0.0]]]]), requires_grad=True)
    nn.MaxPool2d(2)(x).sum().backward()
    np.testing.assert_allclose(x.grad, [[[[0.0, 1.0], [0.0, 0.0]]]])


def test_avgpool2d_backward():
    pool = nn.AvgPool2d(2)
    x = Tensor(np.random.randn(2, 4, 8, 8), requires_grad=True)
    pool(x).sum().backward()
    assert x.grad is not None and x.grad.shape == (2, 4, 8, 8)


# ── Normalization ─────────────────────────────────────────────────────────────

def test_batchnorm1d_backward():
    bn = nn.BatchNorm1d(8)
    x = Tensor(np.random.randn(16, 8), requires_grad=True)
    bn(x).sum().backward()
    assert x.grad is not None and x.grad.shape == (16, 8)
    assert bn.gamma.grad is not None
    assert bn.beta.grad is not None


def test_batchnorm2d_backward():
    bn = nn.BatchNorm2d(4)
    x = Tensor(np.random.randn(8, 4, 6, 6), requires_grad=True)
    bn(x).sum().backward()
    assert x.grad is not None and x.grad.shape == (8, 4, 6, 6)
    assert bn.gamma.grad is not None
    assert bn.beta.grad is not None


def test_layernorm_backward():
    ln = nn.LayerNorm(16)
    x = Tensor(np.random.randn(4, 16), requires_grad=True)
    ln(x).sum().backward()
    assert x.grad is not None and x.grad.shape == (4, 16)
    assert ln.gamma.grad is not None
    assert ln.beta.grad is not None


def test_batchnorm1d_eval_uses_running_stats():
    bn = nn.BatchNorm1d(4)
    x = Tensor(np.random.randn(8, 4), requires_grad=True)
    bn.train()
    bn(x)  # update running stats
    bn.eval()
    out_eval = bn(x)
    assert out_eval.shape == (8, 4)


# ── Dropout ───────────────────────────────────────────────────────────────────

def test_dropout_zeros_in_train():
    drop = nn.Dropout(p=0.99)
    drop.train()
    x = Tensor(np.ones((100, 100)))
    out = drop(x)
    # with p=0.99 virtually all values should be zeroed (survivors are scaled up)
    zero_fraction = (out.data == 0).mean()
    assert zero_fraction > 0.9


def test_dropout_identity_in_eval():
    drop = nn.Dropout(p=0.99)
    drop.eval()
    x = Tensor(np.ones((10, 10)))
    np.testing.assert_allclose(drop(x).data, np.ones((10, 10)))


# ── Embedding ─────────────────────────────────────────────────────────────────

def test_embedding_forward():
    emb = nn.Embedding(50, 8)
    idx = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    assert emb(idx).shape == (2, 3, 8)


def test_embedding_backward():
    emb = nn.Embedding(10, 4)
    idx = Tensor(np.array([0, 2, 5]))
    emb(idx).sum().backward()
    assert emb.weight.grad is not None
    assert emb.weight.grad.shape == emb.weight.shape


# ── RNN / GRU / LSTM ─────────────────────────────────────────────────────────

def test_rnn_shape():
    out, h = nn.RNN(8, 16)(Tensor(np.random.randn(5, 3, 8)))
    assert out.shape == (5, 3, 16)
    assert h[0].shape == (3, 16)


def test_rnn_backward():
    rnn = nn.RNN(input_size=8, hidden_size=16)
    x = Tensor(np.random.randn(5, 3, 8), requires_grad=True)
    out, _ = rnn(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.shape == (5, 3, 8)


def test_gru_shape():
    out, h = nn.GRU(8, 16)(Tensor(np.random.randn(5, 3, 8)))
    assert out.shape == (5, 3, 16)


def test_gru_backward():
    gru = nn.GRU(input_size=8, hidden_size=16)
    x = Tensor(np.random.randn(5, 3, 8), requires_grad=True)
    out, _ = gru(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.shape == (5, 3, 8)


def test_lstm_shape():
    out, h, c = nn.LSTM(8, 16)(Tensor(np.random.randn(5, 3, 8)))
    assert out.shape == (5, 3, 16)
    assert h[0].shape == (3, 16)
    assert c[0].shape == (3, 16)


def test_lstm_backward():
    lstm = nn.LSTM(input_size=8, hidden_size=16)
    x = Tensor(np.random.randn(5, 3, 8), requires_grad=True)
    out, _, _ = lstm(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.shape == (5, 3, 8)


# ── Optimizers ────────────────────────────────────────────────────────────────

def _optimizer_updates_weights(opt_cls, **kwargs):
    model = nn.Linear(4, 2)
    before = model.weight.data.copy()
    opt = opt_cls(model.parameters(), **kwargs)
    x = Tensor(np.random.randn(8, 4), requires_grad=True)
    F.cross_entropy(model(x), Tensor(np.array([0,1,0,1,0,1,0,1]))).backward()
    opt.step()
    assert not np.allclose(model.weight.data, before), f"{opt_cls.__name__} did not update weights"


def test_sgd_updates():     _optimizer_updates_weights(optim.SGD,   lr=0.1)
def test_adam_updates():    _optimizer_updates_weights(optim.Adam,  lr=1e-3)
def test_adamw_updates():   _optimizer_updates_weights(optim.AdamW, lr=1e-3)
def test_rmsprop_updates(): _optimizer_updates_weights(optim.RMSprop, lr=1e-3)


def test_zero_grad_clears():
    model = nn.Linear(4, 2)
    opt = optim.Adam(model.parameters())
    F.cross_entropy(model(Tensor(np.random.randn(4, 4))),
                    Tensor(np.array([0, 1, 0, 1]))).backward()
    assert model.weight.grad is not None
    opt.zero_grad()
    assert model.weight.grad is None


def test_steplr_schedule():
    # last_epoch starts at -1; after 6 steps last_epoch=5, 5//2=2 decays → 1.0 * 0.5**2 = 0.25
    model = nn.Linear(2, 2)
    opt = optim.SGD(model.parameters(), lr=1.0)
    sched = optim.StepLR(opt, step_size=2, gamma=0.5)
    for _ in range(6):
        sched.step()
    np.testing.assert_allclose(opt.lr, 0.25)


# ── Gradient clipping ─────────────────────────────────────────────────────────

def test_clip_grad_norm():
    params = [Tensor(np.random.randn(10, 10), requires_grad=True) for _ in range(3)]
    for p in params:
        p.grad = np.random.randn(*p.shape) * 100   # large grads
    utils.clip_grad_norm_(params, max_norm=1.0)
    total = sum(np.sum(p.grad ** 2) for p in params) ** 0.5
    assert total <= 1.0 + 1e-6


# ── DataLoader ────────────────────────────────────────────────────────────────

def test_dataloader_batch_size():
    ds = data.TensorDataset(Tensor(np.ones((100, 4))), Tensor(np.zeros(100)))
    batches = list(data.DataLoader(ds, batch_size=16))
    assert batches[0][0].shape == (16, 4)


def test_dataloader_covers_all_samples():
    N = 97   # not divisible by batch size
    ds = data.TensorDataset(Tensor(np.arange(N * 4).reshape(N, 4).astype(float)),
                            Tensor(np.zeros(N)))
    total = sum(xb.shape[0] for xb, _ in data.DataLoader(ds, batch_size=16))
    assert total == N


def test_dataloader_shuffle_changes_order():
    ds = data.TensorDataset(Tensor(np.arange(100).reshape(100, 1).astype(float)),
                            Tensor(np.zeros(100)))
    first  = np.concatenate([xb.data for xb, _ in data.DataLoader(ds, batch_size=10, shuffle=True)])
    second = np.concatenate([xb.data for xb, _ in data.DataLoader(ds, batch_size=10, shuffle=True)])
    assert not np.array_equal(first, second)
