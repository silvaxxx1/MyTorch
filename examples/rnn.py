"""
RNN Example — Sequence classification with RNN, GRU, and LSTM.

Demonstrates:
  - Sequence-to-label (many-to-one) classification
  - RNN / GRU / LSTM with the same training loop
  - Using the last hidden state for classification
  - Comparing all three architectures back-to-back

Run:
  python examples/rnn.py
"""

import numpy as np
import mytorch
import mytorch.functional as F
from mytorch import nn, optim, data, utils
from mytorch.tensor import Tensor

mytorch.manual_seed(1)
np.random.seed(1)

# ── Synthetic sequence dataset ────────────────────────────────────────────────
# Task: classify whether the mean of the sequence is positive or negative.
N, SEQ, INPUT, HIDDEN, C = 512, 10, 8, 32, 2

X = np.random.randn(N, SEQ, INPUT).astype(np.float32)
y = (X.mean(axis=(1, 2)) > 0).astype(int)

split = int(0.8 * N)
train_loader = data.DataLoader(
    data.TensorDataset(Tensor(X[:split]), Tensor(y[:split])),
    batch_size=32, shuffle=True,
)
val_loader = data.DataLoader(
    data.TensorDataset(Tensor(X[split:]), Tensor(y[split:])),
    batch_size=32,
)


# ── Sequence classifier (works with any recurrent cell) ───────────────────────
class SeqClassifier(nn.Module):
    """Many-to-one classifier: uses the last hidden state."""

    def __init__(self, cell):
        super().__init__()
        self.cell = cell
        self.fc = nn.Linear(HIDDEN, C)
        self._modules = [self.cell, self.fc]

    def __call__(self, x):
        # x: (batch, seq, input) → transpose to (seq, batch, input) for recurrent cells
        x_t = x.transpose(1, 0, 2)

        if isinstance(self.cell, nn.LSTM):
            _, h, _ = self.cell(x_t)
        else:
            _, h = self.cell(x_t)

        last_h = h[-1]   # last layer hidden state: (batch, hidden)
        return self.fc(last_h)


def train(name, model, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"\n── {name}  ({utils.count_parameters(model):,} params) ──")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Acc':>8}")
    print("-" * 30)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.data)
        train_loss /= len(train_loader)

        model.eval()
        preds, targets = [], []
        for xb, yb in val_loader:
            preds.append(model(xb).data)
            targets.append(yb.data)
        val_acc = utils.accuracy(
            np.concatenate(preds),
            np.concatenate(targets).astype(int),
        )

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_acc:>8.4f}")


# ── Train all three architectures ─────────────────────────────────────────────
train("RNN",  SeqClassifier(nn.RNN(INPUT, HIDDEN)))
train("GRU",  SeqClassifier(nn.GRU(INPUT, HIDDEN)))
train("LSTM", SeqClassifier(nn.LSTM(INPUT, HIDDEN)))
