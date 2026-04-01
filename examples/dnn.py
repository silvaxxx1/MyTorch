"""
DNN Example — Multi-class classification on synthetic data.

Demonstrates:
  - Building a fully-connected network with BatchNorm and Dropout
  - Training loop with Adam, gradient clipping, and early stopping
  - Evaluation with accuracy metric
  - Saving and loading a checkpoint

Run:
  python examples/dnn.py
"""

import numpy as np
import mytorch
import mytorch.functional as F
from mytorch import nn, optim, data, utils
from mytorch.tensor import Tensor

# ── Reproducibility ──────────────────────────────────────────────────────────
mytorch.manual_seed(42)
np.random.seed(42)

# ── Synthetic dataset ─────────────────────────────────────────────────────────
N, D, C = 1024, 20, 5          # samples, features, classes

X = np.random.randn(N, D).astype(np.float32)
y = np.random.randint(0, C, N)

split = int(0.8 * N)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

train_loader = data.DataLoader(
    data.TensorDataset(Tensor(X_train), Tensor(y_train)),
    batch_size=64, shuffle=True,
)
val_loader = data.DataLoader(
    data.TensorDataset(Tensor(X_val), Tensor(y_val)),
    batch_size=64,
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(D, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, C),
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.StepLR(optimizer, step_size=5, gamma=0.5)
early_stop = utils.EarlyStopping(patience=5)

print(f"Parameters: {utils.count_parameters(model):,}")
print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val Acc':>8}")
print("-" * 44)

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(1, 21):
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
    val_loss, val_preds, val_targets = 0.0, [], []
    for xb, yb in val_loader:
        out = model(xb)
        val_loss += float(F.cross_entropy(out, yb).data)
        val_preds.append(out.data)
        val_targets.append(yb.data)
    val_loss /= len(val_loader)
    val_acc = utils.accuracy(
        np.concatenate(val_preds),
        np.concatenate(val_targets).astype(int),
    )

    scheduler.step()
    print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  {val_acc:>8.4f}")

    if early_stop(val_loss):
        print("Early stopping triggered.")
        break

# ── Checkpoint ────────────────────────────────────────────────────────────────
utils.save_checkpoint(model, optimizer, epoch=epoch, filepath="dnn_checkpoint.npz")
utils.load_checkpoint("dnn_checkpoint.npz", model=model, optimizer=optimizer)
print("\nCheckpoint saved and reloaded successfully.")

import os; os.remove("dnn_checkpoint.npz")
