"""
CNN Example — Image classification on synthetic data.

Demonstrates:
  - Conv2d → BatchNorm2d → ReLU → MaxPool2d pipeline
  - Flatten + fully-connected head
  - End-to-end training with backward through conv layers

Run:
  python examples/cnn.py
"""

import numpy as np
import mytorch
import mytorch.functional as F
from mytorch import nn, optim, data, utils
from mytorch.tensor import Tensor

mytorch.manual_seed(0)
np.random.seed(0)

# ── Synthetic image dataset (N, C, H, W) ─────────────────────────────────────
N, C_in, H, W, C_out = 256, 1, 16, 16, 4

X = np.random.randn(N, C_in, H, W).astype(np.float32)
y = np.random.randint(0, C_out, N)

split = int(0.8 * N)
train_loader = data.DataLoader(
    data.TensorDataset(Tensor(X[:split]), Tensor(y[:split])),
    batch_size=32, shuffle=True,
)
val_loader = data.DataLoader(
    data.TensorDataset(Tensor(X[split:]), Tensor(y[split:])),
    batch_size=32,
)


# ── Model ─────────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # → (16, 16, 16)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)                               # → (16, 8, 8)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # → (32, 8, 8)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)                               # → (32, 4, 4)
        self.fc1   = nn.Linear(32 * 4 * 4, 64)
        self.fc2   = nn.Linear(64, C_out)
        self._modules = [self.conv1, self.bn1, self.conv2, self.bn2, self.fc1, self.fc2]

    def __call__(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"Parameters: {utils.count_parameters(model):,}")
print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Acc':>8}")
print("-" * 30)

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(1, 11):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        loss = F.cross_entropy(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
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
