"""Opacus hello-world: one epoch of DP-SGD on random PTB-XL-shaped data."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 64 fake ECGs: 12 leads x 1000 samples, 5 superclasses
X = torch.randn(64, 12, 1000)
y = torch.randint(0, 5, (64,))
loader = DataLoader(TensorDataset(X, y), batch_size=16)

model = nn.Sequential(
    nn.Conv1d(12, 8, kernel_size=7),
    nn.ReLU(),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(8, 5),
).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

engine = PrivacyEngine()
model, optimizer, loader = engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

model.train()
for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(xb), yb)
    loss.backward()
    optimizer.step()

eps = engine.get_epsilon(delta=1e-5)
print(f"Opacus OK | device={device} | loss={loss.item():.4f} | epsilon={eps:.2f}")