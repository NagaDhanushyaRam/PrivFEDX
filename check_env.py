"""Environment sanity check for PrivFedXAI."""
import sys
from importlib.metadata import version

import matplotlib
import numpy
import pandas
import sklearn
import torch
import wfdb

print(f"Python       {sys.version.split()[0]}")
for mod in (numpy, pandas, sklearn, wfdb, matplotlib, torch):
    print(f"{mod.__name__:12s} {mod.__version__}")

# Day 2 packages: report installed versions
for pkg in ("opacus", "shap", "streamlit", "flwr"):
    print(f"{pkg:12s} {version(pkg)}")

# Pick the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device       {device}")

# Smoke test: one Conv1D pass on a PTB-XL-shaped batch
# (batch=2, leads=12, samples=1000 at 100 Hz)
x = torch.randn(2, 12, 1000, device=device)
conv = torch.nn.Conv1d(in_channels=12, out_channels=16, kernel_size=7).to(device)
y = conv(x)
assert y.shape == (2, 16, 994), f"Unexpected shape: {y.shape}"
print(f"Conv1D test  OK, output shape {tuple(y.shape)}")