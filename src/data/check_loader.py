import torch
from torch.utils.data import DataLoader

from src.data.waveform_data import load_split
from src.models.cnn import ECGCNN

ds = load_split("val")
print("len(ds):", len(ds))                          # expect 2146

loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
x, y = next(iter(loader))
print("x:", tuple(x.shape), x.dtype, "| y:", tuple(y.shape), y.dtype)

assert len(ds) == 2146
assert x.shape == (64, 12, 1000) and x.dtype == torch.float32
assert y.shape == (64, 5) and y.dtype == torch.float32
assert torch.isfinite(x).all()
assert set(y.unique().tolist()) <= {0.0, 1.0}      # labels are 0/1 only

model = ECGCNN()
with torch.no_grad():
    logits = model(x)
print("logits:", tuple(logits.shape))               # expect (64, 5)
assert logits.shape == (64, 5)
print("OK")