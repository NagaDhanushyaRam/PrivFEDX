import torch
from torch import nn
from src.models.cnn import ECGCNN

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = ECGCNN().to(dev)
x = torch.randn(8, 12, 1000, device=dev)
print("features", tuple(m.features(x).shape), "logits", tuple(m(x).shape))  
assert m(x).shape == (8, 5)
print("params:", sum(p.numel() for p in m.parameters()))                     

y = torch.randint(0, 2, (8, 5), device=dev).float()
nn.BCEWithLogitsLoss(pos_weight=torch.ones(5, device=dev))(m(x), y).backward()
assert all(torch.isfinite(p.grad).all() for p in m.parameters())

from opacus.validators import ModuleValidator   
print("opacus errors:", ModuleValidator.validate(m, strict=False))              