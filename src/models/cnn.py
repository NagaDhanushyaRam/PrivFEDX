import torch
from torch import nn


def block(c_in, c_out, k, s, groups=8):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, stride=s, padding=k // 2),
        nn.GroupNorm(groups, c_out),
        nn.ReLU(),
    )


class ECGCNN(nn.Module):
    def __init__(self, n_leads=12, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            block(n_leads, 32, 15, 2),
            block(32, 64, 9, 2),
            block(64, 128, 7, 2),
            block(128, 128, 5, 1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):                       
        f = self.features(x)                    
        return self.head(f.mean(dim=2))       