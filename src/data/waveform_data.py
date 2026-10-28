import numpy as np
import torch
from torch.utils.data import Dataset

FEATURES = "data/processed/baseline_features.npz"
CACHE = "data/processed/waveforms_100hz.npy"
CACHE_IDS = "data/processed/waveforms_100hz_ecg_id.npy"


class WaveformSet(Dataset):
    def __init__(self, x, y, ecg_id):          # x (n,12,1000) f32, y (n,5) f32 tensors
        self.x, self.y, self.ecg_id = x, y, ecg_id
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]


def load_split(split):                          # 'train' | 'val' | 'test'
    f = np.load(FEATURES)
    ids = np.load(CACHE_IDS)
    assert (ids == f["ecg_id"]).all(), "cache order != baseline_features order"
    W = np.load(CACHE, mmap_mode="r")
    assert W.shape == (len(ids), 12, 1000)
    m = f["split"].astype(str) == split
    x = torch.from_numpy(W[m])                  # boolean mask copies the rows into RAM
    y = torch.from_numpy(f["Y"][m].astype(np.float32))
    del W
    return WaveformSet(x, y, f["ecg_id"][m])