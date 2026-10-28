import time

import numpy as np

from src.data import features
from src.data.partitions import assign
from src.data.preprocessing import preprocess_record
from src.data.ptbxl_meta import SUPERCLASSES, load_metadata

ROOT = "data/raw/ptbxl"
OUT = "data/processed/baseline_features.npz"

df = assign(load_metadata(ROOT))                 # labeled records only, with client and split
X = np.empty((len(df), len(features.FEATURE_NAMES)), dtype=np.float32)

t = time.time()
for i, fn in enumerate(df["filename_lr"]):
    X[i] = features.extract(preprocess_record(ROOT, fn))
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(df)}  ({time.time() - t:.0f}s)", flush=True)

assert X.shape == (21388, 84) and np.isfinite(X).all()
np.savez(
    OUT,
    X=X,
    Y=df[SUPERCLASSES].to_numpy(dtype=np.int8),
    ecg_id=df.index.to_numpy(),
    client=df["client"].to_numpy(dtype=str),
    split=df["split"].to_numpy(dtype=str),
    feature_names=np.array(features.FEATURE_NAMES),
    label_names=np.array(SUPERCLASSES),
)
print("saved", OUT, X.shape)