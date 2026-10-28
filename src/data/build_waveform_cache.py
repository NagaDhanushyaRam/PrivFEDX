import time
import numpy as np

from src.data.partitions import assign
from src.data.preprocessing import preprocess_record, N_SAMPLES, N_LEADS
from src.data.ptbxl_meta import load_metadata

ROOT = "data/raw/ptbxl"
FEATURES = "data/processed/baseline_features.npz"
OUT_W = "data/processed/waveforms_100hz.npy"
OUT_ID = "data/processed/waveforms_100hz_ecg_id.npy"

df = assign(load_metadata(ROOT))
ecg_id = df.index.to_numpy()
ref = np.load(FEATURES)["ecg_id"]
assert ecg_id.shape == ref.shape and (ecg_id == ref).all(), "order differs from baseline_features.npz"

W = np.empty((len(df), N_LEADS, N_SAMPLES), dtype=np.float32)
t = time.time()
for i, fn in enumerate(df["filename_lr"]):
    W[i] = preprocess_record(ROOT, fn).T          # (1000,12) -> (12,1000): transpose, never reshape
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(df)} ({time.time() - t:.0f}s)", flush=True)

assert np.isfinite(W).all()
np.save(OUT_W, W)
np.save(OUT_ID, ecg_id)

# round trip: reload and recompute a few rows from raw
chk = np.load(OUT_W, mmap_mode="r")
assert chk.shape == (21388, 12, 1000) and chk.dtype == np.float32
assert (np.load(OUT_ID) == ref).all()
for i in (0, 10_000, len(df) - 1):
    assert np.array_equal(chk[i], preprocess_record(ROOT, df["filename_lr"].iloc[i]).T)
j = int(np.abs(chk).max(axis=(1, 2)).argmax())
print("OK", chk.shape, "max |scaled| =", float(np.abs(chk[j]).max()), "ecg_id", ecg_id[j])
del chk