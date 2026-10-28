import numpy as np
from src.data.ptbxl_meta import load_metadata

ROOT = "data/raw/ptbxl"
f = np.load("data/processed/baseline_features.npz")
ids = np.load("data/processed/waveforms_100hz_ecg_id.npy")
assert (ids == f["ecg_id"]).all()
W = np.load("data/processed/waveforms_100hz.npy", mmap_mode="r")

# per-record max |scaled| over all leads/samples, chunked to avoid a 1 GB temp array
peak = np.concatenate([np.abs(W[i:i + 1000]).max(axis=(1, 2)) for i in range(0, len(W), 1000)])

split = f["split"].astype(str)
train, late = split == "train", np.isin(split, ["val", "test"])      # folds 1-8 / folds 9-10
hyp = f["Y"][:, list(f["label_names"]).index("HYP")] == 1

for thr in (20, 30, 50):
    print(f">{thr}: folds1-8 {(peak[train] > thr).sum()}/{train.sum()} "
          f"(HYP+ {(peak[train & hyp] > thr).sum()}) | folds9-10 {(peak[late] > thr).sum()}/{late.sum()}")
for name, m in [("all", train), ("HYP+", train & hyp), ("other", train & ~hyp)]:
    print(name, m.sum(), "p99/p99.9:", np.percentile(peak[m], [99, 99.9]).round(1), "max:", peak[m].max().round(1))
print("HYP+ five largest (folds 1-8):", np.sort(peak[train & hyp])[-5:].round(1))

meta = load_metadata(ROOT)
top = np.argsort(peak)[::-1][:10]                                     # all folds; strat_fold column shows which
cols = ["strat_fold", "device", "baseline_drift", "static_noise", "burst_noise", "electrodes_problems", "pacemaker"]
out = meta.loc[ids[top], cols]
out["peak"] = peak[top].round(1)
out["HYP"] = hyp[top]
print(out.to_string())