import matplotlib.pyplot as plt
import numpy as np

from src.data import preprocessing as pp
from src.data.ptbxl_meta import load_metadata

ROOT = "data/raw/ptbxl"
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

meta = load_metadata(ROOT)
t = np.arange(1000) / 100
fig, axes = plt.subplots(2, 2, figsize=(14, 7))

for col, ecg_id in enumerate([10846, 9574]):
    raw = pp.read_raw(ROOT, meta.loc[ecg_id, "filename_lr"])
    filt = pp.bandpass(raw)
    lead = int(np.abs(filt).max(axis=0).argmax())   # lead with the largest filtered peak
    k = int(np.abs(filt[:, lead]).argmax())         # sample where that peak occurs
    for row, (a, b) in enumerate([(0, 1000), (max(k - 100, 0), min(k + 100, 1000))]):
        ax = axes[row, col]
        ax.plot(t[a:b], raw[a:b, lead], label="raw", lw=1)
        ax.plot(t[a:b], filt[a:b, lead], label="filtered", lw=1)
        ax.set_title(f"{ecg_id}, lead {LEADS[lead]}" + (" (2 s zoom on the peak)" if row else ""))
        ax.set_xlabel("seconds")
        ax.set_ylabel("mV")
        ax.legend()

fig.tight_layout()
fig.savefig("results/eda/waveforms/pacemaker_check.png", dpi=120)
plt.show()