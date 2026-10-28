"""PTB-XL preprocessing: 0.5-40 Hz zero-phase band-pass, then one fixed global scale."""
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
import wfdb

FS = 100
N_SAMPLES, N_LEADS = 1000, 12
SOS = butter(4, [0.5, 40.0], btype="band", fs=FS, output="sos")
FIT_FOLDS_MAX = 8
SCALE_MV = 0.209


def check(x, tag):
    if x.shape != (N_SAMPLES, N_LEADS):
        raise ValueError(f"{tag}: shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{tag}: NaN or Inf")


def read_raw(root, filename_lr):
    x = wfdb.rdrecord(str(Path(root) / filename_lr)).p_signal
    check(x, f"raw {filename_lr}")
    return x


def bandpass(x):
    return sosfiltfilt(SOS, x, axis=0)


def filter_record(root, filename_lr):
    y = bandpass(read_raw(root, filename_lr))
    check(y, f"filtered {filename_lr}")
    return y


def merge_moments(state, y):
    n, mean, m2 = state
    y = y.ravel()
    n_b, mean_b = y.size, float(y.mean())
    m2_b = float(((y - mean_b) ** 2).sum())
    delta = mean_b - mean
    n_tot = n + n_b
    return n_tot, mean + delta * n_b / n_tot, m2 + m2_b + delta**2 * n * n_b / n_tot


def fit_scale_std(meta, root, reader=filter_record):
    fit = meta[(meta["strat_fold"] <= FIT_FOLDS_MAX) & (meta["superclasses"].str.len() > 0)]
    assert (fit["strat_fold"] <= FIT_FOLDS_MAX).all()
    state = (0, 0.0, 0.0)
    for fn in fit["filename_lr"]:
        state = merge_moments(state, reader(root, fn))
    n, _, m2 = state
    return float(np.sqrt(m2 / n)), len(fit)

def preprocess_record(root, filename_lr, scale=SCALE_MV):
    z = (filter_record(root, filename_lr) / scale).astype(np.float32)
    check(z, f"scaled {filename_lr}")
    return z

def scan_all(meta, root, scale=SCALE_MV):
    bad, peak, peak_id = [], 0.0, None
    for i, (ecg_id, fn) in enumerate(meta["filename_lr"].items(), 1):
        try:
            y = bandpass(read_raw(root, fn))   # read_raw already checks the raw array
            check(y, f"filtered {fn}")
            z = (y / scale).astype(np.float32)
            check(z, f"scaled {fn}")
        except Exception as e:                 # also catches a missing file
            bad.append((ecg_id, str(e)))
            continue
        m = float(np.abs(z).max())
        if m > peak:
            peak, peak_id = m, ecg_id
        if i % 5000 == 0:
            print(f"  scanned {i}/{len(meta)}", flush=True)
    return {"n": len(meta), "bad": bad, "peak_abs": peak, "peak_ecg_id": peak_id}