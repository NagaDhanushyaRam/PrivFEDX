"""Small hand-crafted feature set from a preprocessed record: 7 features x 12 leads = 84."""
import numpy as np

from src.data.preprocessing import FS, N_SAMPLES

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
BANDS = [(0.5, 5.0), (5.0, 15.0), (15.0, 40.0)]
STATS = ["std", "min", "max", "absdiff", "bp_0.5_5", "bp_5_15", "bp_15_40"]
FEATURE_NAMES = [f"{lead}_{s}" for lead in LEADS for s in STATS]   # lead-major, 84 names


def extract(z):
    z = z.astype(np.float64)
    spec = np.abs(np.fft.rfft(z, axis=0)) ** 2 / N_SAMPLES    
    freqs = np.fft.rfftfreq(N_SAMPLES, d=1 / FS)
    cols = [z.std(axis=0), z.min(axis=0), z.max(axis=0), np.abs(np.diff(z, axis=0)).mean(axis=0)]
    cols += [spec[(freqs >= lo) & (freqs < hi)].sum(axis=0) for lo, hi in BANDS]
    f = np.stack(cols, axis=1).ravel()                     
    if not np.isfinite(f).all():
        raise ValueError("non-finite feature")
    return f