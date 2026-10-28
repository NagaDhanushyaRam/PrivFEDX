import time

from src.data.ptbxl_meta import load_metadata
from src.data import preprocessing as pp

ROOT = "data/raw/ptbxl"

meta = load_metadata(ROOT)
t = time.time()
std, n = pp.fit_scale_std(meta, ROOT)
print(f"records used: {n}")
print(f"global std (mV): {std:.6f}")
print(f"seconds: {time.time() - t:.0f}")