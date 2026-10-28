from src.data.ptbxl_meta import load_metadata
from src.data import preprocessing as pp

ROOT = "data/raw/ptbxl"

meta = load_metadata(ROOT)
res = pp.scan_all(meta, ROOT)
print(f"{res['n']} records scanned, {len(res['bad'])} bad")
print(f"largest |scaled| value: {res['peak_abs']:.2f} (ecg_id {res['peak_ecg_id']})")
assert not res["bad"], res["bad"][:5]