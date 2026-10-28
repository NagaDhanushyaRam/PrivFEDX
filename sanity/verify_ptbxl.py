# sanity/verify_ptbxl.py
"""Verify PTB-XL download: record count and folder structure."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/ptbxl")

df = pd.read_csv(DATA_DIR / "ptbxl_database.csv", index_col="ecg_id")
print(f"ptbxl_database.csv rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

n_100 = len(list((DATA_DIR / "records100").rglob("*.hea")))
n_500 = len(list((DATA_DIR / "records500").rglob("*.hea")))
print(f"records100 .hea files: {n_100}")
print(f"records500 .hea files: {n_500}")