"""PTB-XL metadata loader: parses scp_codes and maps them to the 5 diagnostic superclasses."""
import ast
from pathlib import Path

import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_metadata(root):
    root = Path(root)
    df = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    diag_class = scp.loc[scp["diagnostic"] == 1, "diagnostic_class"]

    df["superclasses"] = df["scp_codes"].apply(
        lambda codes: sorted({diag_class[c] for c in codes if c in diag_class.index})
    )
    for sc in SUPERCLASSES:
        df[sc] = df["superclasses"].apply(lambda s, sc=sc: int(sc in s))
    return df
