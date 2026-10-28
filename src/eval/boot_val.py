"""Pooled fold-9 bootstrap replicates for every model (paired by construction, D47)."""
from pathlib import Path

import pandas as pd

from src.eval.bootstrap import bootstrap
from src.eval.loading import MODELS

OUT = Path("results/cnn/boot_val_pooled.csv")


def main():
    parts = []
    for model in MODELS:
        b = bootstrap(model, "val", n_boot=1000, seed=0)
        parts.append(b[b["client"] == "pooled"].assign(model=model))
        print(model, "done", flush=True)
    pd.concat(parts, ignore_index=True).to_csv(OUT, index=False)
    print("saved", OUT)


if __name__ == "__main__":
    main() 