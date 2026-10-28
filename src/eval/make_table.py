"""Final baseline table: all three models on TEST, 95% bootstrap CIs. Run once."""
from pathlib import Path

import pandas as pd

from src.eval.bootstrap import bootstrap, summarize
from src.eval.evaluate import evaluate
from src.eval.loading import MODELS

SPLIT = "test"
OUT = Path("results/baselines/baseline_table_test.csv")
KEYS = ["client", "metric", "label"]


def main():
    parts = []
    for model in [m for m in MODELS if m != "cnn"]:
        point = evaluate(model, SPLIT)
        ci = summarize(bootstrap(model, SPLIT, n_boot=1000, seed=0))
        parts.append(point.merge(ci, on=KEYS, how="left"))
        print(model, "done")
    table = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print("saved", OUT, table.shape)


if __name__ == "__main__":
    main()