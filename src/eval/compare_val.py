"""Fold-9 (val) comparison of all models: pooled point estimates + predicted-positive rate."""
from pathlib import Path

import pandas as pd

from src.eval.evaluate import LABELS, evaluate
from src.eval.loading import MODELS, load_split
from src.eval.metrics import predicted_positive_rate

OUT = Path("results/cnn/compare_val.csv")
ORDER = ["cnn", "lr", "lr_smote", "rf", "rf_smote", "svm", "svm_smote"]


def main():
    parts = []
    for model in MODELS:
        parts.append(evaluate(model, "val").query("client == 'pooled'"))
        _, S, _ = load_split(model, "val")
        ppr = predicted_positive_rate(S, MODELS[model][2])
        parts.append(pd.DataFrame({
            "model": model, "client": "pooled", "n": len(S),
            "metric": "ppr", "label": LABELS[:5], "value": ppr}))
    table = pd.concat(parts, ignore_index=True)
    table.to_csv(OUT, index=False)
    wide = table.pivot_table(index=["metric", "label"], columns="model",
                             values="value", sort=False)
    print(wide[ORDER].round(3).to_string())
    print("saved", OUT)


if __name__ == "__main__":
    main()