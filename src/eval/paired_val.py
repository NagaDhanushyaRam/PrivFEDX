"""Paired bootstrap differences on fold 9 (pooled): CNN minus each baseline."""
from pathlib import Path

import pandas as pd

BOOT = Path("results/cnn/boot_val_pooled.csv")
POINT = Path("results/cnn/compare_val.csv")
OUT = Path("results/cnn/paired_val.csv")
KEYS = ["metric", "label"]
LABELS = ["NORM", "MI", "STTC", "CD", "HYP", "macro"]
BASES = ["lr", "lr_smote", "rf", "rf_smote", "svm", "svm_smote"]


def main():
    wide = pd.read_csv(BOOT).pivot_table(index=["rep"] + KEYS, columns="model",
                                         values="value")
    assert wide.shape[0] == 1000 * 18 and not wide.isna().any().any(), "incomplete replicates"
    point = (pd.read_csv(POINT).query("metric != 'ppr'")
             .pivot_table(index=KEYS, columns="model", values="value"))
    rows = []
    for base in BASES:
        d = (wide["cnn"] - wide[base]).groupby(level=KEYS)
        rows.append(pd.DataFrame({"baseline": base,
                                  "diff": point["cnn"] - point[base],
                                  "lo": d.quantile(0.025), "hi": d.quantile(0.975)}))
    out = pd.concat(rows).reset_index()
    out["excl0"] = (out["lo"] > 0) | (out["hi"] < 0)
    out.to_csv(OUT, index=False)

    print("PRIMARY: macro AUC, CNN minus baseline, 95% paired CI")
    prim = out.query("metric == 'auc' and label == 'macro'").set_index("baseline")
    print(prim.loc[BASES, ["diff", "lo", "hi"]].round(3).to_string())
    for metric in ["auc", "f1", "mcc"]:
        t = out[out["metric"] == metric].copy()
        t["cell"] = t["diff"].map("{:+.3f}".format) + t["excl0"].map({True: "*", False: ""})
        print(f"\n{metric}: CNN minus baseline ('*' = unadjusted 95% CI excludes 0)")
        print(t.pivot(index="label", columns="baseline", values="cell")
              .loc[LABELS, BASES].to_string())
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()