"""Random Forest baseline: one forest per class (one-vs-rest), pooled training set."""
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier

NPZ = Path("data/processed/baseline_features.npz")
OUT = Path("results/baselines/rf_probs.npz")


def main():
    d = np.load(NPZ, allow_pickle=True)
    X = d["X"]
    Y = d["Y"]
    split = d["split"].astype(str)
    ecg_id = d["ecg_id"]
    label_names = [str(s) for s in d["label_names"]]

    assert np.isfinite(X).all(), "NaN/Inf in X"
    tr, va, te = (split == s for s in ("train", "val", "test"))
    print("rows train/val/test:", tr.sum(), va.sum(), te.sum())

    t0 = time.perf_counter()
    clf = OneVsRestClassifier(
        RandomForestClassifier(class_weight="balanced", random_state=0, n_jobs=-1)
    ).fit(X[tr], Y[tr])
    print(f"fit {time.perf_counter() - t0:.1f}s")

    P_va = clf.predict_proba(X[va])
    P_te = clf.predict_proba(X[te])  # saved, NOT scored
    print("P_va shape:", P_va.shape)

    auc = roc_auc_score(Y[va], P_va, average=None)
    for name, a in zip(label_names, auc):
        print(f"  val AUC {name:5s} {a:.3f}")
    print("  val AUC macro", round(float(auc.mean()), 3))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, P_val=P_va, P_test=P_te,
             ecg_id_val=ecg_id[va], ecg_id_test=ecg_id[te])
    print("saved", OUT)


if __name__ == "__main__":
    main()