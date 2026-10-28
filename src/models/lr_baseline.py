"""Logistic Regression baseline: pooled training set, one-vs-rest."""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

NPZ = Path("data/processed/baseline_features.npz")
OUT = Path("results/baselines/lr_probs.npz")


def main():
    d = np.load(NPZ, allow_pickle=True)
    X = d["X"].astype(np.float64)
    Y = d["Y"]
    split = d["split"].astype(str)
    ecg_id = d["ecg_id"]
    label_names = [str(s) for s in d["label_names"]]

    assert np.isfinite(X).all(), "NaN/Inf in X"
    tr, va, te = (split == s for s in ("train", "val", "test"))
    print("rows train/val/test:", tr.sum(), va.sum(), te.sum())

    scaler = StandardScaler().fit(X[tr])  # train rows only
    Xtr, Xva, Xte = (scaler.transform(X[m]) for m in (tr, va, te))
    print("max |z| on val:", round(float(np.abs(Xva).max()), 1))

    clf = OneVsRestClassifier(
        LogisticRegression(class_weight="balanced", max_iter=1000)
    ).fit(Xtr, Y[tr])
    print("n_iter per class:", [int(e.n_iter_[0]) for e in clf.estimators_])

    P_va = clf.predict_proba(Xva)
    P_te = clf.predict_proba(Xte)  # saved, NOT scored
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