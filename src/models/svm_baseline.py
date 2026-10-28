"""SVM baseline: RBF SVC, one binary model per class, pooled training set."""
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

NPZ = Path("data/processed/baseline_features.npz")
OUT = Path("results/baselines/svm_scores.npz")


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

    S_va = np.zeros((va.sum(), Y.shape[1]))
    S_te = np.zeros((te.sum(), Y.shape[1]))
    for k, name in enumerate(label_names):
        t0 = time.perf_counter()
        m = SVC(kernel="rbf", class_weight="balanced").fit(Xtr, Y[tr, k])
        S_va[:, k] = m.decision_function(Xva)
        S_te[:, k] = m.decision_function(Xte)  # saved, NOT scored
        frac_sv = m.n_support_.sum() / tr.sum()
        print(f"{name:5s} fit {time.perf_counter() - t0:6.1f}s  "
              f"support vectors {frac_sv:.0%} of train")

    auc = roc_auc_score(Y[va], S_va, average=None)
    for name, a in zip(label_names, auc):
        print(f"  val AUC {name:5s} {a:.3f}")
    print("  val AUC macro", round(float(auc.mean()), 3))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, S_val=S_va, S_test=S_te,
             ecg_id_val=ecg_id[va], ecg_id_test=ecg_id[te])
    print("saved", OUT)


if __name__ == "__main__":
    main()