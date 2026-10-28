"""SVM + per-site SMOTE: one RBF SVC per class, pooled training set."""
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.resample import smote_per_site

NPZ = Path("data/processed/baseline_features.npz")
OUT = Path("results/baselines/svm_smote_scores.npz")


def main():
    d = np.load(NPZ, allow_pickle=True)
    X = d["X"].astype(np.float64)
    Y = d["Y"]
    split = d["split"].astype(str)
    client = d["client"].astype(str)
    ecg_id = d["ecg_id"]
    label_names = [str(s) for s in d["label_names"]]

    assert np.isfinite(X).all(), "NaN/Inf in X"
    tr, va, te = (split == s for s in ("train", "val", "test"))
    print("rows train/val/test:", tr.sum(), va.sum(), te.sum())

    scaler = StandardScaler().fit(X[tr])  # real train rows only
    Xtr, Xva, Xte = (scaler.transform(X[m]) for m in (tr, va, te))

    t_all = time.perf_counter()
    S_va = np.zeros((va.sum(), Y.shape[1]))
    S_te = np.zeros((te.sum(), Y.shape[1]))
    for k, name in enumerate(label_names):
        t0 = time.perf_counter()
        Xa, ya = smote_per_site(Xtr, Y[tr][:, k], client[tr], seed=0)
        m = SVC(kernel="rbf").fit(Xa, ya)  # defaults, no class_weight
        S_va[:, k] = m.decision_function(Xva)
        S_te[:, k] = m.decision_function(Xte)  # saved, NOT scored
        print(f"{name:5s} augmented n={len(ya)} ya.mean()={ya.mean():.3f}  "
              f"fit {time.perf_counter() - t0:6.1f}s  "
              f"support vectors {m.n_support_.sum() / len(ya):.0%} of augmented train")
    print(f"total fit {time.perf_counter() - t_all:.1f}s")

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
