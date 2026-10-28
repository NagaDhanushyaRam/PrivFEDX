"""Random Forest + per-site SMOTE: one forest per class, pooled training set."""
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.resample import smote_per_site

NPZ = Path("data/processed/baseline_features.npz")
OUT = Path("results/baselines/rf_smote_probs.npz")


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

    t0 = time.perf_counter()
    P_va = np.zeros((va.sum(), 5))
    P_te = np.zeros((te.sum(), 5))
    for c in range(5):
        Xa, ya = smote_per_site(Xtr, Y[tr][:, c], client[tr], seed=0)
        clf = RandomForestClassifier(random_state=0, n_jobs=-1).fit(Xa, ya)  # no class_weight
        P_va[:, c] = clf.predict_proba(Xva)[:, 1]
        P_te[:, c] = clf.predict_proba(Xte)[:, 1]  # saved, NOT scored
        print(f"  {label_names[c]:5s} augmented train n={len(ya)}  ya.mean()={ya.mean():.3f}")
    print(f"fit {time.perf_counter() - t0:.1f}s")
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
