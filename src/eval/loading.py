"""Load a saved baseline output aligned with labels and client for one split."""
from pathlib import Path

import numpy as np

FEATURES = Path("data/processed/baseline_features.npz")
RESULTS = Path("results/baselines")

MODELS = {
    "lr": ("lr_probs.npz", "P", 0.5),
    "lr_smote": ("lr_smote_probs.npz", "P", 0.5),
    "svm": ("svm_scores.npz", "S", 0.0),
    "svm_smote": ("svm_smote_scores.npz", "S", 0.0),
    "rf": ("rf_probs.npz", "P", 0.5),
    "rf_smote": ("rf_smote_probs.npz", "P", 0.5),
    "cnn": ("../cnn/cnn_val_probs.npz", "P", 0.5),
}


def load_split(model, split):
    fname, prefix, _ = MODELS[model]
    f = np.load(FEATURES, allow_pickle=True)
    r = np.load(RESULTS / fname)
    m = f["split"].astype(str) == split
    assert (r[f"ecg_id_{split}"] == f["ecg_id"][m]).all(), "row misalignment"
    return f["Y"][m], r[f"{prefix}_{split}"], f["client"][m].astype(str)