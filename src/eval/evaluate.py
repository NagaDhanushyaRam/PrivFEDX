"""Per-client and pooled metrics for one saved model output (point estimates)."""
import numpy as np
import pandas as pd

from src.data.ptbxl_meta import SUPERCLASSES
from src.eval.loading import MODELS, load_split
from src.eval.metrics import per_class_auc, per_class_f1_mcc

CLIENTS = ["site0", "site1", "site2", "site3", "rest"]
LABELS = list(SUPERCLASSES) + ["macro"]


def score_slice(Y, S, thr):
    f1, mcc = per_class_f1_mcc(Y, S, thr)
    return {"auc": per_class_auc(Y, S), "f1": f1, "mcc": mcc}


def evaluate(model, split):
    Y, S, client = load_split(model, split)
    thr = MODELS[model][2]
    rows = []
    for name in ["pooled"] + CLIENTS:
        m = np.ones(len(Y), bool) if name == "pooled" else client == name
        for metric, vals in score_slice(Y[m], S[m], thr).items():
            for label, v in zip(LABELS, list(vals) + [np.nanmean(vals)]):
                rows.append((model, name, int(m.sum()), metric, label, v))
    return pd.DataFrame(
        rows, columns=["model", "client", "n", "metric", "label", "value"])