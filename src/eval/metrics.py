"""Reusable metrics for the multi-label baselines (also used later for the CNN)."""
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score


def per_class_auc(Y, S):
    out = np.full(Y.shape[1], np.nan)
    for k in range(Y.shape[1]):
        if 0 < Y[:, k].sum() < len(Y):
            out[k] = roc_auc_score(Y[:, k], S[:, k])
    return out


def per_class_f1_mcc(Y, S, thr):
    pred = (S > thr).astype(int)
    f1 = np.array([f1_score(Y[:, k], pred[:, k], zero_division=0)
                   for k in range(Y.shape[1])])
    mcc = np.array([matthews_corrcoef(Y[:, k], pred[:, k])
                    for k in range(Y.shape[1])])
    return f1, mcc

def predicted_positive_rate(S, thr):
    return (S > thr).mean(axis=0)