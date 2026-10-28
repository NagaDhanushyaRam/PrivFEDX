import numpy as np
from imblearn.over_sampling import SMOTE

def smote_per_site(X, y, client, seed=0):
    Xs, ys = [], []
    for c in np.unique(client):
        m = client == c
        Xc, yc = SMOTE(random_state=seed).fit_resample(X[m], y[m])
        Xs.append(Xc)
        ys.append(yc)
    return np.vstack(Xs), np.concatenate(ys)