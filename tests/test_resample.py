import numpy as np
from src.data.resample import smote_per_site


def make_two_sites(seed=0):
    rng = np.random.default_rng(seed)
    XA = rng.normal(0, 1, (100, 4))
    XB = rng.normal(100, 1, (60, 4))
    yA = np.r_[np.ones(20), np.zeros(80)].astype(int)
    yB = np.r_[np.ones(12), np.zeros(48)].astype(int)
    X = np.vstack([XA, XB])
    y = np.concatenate([yA, yB])
    client = np.array(["A"] * 100 + ["B"] * 60)
    return X, y, client


def test_each_site_ends_up_balanced():
    X, y, client = make_two_sites()
    for c in ("A", "B"):
        m = client == c
        _, y_aug = smote_per_site(X[m], y[m], client[m])
        assert y_aug.sum() * 2 == len(y_aug)          # exactly 50/50


def test_synthetic_row_count():
    X, y, client = make_two_sites()
    X_aug, y_aug = smote_per_site(X, y, client)
    expected_new = (80 - 20) + (48 - 12)              # n_neg - n_pos per site
    assert len(y_aug) - len(y) == expected_new


def test_original_rows_are_kept_first():
    X, y, client = make_two_sites()
    m = client == "A"
    X_aug, y_aug = smote_per_site(X[m], y[m], client[m])
    assert np.array_equal(X_aug[:m.sum()], X[m])
    assert np.array_equal(y_aug[:m.sum()], y[m])


def test_no_cross_site_mixing():
    X, y, client = make_two_sites()
    X_aug, _ = smote_per_site(X, y, client)
    assert ((X_aug[:, 0] > 10) & (X_aug[:, 0] < 90)).sum() == 0