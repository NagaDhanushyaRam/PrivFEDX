import numpy as np
from src.data import preprocessing as pp
import pandas as pd
import pytest


def test_filter_acts_on_time_axis_only():
    x = np.zeros((1000, 12))
    x[500, 0] = 1.0                      
    y = pp.bandpass(x)
    assert np.abs(y[:, 1:]).max() == 0.0 
    assert np.abs(y[:, 0]).max() > 0.0   
    
def _fake_meta():
    folds = [1, 2, 8, 9, 10, 5, 9, 10]
    return pd.DataFrame({
        "strat_fold": folds,
        "filename_lr": [f"rec{i}" for i in range(len(folds))],
        "superclasses": [["NORM"]] * 5 + [[]] + [["MI"]] * 2,  
    })


def test_fit_never_reads_folds_9_10():
    meta, seen = _fake_meta(), []
    rng = np.random.default_rng(0)

    def spy(root, fn):
        seen.append(fn)
        return rng.normal(size=(1000, 12))

    std, n = pp.fit_scale_std(meta, "unused", reader=spy)
    assert sorted(seen) == ["rec0", "rec1", "rec2"]   # no fold 9/10, no unlabeled
    assert n == 3 and 0.9 < std < 1.1


def test_fit_std_matches_direct_std():
    meta = _fake_meta()
    sigs = {f"rec{i}": np.random.default_rng(i).normal(i, 1 + i, (1000, 12)) for i in range(8)}
    std, _ = pp.fit_scale_std(meta, "unused", reader=lambda r, fn: sigs[fn])
    direct = np.concatenate([sigs[f].ravel() for f in ("rec0", "rec1", "rec2")]).std()
    assert std == pytest.approx(direct, rel=1e-9)
    
def test_preprocess_is_deterministic_float32(monkeypatch):
    raw = np.random.default_rng(1).normal(size=(1000, 12))
    monkeypatch.setattr(pp, "read_raw", lambda root, fn: raw.copy())
    a = pp.preprocess_record("r", "x")
    b = pp.preprocess_record("r", "x")
    assert a.dtype == np.float32 and a.shape == (1000, 12)
    assert np.array_equal(a, b)                      
    assert np.allclose(a, pp.bandpass(raw) / pp.SCALE_MV, atol=1e-5)
    
def test_scan_reports_bad_records(monkeypatch):
    def fake_read(root, fn):
        if fn == "b":
            return np.full((1000, 12), np.nan)     
        if fn == "c":
            return np.zeros((999, 12))             
        return np.zeros((1000, 12))

    monkeypatch.setattr(pp, "read_raw", fake_read)
    meta = pd.DataFrame({"filename_lr": ["a", "b", "c"]}, index=[10, 11, 12])
    res = pp.scan_all(meta, "unused")
    assert [i for i, _ in res["bad"]] == [11, 12]  
    assert res["n"] == 3