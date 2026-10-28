import numpy as np
from src.data import features

def test_extract_shape_and_band_energy():
    t = np.arange(1000) / 100
    z = np.tile(np.sin(2 * np.pi * 10 * t)[:, None], (1, 12))  
    f = features.extract(z)
    assert f.shape == (84,) and len(features.FEATURE_NAMES) == 84
    p = f.reshape(12, 7)
    assert p[:, 5].min() > 10 * max(p[:, 4].max(), p[:, 6].max())  


def test_flat_lead_stays_finite():
    assert np.isfinite(features.extract(np.zeros((1000, 12)))).all()