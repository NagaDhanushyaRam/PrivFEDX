import numpy as np
from src.eval.metrics import per_class_f1_mcc, predicted_positive_rate


def test_ties_at_threshold_count_as_negative():
    S = np.array([[0.5, 0.2], [0.5, 0.9], [0.7, 0.9], [0.1, 0.9]])
    assert np.allclose(predicted_positive_rate(S, 0.5), [0.25, 0.75])
    
def test_f1_mcc_returns_two_arrays():
    Y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    S = np.array([[.9, .1], [.2, .8], [.6, .7], [.1, .3]])
    f1, mcc = per_class_f1_mcc(Y, S, 0.5)
    assert f1.shape == mcc.shape == (2,)
    assert np.allclose(f1, 1.0) and np.allclose(mcc, 1.0)