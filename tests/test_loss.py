import math
import pytest
from src.model.forward import forward, compute_probabilities
from src.model.loss import compute_loss

TOL = 1e-5
E = [[1, 0], [0, 1], [1, 1]]
U = [[0, 1, -1], [1, 0, 1]]


def loss_for(s, t):
    m, a, _ = compute_probabilities(s)
    return compute_loss(s, t, m, a)


def test_mentor_loss():
    _, s = forward(E, U, 0)
    assert abs(loss_for(s, 1) - 0.407606) < TOL


def test_matches_negative_log_probability():
    _, s = forward(E, U, 0)
    _, _, p = compute_probabilities(s)
    assert abs(loss_for(s, 1) - (-math.log(p[1]))) < TOL


def test_equal_scores_give_ln2():
    assert abs(loss_for([2, 2], 0) - math.log(2)) < TOL


def test_extreme_scores_stable():
    assert abs(loss_for([1000, 0], 1) - 1000) < TOL


def test_shift_invariance():
    _, s = forward(E, U, 0)
    assert abs(loss_for([x + 100 for x in s], 1) - loss_for(s, 1)) < TOL


def test_confident_correct_prediction_is_near_zero():
    assert loss_for([50, 0, 0], 0) < TOL


def test_loss_never_negative():
    _, s = forward(E, U, 0)
    assert all(loss_for(s, t) >= 0 for t in range(3))


def test_target_out_of_range():
    m, a, _ = compute_probabilities([0, 1, -1])
    for bad in (-1, 3):
        with pytest.raises(IndexError):
            compute_loss([0, 1, -1], bad, m, a)
