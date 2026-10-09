import pytest
from src.model.forward import (get_embedding, compute_scores, forward,
                               compute_probabilities)

TOL = 1e-5
E = [[1, 0], [0, 1], [1, 1]]
U = [[0, 1, -1], [1, 0, 1]]


def close(xs, ys):
    return len(xs) == len(ys) and all(abs(x - y) < TOL for x, y in zip(xs, ys))


def test_mentor_scores():
    h, s = forward(E, U, 0)
    assert h == [1, 0]
    assert close(s, [0, 1, -1])


def test_center_eat():
    h, s = forward(E, U, 1)
    assert h == [0, 1]
    assert close(s, [1, 0, 1])


def test_generic_size():
    E2 = [[1, 2, 3], [4, 5, 6]]
    U2 = [[1, 0], [0, 1], [1, 1]]
    _, s = forward(E2, U2, 1)
    assert close(s, [10, 11])


def test_lookup_returns_copy():
    E_local = [[1, 0], [0, 1]]
    h = get_embedding(E_local, 0)
    h[0] = 99
    assert E_local[0] == [1, 0]


def test_mentor_probabilities():
    _, s = forward(E, U, 0)
    m, a, p = compute_probabilities(s)
    assert m == 1
    assert close(p, [0.244728, 0.665241, 0.090031])
    assert abs(sum(p) - 1) < TOL
    assert all(x >= 0 for x in p)


def test_equal_scores():
    assert close(compute_probabilities([3, 3])[2], [0.5, 0.5])
    assert close(compute_probabilities([103, 103])[2], [0.5, 0.5])


def test_shift_invariance():
    s = [0, 1, -1]
    _, _, p = compute_probabilities(s)
    _, _, p_shifted = compute_probabilities([x + 100 for x in s])
    assert close(p, p_shifted)


def test_extreme_scores_do_not_crash():
    _, _, p = compute_probabilities([1000, 0])
    assert close(p, [1.0, 0.0])


def test_other_vocabulary_size():
    _, _, p = compute_probabilities([0.5, 0.1, -2, 3, 0])
    assert len(p) == 5 and abs(sum(p) - 1) < TOL


def test_bad_inputs_fail_clearly():
    with pytest.raises(ValueError):
        compute_probabilities([])
    with pytest.raises(ValueError):
        get_embedding([], 0)
    with pytest.raises(IndexError):
        get_embedding(E, 3)
    with pytest.raises(ValueError):
        compute_scores([1], U)
