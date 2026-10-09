import copy
import pytest
from src.model.forward import forward, compute_probabilities
from src.model.gradients import compute_score_error, compute_gradients

TOL = 1e-5
E = [[1, 0], [0, 1], [1, 1]]
U = [[0, 1, -1], [1, 0, 1]]


def close(xs, ys):
    return len(xs) == len(ys) and all(abs(x - y) < TOL for x, y in zip(xs, ys))


def close2d(A, B):
    return len(A) == len(B) and all(close(r1, r2) for r1, r2 in zip(A, B))


def mentor_setup():
    h, s = forward(E, U, 0)
    _, _, p = compute_probabilities(s)
    return h, p


def test_mentor_score_error():
    _, p = mentor_setup()
    e = compute_score_error(p, 1)
    assert close(e, [0.244728, -0.334759, 0.090031])
    assert abs(sum(e)) < TOL


def test_score_error_concept_check():
    e = compute_score_error([0.5, 0.3, 0.2], 2)
    assert close(e, [0.5, 0.3, -0.8]) and abs(sum(e)) < TOL


def test_score_error_does_not_modify_p():
    _, p = mentor_setup()
    p_copy = list(p)
    compute_score_error(p, 1)
    assert p == p_copy


def test_mentor_gradients():
    h, p = mentor_setup()
    grad_U, grad_h = compute_gradients(h, U, p, 1)
    assert close2d(grad_U, [[0.244728, -0.334759, 0.090031], [0, 0, 0]])
    assert close(grad_h, [-0.424790, 0.334759])


def test_gradients_do_not_modify_inputs():
    h, p = mentor_setup()
    U_before, h_before = copy.deepcopy(U), list(h)
    compute_gradients(h, U, p, 1)
    assert U == U_before and h == h_before


def test_gradient_shapes_generic_size():
    E2 = [[1, 2, 3], [4, 5, 6]]
    U2 = [[1, 0], [0, 1], [1, 1]]
    h2, s2 = forward(E2, U2, 1)
    _, _, p2 = compute_probabilities(s2)
    gU, gh = compute_gradients(h2, U2, p2, 0)
    assert len(gU) == 3 and len(gU[0]) == 2 and len(gh) == 3


def test_bad_inputs_fail_clearly():
    h, p = mentor_setup()
    with pytest.raises(IndexError):
        compute_score_error(p, 3)
    with pytest.raises(ValueError):
        compute_gradients([1], U, p, 1)
    with pytest.raises(ValueError):
        compute_gradients(h, U, [0.5, 0.5], 1)
