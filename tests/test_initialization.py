import math
import random
import pytest
from src.model.initialization import initialize_parameters
from src.model.forward import forward, compute_probabilities
from src.model.loss import compute_loss
from src.model.gradients import compute_gradients
from src.training.trainer import train_pair


def test_shapes():
    E, U = initialize_parameters(V=7, d=4, seed=1)
    assert len(E) == 7 and all(len(row) == 4 for row in E)
    assert len(U) == 4 and all(len(row) == 7 for row in U)


def test_values_within_scale():
    E, U = initialize_parameters(V=20, d=5, seed=1, scale=0.05)
    assert all(abs(x) <= 0.05 for row in E for x in row)
    assert all(abs(x) <= 0.05 for row in U for x in row)


def test_same_seed_gives_identical_matrices():
    assert initialize_parameters(10, 3, seed=42) == initialize_parameters(10, 3, seed=42)


def test_different_seed_gives_different_matrices():
    E1, U1 = initialize_parameters(10, 3, seed=1)
    E2, U2 = initialize_parameters(10, 3, seed=2)
    assert E1 != E2 and U1 != U2


def test_rows_are_not_all_identical():
    E, _ = initialize_parameters(10, 3, seed=1)
    assert len({tuple(row) for row in E}) == 10         # symmetry is broken


def test_rows_are_independent_lists():
    E, _ = initialize_parameters(5, 3, seed=1)
    E[0][0] = 99.0
    assert all(row[0] != 99.0 for row in E[1:])


def test_global_random_state_untouched():
    state = random.getstate()
    initialize_parameters(10, 3, seed=1)
    assert random.getstate() == state


def test_initial_loss_is_close_to_ln_V():
    V = 50
    E, U = initialize_parameters(V, d=10, seed=7, scale=0.01)
    _, s = forward(E, U, 0)
    m, a, _ = compute_probabilities(s)
    loss = compute_loss(s, 0, m, a)
    assert abs(loss - math.log(V)) < 0.01


def test_all_zero_matrices_are_stuck():
    V, d = 3, 2
    E = [[0.0] * d for _ in range(V)]
    U = [[0.0] * V for _ in range(d)]
    h, s = forward(E, U, 0)
    _, _, p = compute_probabilities(s)
    grad_U, grad_h = compute_gradients(h, U, p, 1)
    assert h == [0.0, 0.0]
    assert all(x == 0 for row in grad_U for x in row)
    assert all(x == 0 for x in grad_h)
    loss = train_pair(E, U, 0, 1, 0.1)
    assert abs(loss - math.log(3)) < 1e-5               # loss is ln(3), not 0
    assert all(x == 0 for row in E for x in row)        # nothing changed
    assert all(x == 0 for row in U for x in row)


def test_bad_arguments_fail_clearly():
    for args in ((0, 3, 1), (3, 0, 1), (-1, 3, 1), (2.5, 3, 1)):
        with pytest.raises(ValueError):
            initialize_parameters(*args)
    with pytest.raises(ValueError):
        initialize_parameters(3, 3, 1, scale=0)
    with pytest.raises(ValueError):
        initialize_parameters(3, 3, 1, scale=-0.1)
