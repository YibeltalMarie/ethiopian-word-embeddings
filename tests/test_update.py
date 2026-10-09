import copy
import pytest
from src.model.forward import forward, compute_probabilities
from src.model.loss import compute_loss
from src.model.gradients import compute_gradients
from src.model.update import update_parameters
from src.training.trainer import train_pair

TOL = 1e-5


def close(xs, ys):
    return len(xs) == len(ys) and all(abs(x - y) < TOL for x, y in zip(xs, ys))


def close2d(A, B):
    return len(A) == len(B) and all(close(r1, r2) for r1, r2 in zip(A, B))


def fresh():
    return [[1, 0], [0, 1], [1, 1]], [[0, 1, -1], [1, 0, 1]]


def pair_loss(E, U, i, t):
    _, s = forward(E, U, i)
    m, a, _ = compute_probabilities(s)
    return compute_loss(s, t, m, a)


def mentor_gradients():
    E, U = fresh()
    h, s = forward(E, U, 0)
    _, _, p = compute_probabilities(s)
    return compute_gradients(h, U, p, 1)


def test_update_concept_check():
    E = [[1.0, 1.0]]
    U = [[0.0], [0.0]]
    update_parameters(E, U, 0, [[0.0], [0.0]], [-0.5, 0.2], 0.1)
    assert close(E[0], [1.05, 0.98])


def test_mentor_one_update():
    E, U = fresh()
    E_before = copy.deepcopy(E)
    old_loss = pair_loss(E, U, 0, 1)
    grad_U, grad_h = mentor_gradients()
    update_parameters(E, U, 0, grad_U, grad_h, 0.1)
    assert close(E[0], [1.042479, -0.033476])
    assert close2d(U, [[-0.0244728, 1.0334759, -1.0090031], [1, 0, 1]])
    assert E[1] == E_before[1] and E[2] == E_before[2]
    new_loss = pair_loss(E, U, 0, 1)
    assert abs(new_loss - 0.361859) < TOL
    assert new_loss < old_loss


def test_train_pair_matches_manual_update():
    E, U = fresh()
    returned = train_pair(E, U, 0, 1, 0.1)
    assert abs(returned - 0.407606) < TOL          # loss BEFORE the update
    assert close(E[0], [1.042479, -0.033476])
    assert close2d(U, [[-0.0244728, 1.0334759, -1.0090031], [1, 0, 1]])


def test_update_order_matters():
    E, U = fresh()
    h, s = forward(E, U, 0)
    _, _, p = compute_probabilities(s)
    grad_U, grad_h = compute_gradients(h, U, p, 1)
    U_updated = copy.deepcopy(U)
    update_parameters(E, U_updated, 0, grad_U, grad_h, 0.1)
    _, wrong_grad_h = compute_gradients(h, U_updated, p, 1)
    assert not close(wrong_grad_h, grad_h)          # new U would give a wrong gradient
    assert close(grad_h, [-0.424790, 0.334759])     # ours used the old U


def test_only_selected_row_changes():
    E, U = fresh()
    train_pair(E, U, 2, 0, 0.1)
    assert E[0] == [1, 0] and E[1] == [0, 1] and E[2] != [1, 1]


def test_bad_inputs_fail_clearly():
    E, U = fresh()
    grad_U, grad_h = mentor_gradients()
    with pytest.raises(IndexError):
        update_parameters(E, U, 3, grad_U, grad_h, 0.1)
    with pytest.raises(ValueError):
        update_parameters(E, U, 0, [[0, 0]], grad_h, 0.1)
    with pytest.raises(ValueError):
        update_parameters(E, U, 0, grad_U, [0.0], 0.1)
