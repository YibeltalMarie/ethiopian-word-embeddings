"""The assignment's check table (section 3), one test per row, tolerance 0.00001.

Rows 1 to 6 are mandatory; rows 7 and 8 are the optional ones. Run with:
    python -m pytest tests/test_assignment_checks.py -v
"""
import math
import pytest
from src.data.pairs import generate_pairs
from src.evaluation.nearest_neighbors import nearest_neighbors
from src.model.forward import forward, compute_probabilities
from src.model.gradients import compute_gradients
from src.model.loss import compute_loss
from src.model.update import update_parameters
from src.utils.io import save_model, load_model

TOL = 0.00001
VOCAB = ["cats", "eat", "food"]
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}


def tiny_model():
    E = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    U = [[0.0, 1.0, -1.0], [1.0, 0.0, 1.0]]
    return E, U


def close(xs, ys):
    return len(xs) == len(ys) and all(abs(x - y) < TOL for x, y in zip(xs, ys))


def pair_loss(E, U, i, t):
    _, s = forward(E, U, i)
    m, a, _ = compute_probabilities(s)
    return compute_loss(s, t, m, a)


def test_check_1_context_window():
    # Sentence [a,b,c,d], window 1; no cross-sentence pairs.
    assert generate_pairs([["a", "b", "c", "d"]], 1) == [
        ("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("c", "d"), ("d", "c")]
    two = generate_pairs([["a", "b"], ["c", "d"]], 1)
    assert two == [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")]


def test_check_2_forward_pass():
    E, U = tiny_model()
    h, s = forward(E, U, WORD_TO_ID["cats"])
    assert close(s, [0, 1, -1])


def test_check_3_probabilities_and_loss():
    E, U = tiny_model()
    i, t = WORD_TO_ID["cats"], WORD_TO_ID["eat"]
    _, s = forward(E, U, i)
    m, a, p = compute_probabilities(s)
    assert close(p, [0.244728, 0.665241, 0.090031])
    assert all(x >= 0 for x in p) and abs(sum(p) - 1) < TOL
    assert abs(compute_loss(s, t, m, a) - 0.407606) < TOL


def test_check_4_gradients():
    E, U = tiny_model()
    i, t = WORD_TO_ID["cats"], WORD_TO_ID["eat"]
    h, s = forward(E, U, i)
    _, _, p = compute_probabilities(s)
    grad_U, grad_h = compute_gradients(h, U, p, t)
    assert close(grad_U[0], [0.244728, -0.334759, 0.090031])
    assert close(grad_U[1], [0, 0, 0])
    assert close(grad_h, [-0.424790, 0.334759])


def test_check_5_one_update():
    E, U = tiny_model()
    i, t = WORD_TO_ID["cats"], WORD_TO_ID["eat"]
    others_before = [list(E[1]), list(E[2])]
    h, s = forward(E, U, i)
    _, _, p = compute_probabilities(s)
    grad_U, grad_h = compute_gradients(h, U, p, t)       # both from the OLD E and U
    update_parameters(E, U, i, grad_U, grad_h, 0.1)
    assert close(E[i], [1.042479, -0.033476])
    assert abs(pair_loss(E, U, i, t) - 0.361859) < TOL
    assert [E[1], E[2]] == others_before                  # other E rows unchanged


def test_check_6_extreme_scores():
    s = [1000.0, 0.0]
    m, a, p = compute_probabilities(s)
    assert abs(compute_loss(s, 1, m, a) - 1000) < TOL     # stable loss
    assert p[1] == 0.0                                    # the target probability underflowed
    with pytest.raises(ValueError):
        math.log(p[1])                                    # naive -ln(p_target) fails


def test_check_7_shift_invariance_optional():
    E, U = tiny_model()
    _, s = forward(E, U, WORD_TO_ID["cats"])
    shifted = [x + 100 for x in s]
    _, _, p1 = compute_probabilities(s)
    _, _, p2 = compute_probabilities(shifted)
    assert close(p1, p2)
    m1, a1, _ = compute_probabilities(s)
    m2, a2, _ = compute_probabilities(shifted)
    assert abs(compute_loss(s, 1, m1, a1) - compute_loss(shifted, 1, m2, a2)) < TOL


def test_check_8_saved_model_optional(tmp_path):
    E, U = tiny_model()
    save_model(tmp_path, E, U, VOCAB, {"note": "assignment check"})
    E2, U2, vocab2, _, _ = load_model(tmp_path)
    assert vocab2 == VOCAB
    for w in VOCAB:
        n1 = nearest_neighbors(w, E, VOCAB, 2)
        n2 = nearest_neighbors(w, E2, vocab2, 2)
        assert [x for x, _ in n1] == [x for x, _ in n2]
        assert close([x for _, x in n1], [x for _, x in n2])
    for i in range(3):
        p1 = compute_probabilities(forward(E, U, i)[1])[2]
        p2 = compute_probabilities(forward(E2, U2, i)[1])[2]
        assert close(p1, p2)
