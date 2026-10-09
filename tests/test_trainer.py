import copy
import math
import random
import pytest
from src.data.pairs import generate_pairs
from src.model.forward import forward, compute_probabilities
from src.model.loss import compute_loss
from src.model.initialization import initialize_parameters
from src.training.trainer import (train_pair, evaluate_loss,
                                  train_one_epoch, train)

TOL = 1e-5


def fresh():
    return [[1, 0], [0, 1], [1, 1]], [[0, 1, -1], [1, 0, 1]]


def pair_loss(E, U, i, t):
    _, s = forward(E, U, i)
    m, a, _ = compute_probabilities(s)
    return compute_loss(s, t, m, a)


def tiny_corpus_pairs():
    sentences = [[0, 1, 2], [0, 1, 3], [4, 5, 2], [4, 5, 3], [0, 1, 2], [4, 5, 3]]
    return generate_pairs(sentences, 1)


# ---------- evaluate_loss ----------

def test_evaluate_loss_single_pair_matches_mentor_loss():
    E, U = fresh()
    assert abs(evaluate_loss(E, U, [(0, 1)]) - 0.407606) < TOL


def test_evaluate_loss_is_mean_of_pair_losses():
    E, U = fresh()
    pairs = [(0, 1), (0, 2), (1, 0), (2, 2)]
    expected = sum(pair_loss(E, U, i, t) for i, t in pairs) / len(pairs)
    assert abs(evaluate_loss(E, U, pairs) - expected) < TOL


def test_evaluate_loss_does_not_change_weights():
    E, U = fresh()
    E_before, U_before = copy.deepcopy(E), copy.deepcopy(U)
    evaluate_loss(E, U, [(0, 1), (1, 2)])
    assert E == E_before and U == U_before


def test_evaluate_loss_bad_inputs_fail_clearly():
    E, U = fresh()
    with pytest.raises(ValueError):
        evaluate_loss(E, U, [])
    with pytest.raises(IndexError):
        evaluate_loss(E, U, [(5, 1)])
    with pytest.raises(IndexError):
        evaluate_loss(E, U, [(0, 9)])


# ---------- train_one_epoch ----------

def test_epoch_does_not_modify_input_pair_list():
    E, U = initialize_parameters(6, 4, seed=3, scale=0.01)
    pairs = tiny_corpus_pairs()
    pairs_before = list(pairs)
    train_one_epoch(E, U, pairs, 0.1, random.Random(1))
    assert pairs == pairs_before


def test_epoch_order_is_a_permutation_of_the_pairs():
    E, U = initialize_parameters(6, 4, seed=3, scale=0.01)
    pairs = tiny_corpus_pairs()
    order = train_one_epoch(E, U, pairs, 0.1, random.Random(1))
    assert sorted(order) == sorted(pairs) and len(order) == len(pairs)


def test_shuffle_changes_order_across_seeds_and_epochs():
    pairs = tiny_corpus_pairs()
    E, U = initialize_parameters(6, 4, seed=3, scale=0.01)
    orders_by_seed = {tuple(train_one_epoch(copy.deepcopy(E), copy.deepcopy(U),
                                            pairs, 0.1, random.Random(s)))
                      for s in range(5)}
    assert len(orders_by_seed) > 1                       # seeds give different orders
    rng = random.Random(1)
    orders_by_epoch = {tuple(train_one_epoch(E, U, pairs, 0.1, rng))
                       for _ in range(5)}
    assert len(orders_by_epoch) > 1                      # epochs reshuffle


def test_same_seed_gives_identical_training():
    pairs = tiny_corpus_pairs()
    results = []
    for _ in range(2):
        E, U = initialize_parameters(6, 4, seed=3, scale=0.01)
        train(E, U, pairs, 0.1, 5, shuffle_seed=9, verbose=False)
        results.append((E, U))
    assert results[0] == results[1]


def test_epoch_only_changes_rows_of_centers_that_appear():
    sentences = [[0, 1, 2]]                              # words 3 and 4 never appear
    pairs = generate_pairs(sentences, 1)
    E, U = initialize_parameters(5, 3, seed=1, scale=0.1)
    E_before = copy.deepcopy(E)
    train_one_epoch(E, U, pairs, 0.1, random.Random(1))
    assert E[3] == E_before[3] and E[4] == E_before[4]
    assert E[0] != E_before[0]


# ---------- train ----------

def test_history_length_and_first_value():
    E, U = initialize_parameters(6, 4, seed=3, scale=0.01)
    pairs = tiny_corpus_pairs()
    J0 = evaluate_loss(E, U, pairs)
    history = train(E, U, pairs, 0.1, 4, shuffle_seed=5, verbose=False)
    assert len(history) == 5                             # before + 4 epochs
    assert abs(history[0] - J0) < TOL
    assert abs(history[-1] - evaluate_loss(E, U, pairs)) < TOL


def test_initial_loss_is_close_to_ln_V():
    E, U = initialize_parameters(6, 4, seed=3, scale=0.01)
    history = train(E, U, tiny_corpus_pairs(), 0.1, 1, shuffle_seed=5, verbose=False)
    assert abs(history[0] - math.log(6)) < 0.01


def test_loss_decreases_on_tiny_corpus():
    E, U = initialize_parameters(6, 4, seed=3, scale=0.01)
    history = train(E, U, tiny_corpus_pairs(), 0.1, 30, shuffle_seed=5, verbose=False)
    assert history[-1] < 0.6 * history[0]


def test_train_prints_before_and_after_each_epoch(capsys):
    E, U = initialize_parameters(6, 4, seed=3, scale=0.01)
    train(E, U, tiny_corpus_pairs(), 0.1, 3, shuffle_seed=5, verbose=True)
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len(lines) == 4                               # epoch 0 + 3 epochs


def test_train_bad_arguments_fail_clearly():
    E, U = initialize_parameters(6, 4, seed=3, scale=0.01)
    pairs = tiny_corpus_pairs()
    with pytest.raises(ValueError):
        train(E, U, [], 0.1, 3, shuffle_seed=1, verbose=False)
    with pytest.raises(ValueError):
        train(E, U, pairs, 0.0, 3, shuffle_seed=1, verbose=False)
    with pytest.raises(ValueError):
        train(E, U, pairs, -0.1, 3, shuffle_seed=1, verbose=False)
    with pytest.raises(ValueError):
        train(E, U, pairs, 0.1, 0, shuffle_seed=1, verbose=False)
