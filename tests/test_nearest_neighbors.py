import copy
import pytest
from src.evaluation.nearest_neighbors import nearest_neighbors, print_neighbors

TOL = 1e-5
VOCAB = ["cat", "dog", "car", "apple"]
E = [[1, 0], [0.9, 0.1], [0, 1], [-1, 0]]


def test_order_by_similarity():
    result = nearest_neighbors("cat", E, VOCAB, k=3)
    assert [w for w, _ in result] == ["dog", "car", "apple"]
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)
    assert abs(scores[1] - 0.0) < TOL and abs(scores[2] - (-1.0)) < TOL


def test_query_word_is_excluded():
    assert "cat" not in [w for w, _ in nearest_neighbors("cat", E, VOCAB, k=3)]


def test_k_limits_the_result_and_extra_k_is_safe():
    assert [w for w, _ in nearest_neighbors("cat", E, VOCAB, k=1)] == ["dog"]
    assert len(nearest_neighbors("cat", E, VOCAB, k=10)) == 3   # only 3 others exist


def test_unknown_word_returns_empty_list():
    assert nearest_neighbors("ghost", E, VOCAB, k=3) == []


def test_ties_are_broken_by_word_id():
    vocab = ["a", "x", "y", "z"]
    E2 = [[1, 1], [1, 0], [0, 1], [2, 2]]
    result = nearest_neighbors("a", E2, vocab, k=3)
    assert [w for w, _ in result] == ["z", "x", "y"]            # x before y on a tie
    all_tied = nearest_neighbors("a", [[1, 0], [0, 1], [0, 1], [0, 2]], vocab, k=3)
    assert [w for w, _ in all_tied] == ["x", "y", "z"]


def test_zero_norm_vectors_do_not_crash():
    vocab = ["a", "b", "c"]
    E3 = [[0, 0], [1, 0], [0, 1]]                               # query is zero
    assert [s for _, s in nearest_neighbors("a", E3, vocab, k=2)] == [0.0, 0.0]
    E4 = [[1, 0], [0, 0], [1, 0]]                               # a candidate is zero
    result = dict(nearest_neighbors("a", E4, vocab, k=2))
    assert result["b"] == 0.0 and abs(result["c"] - 1.0) < TOL


def test_does_not_modify_E_and_is_deterministic():
    E_before = copy.deepcopy(E)
    first = nearest_neighbors("cat", E, VOCAB, k=3)
    second = nearest_neighbors("cat", E, VOCAB, k=3)
    assert E == E_before and first == second


def test_single_word_vocabulary_has_no_neighbors():
    assert nearest_neighbors("a", [[1, 0]], ["a"], k=3) == []


def test_bad_arguments_fail_clearly():
    with pytest.raises(ValueError):
        nearest_neighbors("cat", E, VOCAB, k=0)
    with pytest.raises(ValueError):
        nearest_neighbors("cat", E, VOCAB[:3], k=3)             # vocab/E length mismatch


def test_print_neighbors_handles_known_and_unknown_words(capsys):
    print_neighbors(["cat", "ghost"], E, VOCAB, k=2)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("cat: dog (") and "car (" in lines[0]
    assert "ghost" in lines[1] and "unknown" in lines[1]
