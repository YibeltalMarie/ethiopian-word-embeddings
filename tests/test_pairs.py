import pytest
from src.data.pairs import generate_pairs


def test_mentor_context_window():
    pairs = generate_pairs([["a", "b", "c", "d"]], 1)
    assert pairs == [("a", "b"), ("b", "a"), ("b", "c"),
                     ("c", "b"), ("c", "d"), ("d", "c")]


def test_no_cross_sentence_pairs():
    pairs = generate_pairs([["a", "b"], ["c", "d"]], 5)
    assert pairs == [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")]
    assert ("b", "c") not in pairs and ("c", "b") not in pairs


def test_concept_check_x_y_z_window_2():
    pairs = generate_pairs([["x", "y", "z"]], 2)
    assert len(pairs) == 6
    assert set(pairs) == {("x", "y"), ("x", "z"), ("y", "x"),
                          ("y", "z"), ("z", "x"), ("z", "y")}


def test_window_2_cat_example():
    sentence = ["the", "cat", "sat", "on", "the", "mat"]
    pairs = generate_pairs([sentence], 2)
    cat_pairs = [p for p in pairs if p[0] == "cat"]
    assert cat_pairs == [("cat", "the"), ("cat", "sat"), ("cat", "on")]


def test_edge_words_get_fewer_contexts():
    pairs = generate_pairs([["a", "b", "c", "d", "e"]], 2)
    assert len([p for p in pairs if p[0] == "a"]) == 2    # b, c
    assert len([p for p in pairs if p[0] == "c"]) == 4    # a, b, d, e


def test_pair_count_formula_for_single_sentence():
    # window 1 on n tokens gives 2 * (n - 1) pairs
    for n in (2, 3, 7):
        pairs = generate_pairs([list(range(n))], 1)
        assert len(pairs) == 2 * (n - 1)


def test_repeated_tokens_keep_duplicate_pairs():
    pairs = generate_pairs([["a", "b", "a"]], 1)
    assert pairs == [("a", "b"), ("b", "a"), ("b", "a"), ("a", "b")]


def test_token_never_paired_with_own_position():
    pairs = generate_pairs([["a", "b", "c"]], 3)
    assert ("a", "a") not in pairs and ("b", "b") not in pairs


def test_short_and_empty_sentences():
    assert generate_pairs([["a"]], 2) == []
    assert generate_pairs([[]], 2) == []
    assert generate_pairs([], 2) == []


def test_works_on_ids_as_well_as_words():
    assert generate_pairs([[0, 1, 2]], 1) == [(0, 1), (1, 0), (1, 2), (2, 1)]


def test_input_is_not_modified():
    sentences = [["a", "b", "c"]]
    generate_pairs(sentences, 1)
    assert sentences == [["a", "b", "c"]]


def test_bad_window_fails_clearly():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            generate_pairs([["a", "b"]], bad)
