import pytest
from src.data.vocabulary import has_ethiopic_letter, build_vocabulary, apply_vocabulary
from src.data.corpus import sample_sentences

S = [["ነው", "ቃል", "the"], ["ነው", "ቃል", "1892"], ["ነው", "ሌላ", "the", "the"]]


def test_has_ethiopic_letter():
    assert has_ethiopic_letter("ከ1892") and has_ethiopic_letter("ነው")
    assert not has_ethiopic_letter("1892") and not has_ethiopic_letter("the")
    assert not has_ethiopic_letter("") and not has_ethiopic_letter("፲፱፻")   # numerals are not letters


def test_order_is_frequency_then_alphabetical():
    vocab, counts = build_vocabulary(S, min_freq=1)
    assert vocab == ["ነው", "ቃል", "ሌላ"] and counts == [3, 2, 1]
    tied, _ = build_vocabulary([["ለ", "ሀ", "መ"]], min_freq=1)
    assert tied == ["ሀ", "ለ", "መ"]                        # equal counts: Unicode order


def test_min_freq_and_max_size():
    assert build_vocabulary(S, min_freq=2)[0] == ["ነው", "ቃል"]
    assert build_vocabulary(S, min_freq=1, max_size=2)[0] == ["ነው", "ቃል"]
    assert build_vocabulary(S, min_freq=99)[0] == []


def test_require_ethiopic_switch():
    on, _ = build_vocabulary(S, min_freq=1, require_ethiopic=True)
    off, _ = build_vocabulary(S, min_freq=1, require_ethiopic=False)
    assert "the" not in on and "1892" not in on
    assert "the" in off and "1892" in off


def test_build_is_deterministic_and_no_special_tokens():
    a = build_vocabulary(S, min_freq=1)
    assert a == build_vocabulary(S, min_freq=1)
    assert not any(w.startswith("<") for w in a[0])


def test_build_bad_arguments_and_empty_input():
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError):
            build_vocabulary(S, min_freq=bad)
    for bad in (0, -3):
        with pytest.raises(ValueError):
            build_vocabulary(S, max_size=bad)
    assert build_vocabulary([], min_freq=1) == ([], [])


def test_apply_deletes_oov_and_makes_neighbors_adjacent():
    ids, stats = apply_vocabulary([["ሀ", "x", "ለ", "y", "z", "መ"]], ["ሀ", "ለ", "መ"])
    assert ids == [[0, 1, 2]]                               # OOV removed, rest adjacent
    assert stats["tokens_in"] == 6 and stats["tokens_kept"] == 3


def test_apply_drops_sentences_that_become_too_short():
    ids, stats = apply_vocabulary([["ሀ", "x"], ["ሀ", "ለ"], ["x", "y"]], ["ሀ", "ለ"])
    assert ids == [[0, 1]]
    assert stats["sentences_in"] == 3 and stats["sentences_kept"] == 1


def test_apply_reports_unused_words():
    _, stats = apply_vocabulary([["ሀ", "ለ"]], ["ሀ", "ለ", "መ"])
    assert stats["words_unused"] == 1
    _, stats2 = apply_vocabulary([["ሀ", "x"], ["ሀ", "ለ"]], ["ሀ", "ለ"])
    assert stats2["words_unused"] == 0


def test_apply_ids_follow_vocabulary_positions_and_input_unchanged():
    sents = [["ለ", "ሀ"]]
    before = [list(s) for s in sents]
    ids, _ = apply_vocabulary(sents, ["ሀ", "ለ"])
    assert ids == [[1, 0]] and sents == before


def test_apply_bad_arguments():
    with pytest.raises(ValueError):
        apply_vocabulary([["ሀ"]], ["ሀ", "ሀ"])
    with pytest.raises(ValueError):
        apply_vocabulary([["ሀ"]], ["ሀ"], min_tokens=0)


def test_vocabulary_ids_feed_the_model_and_pairs():
    from src.data.pairs import generate_pairs
    vocab, _ = build_vocabulary(S, min_freq=1)
    ids, _ = apply_vocabulary(S, vocab)
    pairs = generate_pairs(ids, 1)
    assert all(0 <= i < len(vocab) and 0 <= t < len(vocab) for i, t in pairs)


def test_sample_sentences_is_seeded_ordered_and_private():
    import random
    data = [[str(i), "x"] for i in range(100)]
    a = sample_sentences(data, 10, seed=7)
    assert a == sample_sentences(data, 10, seed=7)
    assert a != sample_sentences(data, 10, seed=8)
    assert len(a) == 10 and a == sorted(a, key=lambda s: int(s[0]))
    assert all(s in data for s in a)
    state = random.getstate()
    sample_sentences(data, 10, seed=1)
    assert random.getstate() == state


def test_sample_sentences_edge_cases():
    data = [["a", "b"], ["c", "d"]]
    assert sample_sentences(data, 5, seed=1) == data
    assert sample_sentences(data, 2, seed=1) == data
    with pytest.raises(ValueError):
        sample_sentences(data, 0, seed=1)
