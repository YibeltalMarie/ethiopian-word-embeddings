import json
import os
import pytest
from src.data.pairs import generate_pairs
from src.evaluation.nearest_neighbors import nearest_neighbors
from src.model.forward import forward, compute_probabilities
from src.model.initialization import initialize_parameters
from src.training.trainer import train
from src.utils.io import save_model, load_model, FILE_NAMES

TOL = 1e-5
VOCAB = ["cat", "dog", "car"]
E = [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]]
U = [[0.7, 0.8, -0.9], [0.0, 1.0, -1.0]]


def test_round_trip_is_exact(tmp_path):
    config = {"seed": 7, "learning_rate": 0.1}
    metadata = {"language": "toy", "epochs": 3}
    save_model(tmp_path, E, U, VOCAB, config, metadata)
    E2, U2, vocab2, config2, metadata2 = load_model(tmp_path)
    assert E2 == E and U2 == U and vocab2 == VOCAB      # exact, not just close
    assert config2 == config and metadata2 == metadata


def test_all_five_files_are_created(tmp_path):
    save_model(tmp_path, E, U, VOCAB)
    assert sorted(os.listdir(tmp_path)) == sorted(FILE_NAMES)


def test_default_config_and_metadata_are_empty_dicts(tmp_path):
    save_model(tmp_path, E, U, VOCAB)
    *_, config, metadata = load_model(tmp_path)
    assert config == {} and metadata == {}


def test_awkward_floats_survive(tmp_path):
    E_hard = [[0.1 + 0.2, 1e-12], [123456789.123456789, -1e-300]]
    U_hard = [[1 / 3, 2 / 3], [1e300, -0.0]]
    save_model(tmp_path, E_hard, U_hard, ["a", "b"])
    E2, U2, *_ = load_model(tmp_path)
    assert E2 == E_hard and U2 == U_hard


def test_geez_vocabulary_round_trips_as_real_utf8(tmp_path):
    vocab = ["ሰላም", "ቤት", "ውሃ"]
    save_model(tmp_path, E, U, vocab)
    assert load_model(tmp_path)[2] == vocab
    raw = (tmp_path / "vocabulary.json").read_text(encoding="utf-8")
    assert "ሰላም" in raw and "\\u" not in raw


def test_reloaded_model_reproduces_neighbors_and_probabilities(tmp_path):
    words = ["the", "cat", "dog", "sat", "ran"]
    idx = {w: i for i, w in enumerate(words)}
    sents = [["the", "cat", "sat"], ["the", "dog", "ran"], ["the", "cat", "ran"]]
    pairs = generate_pairs([[idx[w] for w in s] for s in sents], 1)
    E1, U1 = initialize_parameters(len(words), 4, seed=3, scale=0.1)
    train(E1, U1, pairs, 0.1, 20, shuffle_seed=5, verbose=False)
    save_model(tmp_path, E1, U1, words, {"seed": 3})
    E2, U2, words2, _, _ = load_model(tmp_path)
    assert words2 == words
    for w in words:
        n1 = nearest_neighbors(w, E1, words, 3)
        n2 = nearest_neighbors(w, E2, words2, 3)
        assert [x for x, _ in n1] == [x for x, _ in n2]              # same order
        assert all(abs(a - b) < TOL for (_, a), (_, b) in zip(n1, n2))
    for i, _ in pairs:
        p1 = compute_probabilities(forward(E1, U1, i)[1])[2]
        p2 = compute_probabilities(forward(E2, U2, i)[1])[2]
        assert all(abs(a - b) < TOL for a, b in zip(p1, p2))


def test_nested_directory_is_created(tmp_path):
    target = tmp_path / "models" / "run_1"
    save_model(target, E, U, VOCAB)
    assert load_model(target)[2] == VOCAB


def test_refuses_to_overwrite_unless_asked(tmp_path):
    save_model(tmp_path, E, U, VOCAB)
    with pytest.raises(FileExistsError):
        save_model(tmp_path, E, U, VOCAB)
    save_model(tmp_path, E, U, ["x", "y", "z"], overwrite=True)
    assert load_model(tmp_path)[2] == ["x", "y", "z"]


def test_bad_models_are_rejected_and_nothing_is_written(tmp_path):
    bad_calls = [
        dict(E=E, U=U, vocabulary=VOCAB[:2]),                      # vocab/E mismatch
        dict(E=E, U=[[1, 2], [3, 4]], vocabulary=VOCAB),           # U wrong shape
        dict(E=E, U=U, vocabulary=["a", "a", "b"]),                # duplicate words
        dict(E=E, U=U, vocabulary=["a", "b", 3]),                  # non-string word
        dict(E=[], U=U, vocabulary=VOCAB),                         # empty E
        dict(E=[[0.1, 0.2], [0.3]], U=U, vocabulary=VOCAB[:2]),    # ragged E
        dict(E=[[float("nan"), 0], [0, 0], [0, 0]], U=U, vocabulary=VOCAB),
        dict(E=E, U=[[float("inf"), 0, 0], [0, 0, 0]], vocabulary=VOCAB),
    ]
    for kwargs in bad_calls:
        target = tmp_path / "bad"
        with pytest.raises(ValueError):
            save_model(target, **kwargs)
        assert not target.exists()                                  # disk untouched


def test_unserializable_config_writes_nothing(tmp_path):
    target = tmp_path / "bad"
    with pytest.raises(TypeError):
        save_model(target, E, U, VOCAB, config={"f": object()})
    assert not target.exists()


def test_load_missing_file_fails_clearly(tmp_path):
    save_model(tmp_path, E, U, VOCAB)
    os.remove(tmp_path / "U.json")
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does_not_exist")


def test_load_corrupt_json_fails_clearly(tmp_path):
    save_model(tmp_path, E, U, VOCAB)
    (tmp_path / "E.json").write_text("[[0.1, 0.2", encoding="utf-8")
    with pytest.raises(ValueError):
        load_model(tmp_path)


def test_load_inconsistent_artifacts_fail_clearly(tmp_path):
    save_model(tmp_path, E, U, VOCAB)
    (tmp_path / "vocabulary.json").write_text(json.dumps(["cat", "dog"]),
                                              encoding="utf-8")
    with pytest.raises(ValueError):
        load_model(tmp_path)
    save_model(tmp_path, E, U, VOCAB, overwrite=True)
    (tmp_path / "config.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_model(tmp_path)
