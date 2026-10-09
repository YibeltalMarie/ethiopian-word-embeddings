import json
import os
import pytest
import src.training.trainer as trainer_module
import src.training.checkpointing as ckpt
from src.data.pairs import generate_pairs
from src.model.initialization import initialize_parameters
from src.training.trainer import evaluate_loss
from src.training.checkpointing import (train_with_checkpoints, save_checkpoint,
                                        load_checkpoint, pairs_fingerprint,
                                        epoch_shuffle_seed, CHECKPOINT_NAME,
                                        BACKUP_NAME)

LR, SEED = 0.1, 9


def setup():
    sentences = [[0, 1, 2], [0, 1, 3], [4, 5, 2], [4, 5, 3], [0, 1, 2], [4, 5, 3]]
    pairs = generate_pairs(sentences, 1)
    E, U = initialize_parameters(6, 4, seed=3, scale=0.1)
    return E, U, pairs


def run(directory, epochs, E=None, U=None, pairs=None, **kw):
    e, u, p = setup()
    E, U, pairs = (E or e), (U or u), (pairs or p)
    result = train_with_checkpoints(E, U, pairs, kw.pop("lr", LR), epochs,
                                    kw.pop("seed", SEED), directory,
                                    verbose=kw.pop("verbose", False), **kw)
    return E, U, result


def test_straight_run_basics(tmp_path):
    E0, U0, pairs = setup()
    j0 = evaluate_loss(E0, U0, pairs)
    E, U, r = run(tmp_path, 4)
    assert r["epochs_done"] == 4 and len(r["history"]) == 5 and len(r["epoch_seconds"]) == 4
    assert abs(r["history"][0] - j0) < 1e-12
    assert r["history"][-1] < r["history"][0]
    assert abs(r["total_seconds"] - sum(r["epoch_seconds"])) < 1e-9
    assert not os.path.exists(os.path.join(tmp_path, CHECKPOINT_NAME + ".tmp"))
    assert os.path.exists(os.path.join(tmp_path, BACKUP_NAME))


def test_clean_stop_then_resume_equals_uninterrupted_run(tmp_path):
    E_a, U_a, r_a = run(tmp_path / "straight", 6)
    run(tmp_path / "split", 3)                              # "session 1" ends after 3 epochs
    E_b, U_b, r_b = run(tmp_path / "split", 6)              # "session 2" rebuilds E, U and resumes
    assert E_a == E_b and U_a == U_b
    assert r_a["history"] == r_b["history"]
    assert len(r_b["epoch_seconds"]) == 6


def test_crash_in_the_middle_of_an_epoch_then_resume(tmp_path, monkeypatch):
    E_a, U_a, r_a = run(tmp_path / "straight", 6)
    original = trainer_module.train_pair
    calls = {"n": 0}
    n_pairs = len(setup()[2])

    def crashing(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3 * n_pairs + 5:                   # inside epoch 4
            raise RuntimeError("simulated disconnect")
        return original(*a, **k)

    monkeypatch.setattr(trainer_module, "train_pair", crashing)
    with pytest.raises(RuntimeError):
        run(tmp_path / "crash", 6)
    monkeypatch.undo()
    assert load_checkpoint(tmp_path / "crash")["epochs_done"] == 3
    E_b, U_b, r_b = run(tmp_path / "crash", 6)
    assert E_a == E_b and U_a == U_b and r_a["history"] == r_b["history"]


def test_damaged_checkpoint_falls_back_to_the_backup(tmp_path):
    E_a, U_a, r_a = run(tmp_path / "straight", 6)
    d = tmp_path / "damaged"
    run(d, 4)
    (d / CHECKPOINT_NAME).write_text("{broken", encoding="utf-8")
    assert load_checkpoint(d)["epochs_done"] == 3           # backup is one epoch older
    E_b, U_b, r_b = run(d, 6)
    assert E_a == E_b and U_a == U_b and r_a["history"] == r_b["history"]


def test_missing_current_file_also_falls_back(tmp_path):
    run(tmp_path, 3)
    os.remove(tmp_path / CHECKPOINT_NAME)
    assert load_checkpoint(tmp_path)["epochs_done"] == 2


def test_no_checkpoint_gives_none_and_all_damaged_raises(tmp_path):
    assert load_checkpoint(tmp_path) is None
    assert load_checkpoint(tmp_path / "does_not_exist") is None
    run(tmp_path, 2)
    (tmp_path / CHECKPOINT_NAME).write_text("nope", encoding="utf-8")
    (tmp_path / BACKUP_NAME).write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path)


def test_resume_is_refused_when_the_run_definition_changed(tmp_path):
    run(tmp_path, 2, config={"window": 1})
    with pytest.raises(ValueError):
        run(tmp_path, 3, config={"window": 1}, lr=0.05)
    with pytest.raises(ValueError):
        run(tmp_path, 3, config={"window": 1}, seed=10)
    with pytest.raises(ValueError):
        run(tmp_path, 3, config={"window": 2})
    with pytest.raises(ValueError):
        run(tmp_path, 3)                                    # config missing = different
    E, U, pairs = setup()
    with pytest.raises(ValueError):
        run(tmp_path, 3, config={"window": 1}, pairs=pairs[:-1])
    E_big, U_big = initialize_parameters(7, 4, seed=3, scale=0.1)
    with pytest.raises(ValueError):
        run(tmp_path, 3, E=E_big, U=U_big, config={"window": 1})


def test_requesting_fewer_epochs_fails_and_equal_epochs_changes_nothing(tmp_path):
    E_a, U_a, r_a = run(tmp_path, 4)
    with pytest.raises(ValueError):
        run(tmp_path, 3)
    E_b, U_b, r_b = run(tmp_path, 4)
    assert E_a == E_b and U_a == U_b and r_a["history"] == r_b["history"]
    assert r_b["epoch_seconds"] == r_a["epoch_seconds"]     # no extra training happened


def test_resume_updates_the_callers_lists_in_place(tmp_path):
    run(tmp_path, 2)
    E, U, pairs = setup()
    E_id, U_id = id(E), id(U)
    train_with_checkpoints(E, U, pairs, LR, 3, SEED, tmp_path, verbose=False)
    assert id(E) == E_id and id(U) == U_id
    assert load_checkpoint(tmp_path)["E"] == E


def test_divergence_stops_without_overwriting_the_good_checkpoint(tmp_path, monkeypatch):
    real = ckpt.evaluate_loss
    calls = {"n": 0}

    def sabotaged(E, U, pairs):
        calls["n"] += 1
        return float("nan") if calls["n"] == 3 else real(E, U, pairs)

    monkeypatch.setattr(ckpt, "evaluate_loss", sabotaged)
    with pytest.raises(ValueError, match="diverged"):
        run(tmp_path, 5)
    assert load_checkpoint(tmp_path)["epochs_done"] == 1    # epochs 0 and 1 saved, epoch 2 refused


def test_save_checkpoint_rejects_bad_states_and_writes_nothing(tmp_path):
    E, U, pairs = setup()
    good = {"epochs_done": 0, "E": E, "U": U, "history": [1.0], "epoch_seconds": [],
            "learning_rate": LR, "shuffle_seed": SEED, "pairs_sha256": "x", "config": {}}
    bad_states = [
        {k: v for k, v in good.items() if k != "history"},
        dict(good, history=[1.0, 2.0]),
        dict(good, epoch_seconds=[1.0]),
        dict(good, epochs_done=-1),
        dict(good, U=[[0.0]]),
        dict(good, history=[float("nan")]),
        "not a dict",
    ]
    for state in bad_states:
        with pytest.raises(ValueError):
            save_checkpoint(tmp_path / "x", state)
    assert not (tmp_path / "x").exists()


def test_shuffle_policy_is_per_epoch_and_reproducible():
    seeds = {epoch_shuffle_seed(9, e) for e in range(1, 50)}
    assert len(seeds) == 49
    assert epoch_shuffle_seed(9, 3) == epoch_shuffle_seed(9, 3)
    assert epoch_shuffle_seed(9, 1) != epoch_shuffle_seed(10, 1)


def test_different_shuffle_seed_gives_a_different_model(tmp_path):
    E_a, _, _ = run(tmp_path / "a", 3, seed=1)
    E_b, _, _ = run(tmp_path / "b", 3, seed=2)
    assert E_a != E_b


def test_fingerprint_changes_with_any_pair():
    pairs = [(0, 1), (1, 0), (2, 3)]
    assert pairs_fingerprint(pairs) == pairs_fingerprint(list(pairs))
    assert pairs_fingerprint(pairs) != pairs_fingerprint(pairs[:-1])
    assert pairs_fingerprint(pairs) != pairs_fingerprint([(0, 1), (1, 0), (2, 4)])
    assert pairs_fingerprint([(1, 23)]) != pairs_fingerprint([(12, 3)])


def test_checkpoint_file_is_plain_json_with_the_run_definition(tmp_path):
    run(tmp_path, 2, config={"window": 1, "d": 4})
    data = json.loads((tmp_path / CHECKPOINT_NAME).read_text(encoding="utf-8"))
    assert data["epochs_done"] == 2 and data["config"] == {"window": 1, "d": 4}
    assert data["learning_rate"] == LR and data["shuffle_seed"] == SEED


def test_verbose_output(tmp_path, capsys):
    run(tmp_path, 2, verbose=True)
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("epoch 0 (before training)") and len(out) == 3
    run(tmp_path, 3, verbose=True)
    out = capsys.readouterr().out
    assert "resuming after epoch 2" in out and "epoch 3/3" in out


def test_bad_arguments(tmp_path):
    E, U, pairs = setup()
    for kwargs in (dict(pairs=[]), dict(learning_rate=0), dict(learning_rate=-1),
                   dict(epochs=0), dict(epochs=2.5), dict(epochs=True),
                   dict(shuffle_seed=1.5), dict(shuffle_seed=True)):
        args = dict(E=E, U=U, pairs=pairs, learning_rate=LR, epochs=2,
                    shuffle_seed=SEED, checkpoint_dir=tmp_path / "bad", verbose=False)
        args.update(kwargs)
        with pytest.raises(ValueError):
            train_with_checkpoints(**args)
    assert not (tmp_path / "bad").exists()
