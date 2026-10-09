"""Checkpointed training: save after every epoch, resume after a disconnect.

Why this exists: one baseline epoch takes about 12 minutes on Colab, so a full
run takes hours, and a Colab session can end at any time. After every epoch the
complete training state is written to disk, and a rerun with the same
arguments continues from the last saved epoch and ends up with EXACTLY the
model an uninterrupted run would have produced.

Shuffle policy (recorded for the experiment log): epoch e (counting from 1)
shuffles the pair list with random.Random(shuffle_seed * 1000003 + e). Each
epoch's order depends only on (shuffle_seed, e), never on earlier epochs, which
is what makes resuming exact. This differs from trainer.train, which uses one
generator across all epochs, so the two functions give different (equally
valid) shuffles for the same seed.
"""
import hashlib
import json
import math
import os
import random
import time

from src.training.trainer import evaluate_loss, train_one_epoch

CHECKPOINT_NAME = "checkpoint.json"
BACKUP_NAME = "checkpoint.prev.json"
_REQUIRED = ("epochs_done", "E", "U", "history", "epoch_seconds",
             "learning_rate", "shuffle_seed", "pairs_sha256", "config")


def pairs_fingerprint(pairs):
    """SHA-256 of a pair list, so a resume can prove it uses the same pairs."""
    digest = hashlib.sha256()
    for i, t in pairs:
        digest.update(f"{i},{t};".encode("ascii"))
    return digest.hexdigest()


def epoch_shuffle_seed(shuffle_seed, epoch):
    """Seed of the shuffle used in epoch `epoch` (1-based). See module docstring."""
    return shuffle_seed * 1_000_003 + epoch


def _validate_state(state):
    if not isinstance(state, dict):
        raise ValueError("checkpoint must be a JSON object")
    missing = [k for k in _REQUIRED if k not in state]
    if missing:
        raise ValueError(f"checkpoint is missing fields: {missing}")
    done = state["epochs_done"]
    if isinstance(done, bool) or not isinstance(done, int) or done < 0:
        raise ValueError("epochs_done must be an integer >= 0")
    if len(state["history"]) != done + 1:
        raise ValueError("history must hold epochs_done + 1 values")
    if len(state["epoch_seconds"]) != done:
        raise ValueError("epoch_seconds must hold epochs_done values")
    E, U = state["E"], state["U"]
    if not E or not E[0] or not U or not U[0]:
        raise ValueError("E and U must be non-empty")
    d, V = len(E[0]), len(E)
    if any(len(r) != d for r in E) or len(U) != d or any(len(r) != V for r in U):
        raise ValueError("E and U shapes are inconsistent")


def save_checkpoint(directory, state):
    """Write the state atomically and keep the previous checkpoint as a backup.

    The new state goes to a temporary file first. Only then does the current
    checkpoint become checkpoint.prev.json and the temporary file become
    checkpoint.json, so a crash while writing never destroys the last good
    checkpoint.

    Raises:  ValueError if the state is malformed or contains NaN/infinity
             (nothing is written in that case)
    """
    _validate_state(state)
    text = json.dumps(state, ensure_ascii=False, allow_nan=False)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, CHECKPOINT_NAME)
    backup = os.path.join(directory, BACKUP_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        os.replace(path, backup)
    os.replace(tmp, path)


def load_checkpoint(directory):
    """Load the latest usable checkpoint.

    Output:  the state dict, or None if the directory has no checkpoint files
    Policy:  checkpoint.json is tried first; if it is missing or damaged,
             checkpoint.prev.json (one epoch older) is used instead
    Raises:  ValueError if checkpoint files exist but none of them is usable
    """
    problems = []
    for name in (CHECKPOINT_NAME, BACKUP_NAME):
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            _validate_state(state)
            return state
        except (ValueError, OSError) as err:
            problems.append(f"{name}: {err}")
    if problems:
        raise ValueError("no usable checkpoint: " + "; ".join(problems))
    return None


def train_with_checkpoints(E, U, pairs, learning_rate, epochs, shuffle_seed,
                           checkpoint_dir, config=None, verbose=True):
    """Train for `epochs` epochs, saving after every one; resume if possible.

    E and U are changed IN PLACE. On a fresh start they are the initial
    parameters. On a resume their values are replaced by the checkpoint's
    (only their shapes matter), so a new session can simply rebuild them with
    initialize_parameters and call this function again.

    J (history) is measured with evaluate_loss at frozen weights on the
    original pair list: history[0] before training, history[k] after epoch k.

    A resume is refused (ValueError) unless learning_rate, shuffle_seed, the
    pair list (by fingerprint), `config` and the matrix shapes all match the
    checkpoint. `config` is a JSON-serializable dict of everything else that
    defines the run (window, d, init seed, corpus...). Do not put `epochs` in
    it: asking for more epochs than were saved is how a run is extended.

    If J ever becomes NaN or infinite, training stops with ValueError and the
    last good checkpoint stays untouched (lower the learning rate).

    Output:  dict with history, epoch_seconds, epochs_done, total_seconds
    Raises:  ValueError for bad arguments, a mismatched or unusable
             checkpoint, a checkpoint with more epochs than requested, or
             divergence
    """
    if not pairs:
        raise ValueError("cannot train on an empty pair list")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"epochs must be an integer >= 1, got {epochs!r}")
    if isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, int):
        raise ValueError(f"shuffle_seed must be an integer, got {shuffle_seed!r}")
    config = {} if config is None else config
    fingerprint = pairs_fingerprint(pairs)

    state = load_checkpoint(checkpoint_dir)
    if state is None:
        j0 = evaluate_loss(E, U, pairs)
        if not math.isfinite(j0):
            raise ValueError("initial J is not finite")
        state = {"epochs_done": 0, "E": E, "U": U, "history": [j0],
                 "epoch_seconds": [], "learning_rate": learning_rate,
                 "shuffle_seed": shuffle_seed, "pairs_sha256": fingerprint,
                 "config": config}
        save_checkpoint(checkpoint_dir, state)
        if verbose:
            print(f"epoch 0 (before training): J = {j0:.6f}")
    else:
        for key, mine in (("learning_rate", learning_rate),
                          ("shuffle_seed", shuffle_seed),
                          ("pairs_sha256", fingerprint), ("config", config)):
            if state[key] != mine:
                raise ValueError(f"cannot resume: {key} differs from the checkpoint")
        if len(state["E"]) != len(E) or len(state["E"][0]) != len(E[0]) \
                or len(state["U"]) != len(U) or len(state["U"][0]) != len(U[0]):
            raise ValueError("cannot resume: E/U shapes differ from the checkpoint")
        if state["epochs_done"] > epochs:
            raise ValueError(
                f"checkpoint already has {state['epochs_done']} epochs, "
                f"more than the {epochs} requested")
        E[:] = [list(row) for row in state["E"]]
        U[:] = [list(row) for row in state["U"]]
        state["E"], state["U"] = E, U
        if verbose:
            print(f"resuming after epoch {state['epochs_done']} "
                  f"(J = {state['history'][-1]:.6f})")

    for epoch in range(state["epochs_done"] + 1, epochs + 1):
        start = time.perf_counter()
        rng = random.Random(epoch_shuffle_seed(shuffle_seed, epoch))
        train_one_epoch(E, U, pairs, learning_rate, rng)
        j = evaluate_loss(E, U, pairs)
        seconds = time.perf_counter() - start
        if not math.isfinite(j):
            raise ValueError(
                f"training diverged in epoch {epoch} (J is {j}); the last good "
                f"checkpoint (epoch {epoch - 1}) is untouched. Lower the learning rate.")
        state["epochs_done"] = epoch
        state["history"].append(j)
        state["epoch_seconds"].append(seconds)
        save_checkpoint(checkpoint_dir, state)
        if verbose:
            print(f"epoch {epoch}/{epochs}: J = {j:.6f} | {seconds:.0f} s")

    return {"history": list(state["history"]),
            "epoch_seconds": list(state["epoch_seconds"]),
            "epochs_done": state["epochs_done"],
            "total_seconds": sum(state["epoch_seconds"])}
