"""Save and load a trained model as five JSON files.

Files written to the model directory:
    E.json, U.json, vocabulary.json, config.json, metadata.json
"""
import json
import math
import os

FILE_NAMES = ("E.json", "U.json", "vocabulary.json", "config.json", "metadata.json")


def _validate_model(E, U, vocabulary):
    """Raise ValueError unless E, U and vocabulary are consistent.

    Required: E is V x d (rectangular, non-empty), U is d x V, vocabulary is a
    list of V unique strings, and every number is finite.
    """
    if not E or not E[0]:
        raise ValueError("E is empty")
    V, d = len(E), len(E[0])
    if any(len(row) != d for row in E):
        raise ValueError("E rows have different lengths")
    if len(U) != d or any(len(row) != V for row in U):
        raise ValueError(f"U must have shape d x V = {d} x {V}")
    if len(vocabulary) != V:
        raise ValueError(f"vocabulary has {len(vocabulary)} words but E has {V} rows")
    if not all(isinstance(w, str) for w in vocabulary):
        raise ValueError("vocabulary must contain only strings")
    if len(set(vocabulary)) != V:
        raise ValueError("vocabulary contains duplicate words")
    for matrix in (E, U):
        for row in matrix:
            for x in row:
                if not math.isfinite(x):
                    raise ValueError("E and U must contain only finite numbers")


def save_model(directory, E, U, vocabulary, config=None, metadata=None,
               overwrite=False):
    """Write the model to `directory` as five UTF-8 JSON files.

    Everything is validated and converted to JSON text BEFORE any file is
    written, so a bad call leaves the disk untouched. The directory is created
    if needed. Existing model files are never replaced unless overwrite=True.
    Floats round-trip exactly, and Ge'ez (and other non-ASCII) text is written
    as real UTF-8, not \\u escapes.

    Inputs:  directory (path), E (V x d), U (d x V), vocabulary (V unique words),
             config / metadata (JSON-serializable dicts, default empty),
             overwrite (bool)
    Output:  None
    Raises:  ValueError for inconsistent or non-finite data,
             FileExistsError if model files exist and overwrite is False,
             TypeError if config or metadata are not JSON-serializable
    """
    config = {} if config is None else config
    metadata = {} if metadata is None else metadata
    _validate_model(E, U, vocabulary)
    payloads = {
        "E.json": E, "U.json": U, "vocabulary.json": vocabulary,
        "config.json": config, "metadata.json": metadata,
    }
    texts = {name: json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=1)
             for name, obj in payloads.items()}
    if not overwrite:
        existing = [n for n in FILE_NAMES if os.path.exists(os.path.join(directory, n))]
        if existing:
            raise FileExistsError(
                f"{directory} already contains {existing}; pass overwrite=True")
    os.makedirs(directory, exist_ok=True)
    for name, text in texts.items():
        with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
            f.write(text)


def load_model(directory):
    """Load a model saved by save_model and check that it is consistent.

    Inputs:  directory (path)
    Output:  (E, U, vocabulary, config, metadata)
    Raises:  FileNotFoundError if a model file is missing,
             ValueError if a file is not valid JSON or the artifacts disagree
             (for example a vocabulary that does not match E)
    """
    loaded = {}
    for name in FILE_NAMES:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing model file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            try:
                loaded[name] = json.load(f)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path} is not valid JSON: {err}") from err
    E, U, vocabulary = loaded["E.json"], loaded["U.json"], loaded["vocabulary.json"]
    _validate_model(E, U, vocabulary)
    if not isinstance(loaded["config.json"], dict) or \
            not isinstance(loaded["metadata.json"], dict):
        raise ValueError("config.json and metadata.json must contain JSON objects")
    return E, U, vocabulary, loaded["config.json"], loaded["metadata.json"]
