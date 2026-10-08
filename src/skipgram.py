"""Skip-gram model core (pure Python: lists, loops, math, random).

Row-vector notation:
    E has shape V x d   (one row per word, used for the center word)
    U has shape d x V   (one column per word, used for the context word)
"""


def get_embedding(E, i):
    """Return h = E[i], a list of length d.

    Inputs:  E (V x d list of lists), i (int center word ID)
    Output:  a copy of row i, so later changes to h never alter E
    Raises:  ValueError if E is empty, IndexError if i is out of range
    """
    if not E:
        raise ValueError("E is empty")
    if not 0 <= i < len(E):
        raise IndexError(f"center ID {i} out of range for V={len(E)}")
    return list(E[i])


def compute_scores(h, U):
    """Return s = hU, a list of length V.

    Inputs:  h (length d), U (d x V list of lists)
    Output:  s[j] = sum over k of h[k] * U[k][j]
    Raises:  ValueError if len(h) != len(U) (the inner dimensions must match)
    """
    d = len(U)
    if d == 0:
        raise ValueError("U is empty")
    if len(h) != d:
        raise ValueError(f"shape mismatch: len(h)={len(h)} but U has {d} rows")
    V = len(U[0])
    return [sum(h[k] * U[k][j] for k in range(d)) for j in range(V)]


def forward(E, U, i):
    """Run lookup then scores. Returns (h, s)."""
    h = get_embedding(E, i)
    s = compute_scores(h, U)
    return h, s
