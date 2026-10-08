"""Skip-gram model core (pure Python: lists, loops, math, random).

Row-vector notation:
    E has shape V x d   (one row per word, used for the center word)
    U has shape d x V   (one column per word, used for the context word)
"""
import math


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


def compute_probabilities(s):
    """Numerically stable softmax.

    Inputs:  s (list of V scores)
    Output:  (m, a, p) where
                 m = max(s)
                 a[j] = exp(s[j] - m)
                 p[j] = a[j] / sum(a)
             m and a are returned because the stable loss reuses them.
    Raises:  ValueError if s is empty
    """
    if not s:
        raise ValueError("scores are empty")
    m = max(s)
    a = [math.exp(x - m) for x in s]
    total = sum(a)
    p = [x / total for x in a]
    return m, a, p


def compute_loss(s, t, m, a):
    """Stable one-pair loss: L = (m - s[t]) + ln(sum(a)).

    Inputs:  s (scores), t (int target word ID),
             m and a exactly as returned by compute_probabilities(s)
    Output:  float loss, never ln(0), because sum(a) >= 1
    Raises:  IndexError if t is out of range
    """
    if not 0 <= t < len(s):
        raise IndexError(f"target ID {t} out of range for V={len(s)}")
    return (m - s[t]) + math.log(sum(a))


def compute_score_error(p, t):
    """Score error e = p - y, where y is one-hot at the target t.

    Inputs:  p (list of V probabilities), t (int target word ID)
    Output:  e (list of V), e[t] = p[t] - 1, e[j] = p[j] otherwise.
             e always sums to 0 (up to rounding).
    Raises:  IndexError if t is out of range
    """
    if not 0 <= t < len(p):
        raise IndexError(f"target ID {t} out of range for V={len(p)}")
    e = list(p)
    e[t] -= 1
    return e


def compute_gradients(h, U, p, t):
    """Compute both gradients from the CURRENT (old) h and U.

    This function only reads its inputs and never modifies h or U, so both
    gradients are always based on the old values. Updating happens elsewhere.

    Inputs:  h (length d), U (d x V), p (length V), t (int target ID)
    Output:  (grad_U, grad_h)
                 grad_U[k][j] = h[k] * e[j]            shape d x V
                 grad_h[k]    = sum_j U[k][j] * e[j]   length d
    Raises:  ValueError on shape mismatch, IndexError if t is out of range
    """
    d = len(U)
    if d == 0:
        raise ValueError("U is empty")
    V = len(U[0])
    if len(h) != d:
        raise ValueError(f"shape mismatch: len(h)={len(h)} but U has {d} rows")
    if len(p) != V:
        raise ValueError(f"shape mismatch: len(p)={len(p)} but U has {V} columns")
    e = compute_score_error(p, t)
    grad_U = [[h[k] * e[j] for j in range(V)] for k in range(d)]
    grad_h = [sum(U[k][j] * e[j] for j in range(V)) for k in range(d)]
    return grad_U, grad_h
