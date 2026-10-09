"""Score error and gradients for the skip-gram model."""


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
