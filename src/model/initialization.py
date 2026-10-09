"""Reproducible initialization of the two parameter matrices."""
import random


def initialize_parameters(V, d, seed, scale=0.1):
    """Create E (V x d) and U (d x V) with small random values.

    Each entry is drawn from uniform(-scale, scale) using a private
    random.Random(seed) generator, so:
        - the same (V, d, seed, scale) always gives identical E and U
        - the global random state is never touched
    The fill order is fixed: all of E first (row by row), then all of U
    (row by row). Changing this order would change the matrices for a
    given seed, so it must stay the same.

    Inputs:  V (int >= 1 vocabulary size), d (int >= 1 embedding dimension),
             seed (int, record it with every run), scale (float > 0)
    Output:  (E, U) as lists of lists of floats
    Raises:  ValueError if V or d is not a positive integer, or scale <= 0
    """
    if not isinstance(V, int) or V < 1:
        raise ValueError(f"V must be a positive integer, got {V!r}")
    if not isinstance(d, int) or d < 1:
        raise ValueError(f"d must be a positive integer, got {d!r}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    rng = random.Random(seed)
    E = [[rng.uniform(-scale, scale) for _ in range(d)] for _ in range(V)]
    U = [[rng.uniform(-scale, scale) for _ in range(V)] for _ in range(d)]
    return E, U
