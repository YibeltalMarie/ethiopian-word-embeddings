"""Cosine similarity between two vectors (pure Python)."""
import math


def cosine_similarity(a, b):
    """Cosine of the angle between vectors a and b.

        cosine(a, b) = sum_k(a[k] * b[k]) / (sqrt(sum_k a[k]^2) * sqrt(sum_k b[k]^2))

    Range: -1 (opposite directions) through 0 (right angle) to 1 (same
    direction). Only direction matters, so scaling a vector does not change it.

    Zero-norm policy: if either vector is all zeros the cosine is mathematically
    undefined, so this function returns 0.0 ("no measurable similarity") instead
    of dividing by zero.

    Rounding: the result is clamped to [-1, 1], because floating-point rounding
    can produce values like 1.0000000000000002 for identical directions.

    Inputs:  a, b (lists of numbers with the same, non-zero length)
    Output:  float in [-1, 1]
    Raises:  ValueError if the lengths differ or the vectors are empty
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: len(a)={len(a)} but len(b)={len(b)}")
    if len(a) == 0:
        raise ValueError("vectors are empty")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))
