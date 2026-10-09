"""One-pair loss, written in the numerically stable form."""
import math


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
