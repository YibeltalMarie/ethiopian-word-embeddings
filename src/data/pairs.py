"""Center-context pair generation for skip-gram training."""


def generate_pairs(sentences, window):
    """Generate (center, context) pairs, one sentence at a time.

    Rules:
        - Direction: each pair is (center, context). Every token in a sentence
          takes a turn as the center, so pairs exist in both directions
          (a is a context of b, and b is a context of a).
        - Window: context positions q satisfy 1 <= abs(q - c) <= window,
          where c is the center position.
        - Edges: positions outside the sentence are skipped. Nothing is padded.
        - No cross-sentence pairs: the window is only checked inside the
          current sentence.
        - Repeated tokens: each occurrence creates its own pair, so duplicate
          pairs are kept (they reflect how often words co-occur). A token is
          never paired with its own position, but a repeated word can be paired
          with another occurrence of itself.
        - Order: sentences in input order, centers left to right, contexts
          left to right.
        - A one-token or empty sentence produces no pairs.

    Inputs:  sentences (list of lists of tokens; tokens can be words or IDs),
             window (int >= 1)
    Output:  list of (center, context) tuples
    Raises:  ValueError if window < 1
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    pairs = []
    for sentence in sentences:
        n = len(sentence)
        for c in range(n):
            start = max(0, c - window)
            end = min(n - 1, c + window)
            for q in range(start, end + 1):
                if q != c:
                    pairs.append((sentence[c], sentence[q]))
    return pairs
