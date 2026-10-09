"""Nearest neighbors by cosine similarity over the rows of E."""
from src.evaluation.cosine import cosine_similarity


def nearest_neighbors(word, E, vocabulary, k=3):
    """Return the k words whose vectors in E are most similar to `word`.

    Rules:
        - Vectors come from E (the final embeddings), never from U.
        - vocabulary[i] is the word whose vector is E[i] (position = ID).
        - The query word itself is excluded.
        - Results are sorted by cosine similarity, highest first. Ties are
          broken by word ID (lower ID first), so output is deterministic.
        - Unknown word: returns an empty list instead of raising.
        - Zero-norm vectors: they score 0.0 against everything (see
          cosine_similarity), so nothing crashes.
        - If fewer than k other words exist, all of them are returned.

    Inputs:  word (str), E (V x d list of lists), vocabulary (list of V words),
             k (int >= 1)
    Output:  list of (neighbor word, cosine score), at most k long
    Raises:  ValueError if k < 1 or len(vocabulary) != len(E)
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if len(vocabulary) != len(E):
        raise ValueError(
            f"vocabulary has {len(vocabulary)} words but E has {len(E)} rows")
    if word not in vocabulary:
        return []
    query_id = vocabulary.index(word)
    query_vector = E[query_id]
    scored = [(j, cosine_similarity(query_vector, E[j]))
              for j in range(len(E)) if j != query_id]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [(vocabulary[j], score) for j, score in scored[:k]]


def print_neighbors(words, E, vocabulary, k=3):
    """Print the k nearest neighbors of each word in `words`.

    Unknown words print a short message instead of crashing.
    """
    for word in words:
        neighbors = nearest_neighbors(word, E, vocabulary, k)
        if not neighbors:
            if word not in vocabulary:
                print(f"{word}: unknown word (not in vocabulary)")
            else:
                print(f"{word}: no other words to compare")
            continue
        formatted = ", ".join(f"{w} ({score:.3f})" for w, score in neighbors)
        print(f"{word}: {formatted}")
