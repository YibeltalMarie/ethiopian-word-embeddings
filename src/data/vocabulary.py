"""Deterministic vocabulary construction and application."""
from collections import Counter


def has_ethiopic_letter(token):
    """True if the token contains at least one Ethiopic letter (U+1200 to U+135A)."""
    return any("\u1200" <= ch <= "\u135A" for ch in token)


def build_vocabulary(sentences, min_freq=5, max_size=None, require_ethiopic=True):
    """Build a vocabulary from tokenized sentences.

    Rules:
        - Count every token over all sentences.
        - Keep tokens seen at least min_freq times.
        - If require_ethiopic, keep only tokens containing an Ethiopic letter
          (drops English fragments, filenames and digit-only tokens).
        - Order: highest frequency first; ties broken by the token itself
          (Unicode order), so the result is identical on every run.
        - If max_size is given, keep only the first max_size tokens.
        - No special tokens (no <UNK>) are added.

    Inputs:  sentences (list of token lists), min_freq (int >= 1),
             max_size (None or int >= 1), require_ethiopic (bool)
    Output:  (vocabulary, counts): the word list (position = ID) and the
             matching list of corpus counts
    Raises:  ValueError for bad min_freq or max_size
    """
    if not isinstance(min_freq, int) or min_freq < 1:
        raise ValueError(f"min_freq must be an integer >= 1, got {min_freq!r}")
    if max_size is not None and (not isinstance(max_size, int) or max_size < 1):
        raise ValueError(f"max_size must be None or an integer >= 1, got {max_size!r}")
    freq = Counter(tok for sentence in sentences for tok in sentence)
    kept = [(tok, n) for tok, n in freq.items()
            if n >= min_freq and (not require_ethiopic or has_ethiopic_letter(tok))]
    kept.sort(key=lambda item: (-item[1], item[0]))
    if max_size is not None:
        kept = kept[:max_size]
    return [tok for tok, _ in kept], [n for _, n in kept]


def apply_vocabulary(sentences, vocabulary, min_tokens=2):
    """Convert token sentences to ID sentences, deleting out-of-vocabulary tokens.

    Out-of-vocabulary policy: such tokens are deleted from the sentence, so the
    tokens around them become adjacent (the original word2vec behaviour).
    Sentences left with fewer than min_tokens IDs are dropped, since they
    cannot produce pairs. ID = position in `vocabulary`.

    Inputs:  sentences (list of token lists), vocabulary (list of unique words)
    Output:  (id_sentences, stats) where stats is a dict with
             tokens_in, tokens_kept, sentences_in, sentences_kept,
             words_unused (vocabulary words that never occur in the output)
    Raises:  ValueError if the vocabulary has duplicates or min_tokens < 1
    """
    if min_tokens < 1:
        raise ValueError(f"min_tokens must be >= 1, got {min_tokens}")
    word_to_id = {w: i for i, w in enumerate(vocabulary)}
    if len(word_to_id) != len(vocabulary):
        raise ValueError("vocabulary contains duplicate words")
    out = []
    tokens_in = tokens_kept = 0
    used = set()
    for sentence in sentences:
        tokens_in += len(sentence)
        ids = [word_to_id[t] for t in sentence if t in word_to_id]
        tokens_kept += len(ids)
        if len(ids) >= min_tokens:
            out.append(ids)
            used.update(ids)
    stats = {"tokens_in": tokens_in, "tokens_kept": tokens_kept,
             "sentences_in": len(sentences), "sentences_kept": len(out),
             "words_unused": len(vocabulary) - len(used)}
    return out, stats
