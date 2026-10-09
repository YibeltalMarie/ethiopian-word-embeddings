"""Corpus quality report (plan section 10): plain numbers, JSON-serializable."""
import statistics
from collections import Counter


def _bucket(ch):
    cp = ord(ch)
    if 0x1200 <= cp <= 0x135A:
        return "ethiopic_letters"
    if 0x1369 <= cp <= 0x137C:
        return "ethiopic_numerals"
    if ch.isascii() and ch.isalpha():
        return "ascii_letters"
    if ch.isascii() and ch.isdigit():
        return "ascii_digits"
    return "other"


def build_quality_report(sentences_before_dedup, sentences, n_pages,
                         pages_without_sentences, raw_characters):
    """Measure the final corpus and what preprocessing removed.

    Inputs:
        sentences_before_dedup  token sentences after cleaning, before duplicate removal
        sentences               token sentences after duplicate removal (the final corpus)
        n_pages                 article pages read from the dump
        pages_without_sentences pages that produced no usable sentence
        raw_characters          total characters of raw wikitext over those pages
    Output:  dict (all values are ints, floats, strings, lists or dicts)
    Raises:  ValueError for an empty corpus, raw_characters < 1, or inconsistent counts

    retained_character_percent compares the letters/digits kept inside final
    tokens with the raw text; markup, whitespace, punctuation and removed
    duplicates all count as discarded.
    """
    if not sentences:
        raise ValueError("final corpus has no sentences")
    if raw_characters < 1:
        raise ValueError("raw_characters must be >= 1")
    if len(sentences) > len(sentences_before_dedup):
        raise ValueError("final corpus is larger than the pre-deduplication corpus")
    if not 0 <= pages_without_sentences <= n_pages:
        raise ValueError("pages_without_sentences must be between 0 and n_pages")

    freq = Counter(tok for s in sentences for tok in s)
    total_tokens = sum(freq.values())
    lengths = sorted(len(s) for s in sentences)
    chars = Counter()
    other = Counter()
    for tok, n in freq.items():
        for ch in tok:
            b = _bucket(ch)
            chars[b] += n
            if b == "other":
                other[ch] += n
    kept_characters = sum(chars.values())
    removed = len(sentences_before_dedup) - len(sentences)
    return {
        "documents": {
            "article_pages": n_pages,
            "pages_without_usable_sentences": pages_without_sentences,
            "pages_with_usable_sentences": n_pages - pages_without_sentences,
        },
        "sentences": {
            "before_deduplication": len(sentences_before_dedup),
            "after_deduplication": len(sentences),
            "exact_duplicates_removed": removed,
            "duplicate_percent": round(100 * removed / len(sentences_before_dedup), 2),
        },
        "tokens": {
            "total": total_tokens,
            "distinct": len(freq),
            "seen_once": sum(1 for n in freq.values() if n == 1),
            "seen_at_least_5": sum(1 for n in freq.values() if n >= 5),
            "seen_at_least_10": sum(1 for n in freq.values() if n >= 10),
            "seen_at_least_20": sum(1 for n in freq.values() if n >= 20),
            "most_frequent_20": freq.most_common(20),
        },
        "sentence_length_tokens": {
            "min": lengths[0],
            "mean": round(total_tokens / len(sentences), 2),
            "median": statistics.median(lengths),
            "p95": lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))],
            "max": lengths[-1],
        },
        "characters_in_final_tokens": dict(chars),
        "most_common_other_characters": [
            [ch, f"U+{ord(ch):04X}", n] for ch, n in other.most_common(15)],
        "discarded_text": {
            "raw_characters": raw_characters,
            "characters_in_final_tokens": kept_characters,
            "retained_character_percent": round(100 * kept_characters / raw_characters, 2),
        },
    }
