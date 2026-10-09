import json
import pytest
from src.data.quality import build_quality_report

BEFORE = [["ነው", "ቃል"], ["ነው", "ቃል"], ["ሀ", "ለ", "the", "1"], ["ነው", "שלום"]]
FINAL = [["ነው", "ቃል"], ["ሀ", "ለ", "the", "1"], ["ነው", "שלום"]]


def report(**kw):
    args = dict(sentences_before_dedup=BEFORE, sentences=FINAL, n_pages=5,
                pages_without_sentences=2, raw_characters=100)
    args.update(kw)
    return build_quality_report(**args)


def test_counts():
    r = report()
    assert r["documents"] == {"article_pages": 5, "pages_without_usable_sentences": 2,
                              "pages_with_usable_sentences": 3}
    assert r["sentences"]["before_deduplication"] == 4
    assert r["sentences"]["after_deduplication"] == 3
    assert r["sentences"]["exact_duplicates_removed"] == 1
    assert r["sentences"]["duplicate_percent"] == 25.0
    assert r["tokens"]["total"] == 8 and r["tokens"]["distinct"] == 7
    assert r["tokens"]["seen_once"] == 6 and r["tokens"]["seen_at_least_5"] == 0
    assert r["tokens"]["most_frequent_20"][0] == ["ነው", 2] or r["tokens"]["most_frequent_20"][0] == ("ነው", 2)


def test_sentence_length_stats():
    s = report()["sentence_length_tokens"]
    assert s["min"] == 2 and s["max"] == 4 and s["median"] == 2
    assert abs(s["mean"] - 8 / 3) < 0.01


def test_character_buckets_and_unusual_characters():
    r = report()
    c = r["characters_in_final_tokens"]
    assert c["ethiopic_letters"] == 2 + 2 + 1 + 1 + 1 + 1 and c["ascii_letters"] == 3
    assert c["ascii_digits"] == 1 and c["other"] == 4              # four Hebrew letters
    top = r["most_common_other_characters"]
    assert len(top) == 4 and top[0][1].startswith("U+05")


def test_discarded_text_and_json_serializable():
    r = report(raw_characters=200)
    kept = sum(r["characters_in_final_tokens"].values())
    assert r["discarded_text"]["characters_in_final_tokens"] == kept
    assert abs(r["discarded_text"]["retained_character_percent"] - 100 * kept / 200) < 0.01
    json.dumps(r, ensure_ascii=False)                               # must not raise


def test_bad_inputs_fail_clearly():
    with pytest.raises(ValueError):
        report(sentences=[])
    with pytest.raises(ValueError):
        report(raw_characters=0)
    with pytest.raises(ValueError):
        report(sentences=BEFORE + [["x", "y"]])
    with pytest.raises(ValueError):
        report(pages_without_sentences=6)
