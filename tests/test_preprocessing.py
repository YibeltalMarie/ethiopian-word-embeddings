import pytest
from src.data.preprocessing import (remove_nested, strip_markup, split_sentences,
                                    tokenize, page_to_sentences,
                                    deduplicate_sentences)


# ---------- remove_nested ----------

def test_remove_nested_simple_and_nested():
    assert remove_nested("a {{x}} b", "{{", "}}") == "a  b"
    assert remove_nested("a {{x|{{y}}|z}} b", "{{", "}}") == "a  b"
    assert remove_nested("{{a}}{{b}}c", "{{", "}}") == "c"


def test_remove_nested_unclosed_drops_the_rest():
    assert remove_nested("keep {{broken| x y", "{{", "}}") == "keep "


def test_remove_nested_stray_closer_is_left_alone():
    assert remove_nested("a }} b", "{{", "}}") == "a }} b"


def test_remove_nested_tables():
    assert remove_nested("a {| x {| y |} z |} b", "{|", "|}") == "a  b"


# ---------- strip_markup: real patterns seen in the dump ----------

def test_pagename_is_replaced_by_title_before_templates_are_removed():
    text = "{{የቦታ መረጃ\n | ስም = {{PAGENAME}}\n}}\n'''{{PAGENAME}}''' በ[[ኦሮሚያ ክልል]] የሚገኝ [[ወረዳ]] ነው።"
    assert strip_markup(text, "ጦሌ") == "ጦሌ በኦሮሚያ ክልል የሚገኝ ወረዳ ነው።"


def test_piped_link_keeps_label_and_plain_link_keeps_target():
    assert strip_markup("ዘፈኑ [[ኩባ|በኩባ]] እና [[ቬት ናም|በቬትናም]]") == "ዘፈኑ በኩባ እና በቬትናም"
    assert strip_markup("የ[[አማርኛ]] ምሳሌ") == "የአማርኛ ምሳሌ"


def test_category_and_file_links_are_removed_including_nested_captions():
    text = ("ጽሑፍ ነው።\n[[መደብ: ተረትና ምሳሌ]]\n[[Category:Foo]]\n"
            "[[File:a.jpg|thumb|ስዕል [[ኩባ]] መግለጫ]]\n[[ፋይል:b.png|x]]")
    assert strip_markup(text) == "ጽሑፍ ነው።"


def test_interlanguage_links_are_removed_but_normal_colon_titles_stay():
    assert strip_markup("ቃል [[en:Austria]] [[zh-yue:Foo]] ቃል").split() == ["ቃል", "ቃል"]
    assert strip_markup("[[ሰዓት: ሰዓት]]") == "ሰዓት: ሰዓት"   # Amharic prefix is kept, not treated as interwiki


def test_headings_dropped_and_list_markers_stripped():
    text = "== ትርጉሙ ==\n=== ንዑስ ===\n: ፋኖ ተሰማራ፤\n* ንጥል አንድ\n## ንጥል ሁለት"
    assert strip_markup(text) == "ፋኖ ተሰማራ፤\nንጥል አንድ\nንጥል ሁለት"


def test_bold_italic_comments_refs_and_tags():
    text = ("'''ደማቅ''' እና ''ሰያፍ'' ቃል<ref name=a>ምንጭ</ref> <ref name=b /> "
            "<!-- አስተያየት --> <b>ጽሑፍ</b>")
    assert strip_markup(text).split() == ["ደማቅ", "እና", "ሰያፍ", "ቃል", "ጽሑፍ"]


def test_br_becomes_a_line_break():
    assert strip_markup("አንድ<br />ሁለት<br>ሦስት") == "አንድ\nሁለት\nሦስት"


def test_external_links_keep_label_only():
    assert strip_markup("[http://x.org/a መጽሐፍ ስም] እና [https://y.org] እና http://z.org/q ቃል") \
        .split() == ["መጽሐፍ", "ስም", "እና", "እና", "ቃል"]


def test_tables_templates_and_remnants_removed():
    text = ("{{አሞሌ ቻርት\n| title = {{PAGENAME}}\n}}\n"
            "{| class=wikitable\n|-\n| a || b\n|}\nእውነተኛ ዓረፍተ ነገር።\n| remnant\n! remnant2")
    assert strip_markup(text) == "እውነተኛ ዓረፍተ ነገር።"


def test_entities_decoded_and_magic_words_removed():
    assert strip_markup("ሀ &amp; ለ&nbsp;መ __TOC__") == "ሀ & ለ መ"


def test_unclosed_link_loses_only_its_marker():
    assert strip_markup("ቃል [[ተሰበረ እና ጽሑፍ ቀጠለ።") == "ቃል ተሰበረ እና ጽሑፍ ቀጠለ።"


def test_unclosed_template_drops_the_rest_and_empty_input_is_empty():
    assert strip_markup("ጽሑፍ።\n{{የተሰበረ | a = b") == "ጽሑፍ።"
    assert strip_markup("") == "" and strip_markup("{{a}}") == ""


def test_main_page_style_markup_leaves_no_prose():
    text = ('<templatestyles src="x.css" />\n{{እንኳን ደህና መጡ}}\n'
            '{| id="mp-upper" style="width:100%"\n| style="a" |\n|}')
    assert strip_markup(text) == ""


# ---------- sentences and tokens ----------

def test_split_on_ethiopic_full_stop_question_and_newline():
    assert split_sentences("አንድ ነው። ሁለት ነው፧ ሦስት!\nአራት") == [
        "አንድ ነው", "ሁለት ነው", "ሦስት", "አራት"]
    assert split_sentences("ዋጋ 3.5 ብር ነው።") == ["ዋጋ 3.5 ብር ነው"]   # ASCII . is not a stop
    assert split_sentences("።።\n  \n") == []


def test_tokenize_separators_and_kept_characters():
    assert tokenize("ነው፡ ቃል፣ ቃል፤ (ሌላ) 'ቃል'") == ["ነው", "ቃል", "ቃል", "ሌላ", "ቃል"]
    assert tokenize("ደቡብ-ምዕራብ") == ["ደቡብ", "ምዕራብ"]
    assert tokenize("በ1960ዎቹ ESM ፲፱፻") == ["በ1960ዎቹ", "ESM", "፲፱፻"]
    assert tokenize("a\u200bb  c\t d") == ["a", "b", "c", "d"]
    assert tokenize("") == [] and tokenize("፡ ። ,") == []


def test_homophone_letters_are_not_merged():
    assert tokenize("ሀገር ሐገር") == ["ሀገር", "ሐገር"]


# ---------- full pipeline ----------

def test_page_to_sentences_end_to_end():
    text = ("{{info|x=1}}\n'''{{PAGENAME}}''' የኢትዮጵያ [[ዘፈን]] ነው። ተወዳጅ ነበር።\n"
            "== ግጥሞች ==\n: ፋኖ ተሰማራ፤\n: ፋኖ\n[[መደብ: ዘፈን]]")
    assert page_to_sentences(text, "ፋኖ ተሰማራ") == [
        ["ፋኖ", "ተሰማራ", "የኢትዮጵያ", "ዘፈን", "ነው"],
        ["ተወዳጅ", "ነበር"],
        ["ፋኖ", "ተሰማራ"]]                                  # one-token line "ፋኖ" dropped


def test_min_tokens_and_bad_argument():
    assert page_to_sentences("አንድ። ሁለት ሦስት።", min_tokens=1) == [["አንድ"], ["ሁለት", "ሦስት"]]
    with pytest.raises(ValueError):
        page_to_sentences("አንድ", min_tokens=0)


def test_sentences_feed_generate_pairs_without_cross_sentence_pairs():
    from src.data.pairs import generate_pairs
    sents = page_to_sentences("ሀ ለ። መ ሠ።")
    assert generate_pairs(sents, 5) == [("ሀ", "ለ"), ("ለ", "ሀ"), ("መ", "ሠ"), ("ሠ", "መ")]


# ---------- initialisms and duplicates ----------

def test_initialisms_are_joined_into_one_token():
    assert tokenize("ከ1892 ዓ.ም አስቀድሞ") == ["ከ1892", "ዓም", "አስቀድሞ"]
    assert tokenize("በ2010 እ.ኤ.አ. ተወለደ") == ["በ2010", "እኤአ", "ተወለደ"]
    assert tokenize("ዶ.ር ተስፋዬ") == ["ዶር", "ተስፋዬ"]


def test_initialism_rule_does_not_merge_ordinary_words():
    assert tokenize("ቃል.ቃል") == ["ቃል", "ቃል"]          # multi-letter pieces stay split
    assert tokenize("ዓ.ምሕረት") == ["ዓ", "ምሕረት"]       # not a complete initialism
    assert tokenize("ነው. ሀ ለ") == ["ነው", "ሀ", "ለ"]     # ordinary period + space


def test_deduplicate_keeps_first_occurrence_and_order():
    s = [["a", "b"], ["c", "d"], ["a", "b"], ["e", "f"], ["c", "d"]]
    s_before = [list(x) for x in s]
    assert deduplicate_sentences(s) == [["a", "b"], ["c", "d"], ["e", "f"]]
    assert s == s_before                                  # input not modified
    assert deduplicate_sentences([]) == []
    assert deduplicate_sentences([["a", "b"], ["b", "a"]]) == [["a", "b"], ["b", "a"]]
