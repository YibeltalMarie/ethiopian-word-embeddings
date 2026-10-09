"""Wikipedia markup cleaning, sentence splitting and tokenization.

Pipeline for one page:
    strip_markup -> split_sentences -> tokenize
and page_to_sentences() runs all three. Every rule below is justified by
measurements on the amwiki-20260901 dump (see docs and the experiment log).
Nothing here uses the model code; only the standard library.
"""
import functools
import html
import re
import unicodedata

# Link prefixes that mean "not running text" (files, categories, media).
NAMESPACE_PREFIXES = frozenset({"file", "image", "media", "category", "ፋይል", "መደብ"})
_INTERWIKI = re.compile(r"[a-z]{2,3}(-[a-z0-9]+)*")   # e.g. "en", "de", "zh-yue"

_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.DOTALL)
_REF_SELF_CLOSING = re.compile(r"<ref\b[^>]*/>", re.IGNORECASE)
_REF_BLOCK = re.compile(r"<ref\b[^>]*>.*?(?:</ref\s*>|\Z)", re.DOTALL | re.IGNORECASE)
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_EXTERNAL_LINK = re.compile(r"\[(?:https?|ftp)://[^\s\]]+(?:[ \t]+([^\]]*))?\]")
_BARE_URL = re.compile(r"(?:https?|ftp)://\S+")
_MAGIC_WORD = re.compile(r"__[A-Z]+__")
_QUOTES = re.compile(r"'{2,}")
_HEADING = re.compile(r"=+.*?=+")
_LIST_MARKERS = re.compile(r"^[*#:;]+")
_SENTENCE_END = re.compile(r"[።፧?!]+")
# Initialisms such as ዓ.ም, እ.ኤ.አ, ዶ.ር: single Ethiopic letters joined by periods.
_INITIALISM = re.compile(
    r"(?<![\u1200-\u135A])[\u1200-\u135A](?:\.[\u1200-\u135A])+(?![\u1200-\u135A])")


def _match_end(text, start, open_tok, close_tok):
    """Index just after the closer that matches the opener at `start`.

    Nesting is counted, so {{a|{{b}}|c}} is one unit. Returns -1 if the
    opener is never closed.
    """
    pattern = re.compile(re.escape(open_tok) + "|" + re.escape(close_tok))
    depth = 0
    pos = start
    while True:
        m = pattern.search(text, pos)
        if m is None:
            return -1
        depth += 1 if m.group() == open_tok else -1
        pos = m.end()
        if depth == 0:
            return pos


def remove_nested(text, open_tok, close_tok):
    """Delete every open_tok ... close_tok block, honouring nesting.

    Policy for an opener that is never closed: the block is assumed to run to
    the end of the text, so everything from that opener onward is dropped
    (an unclosed template or table is broken markup, and keeping its
    key=value fragments would add noise tokens). Stray closers are left in
    place; the tokenizer removes them as punctuation.
    """
    out = []
    pos = 0
    while True:
        i = text.find(open_tok, pos)
        if i == -1:
            out.append(text[pos:])
            break
        out.append(text[pos:i])
        end = _match_end(text, i, open_tok, close_tok)
        if end == -1:
            break
        pos = end
    return "".join(out)


def _is_dropped_link(target):
    if ":" not in target:
        return False
    prefix = target.split(":", 1)[0].strip().lower()
    return prefix in NAMESPACE_PREFIXES or bool(_INTERWIKI.fullmatch(prefix))


def _process_links(text):
    """[[target|label]] -> label; [[target]] -> target.

    File, category, media and interlanguage links are removed entirely
    (nested captions included). An unclosed [[ loses only its own marker.
    """
    out = []
    pos = 0
    while True:
        i = text.find("[[", pos)
        if i == -1:
            out.append(text[pos:])
            break
        out.append(text[pos:i])
        end = _match_end(text, i, "[[", "]]")
        if end == -1:
            pos = i + 2
            continue
        inner = text[i + 2:end - 2]
        target = inner.split("|", 1)[0]
        if _is_dropped_link(target):
            pass
        else:
            label = inner.rsplit("|", 1)[-1] if "|" in inner else inner
            if not label.strip():
                label = target
            out.append(_process_links(label))
        pos = end
    return "".join(out)


def strip_markup(text, title=""):
    """Turn raw wikitext into plain lines of prose, one segment per line.

    Steps, in order:
      1. HTML comments and <ref> citations removed
      2. {{PAGENAME}} replaced by the page title (it means exactly that, and
         many stub pages use it as the sentence subject)
      3. all other {{templates}} and {| tables |} removed (infobox and layout
         markup, not sentences)
      4. internal links: keep the visible label; file/category/interlanguage
         links removed
      5. external links: keep the label, drop the URL; bare URLs dropped
      6. remaining HTML tags removed (<br> becomes a line break), entities
         decoded, bold/italic quote marks removed, magic words like __TOC__
         removed
      7. per line: headings dropped, list markers (* # : ;) stripped, lines
         that still start with | or ! (table remnants) dropped

    Output: text with one segment per line (blank lines removed).
    """
    text = _COMMENT.sub("", text)
    text = _REF_SELF_CLOSING.sub("", text)
    text = _REF_BLOCK.sub("", text)
    text = text.replace("{{PAGENAME}}", title)
    text = remove_nested(text, "{{", "}}")
    text = remove_nested(text, "{|", "|}")
    text = _process_links(text)
    text = _EXTERNAL_LINK.sub(lambda m: m.group(1) or "", text)
    text = _BARE_URL.sub(" ", text)
    text = _BR.sub("\n", text)
    text = _TAG.sub("", text)
    text = html.unescape(text).replace("\xa0", " ")
    text = _QUOTES.sub("", text)
    text = _MAGIC_WORD.sub("", text)
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s or _HEADING.fullmatch(s) or s[0] in "|!":
            continue
        s = _LIST_MARKERS.sub("", s).strip()
        if s:
            lines.append(s)
    return "\n".join(lines)


def split_sentences(clean_text):
    """Split cleaned text into sentence strings.

    A line break always ends a segment (list items and lyric lines are not
    one sentence). Within a line, the Ethiopic full stop ።, the Ethiopic
    question mark ፧, and ASCII ? and ! end a sentence. The ASCII period does
    not (abbreviations and decimals). Terminators are dropped; empty pieces
    are skipped.
    """
    sentences = []
    for line in clean_text.split("\n"):
        for piece in _SENTENCE_END.split(line):
            if piece.strip():
                sentences.append(piece.strip())
    return sentences


@functools.lru_cache(maxsize=None)
def _is_separator(ch):
    cat = unicodedata.category(ch)
    return ch.isspace() or cat[0] in "PSZ" or cat in ("Cc", "Cf")


def tokenize(sentence):
    """Split a sentence into tokens.

    Separators: whitespace, every Unicode punctuation (P*) and symbol (S*)
    character (this includes the Ethiopic ፡ ፣ ፤ ፥ ፦ and ASCII hyphen, so
    ደቡብ-ምዕራብ becomes two tokens), and invisible control/format characters.
    Exception: initialisms made of single Ethiopic letters joined by periods
    (ዓ.ም, እ.ኤ.አ) are first joined into one token (ዓም, እኤአ), because
    splitting them produced meaningless single-letter tokens among the most
    frequent in the corpus.
    Kept inside tokens: letters, digits, Ethiopic numerals, combining marks.
    No lowercasing and no Unicode normalization (baseline; tested later as a
    controlled experiment).

    Output: list of non-empty strings.
    """
    sentence = _INITIALISM.sub(lambda m: m.group().replace(".", ""), sentence)
    spaced = "".join(" " if _is_separator(ch) else ch for ch in sentence)
    return spaced.split()


def page_to_sentences(text, title="", min_tokens=2):
    """Full pipeline for one page: list of token lists.

    Sentences shorter than min_tokens are dropped. The default 2 loses nothing
    useful: a one-token sentence produces no (center, context) pairs anyway.
    """
    if min_tokens < 1:
        raise ValueError(f"min_tokens must be >= 1, got {min_tokens}")
    result = []
    for sentence in split_sentences(strip_markup(text, title)):
        tokens = tokenize(sentence)
        if len(tokens) >= min_tokens:
            result.append(tokens)
    return result


def deduplicate_sentences(sentences):
    """Drop sentences identical to an earlier one; keep the first occurrence.

    Reason: templated pages repeat the same boilerplate hundreds of times
    (for example a calendar-converter note 366 times), which would overweight
    those word pairs. Cost: a genuinely repeated short sentence is also
    collapsed to one copy.

    Inputs:  sentences (list of token lists)
    Output:  new list in the original order; the input is not modified
    """
    seen = set()
    unique = []
    for sentence in sentences:
        key = tuple(sentence)
        if key not in seen:
            seen.add(key)
            unique.append(sentence)
    return unique
