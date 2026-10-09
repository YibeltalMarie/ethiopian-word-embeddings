import bz2
import pytest
from src.data.corpus import iter_wiki_pages

NS = 'xmlns="http://www.mediawiki.org/xml/export-0.11/"'


def page(title, ns, text, redirect=False):
    r = '<redirect title="x" />' if redirect else ""
    t = f"<text>{text}</text>" if text is not None else "<text />"
    return (f"<page><title>{title}</title><ns>{ns}</ns>{r}"
            f"<revision>{t}</revision></page>")


def dump(*pages):
    return f'<mediawiki {NS}><siteinfo><sitename>x</sitename></siteinfo>{"".join(pages)}</mediawiki>'


def write(tmp_path, xml, compressed=True):
    p = tmp_path / ("d.xml.bz2" if compressed else "d.xml")
    data = xml.encode("utf-8")
    p.write_bytes(bz2.compress(data) if compressed else data)
    return p


def test_keeps_only_real_articles(tmp_path):
    xml = dump(page("አንድ", 0, "የመጀመሪያ ጽሑፍ።"),
               page("Talk:አንድ", 1, "talk page"),
               page("ወደ", 0, "#REDIRECT [[አንድ]]", redirect=True),
               page("ባዶ", 0, None),
               page("ሁለት", 0, "ሁለተኛ ጽሑፍ።"))
    assert list(iter_wiki_pages(write(tmp_path, xml))) == [
        ("አንድ", "የመጀመሪያ ጽሑፍ።"), ("ሁለት", "ሁለተኛ ጽሑፍ።")]


def test_plain_and_compressed_files_agree(tmp_path):
    xml = dump(page("a", 0, "text one"), page("b", 0, "text two"))
    assert (list(iter_wiki_pages(write(tmp_path, xml, True)))
            == list(iter_wiki_pages(write(tmp_path, xml, False))))


def test_entities_are_decoded(tmp_path):
    xml = dump(page("a", 0, "x &amp; y &lt;b&gt;"))
    assert list(iter_wiki_pages(write(tmp_path, xml))) == [("a", "x & y <b>")]


def test_order_is_dump_order(tmp_path):
    xml = dump(*[page(f"t{i}", 0, f"text {i}") for i in range(5)])
    assert [t for t, _ in iter_wiki_pages(write(tmp_path, xml))] == [f"t{i}" for i in range(5)]


def test_empty_dump_yields_nothing(tmp_path):
    assert list(iter_wiki_pages(write(tmp_path, dump()))) == []


def test_missing_file_and_bad_xml(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_wiki_pages(tmp_path / "missing.xml.bz2"))
    bad = tmp_path / "bad.xml"
    bad.write_bytes(b"<mediawiki><page><title>x")
    with pytest.raises(ValueError):
        list(iter_wiki_pages(bad))
