import pytest
from src.data.corpus import (sha256_file, verify_checksum, build_provenance_record,
                             save_provenance, load_provenance, validate_record)

ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def good(**overrides):
    fields = dict(source_id="amwiki-20260901", source_name="Amharic Wikipedia dump",
                  url="https://example.org/dump", language="Amharic (am)",
                  license="CC BY-SA (to verify)", access_date="2026-10-09",
                  original_format="xml.bz2", original_size=1234, sha256=ABC,
                  description="Test record", processing_steps=["none yet"],
                  notes="")
    fields.update(overrides)
    return build_provenance_record(**fields)


def test_known_checksums(tmp_path):
    (tmp_path / "abc.txt").write_bytes(b"abc")
    (tmp_path / "empty.txt").write_bytes(b"")
    assert sha256_file(tmp_path / "abc.txt") == ABC
    assert sha256_file(tmp_path / "empty.txt") == EMPTY


def test_chunking_does_not_change_the_result(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(bytes(range(256)) * 1000)
    assert sha256_file(p, chunk_size=1) == sha256_file(p, chunk_size=1 << 20)


def test_one_changed_byte_changes_the_checksum(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"abc")
    q = tmp_path / "g.txt"
    q.write_bytes(b"abd")
    assert sha256_file(p) != sha256_file(q)


def test_verify_checksum(tmp_path):
    p = tmp_path / "abc.txt"
    p.write_bytes(b"abc")
    assert verify_checksum(p, ABC) and verify_checksum(p, ABC.upper())
    assert not verify_checksum(p, EMPTY)


def test_checksum_bad_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing")
    (tmp_path / "x").write_bytes(b"x")
    with pytest.raises(ValueError):
        sha256_file(tmp_path / "x", chunk_size=0)


def test_record_has_all_plan_fields():
    record = good()
    for key in ("source_id", "source_name", "URL", "language", "license",
                "access_date", "original_format", "original_size", "description",
                "processing_steps", "notes", "sha256"):
        assert key in record


def test_record_round_trip_with_amharic_text(tmp_path):
    record = good(description="የአማርኛ ዊኪፔዲያ", notes="ማስታወሻ")
    path = tmp_path / "meta" / "prov.json"
    save_provenance(path, record)
    assert load_provenance(path) == record
    assert "የአማርኛ" in path.read_text(encoding="utf-8")      # real UTF-8, no \u escapes


def test_save_refuses_overwrite(tmp_path):
    path = tmp_path / "p.json"
    save_provenance(path, good())
    with pytest.raises(FileExistsError):
        save_provenance(path, good())
    save_provenance(path, good(notes="changed"), overwrite=True)
    assert load_provenance(path)["notes"] == "changed"


def test_bad_records_are_rejected(tmp_path):
    bad = [dict(source_id=""), dict(language="  "), dict(access_date="20261009"),
           dict(access_date="2026-13-40"), dict(original_size=-1),
           dict(original_size=True), dict(original_size="12"),
           dict(sha256="abc"), dict(sha256=ABC.upper()),
           dict(processing_steps="clean"), dict(processing_steps=[1]),
           dict(notes=None)]
    for override in bad:
        with pytest.raises(ValueError):
            good(**override)
    with pytest.raises(ValueError):
        validate_record({"source_id": "x"})
    with pytest.raises(ValueError):
        validate_record("not a dict")
    assert not (tmp_path / "p.json").exists()


def test_steps_list_is_copied():
    steps = ["a"]
    record = good(processing_steps=steps)
    steps.append("b")
    assert record["processing_steps"] == ["a"]


def test_load_bad_files(tmp_path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_provenance(tmp_path / "bad.json")
    (tmp_path / "partial.json").write_text('{"source_id": "x"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_provenance(tmp_path / "partial.json")
    with pytest.raises(FileNotFoundError):
        load_provenance(tmp_path / "missing.json")
