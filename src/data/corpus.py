"""Corpus provenance helpers: file checksums and provenance records."""
import datetime
import hashlib
import json
import os

REQUIRED_FIELDS = (
    "source_id", "source_name", "URL", "language", "license", "access_date",
    "original_format", "original_size", "description", "processing_steps",
    "notes",
)
TEXT_FIELDS = ("source_id", "source_name", "URL", "language", "license",
               "original_format", "description")


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA-256 fingerprint of a file as 64 lowercase hex characters.

    The file is read in chunks, so large files do not need to fit in memory.

    Raises:  FileNotFoundError if the file is missing, ValueError if chunk_size < 1
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path, expected):
    """True if the file's SHA-256 equals `expected` (case-insensitive)."""
    return sha256_file(path) == str(expected).strip().lower()


def validate_record(record):
    """Raise ValueError unless `record` is a complete, well-formed record.

    Required: every field in REQUIRED_FIELDS plus "sha256".
    Text fields are non-empty strings, access_date is ISO YYYY-MM-DD,
    original_size is an int >= 0, processing_steps is a list of strings,
    notes is a string (may be empty), sha256 is 64 lowercase hex characters.
    """
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")
    missing = [k for k in REQUIRED_FIELDS + ("sha256",) if k not in record]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    for key in TEXT_FIELDS:
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if not isinstance(record["notes"], str):
        raise ValueError("notes must be a string")
    date = record["access_date"]
    try:
        ok = isinstance(date, str) and datetime.date.fromisoformat(date).isoformat() == date
    except ValueError:
        ok = False
    if not ok:
        raise ValueError(f"access_date must be YYYY-MM-DD, got {date!r}")
    size = record["original_size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"original_size must be an int >= 0, got {size!r}")
    steps = record["processing_steps"]
    if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
        raise ValueError("processing_steps must be a list of strings")
    digest = record["sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("sha256 must be 64 lowercase hex characters")


def build_provenance_record(source_id, source_name, url, language, license,
                            access_date, original_format, original_size,
                            sha256, description, processing_steps=None,
                            notes=""):
    """Build a provenance record with the plan's fields plus the checksum.

    Output:  dict (validated before it is returned)
    Raises:  ValueError if any field is missing or malformed
    """
    if processing_steps is None:
        steps = []
    elif isinstance(processing_steps, (str, bytes)):
        raise ValueError("processing_steps must be a list of strings, not a string")
    else:
        steps = list(processing_steps)
    record = {
        "source_id": source_id,
        "source_name": source_name,
        "URL": url,
        "language": language,
        "license": license,
        "access_date": access_date,
        "original_format": original_format,
        "original_size": original_size,
        "sha256": sha256,
        "description": description,
        "processing_steps": steps,
        "notes": notes,
    }
    validate_record(record)
    return record


def save_provenance(path, record, overwrite=False):
    """Validate and write a record as UTF-8 JSON. Never overwrites by default.

    Raises:  ValueError for a bad record, FileExistsError if the file exists
             and overwrite is False
    """
    validate_record(record)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True")
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def load_provenance(path):
    """Load and validate a provenance record.

    Raises:  FileNotFoundError, or ValueError for invalid JSON or a bad record
    """
    with open(path, "r", encoding="utf-8") as f:
        try:
            record = json.load(f)
        except json.JSONDecodeError as err:
            raise ValueError(f"{path} is not valid JSON: {err}") from err
    validate_record(record)
    return record
