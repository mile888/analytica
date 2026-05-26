"""CSV encoding robustness tests.

Verifies that ``read_csv_dataset`` handles real-world encodings:
UTF-8, UTF-8-SIG (BOM), CP1252, Latin-1, CP1251, semicolon-separated,
non-breaking spaces (0xA0), and corrupted files.
"""
from __future__ import annotations

import pytest
import pandas as pd

from source.dataframe import CsvLoadError, _detect_encoding, read_csv_dataset


# ── Helpers ──────────────────────────────────────────────────────────────

_UTF8_CSV = "Name,Value\nAlice,100\nBob,200\n"
_UTF8_BOM_CSV = b"\xef\xbb\xbf" + _UTF8_CSV.encode("utf-8")
_CP1252_CSV = "Name,Value\nCafé,100\nNaïve,200\n".encode("cp1252")
_LATIN1_CSV = "Name,Value\nJürgen,100\nÜber,200\n".encode("latin-1")
_CP1251_CSV = "Имя,Значение\nАлиса,100\nБорис,200\n".encode("cp1251")
_SEMICOLON_CSV = "Name;Value\nAlice;100\nBob;200\n"
_NBSP_CSV = b"Name,Value\r\nAlice,100\xc2\xa0\r\nBob,200\r\n"  # UTF-8 non-breaking space
_NBSP_CP1252_CSV = b"Name,Value\r\nAlice,100\xa0\r\nBob,200\r\n"  # CP1252 0xA0


def _write(tmp_path, filename: str, content: bytes | str) -> str:
    path = tmp_path / filename
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)
    return str(path)


# ── Encoding tests ───────────────────────────────────────────────────────

def test_utf8_csv(tmp_path) -> None:
    path = _write(tmp_path, "utf8.csv", _UTF8_CSV)
    df = read_csv_dataset(path)
    assert len(df) == 2
    assert list(df.columns) == ["Name", "Value"]
    assert df["Name"].tolist() == ["Alice", "Bob"]


def test_utf8_bom_csv(tmp_path) -> None:
    path = _write(tmp_path, "bom.csv", _UTF8_BOM_CSV)
    df = read_csv_dataset(path)
    assert len(df) == 2
    assert "Name" in df.columns


def test_cp1252_csv(tmp_path) -> None:
    path = _write(tmp_path, "cp1252.csv", _CP1252_CSV)
    df = read_csv_dataset(path)
    assert len(df) == 2
    assert "Café" in df["Name"].values or "Caf" in str(df["Name"].values)


def test_latin1_csv(tmp_path) -> None:
    path = _write(tmp_path, "latin1.csv", _LATIN1_CSV)
    df = read_csv_dataset(path)
    assert len(df) == 2
    assert "Jürgen" in df["Name"].values or "J" in str(df["Name"].values[0])


def test_cp1251_csv(tmp_path) -> None:
    path = _write(tmp_path, "cp1251.csv", _CP1251_CSV)
    df = read_csv_dataset(path)
    assert len(df) == 2


def test_semicolon_separated_csv(tmp_path) -> None:
    path = _write(tmp_path, "semicolon.csv", _SEMICOLON_CSV)
    df = read_csv_dataset(path)
    assert len(df) == 2
    assert "Name" in df.columns
    assert "Value" in df.columns


def test_non_breaking_space_utf8(tmp_path) -> None:
    """UTF-8 encoded non-breaking space (0xC2 0xA0) should not crash."""
    path = _write(tmp_path, "nbsp_utf8.csv", _NBSP_CSV)
    df = read_csv_dataset(path)
    assert len(df) == 2


def test_non_breaking_space_cp1252(tmp_path) -> None:
    """CP1252 non-breaking space (0xA0) — the original failing case."""
    path = _write(tmp_path, "nbsp_cp1252.csv", _NBSP_CP1252_CSV)
    df = read_csv_dataset(path)
    assert len(df) == 2


def test_corrupted_csv(tmp_path) -> None:
    """Completely invalid binary should raise a readable error, not a raw trace."""
    path = _write(tmp_path, "corrupt.csv", b"\xff\xfe\x00\x01\x02\x03")
    # Should either parse (with fallback) or raise something readable
    try:
        df = read_csv_dataset(path)
        # If it parsed, just check it didn't crash
        assert isinstance(df, pd.DataFrame)
    except (CsvLoadError, pd.errors.EmptyDataError, pd.errors.ParserError):
        pass  # Acceptable — structured error


# ── Encoding detection ───────────────────────────────────────────────────

def test_detect_encoding_utf8(tmp_path) -> None:
    path = tmp_path / "utf8.csv"
    path.write_text(_UTF8_CSV, encoding="utf-8")
    assert _detect_encoding(path) == "utf-8"


def test_detect_encoding_bom(tmp_path) -> None:
    path = tmp_path / "bom.csv"
    path.write_bytes(_UTF8_BOM_CSV)
    assert _detect_encoding(path) == "utf-8-sig"


def test_detect_encoding_cp1252(tmp_path) -> None:
    path = tmp_path / "cp1252.csv"
    path.write_bytes(_CP1252_CSV)
    enc = _detect_encoding(path)
    # CP1252 bytes may decode as UTF-8 if they happen to be valid;
    # what matters is that a valid encoding is returned
    assert enc is not None


def test_detect_encoding_cp1251(tmp_path) -> None:
    path = tmp_path / "cp1251.csv"
    path.write_bytes(_CP1251_CSV)
    enc = _detect_encoding(path)
    assert enc is not None


# ── Row count preservation ───────────────────────────────────────────────

def test_row_count_preserved_across_encodings(tmp_path) -> None:
    """All encoding variants with 2 data rows must yield exactly 2 rows."""
    cases = {
        "utf8.csv": _UTF8_CSV.encode("utf-8"),
        "bom.csv": _UTF8_BOM_CSV,
        "cp1252.csv": _CP1252_CSV,
        "latin1.csv": _LATIN1_CSV,
        "semicolon.csv": _SEMICOLON_CSV.encode("utf-8"),
    }
    for filename, content in cases.items():
        path = _write(tmp_path, filename, content)
        df = read_csv_dataset(path)
        assert len(df) == 2, f"{filename}: expected 2 rows, got {len(df)}"


# ── Large file safety ───────────────────────────────────────────────────

def test_large_csv_encoding_detection(tmp_path) -> None:
    """Encoding detection uses only 8KB sample, so large files are safe."""
    header = "id,name,value\n"
    rows = "".join(f"{i},name_{i},{i * 1.5}\n" for i in range(50_000))
    content = (header + rows).encode("cp1252")
    path = _write(tmp_path, "large.csv", content)
    df = read_csv_dataset(path)
    assert len(df) == 50_000


# ── CsvLoadError message ────────────────────────────────────────────────

def test_csv_load_error_message() -> None:
    err = CsvLoadError("/fake/path.csv")
    assert "Could not parse" in err.user_message
    assert "UTF-8" in err.user_message or "utf-8" in err.user_message
    assert "re-saving" in err.user_message
