from __future__ import annotations

import pytest
import pandas as pd

from source.dataframe import CsvLoadError, _detect_encoding, read_csv_dataset

_UTF8_CSV = "Name,Value\nAlice,100\nBob,200\n"
_UTF8_BOM_CSV = b"\xef\xbb\xbf" + _UTF8_CSV.encode("utf-8")
_CP1252_CSV = "Name,Value\nCafé,100\nNaïve,200\n".encode("cp1252")
_LATIN1_CSV = "Name,Value\nJürgen,100\nÜber,200\n".encode("latin-1")
_CP1251_CSV = "Имя,Значение\nАлиса,100\nБорис,200\n".encode("cp1251")
_SEMICOLON_CSV = "Name;Value\nAlice;100\nBob;200\n"
_NBSP_CSV = b"Name,Value\r\nAlice,100\xc2\xa0\r\nBob,200\r\n"
_NBSP_CP1252_CSV = b"Name,Value\r\nAlice,100\xa0\r\nBob,200\r\n"

def _write(tmp_path, filename: str, content: bytes | str) -> str:
    path = tmp_path / filename
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)
    return str(path)

def test_cp1251_csv(tmp_path) -> None:
    path = _write(tmp_path, "cp1251.csv", _CP1251_CSV)
    df = read_csv_dataset(path)
    assert len(df) == 2

def test_csv_load_error_message() -> None:
    err = CsvLoadError("/fake/path.csv")
    assert "Could not parse" in err.user_message
    assert "UTF-8" in err.user_message or "utf-8" in err.user_message
    assert "re-saving" in err.user_message
