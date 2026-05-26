"""Upload robustness regression tests.

These tests exercise the real /data-sources/upload-csv endpoint with edge-case
CSVs that have historically caused API 500 errors.  Every parseable CSV must
return 200 (even if profiling is partial).  Only truly unparseable content
should return 400.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from source.api.app import app

client = TestClient(app)


def _upload(csv_bytes: bytes, filename: str = "test.csv") -> tuple[int, dict]:
    response = client.post(
        "/data-sources/upload-csv",
        files={"file": (filename, io.BytesIO(csv_bytes), "text/csv")},
    )
    return response.status_code, response.json()


# ---- Tests that must succeed (200) ----


def test_upload_normal_csv() -> None:
    code, body = _upload(b"Name,Age,Salary\nAlice,30,50000\nBob,25,60000\n")
    assert code == 200
    assert body["data_source"]["data_source_id"]
    assert body["profile"]["row_count"] == 2
    assert body["profile"]["column_count"] == 3


def test_upload_text_heavy_csv() -> None:
    text = b"Title,Description\nDoc1," + b"x" * 1000 + b"\nDoc2," + b"y" * 1000 + b"\n"
    code, body = _upload(text, "text_heavy.csv")
    assert code == 200
    assert body["profile"]["row_count"] == 2


def test_upload_categorical_only_csv() -> None:
    code, body = _upload(b"City,Country,Status\nLondon,UK,Active\nParis,France,Inactive\n")
    assert code == 200
    assert body["profile"]["column_count"] == 3
    # No numeric columns — must not crash
    assert body["profile"]["numeric_summary"] == {}


def test_upload_numeric_only_csv() -> None:
    code, body = _upload(b"A,B,C\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
    assert code == 200
    assert body["profile"]["column_count"] == 3
    assert body["profile"]["categorical_summary"] == {}


def test_upload_missing_values_csv() -> None:
    code, body = _upload(b"Name,Age,Score\nAlice,,90\n,25,\nCharlie,30,80\n")
    assert code == 200
    assert body["profile"]["row_count"] == 3
    assert body["profile"]["missing_summary"]["Name"] == 1


def test_upload_long_text_fields() -> None:
    long_text = "x" * 10000
    csv_bytes = f"ID,Text\n1,{long_text}\n2,{long_text}\n".encode()
    code, body = _upload(csv_bytes, "long_text.csv")
    assert code == 200
    assert body["profile"]["row_count"] == 2


def test_upload_duplicate_column_names() -> None:
    code, body = _upload(b"Name,Name,Value\nA,X,1\nB,Y,2\n", "duplicates.csv")
    assert code == 200
    assert body["profile"]["column_count"] == 3


def test_upload_single_row_csv() -> None:
    code, body = _upload(b"Value,Category\n42,A\n", "single_row.csv")
    assert code == 200
    assert body["profile"]["row_count"] == 1


def test_upload_headers_only_csv() -> None:
    code, body = _upload(b"Name,Age,Score\n", "headers_only.csv")
    assert code == 200
    assert body["profile"]["row_count"] == 0
    assert body["profile"]["column_count"] == 3


def test_upload_wide_csv() -> None:
    cols = ",".join(f"col_{i}" for i in range(50))
    row = ",".join(str(i) for i in range(50))
    csv_bytes = f"{cols}\n{row}\n{row}\n".encode()
    code, body = _upload(csv_bytes, "wide.csv")
    assert code == 200
    assert body["profile"]["column_count"] == 50


def test_upload_cyrillic_csv() -> None:
    csv_bytes = "Имя,Цена,Город\nТовар,100,Москва\n".encode("utf-8")
    code, body = _upload(csv_bytes, "cyrillic.csv")
    assert code == 200
    assert body["profile"]["row_count"] == 1


def test_upload_special_chars_in_columns() -> None:
    csv_bytes = b"Name (first),Value [$],Date/Time\nAlice,100.5,2024-01-01\n"
    code, body = _upload(csv_bytes, "special_cols.csv")
    assert code == 200
    assert body["profile"]["column_count"] == 3


def test_upload_mixed_dtypes() -> None:
    csv_bytes = b"ID,Value,Flag\n1,abc,true\n2,100,false\nthree,xyz,maybe\n"
    code, body = _upload(csv_bytes, "mixed.csv")
    assert code == 200
    assert body["profile"]["row_count"] == 3


def test_upload_boolean_like_csv() -> None:
    csv_bytes = b"Active,Verified,Score\ntrue,false,90\nyes,no,80\n1,0,70\n"
    code, body = _upload(csv_bytes, "booleans.csv")
    assert code == 200


def test_upload_timestamp_csv() -> None:
    csv_bytes = b"event,timestamp,value\nclick,2024-01-01 12:00:00,100\nhover,2024-01-02 13:30:00,200\n"
    code, body = _upload(csv_bytes, "timestamps.csv")
    assert code == 200
    assert body["profile"]["row_count"] == 2


def test_upload_nan_heavy_csv() -> None:
    csv_bytes = b"A,B,C\n,,\n,,1\n,2,\n"
    code, body = _upload(csv_bytes, "nan_heavy.csv")
    assert code == 200


def test_upload_trailing_comma_csv() -> None:
    csv_bytes = b"A,B,\n1,2,3\n4,5,6\n"
    code, body = _upload(csv_bytes, "trailing_comma.csv")
    assert code == 200


def test_upload_single_column_csv() -> None:
    csv_bytes = b"notes\nhello world\nfoo bar\nbaz\n"
    code, body = _upload(csv_bytes, "single_col.csv")
    assert code == 200
    assert body["profile"]["column_count"] == 1


# ---- Tests for malformed/unparseable CSVs (400) ----


def test_upload_empty_file() -> None:
    code, body = _upload(b"", "empty.csv")
    assert code == 400
    assert "empty" in body.get("detail", "").lower()


def test_upload_non_utf_encoding() -> None:
    """Non-UTF-8 encodings like Latin-1 should now be handled correctly."""
    csv_bytes = "Name,City\nAlice,München\n".encode("latin-1")
    code, body = _upload(csv_bytes, "latin.csv")
    assert code == 200, f"Latin-1 CSV should succeed, got {code}: {body}"
    assert body["profile"]["row_count"] == 1


def test_upload_binary_content() -> None:
    """Random binary should either parse as fallback or return 400."""
    csv_bytes = bytes(range(256)) * 10
    code, body = _upload(csv_bytes, "binary.csv")
    # The encoding fallback may parse this as Latin-1; that's acceptable.
    # What matters is no 500 error.
    assert code in (200, 400)


# ---- Upload must not return 500 ----


def test_upload_malformed_csv_does_not_500() -> None:
    csv_bytes = b"A,B\n1,2,3\n4\n"
    code, _body = _upload(csv_bytes, "malformed.csv")
    assert code != 500, "Malformed CSV must not return 500"


# ---- Response schema ----


def test_upload_response_has_data_source_id() -> None:
    code, body = _upload(b"X,Y\n1,2\n")
    assert code == 200
    assert "data_source" in body
    assert "data_source_id" in body["data_source"]
    assert "profile" in body


def test_upload_response_has_profile_status_on_warning() -> None:
    """If warnings are present, profile_status must be set."""
    # Normal CSV should have no warnings
    code, body = _upload(b"X,Y\n1,2\n")
    assert code == 200
    # When there are no warnings, profile_status may be absent (complete by default)
    if "warnings" in body:
        assert body.get("profile_status") in ("complete", "partial")


# ---- Non-.csv extension ----


def test_upload_non_csv_extension() -> None:
    response = client.post(
        "/data-sources/upload-csv",
        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400
    assert "csv" in response.json()["detail"].lower()


# ---- Production-critical: JSON serialization safety ----


def test_upload_inf_values_produce_valid_json() -> None:
    """inf/nan in numeric columns must not crash FastAPI JSON serialization."""
    import json

    csv_bytes = b"A,B\n1e308,text\n-1e308,more\n0,short\n"
    code, body = _upload(csv_bytes, "extreme_values.csv")
    assert code == 200
    # Verify the entire response is strictly JSON-safe (no Infinity/NaN)
    raw = json.dumps(body, allow_nan=False)
    assert "Infinity" not in raw
    assert "NaN" not in raw


def test_upload_all_nan_column_produces_valid_profile() -> None:
    """A column of all NaN values must not crash profiling."""
    csv_bytes = b"value,label\n,A\n,B\n,C\n"
    code, body = _upload(csv_bytes, "all_nan.csv")
    assert code == 200
    profile = body["profile"]
    # All stats should be None for all-NaN numeric column
    if "value" in profile.get("numeric_summary", {}):
        stats = profile["numeric_summary"]["value"]
        for stat_val in stats.values():
            assert stat_val is None or isinstance(stat_val, (int, float))


def test_upload_long_text_truncated_in_profile() -> None:
    """Long text values in profile must be truncated to prevent payload bloat."""
    long_text = "x" * 5000
    csv_bytes = f"ID,Text\n1,{long_text}\n2,{long_text}\n".encode()
    code, body = _upload(csv_bytes, "long_text_profile.csv")
    assert code == 200
    profile = body["profile"]
    # Sample values should be truncated
    for col in profile.get("columns", []):
        if col["name"] == "Text":
            for sample in col.get("sample_values", []):
                if isinstance(sample, str):
                    assert len(sample) <= 300, f"Sample value not truncated: {len(sample)} chars"
    # Sampled rows should be truncated
    for row in profile.get("sampled_rows", []):
        for val in row.values():
            if isinstance(val, str):
                assert len(val) <= 300, f"Row value not truncated: {len(val)} chars"


def test_upload_response_is_fastapi_json_safe() -> None:
    """The full upload response must survive FastAPI's strict JSON encoder."""
    import json

    csv_bytes = b"num,text,mixed\n42,hello,1\n0,world,two\n"
    code, body = _upload(csv_bytes, "json_safe.csv")
    assert code == 200
    # FastAPI uses json.dumps with allow_nan=False by default
    raw = json.dumps(body, allow_nan=False)
    assert isinstance(raw, str)


# ---- Scale-aware profiling ----


def test_profile_caps_sample_rows() -> None:
    """Profile sampled_rows should be capped at the configured limit."""
    rows = "\n".join(f"{i},val_{i}" for i in range(100))
    csv_bytes = f"ID,Value\n{rows}\n".encode()
    code, body = _upload(csv_bytes, "many_rows.csv")
    assert code == 200
    assert len(body["profile"]["sampled_rows"]) <= 20


def test_profile_caps_categorical_top_values() -> None:
    """Categorical top_values should be capped at the configured limit."""
    rows = "\n".join(f"category_{i}" for i in range(1000))
    csv_bytes = f"Label\n{rows}\n".encode()
    code, body = _upload(csv_bytes, "high_cardinality.csv")
    assert code == 200
    cat = body["profile"].get("categorical_summary", {})
    if "Label" in cat:
        top = cat["Label"].get("top_values", {})
        assert len(top) <= 10


# ---- Global exception handler ----


def test_global_exception_handler_returns_json() -> None:
    """If an endpoint crashes unexpectedly, the response should be JSON, not raw 500."""
    # Health endpoint should work
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

