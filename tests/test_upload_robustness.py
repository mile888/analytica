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

def test_upload_normal_csv() -> None:
    code, body = _upload(b"Name,Age,Salary\nAlice,30,50000\nBob,25,60000\n")
    assert code == 200
    assert body["data_source"]["data_source_id"]
    assert body["profile"]["row_count"] == 2
    assert body["profile"]["column_count"] == 3

def test_upload_malformed_csv_does_not_500() -> None:
    csv_bytes = b"A,B\n1,2,3\n4\n"
    code, _body = _upload(csv_bytes, "malformed.csv")
    assert code != 500, "Malformed CSV must not return 500"

