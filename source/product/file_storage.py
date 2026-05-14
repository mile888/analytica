from __future__ import annotations

import re
from pathlib import Path


DEFAULT_UPLOAD_DIR = Path(".analytica/uploads")


def ensure_upload_dir(upload_dir: str | Path = DEFAULT_UPLOAD_DIR) -> Path:
    path = Path(upload_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(filename: str | None) -> str:
    name = Path(filename or "uploaded.csv").name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "uploaded.csv"


def save_uploaded_csv(
    file_bytes: bytes,
    filename: str | None,
    data_source_id: str,
    upload_dir: str | Path = DEFAULT_UPLOAD_DIR,
) -> str:
    directory = ensure_upload_dir(upload_dir)
    original = sanitize_filename(filename)
    suffix = Path(original).suffix.lower() or ".csv"
    path = directory / f"{data_source_id}{suffix}"
    path.write_bytes(file_bytes)
    return str(path)
