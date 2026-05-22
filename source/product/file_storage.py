from __future__ import annotations

import re
from pathlib import Path

from source.config import UPLOAD_DIR


DEFAULT_UPLOAD_DIR = UPLOAD_DIR


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


def delete_uploaded_file_if_safe(
    location: str | Path | None,
    upload_dir: str | Path = DEFAULT_UPLOAD_DIR,
) -> bool:
    """Delete a local upload only when it is inside the managed upload dir."""
    if not location:
        return False
    target = Path(location).expanduser()
    root = Path(upload_dir).expanduser()
    target_resolved = target.resolve()
    root_resolved = root.resolve()
    try:
        is_managed_upload = target_resolved.is_relative_to(root_resolved)
    except AttributeError:
        is_managed_upload = str(target_resolved).startswith(str(root_resolved) + "/")
    if not is_managed_upload or not target_resolved.is_file():
        return False
    target_resolved.unlink()
    return True
