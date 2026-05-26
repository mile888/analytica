from __future__ import annotations

import os
from typing import Literal

from source.product.sqlite_store import DEFAULT_INVESTIGATION_DB_PATH, SQLiteInvestigationStore
from source.product.store import InvestigationStore


StoreKind = Literal["memory", "sqlite"]


def create_investigation_store(kind: str | None = None, db_path: str | None = None):
    resolved_kind = (kind or os.getenv("ANALYTICA_INVESTIGATION_STORE") or "sqlite").strip().lower()
    if resolved_kind == "memory":
        return InvestigationStore()
    if resolved_kind == "sqlite":
        resolved_path = db_path or os.getenv("ANALYTICA_INVESTIGATION_DB_PATH") or DEFAULT_INVESTIGATION_DB_PATH
        return SQLiteInvestigationStore(resolved_path)
    raise ValueError(f"Unsupported investigation store: {resolved_kind}")
