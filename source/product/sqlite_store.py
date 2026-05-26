"""Compatibility wrapper for the SQLite-backed product store.

New code should import from `source.product.sqlite`, while this module keeps
older imports stable.
"""

from source.product.sqlite import DEFAULT_INVESTIGATION_DB_PATH, SQLiteInvestigationStore

__all__ = ["DEFAULT_INVESTIGATION_DB_PATH", "SQLiteInvestigationStore"]
