from __future__ import annotations

from source.product.store_factory import create_investigation_store


_store = None


def get_store():
    global _store
    if _store is None:
        _store = create_investigation_store()
    return _store


def set_store_for_testing(store) -> None:
    global _store
    _store = store
