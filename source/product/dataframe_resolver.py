from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from source.config import PROJECT_ROOT
from source.dataframe import read_csv_dataset
from source.product.data_sources import DataSourceType

if TYPE_CHECKING:
    from source.product.investigation import Investigation
    from source.product.store import InvestigationStore


def resolve_dataframe_for_investigation(
    store: InvestigationStore,
    investigation: Investigation,
) -> pd.DataFrame | None:
    """Load the first available CSV dataframe linked to an investigation."""

    for data_source_id in investigation.linked_data_source_ids:
        df = resolve_dataframe_from_data_source(store, data_source_id)
        if df is not None:
            return df
    return None


def resolve_dataframe_from_data_source(
    store: InvestigationStore,
    data_source_id: str,
) -> pd.DataFrame | None:
    """Load a registered CSV data source as a dataframe when its file exists.

    This resolver intentionally only reads files through registered DataSource
    records. Relative upload paths are resolved from the project root so API,
    tests, notebooks, and Streamlit do not depend on the current process cwd.
    """

    try:
        source = store.get_data_source(data_source_id)
    except KeyError:
        return None
    if source.data_source_type != DataSourceType.CSV or not source.location:
        return None

    path = resolve_data_source_file_path(source.location)
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        return read_csv_dataset(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return None


def resolve_data_source_file_path(location: str | Path | None) -> Path | None:
    """Resolve a registered data-source file path without relying on cwd."""

    if not location:
        return None
    path = Path(location).expanduser()
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return path
