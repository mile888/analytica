from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from source.dataframe import read_csv_dataset
from source.product.data_sources import DataSourceColumn, DataSourceProfile
from source.product.investigation import utc_now


# ---------------------------------------------------------------------------
# Configurable profiling limits (production-safe defaults)
# ---------------------------------------------------------------------------
MAX_PROFILE_SAMPLE_ROWS: int = 20
"""Maximum number of rows included in the sampled_rows preview."""

MAX_SAMPLE_VALUES: int = 5
"""Maximum number of sample values per column."""

MAX_SAMPLE_VALUE_LENGTH: int = 200
"""Maximum character length for a single sample value (text truncation)."""

MAX_CATEGORICAL_TOP_VALUES: int = 10
"""Maximum number of top values in categorical summary."""

MAX_UNIQUE_COUNT_SCAN_ROWS: int = 200_000
"""Beyond this row count, nunique is estimated from a sample."""


def profile_csv(path: str | Path) -> DataSourceProfile:
    try:
        return profile_dataframe(_read_csv_for_profile(path))
    except pd.errors.EmptyDataError:
        return profile_dataframe(pd.DataFrame())


def _read_csv_for_profile(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if csv_path.exists():
        try:
            return read_csv_dataset(csv_path)
        except (pd.errors.ParserError, UnicodeDecodeError):
            retry_kwargs = {"escapechar": chr(92)}
            try:
                return read_csv_dataset(csv_path, **retry_kwargs)
            except (pd.errors.ParserError, UnicodeDecodeError):
                retry_kwargs["on_bad_lines"] = "skip"
                return read_csv_dataset(csv_path, **retry_kwargs)
    return read_csv_dataset(path)


def profile_dataframe(df: pd.DataFrame) -> DataSourceProfile:
    if df is None:
        df = pd.DataFrame()
    row_count, column_count = df.shape
    columns: list[DataSourceColumn] = []
    missing_summary: dict[str, int] = {}
    numeric_summary: dict[str, dict[str, Any]] = {}
    categorical_summary: dict[str, dict[str, Any]] = {}

    for column in df.columns:
        series = df[column]
        name = str(column)
        non_null = series.dropna()
        missing_summary[name] = int(series.isna().sum())
        columns.append(
            DataSourceColumn(
                name=name,
                dtype=str(series.dtype),
                nullable=bool(series.isna().any()),
                unique_count=_safe_unique_count(series),
                sample_values=_sample_values(non_null),
            )
        )

        if pd.api.types.is_numeric_dtype(series):
            numeric_summary[name] = _numeric_stats(series)
        else:
            categorical_summary[name] = _categorical_stats(non_null)

    sampled_rows = _safe_records(df.head(MAX_PROFILE_SAMPLE_ROWS))
    return DataSourceProfile(
        row_count=int(row_count),
        column_count=int(column_count),
        columns=columns,
        missing_summary=missing_summary,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        sampled_rows=sampled_rows,
        generated_at=utc_now(),
    )


def _safe_unique_count(series: pd.Series) -> int:
    try:
        if len(series) > MAX_UNIQUE_COUNT_SCAN_ROWS:
            # Estimate from a sample for very large series
            sample = series.sample(n=MAX_UNIQUE_COUNT_SCAN_ROWS, random_state=42)
            return int(sample.nunique(dropna=True))
        return int(series.nunique(dropna=True))
    except Exception:
        return 0


def _sample_values(series: pd.Series) -> list[Any]:
    values: list[Any] = []
    for value in series.head(MAX_SAMPLE_VALUES).tolist():
        safe = _json_safe_scalar(value)
        # Truncate long text values
        if isinstance(safe, str) and len(safe) > MAX_SAMPLE_VALUE_LENGTH:
            safe = safe[:MAX_SAMPLE_VALUE_LENGTH] + "…"
        values.append(safe)
    return values


def _numeric_stats(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    stats: dict[str, Any] = {}
    for name in ["min", "max", "mean", "median", "std"]:
        try:
            value = getattr(numeric, name)()
            stats[name] = _finite_float_or_none(value)
        except Exception:
            stats[name] = None
    return stats


def _categorical_stats(series: pd.Series) -> dict[str, Any]:
    try:
        counts = series.astype(str).value_counts(dropna=True).head(MAX_CATEGORICAL_TOP_VALUES)
        top_values: dict[str, int] = {}
        for key, value in counts.items():
            key_str = str(key)
            if len(key_str) > MAX_SAMPLE_VALUE_LENGTH:
                key_str = key_str[:MAX_SAMPLE_VALUE_LENGTH] + "…"
            top_values[key_str] = int(value)
        return {"top_values": top_values}
    except Exception:
        return {"top_values": {}}


def _safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in df.to_dict(orient="records"):
        safe_row: dict[str, Any] = {}
        for key, value in row.items():
            safe_val = _json_safe_scalar(value)
            # Truncate long text in preview rows
            if isinstance(safe_val, str) and len(safe_val) > MAX_SAMPLE_VALUE_LENGTH:
                safe_val = safe_val[:MAX_SAMPLE_VALUE_LENGTH] + "…"
            safe_row[str(key)] = safe_val
        records.append(safe_row)
    return records


def _finite_float_or_none(value: Any) -> float | None:
    """Convert a numeric value to a finite Python float, or None."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _json_safe_scalar(value: Any) -> Any:
    """Convert a pandas/numpy scalar to a JSON-safe Python primitive."""
    # Handle missing values first
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass

    # Convert numpy scalars to Python primitives
    if hasattr(value, "item"):
        try:
            native = value.item()
            # Ensure the converted value is a proper Python type
            if isinstance(native, float):
                return native if math.isfinite(native) else None
            if isinstance(native, (int, bool, str)):
                return native
            return str(native)
        except Exception:
            pass

    # Check for float safety (inf/nan)
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    # Accept standard JSON types
    if isinstance(value, (str, int, bool)) or value is None:
        return value

    # Everything else → string
    return str(value)
