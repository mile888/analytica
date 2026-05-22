from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from source.dataframe import read_csv_dataset
from source.product.data_sources import DataSourceColumn, DataSourceProfile
from source.product.investigation import utc_now


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
        except pd.errors.ParserError:
            retry_kwargs = {"escapechar": chr(92)}
            try:
                return read_csv_dataset(csv_path, **retry_kwargs)
            except pd.errors.ParserError:
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

    sampled_rows = _safe_records(df.head(20))
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
        return int(series.nunique(dropna=True))
    except Exception:
        return 0


def _sample_values(series: pd.Series) -> list[Any]:
    values: list[Any] = []
    for value in series.head(5).tolist():
        values.append(_json_safe_scalar(value))
    return values


def _numeric_stats(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    stats: dict[str, Any] = {}
    for name in ["min", "max", "mean", "median", "std"]:
        try:
            value = getattr(numeric, name)()
            stats[name] = None if pd.isna(value) else float(value)
        except Exception:
            stats[name] = None
    return stats


def _categorical_stats(series: pd.Series) -> dict[str, Any]:
    try:
        counts = series.astype(str).value_counts(dropna=True).head(10)
        return {"top_values": {str(key): int(value) for key, value in counts.items()}}
    except Exception:
        return {"top_values": {}}


def _safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in df.to_dict(orient="records"):
        records.append({str(key): _json_safe_scalar(value) for key, value in row.items()})
    return records


def _json_safe_scalar(value: Any) -> Any:
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
