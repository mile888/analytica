"""Semantic metric and grouping validation.

Pre-execution validation layer that prevents nonsense metric/dimension
combinations from reaching artifact generation.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MetricValidation:
    valid: bool
    reason: str
    suggested_alternative: str | None = None


@dataclass(frozen=True)
class GroupingValidation:
    valid: bool
    reason: str
    suggested_metric: str | None = None
    suggested_dimension: str | None = None


_IDENTIFIER_MARKERS = (
    "row id", "row_id", "rowid", "row number",
    "order id", "order_id", "orderid", "order number",
    "customer id", "customer_id", "customerid",
    "product id", "product_id", "productid",
    "invoice id", "invoice_id", "invoice number",
    "transaction id", "transaction_id",
    "ticket id", "ticket_id", "ticket number",
    "employee id", "employee_id",
    "user id", "user_id",
    "item id", "item_id",
    "record id", "record_id",
)

_MEANINGFUL_METRIC_MARKERS = (
    "amount", "value", "price", "cost", "revenue", "sales", "profit",
    "score", "rating", "salary", "duration", "quantity", "count",
    "income", "spend", "total", "rate", "margin", "discount",
    "weight", "volume", "frequency", "age", "balance",
)


def _normalize(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").casefold().split())


def validate_metric_for_analysis(metric: str | None, df: pd.DataFrame | None) -> MetricValidation:
    """Validate that a metric column is semantically meaningful for analysis."""
    if not metric:
        return MetricValidation(valid=False, reason="No metric column specified.")
    if not isinstance(df, pd.DataFrame) or metric not in df.columns:
        return MetricValidation(valid=True, reason="Cannot validate without dataframe; allowing.")

    normalized = _normalize(metric)

    if _is_identifier_name(normalized):
        alternative = _find_alternative_metric(df, exclude={metric})
        return MetricValidation(
            valid=False,
            reason=f"`{metric}` is an identifier column — aggregating it (sum, average) is semantically meaningless.",
            suggested_alternative=alternative,
        )

    series = df[metric]
    if not pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() < 0.5:
            return MetricValidation(
                valid=False,
                reason=f"`{metric}` is not a numeric column and cannot be aggregated as a metric.",
            )

    non_null = series.dropna()
    if non_null.empty:
        return MetricValidation(valid=False, reason=f"`{metric}` contains no non-null values.")

    if pd.api.types.is_numeric_dtype(series):
        unique_ratio = non_null.nunique(dropna=True) / max(len(non_null), 1)
        if unique_ratio > 0.95 and len(non_null) > 20 and not any(
            marker in normalized for marker in _MEANINGFUL_METRIC_MARKERS
        ):
            alternative = _find_alternative_metric(df, exclude={metric})
            return MetricValidation(
                valid=False,
                reason=f"`{metric}` has near-unique values ({unique_ratio:.0%} unique) — likely an identifier, not a metric.",
                suggested_alternative=alternative,
            )

    return MetricValidation(valid=True, reason="Metric passes semantic validation.")


def validate_grouping(
    metric: str | None,
    dimension: str | None,
    df: pd.DataFrame | None,
) -> GroupingValidation:
    """Validate that a metric+dimension grouping is semantically coherent."""
    if not metric or not dimension:
        return GroupingValidation(valid=True, reason="Incomplete grouping; skipping validation.")
    if not isinstance(df, pd.DataFrame):
        return GroupingValidation(valid=True, reason="No dataframe available; allowing.")

    metric_validation = validate_metric_for_analysis(metric, df)
    if not metric_validation.valid:
        return GroupingValidation(
            valid=False,
            reason=metric_validation.reason,
            suggested_metric=metric_validation.suggested_alternative,
        )

    norm_dim = _normalize(dimension)
    if _is_identifier_name(norm_dim):
        alt_dim = _find_alternative_dimension(df, exclude={metric, dimension})
        return GroupingValidation(
            valid=False,
            reason=f"Grouping by `{dimension}` (an identifier) produces meaningless aggregations.",
            suggested_dimension=alt_dim,
        )

    if dimension in df.columns:
        dim_series = df[dimension]
        unique_count = int(dim_series.nunique(dropna=True))
        if unique_count <= 1:
            return GroupingValidation(
                valid=False,
                reason=f"`{dimension}` has only {unique_count} unique value(s) — grouping produces no comparison.",
            )
        if unique_count > max(80, int(len(df) * 0.75)) and not _is_identifier_name(norm_dim):
            alt_dim = _find_alternative_dimension(df, exclude={metric, dimension})
            return GroupingValidation(
                valid=False,
                reason=f"`{dimension}` has {unique_count} unique values — too granular for meaningful grouping.",
                suggested_dimension=alt_dim,
            )

    return GroupingValidation(valid=True, reason="Grouping passes semantic validation.")


def _is_identifier_name(normalized: str) -> bool:
    """Check if a normalized column name looks like an identifier."""
    if normalized in ("id", "uuid", "key"):
        return True
    if normalized.endswith(" id") or normalized.endswith("_id"):
        return True
    return any(marker in normalized for marker in _IDENTIFIER_MARKERS)


def _find_alternative_metric(df: pd.DataFrame, exclude: set[str]) -> str | None:
    """Find the best alternative metric column."""
    for column in df.select_dtypes(include="number").columns:
        name = str(column)
        if name in exclude:
            continue
        normalized = _normalize(name)
        if _is_identifier_name(normalized):
            continue
        if any(marker in normalized for marker in _MEANINGFUL_METRIC_MARKERS):
            return name
    for column in df.select_dtypes(include="number").columns:
        name = str(column)
        if name in exclude:
            continue
        normalized = _normalize(name)
        if not _is_identifier_name(normalized):
            return name
    return None


def _find_alternative_dimension(df: pd.DataFrame, exclude: set[str]) -> str | None:
    """Find the best alternative dimension column."""
    for column in df.columns:
        name = str(column)
        if name in exclude:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        normalized = _normalize(name)
        if _is_identifier_name(normalized):
            continue
        unique_count = int(series.nunique(dropna=True))
        if 2 <= unique_count <= max(30, int(len(df) * 0.5)):
            return name
    return None
