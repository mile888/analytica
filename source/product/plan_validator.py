"""Validate semantic plans before deterministic execution."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from source.product.llm_semantic_planner import SemanticPlan, SchemaContext, parse_ordered_range_value


@dataclass
class ValidatedPlan:
    """A plan that can run, with any safe repairs noted."""

    plan: SemanticPlan
    repairs: list[str] = field(default_factory=list)


@dataclass
class PlanRejection:
    """A plan that cannot run safely."""

    reasons: list[str] = field(default_factory=list)
    fallback_to_deterministic: bool = True


def validate_plan(
    plan: SemanticPlan,
    df: pd.DataFrame,
    schema: SchemaContext | None = None,
) -> ValidatedPlan | PlanRejection:
    """Check a semantic plan against the actual dataset."""
    columns = [str(c) for c in df.columns]
    columns_lower = {c.lower(): c for c in columns}
    numeric_cols = set(str(c) for c in df.select_dtypes(include="number").columns)
    errors: list[str] = []
    repairs: list[str] = []
    plan.metric_columns = [item for item in plan.metric_columns if item]
    plan.grouping_columns = [item for item in plan.grouping_columns if item]
    if not plan.metric and plan.metric_columns:
        plan.metric = plan.metric_columns[0]
    if not plan.dimension and plan.grouping_columns:
        plan.dimension = plan.grouping_columns[0]
    metric_locked = bool(plan.constraints_locked.get("metric") or plan.constraints_locked.get("metric_columns"))
    grouping_locked = bool(plan.constraints_locked.get("dimension") or plan.constraints_locked.get("grouping_columns"))
    aggregation_locked = bool(plan.constraints_locked.get("aggregation"))

    plan = _repair_column_cases(plan, columns_lower, repairs)

    for ref_name, ref_value in _column_references(plan):
        if ref_value and ref_value not in columns:
            errors.append(f"{ref_name} '{ref_value}' does not exist in dataset. Available: {columns[:10]}")
    for ref_name, ref_values in (("metric_columns", plan.metric_columns), ("grouping_columns", plan.grouping_columns), ("target_values", [])):
        if ref_name == "target_values":
            continue
        for ref_value in ref_values:
            if ref_value and ref_value not in columns:
                errors.append(f"{ref_name} contains '{ref_value}', which does not exist in dataset. Available: {columns[:10]}")

    if errors:
        return PlanRejection(reasons=errors)

    # 3. Identifier fields should not be used as metrics (check early to prevent false rejection)
    if plan.metric and _is_identifier_column(plan.metric, df):
        if metric_locked:
            errors.append(f"Locked metric '{plan.metric}' appears to be an identifier, so it cannot be aggregated.")
        else:
            plan.metric = None
            plan.aggregation = "count"
            if plan.operation in {"METRIC_AGGREGATION", "GROUPED_AGGREGATION"}:
                plan.operation = "COUNT_DISTRIBUTION"
            repairs.append("Removed identifier column as metric")

    # 4. Metric must be numeric for aggregation operations
    needs_numeric = plan.aggregation in ("sum", "mean", "median", "min", "max")
    if needs_numeric and plan.metric:
        if plan.metric not in numeric_cols:
            if _is_ordered_range_column(plan.metric, df):
                note = f"Metric '{plan.metric}' is an ordered/range category; deterministic execution will use approximate numeric midpoints."
                if note not in plan.warnings:
                    plan.warnings.append(note)
                if note not in plan.limitations:
                    plan.limitations.append(note)
            # Try to repair: if it's a COUNT operation, drop metric
            elif plan.operation == "COUNT_DISTRIBUTION":
                plan.metric = None
                plan.aggregation = "count"
                repairs.append("Removed non-numeric metric for COUNT operation")
            elif metric_locked:
                errors.append(f"Locked metric '{plan.metric}' is not numeric and cannot be parsed as an ordered/range metric for aggregation '{plan.aggregation}'.")
            else:
                errors.append(f"Metric '{plan.metric}' is not numeric but aggregation '{plan.aggregation}' requires numeric")

    # 5. COUNT operations should not require metric
    if plan.operation == "COUNT_DISTRIBUTION" and plan.aggregation == "count":
        if plan.metric and plan.metric not in numeric_cols:
            plan.metric = None
            repairs.append("Cleared non-numeric metric for count-based operation")

    # 6. Year fields should not be summed/averaged
    if plan.metric and _is_year_column(plan.metric, df):
        if plan.aggregation in ("sum", "mean", "median"):
            if metric_locked or aggregation_locked:
                errors.append(f"Locked metric '{plan.metric}' is year-like and cannot be aggregated with '{plan.aggregation}'.")
            else:
                original = plan.aggregation
                plan.aggregation = "count"
                plan.operation = "COUNT_DISTRIBUTION"
                repairs.append(f"Changed aggregation from {original} to count — year columns should not be aggregated")

    # 6a. Locked grouping columns must remain usable group dimensions.
    if grouping_locked:
        for group in plan.grouping_columns or ([plan.dimension] if plan.dimension else []):
            if group and group in df.columns and _is_identifier_column(group, df):
                errors.append(f"Locked grouping column '{group}' appears identifier-like; choose a real segment/category field.")

    # 7. Time field should be year/date-like
    if plan.time_field:
        if not _is_year_column(plan.time_field, df) and not _is_date_column(plan.time_field, df):
            # Try to find an actual year column
            actual_year = _find_year_column(df)
            if actual_year:
                plan.time_field = actual_year
                repairs.append(f"Replaced time_field with detected year column '{actual_year}'")
            else:
                plan.time_field = None
                repairs.append("Removed invalid time_field — no year/date column found")

    # 8. Operation must be in allowed set (already checked by planner, but belt & suspenders)
    from source.product.llm_semantic_planner import ALLOWED_OPERATIONS
    if plan.operation not in ALLOWED_OPERATIONS:
        errors.append(f"Operation '{plan.operation}' is not allowed")
    if plan.aggregation in set(plan.forbidden_aggregations):
        errors.append(f"Aggregation '{plan.aggregation}' was explicitly forbidden by the user.")

    if errors:
        return PlanRejection(reasons=errors)

    return ValidatedPlan(plan=plan, repairs=repairs)


def _repair_column_cases(
    plan: SemanticPlan,
    columns_lower: dict[str, str],
    repairs: list[str],
) -> SemanticPlan:
    """Fix safe column-name case mismatches."""
    for attr in ("metric", "dimension", "time_field", "category_field", "duration_field", "target_column", "ordered_column"):
        value = getattr(plan, attr)
        if value and value not in columns_lower.values():
            corrected = columns_lower.get(value.lower())
            if corrected:
                setattr(plan, attr, corrected)
                repairs.append(f"Fixed case: {attr} '{value}' → '{corrected}'")
    for attr in ("metric_columns", "grouping_columns"):
        values = list(getattr(plan, attr) or [])
        repaired_values: list[str] = []
        for value in values:
            corrected = columns_lower.get(str(value).lower())
            repaired_values.append(corrected or value)
            if corrected and corrected != value:
                repairs.append(f"Fixed case: {attr} '{value}' → '{corrected}'")
        setattr(plan, attr, repaired_values)
    return plan


def _column_references(plan: SemanticPlan) -> list[tuple[str, str | None]]:
    """Return all column names referenced by a plan."""
    return [
        ("metric", plan.metric),
        ("dimension", plan.dimension),
        ("time_field", plan.time_field),
        ("category_field", plan.category_field),
        ("duration_field", plan.duration_field),
        ("target_column", plan.target_column),
        ("ordered_column", plan.ordered_column),
    ]


def _is_year_column(col: str, df: pd.DataFrame) -> bool:
    if col not in df.columns:
        return False
    series = df[col]
    if not pd.api.types.is_numeric_dtype(series):
        return False
    try:
        desc = series.describe()
        smin, smax = float(desc.get("min", 0)), float(desc.get("max", 0))
        return 1900 <= smin <= 2100 and 1900 <= smax <= 2100
    except (TypeError, ValueError):
        return False


def _is_date_column(col: str, df: pd.DataFrame) -> bool:
    if col not in df.columns:
        return False
    series = df[col]
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(series.dropna().head(20), errors="coerce")
            return len(parsed) > 0 and parsed.notna().mean() >= 0.7
        except Exception:
            return False


def _is_identifier_column(col: str, df: pd.DataFrame) -> bool:
    if col not in df.columns:
        return False
    norm = col.lower().replace(" ", "_")
    id_markers = ("_id", "id_", "uuid", "guid", "key", "index")
    if any(marker in norm for marker in id_markers):
        return True
    nunique = df[col].nunique()
    return len(df) > 10 and nunique > len(df) * 0.8


def _find_year_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if _is_year_column(str(col), df):
            return str(col)
    return None


def _is_ordered_range_column(col: str, df: pd.DataFrame) -> bool:
    if col not in df.columns:
        return False
    sample = [value for value in df[col].dropna().head(30).tolist()]
    if len(sample) < 2:
        return False
    parsed = [parse_ordered_range_value(value) for value in sample]
    return sum(value is not None for value in parsed) / max(len(sample), 1) >= 0.6
