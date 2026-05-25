"""Run validated semantic plans with deterministic pandas code."""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from source.product.llm_semantic_planner import SemanticPlan, parse_ordered_range_value
from source.product.plan_validator import ValidatedPlan


@dataclass
class EvidencePackage:
    """Stores computed tables, statistics, artifacts, and limits."""

    question: str
    operation: str
    columns_used: list[str] = field(default_factory=list)
    filters_applied: list[dict[str, Any]] = field(default_factory=list)
    record_count: int = 0
    computed_tables: list[dict[str, Any]] = field(default_factory=list)
    summary_statistics: dict[str, Any] = field(default_factory=dict)
    chart_data: dict[str, Any] | None = None
    limitations: list[str] = field(default_factory=list)
    validation_notes: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "question": self.question,
            "operation": self.operation,
            "columns_used": self.columns_used,
            "filters_applied": self.filters_applied,
            "record_count": self.record_count,
            "computed_tables": self.computed_tables,
            "summary_statistics": self.summary_statistics,
            "limitations": self.limitations,
            "validation_notes": self.validation_notes,
        }
        if self.chart_data:
            result["chart_data"] = self.chart_data
        return result


def execute_plan(
    validated: ValidatedPlan,
    df: pd.DataFrame,
    question: str = "",
) -> EvidencePackage:
    """Execute a validated plan and return computed evidence."""
    plan = validated.plan
    evidence = EvidencePackage(
        question=question,
        operation=plan.operation,
        record_count=len(df),
        validation_notes=list(validated.repairs),
        limitations=list(plan.limitations),
    )

    working_df = _apply_filters(df, plan.filters)
    if len(working_df) < len(df):
        evidence.filters_applied = plan.filters
        evidence.record_count = len(working_df)

    cols_used: set[str] = set()
    for col in (plan.metric, plan.dimension, plan.time_field, plan.category_field, plan.duration_field):
        if col and col in df.columns:
            cols_used.add(col)
    evidence.columns_used = sorted(cols_used)

    # Route to operation handler
    handlers = {
        "COUNT_DISTRIBUTION": _exec_count_distribution,
        "METRIC_AGGREGATION": _exec_metric_aggregation,
        "GROUPED_AGGREGATION": _exec_metric_aggregation,
        "CATEGORICAL_OUTCOME_BREAKDOWN": _exec_categorical_outcome_breakdown,
        "ORDERED_OUTCOME_RISK": _exec_ordered_outcome_risk,
        "TEMPORAL_TREND": _exec_temporal_trend,
        "CATEGORY_MIX_SHIFT": _exec_category_mix_shift,
        "DIVERSITY_ANALYSIS": _exec_diversity_analysis,
        "DURATION_ANALYSIS": _exec_duration_analysis,
        "OUTLIER_ANALYSIS": _exec_outlier_analysis,
        "COMPARISON_ANALYSIS": _exec_comparison_analysis,
        "STRATEGIC_SYNTHESIS": _exec_strategic_synthesis,
        "CORRELATION_ANALYSIS": _exec_correlation_analysis,
        "DATA_QUALITY_CHECK": _exec_data_quality,
        "DATASET_OVERVIEW": _exec_dataset_overview,
    }

    handler = handlers.get(plan.operation, _exec_dataset_overview)
    handler(plan, working_df, evidence)

    return evidence


# ── Operation handlers ──────────────────────────────────────────────────────

def _exec_count_distribution(plan: SemanticPlan, df: pd.DataFrame, evidence: EvidencePackage) -> None:
    dim = plan.dimension or plan.category_field
    if not dim or dim not in df.columns:
        # Find best categorical column
        dim = _best_categorical_column(df)
    if not dim:
        evidence.limitations.append("No suitable categorical column found for count distribution")
        return

    working = df
    # Handle multi-label columns
    if _is_multi_label_column(df[dim]):
        working = _explode_multi_label(df, dim)

    counts = working[dim].value_counts().head(15)
    total = len(working)
    table = [{"value": str(k), "count": int(v), "share": round(v / max(total, 1) * 100, 1)} for k, v in counts.items()]

    evidence.computed_tables.append({
        "name": f"count_by_{dim}",
        "dimension": dim,
        "aggregation": "count",
        "rows": table,
    })
    evidence.summary_statistics = {
        "dimension": dim,
        "total_records": total,
        "unique_values": int(working[dim].nunique()),
        "top_value": str(counts.index[0]) if len(counts) > 0 else None,
        "top_count": int(counts.iloc[0]) if len(counts) > 0 else 0,
    }

    evidence.artifacts.append({
        "type": "table",
        "title": f"Distribution: {dim}",
        "metadata": {"dimension": dim, "aggregation": "count", "chart_type": "bar"},
        "content": table,
    })


def _exec_metric_aggregation(plan: SemanticPlan, df: pd.DataFrame, evidence: EvidencePackage) -> None:
    metric = plan.metric or (plan.metric_columns[0] if plan.metric_columns else None)
    grouping_columns = [column for column in (plan.grouping_columns or []) if column in df.columns]
    dim = plan.dimension or plan.category_field
    if dim and dim in df.columns and dim not in grouping_columns:
        grouping_columns.insert(0, dim)
    agg = plan.aggregation or "mean"
    limit = int(plan.limit or 15)
    ascending = str(plan.sorting or "").lower() == "asc"

    if not metric or metric not in df.columns:
        evidence.limitations.append("No valid metric column for aggregation")
        return
    metric_values, metric_note = _metric_numeric_series(df[metric])
    if metric_values is None:
        evidence.limitations.append(f"Metric '{metric}' could not be converted to numeric values; no substitute metric was used.")
        return
    if metric_note:
        evidence.validation_notes.append(metric_note)

    if grouping_columns:
        working = df.copy()
        working["_semantic_metric_value"] = metric_values
        working = working.dropna(subset=["_semantic_metric_value"])
        if working.empty:
            evidence.limitations.append(f"Metric '{metric}' contains no usable numeric values.")
            return
        grouped = (
            working.groupby(grouping_columns, observed=True, dropna=False)["_semantic_metric_value"]
            .agg(agg)
            .sort_values(ascending=ascending)
            .head(limit)
        )
        table = []
        for key, value in grouped.items():
            key_values = key if isinstance(key, tuple) else (key,)
            row = {column: str(key_values[idx]) for idx, column in enumerate(grouping_columns)}
            row["group"] = " / ".join(str(item) for item in key_values)
            rounded = round(float(value), 2)
            row[agg] = rounded
            if agg == "sum":
                row["total"] = rounded
            table.append(row)
        evidence.computed_tables.append({
            "name": f"{agg}_{metric}_by_{dim}",
            "metric": metric,
            "dimension": grouping_columns[0],
            "grouping_columns": grouping_columns,
            "aggregation": agg,
            "rows": table,
        })
        evidence.chart_data = {
            "chart_type": plan.artifact_type if plan.artifact_type != "none" else "bar",
            "metric": metric,
            "grouping_columns": grouping_columns,
            "aggregation": agg,
            "rows": table,
        }
        title = f"{_aggregation_title(agg)} {metric} by {' and '.join(grouping_columns)}"
        evidence.artifacts.append({
            "artifact_type": "table",
            "type": "table",
            "title": title,
            "metadata": {"metric": metric, "dimension": grouping_columns[0], "grouping_columns": grouping_columns, "aggregation": agg, "chart_type": "bar", "query_plan": plan.to_dict()},
            "content": table,
        })
        if plan.artifact_required or plan.visualization_requested or plan.operation == "GROUPED_AGGREGATION":
            evidence.artifacts.append({
                "artifact_type": "chart",
                "type": "chart",
                "title": title,
                "metadata": {"metric": metric, "dimension": grouping_columns[0], "grouping_columns": grouping_columns, "aggregation": agg, "chart_type": "bar", "query_plan": plan.to_dict()},
                "content": {"chart_type": "bar", "x": "group", "y": agg, "rows": table, "metric": metric, "aggregation": agg},
            })
    else:
        desc = metric_values.describe()
        evidence.summary_statistics = {
            "metric": metric,
            "count": int(desc.get("count", 0)),
            "mean": round(float(desc.get("mean", 0)), 2),
            "median": round(float(metric_values.median()), 2),
            "min": round(float(desc.get("min", 0)), 2),
            "max": round(float(desc.get("max", 0)), 2),
            "std": round(float(desc.get("std", 0)), 2),
        }


def _exec_categorical_outcome_breakdown(plan: SemanticPlan, df: pd.DataFrame, evidence: EvidencePackage) -> None:
    target = plan.target_column
    groups = [column for column in plan.grouping_columns if column in df.columns]
    values = [str(value) for value in plan.target_values]
    if not target or target not in df.columns or not groups or not values:
        evidence.limitations.append("Outcome breakdown needs a target column, target values, and grouping columns.")
        return

    working = df[groups + [target]].dropna(subset=[target]).copy()
    grouped = working.groupby(groups + [target], observed=True, dropna=False).size().reset_index(name="count")
    totals = working.groupby(groups, observed=True, dropna=False).size().reset_index(name="group_total")
    merged = grouped.merge(totals, on=groups, how="left")
    merged["share"] = (merged["count"] / merged["group_total"].clip(lower=1) * 100).round(2)
    if values:
        normalized_values = {_normalize_value(value) for value in values}
        merged = merged[merged[target].map(lambda item: _normalize_value(item) in normalized_values)]
    rows = merged.head(200).to_dict(orient="records")
    evidence.computed_tables.append({
        "name": f"outcome_breakdown_{target}_by_{'_'.join(groups)}",
        "target_column": target,
        "target_values": values,
        "grouping_columns": groups,
        "aggregation": "share",
        "rows": rows,
    })
    title = f"{target} by {' and '.join(groups)}"
    evidence.chart_data = {"chart_type": plan.artifact_type if plan.artifact_type != "none" else "heatmap", "rows": rows, "target_column": target, "grouping_columns": groups}
    evidence.artifacts.append({
        "artifact_type": "chart",
        "type": "chart",
        "title": title,
        "metadata": {"target_column": target, "target_values": values, "grouping_columns": groups, "aggregation": "share", "chart_type": evidence.chart_data["chart_type"], "query_plan": plan.to_dict()},
        "content": {"chart_type": evidence.chart_data["chart_type"], "x": groups[0], "series": target, "rows": rows, "target_column": target, "grouping_columns": groups},
    })
    evidence.artifacts.append({
        "artifact_type": "table",
        "type": "table",
        "title": f"{title} table",
        "metadata": {"target_column": target, "target_values": values, "grouping_columns": groups, "aggregation": "share", "query_plan": plan.to_dict()},
        "content": rows,
    })


def _exec_ordered_outcome_risk(plan: SemanticPlan, df: pd.DataFrame, evidence: EvidencePackage) -> None:
    target = plan.target_column
    ordered = plan.ordered_column
    groups = [column for column in plan.grouping_columns if column in df.columns]
    target_values = {_normalize_value(value) for value in plan.target_values}
    if not target or target not in df.columns or not ordered or ordered not in df.columns or not target_values:
        evidence.limitations.append("Ordered risk analysis needs an outcome target, target value, and ordered/range column.")
        return

    working = df.copy()
    working["_ordered_score"] = working[ordered].map(parse_ordered_range_value)
    working["_target_hit"] = working[target].map(lambda item: _normalize_value(item) in target_values)
    working = working.dropna(subset=["_ordered_score"])
    if working.empty:
        evidence.limitations.append(f"Ordered column '{ordered}' could not be parsed into ordered scores.")
        return
    if not groups:
        groups = [column for column in df.columns if column not in {target, ordered} and not pd.api.types.is_numeric_dtype(df[column])][:2]
    rows: list[dict[str, Any]] = []
    for group_col in groups:
        for group_value, subset in working.groupby(group_col, observed=True, dropna=False):
            if subset.empty:
                continue
            low_score = subset["_ordered_score"].min()
            high_score = subset["_ordered_score"].max()
            low_subset = subset[subset["_ordered_score"] == low_score]
            high_subset = subset[subset["_ordered_score"] == high_score]
            if low_subset.empty or high_subset.empty:
                continue
            low_rate = float(low_subset["_target_hit"].mean() * 100)
            high_rate = float(high_subset["_target_hit"].mean() * 100)
            rows.append({
                "grouping_column": group_col,
                "group": str(group_value),
                "ordered_column": ordered,
                "lowest_ordered_value": str(low_subset[ordered].iloc[0]),
                "highest_ordered_value": str(high_subset[ordered].iloc[0]),
                "risk_at_lowest": round(low_rate, 2),
                "risk_at_highest": round(high_rate, 2),
                "increase_as_order_decreases": round(low_rate - high_rate, 2),
                "records": int(len(subset)),
            })
    rows.sort(key=lambda row: row["increase_as_order_decreases"], reverse=True)
    limit = int(plan.limit or 20)
    rows = rows[:limit]
    if not rows:
        evidence.limitations.append("No demographic group had enough ordered-band data to compute a risk gradient.")
        return
    evidence.computed_tables.append({
        "name": f"ordered_risk_{target}_by_{ordered}",
        "target_column": target,
        "target_values": list(plan.target_values),
        "ordered_column": ordered,
        "grouping_columns": groups,
        "aggregation": "share_gradient",
        "rows": rows,
    })
    evidence.validation_notes.append(f"Parsed '{ordered}' as an ordered/range category using approximate numeric midpoints.")
    title = f"{target} risk as {ordered} decreases"
    evidence.artifacts.append({
        "artifact_type": "table",
        "type": "table",
        "title": title,
        "metadata": {"target_column": target, "ordered_column": ordered, "grouping_columns": groups, "aggregation": "share_gradient", "query_plan": plan.to_dict()},
        "content": rows,
    })
    evidence.artifacts.append({
        "artifact_type": "chart",
        "type": "chart",
        "title": title,
        "metadata": {"target_column": target, "ordered_column": ordered, "grouping_columns": groups, "aggregation": "share_gradient", "chart_type": "bar", "query_plan": plan.to_dict()},
        "content": {"chart_type": "bar", "x": "group", "y": "increase_as_order_decreases", "rows": rows},
    })


def _exec_category_mix_shift(plan: SemanticPlan, df: pd.DataFrame, evidence: EvidencePackage) -> None:
    time_col = plan.time_field
    cat_col = plan.category_field or plan.dimension

    if not time_col or time_col not in df.columns:
        time_col = _find_year_column(df)
    if not cat_col or cat_col not in df.columns:
        cat_col = _best_categorical_column(df, exclude={time_col} if time_col else set())

    if not time_col or not cat_col:
        evidence.limitations.append("Cannot perform temporal mix shift without time and category columns")
        return

    working = df
    if _is_multi_label_column(df[cat_col]):
        working = _explode_multi_label(df, cat_col)

    # Find midpoint
    years = pd.to_numeric(working[time_col], errors="coerce").dropna()
    if years.empty:
        evidence.limitations.append("Time field contains no valid numeric values")
        return
    threshold = int(years.median())

    before = working[pd.to_numeric(working[time_col], errors="coerce") <= threshold]
    after = working[pd.to_numeric(working[time_col], errors="coerce") > threshold]

    before_dist = before[cat_col].value_counts().head(10)
    after_dist = after[cat_col].value_counts().head(10)

    table_before = [{"category": str(k), "count": int(v), "period": f"≤{threshold}"} for k, v in before_dist.items()]
    table_after = [{"category": str(k), "count": int(v), "period": f">{threshold}"} for k, v in after_dist.items()]

    # Find emerging and declining categories
    all_cats = set(before_dist.index) | set(after_dist.index)
    shifts = []
    for cat in all_cats:
        b = before_dist.get(cat, 0) / max(len(before), 1) * 100
        a = after_dist.get(cat, 0) / max(len(after), 1) * 100
        shifts.append({"category": str(cat), "before_share": round(b, 1), "after_share": round(a, 1), "change": round(a - b, 1)})
    shifts.sort(key=lambda x: abs(x["change"]), reverse=True)

    evidence.computed_tables.append({
        "name": f"mix_shift_{cat_col}_by_{time_col}",
        "category_field": cat_col,
        "time_field": time_col,
        "threshold": threshold,
        "before_count": len(before),
        "after_count": len(after),
        "before_distribution": table_before,
        "after_distribution": table_after,
        "shifts": shifts[:10],
    })

    evidence.summary_statistics = {
        "time_field": time_col,
        "category_field": cat_col,
        "threshold": threshold,
        "before_count": len(before),
        "after_count": len(after),
        "biggest_gainer": shifts[0]["category"] if shifts and shifts[0]["change"] > 0 else None,
        "biggest_decliner": next((s["category"] for s in shifts if s["change"] < 0), None),
    }


def _exec_temporal_trend(plan: SemanticPlan, df: pd.DataFrame, evidence: EvidencePackage) -> None:
    """Use the category mix shift executor for this plan."""
    _exec_category_mix_shift(plan, df, evidence)


def _exec_diversity_analysis(plan: SemanticPlan, df: pd.DataFrame, evidence: EvidencePackage) -> None:
    group_col = plan.dimension
    entity_col = plan.category_field

    if not group_col or group_col not in df.columns:
        group_col = _best_categorical_column(df, max_unique=15)
    if not entity_col or entity_col not in df.columns:
        entity_col = _best_categorical_column(df, exclude={group_col} if group_col else set())

    if not group_col or not entity_col:
        evidence.limitations.append("Cannot perform diversity analysis without group and entity columns")
        return

    diversity = df.groupby(group_col, observed=True)[entity_col].nunique().sort_values(ascending=False)
    table = [{"group": str(k), "unique_count": int(v)} for k, v in diversity.items()]

    evidence.computed_tables.append({
        "name": f"diversity_{entity_col}_by_{group_col}",
        "group_column": group_col,
        "entity_column": entity_col,
        "rows": table,
    })
    evidence.summary_statistics = {
        "group_column": group_col,
        "entity_column": entity_col,
        "most_diverse": str(diversity.index[0]) if len(diversity) > 0 else None,
        "least_diverse": str(diversity.index[-1]) if len(diversity) > 0 else None,
        "max_unique": int(diversity.iloc[0]) if len(diversity) > 0 else 0,
        "min_unique": int(diversity.iloc[-1]) if len(diversity) > 0 else 0,
    }


def _exec_duration_analysis(plan: SemanticPlan, df: pd.DataFrame, evidence: EvidencePackage) -> None:
    dur_col = plan.duration_field
    cat_col = plan.category_field or plan.dimension

    if not dur_col or dur_col not in df.columns:
        dur_col = _find_duration_column(df)
    if not cat_col or cat_col not in df.columns:
        cat_col = _best_categorical_column(df, exclude={dur_col} if dur_col else set())

    if not dur_col:
        evidence.limitations.append("No duration column found")
        return

    parsed = df[dur_col].apply(_parse_duration_value)
    valid = parsed.dropna()

    if valid.empty:
        evidence.limitations.append(f"Could not parse durations from '{dur_col}'")
        return

    if cat_col and cat_col in df.columns:
        working = df.copy()
        working["_parsed_duration"] = parsed

        if _is_multi_label_column(df[cat_col]):
            working = _explode_multi_label(working, cat_col)

        grouped = working.dropna(subset=["_parsed_duration"]).groupby(cat_col, observed=True)["_parsed_duration"]
        table = [
            {"category": str(k), "mean": round(float(v), 1), "count": int(grouped.get_group(k).count())}
            for k, v in grouped.mean().sort_values(ascending=False).head(15).items()
        ]
        evidence.computed_tables.append({
            "name": f"duration_by_{cat_col}",
            "duration_field": dur_col,
            "category_field": cat_col,
            "rows": table,
        })
    else:
        evidence.summary_statistics = {
            "duration_field": dur_col,
            "mean": round(float(valid.mean()), 1),
            "median": round(float(valid.median()), 1),
            "min": round(float(valid.min()), 1),
            "max": round(float(valid.max()), 1),
        }


def _exec_outlier_analysis(plan: SemanticPlan, df: pd.DataFrame, evidence: EvidencePackage) -> None:
    metric = plan.metric
    dim = plan.dimension

    if not metric or metric not in df.columns:
        numeric_cols = [str(c) for c in df.select_dtypes(include="number").columns if not _is_year_column(str(c), df)]
        metric = numeric_cols[0] if numeric_cols else None

    if not metric:
        evidence.limitations.append("No numeric column for outlier analysis")
        return

    series = pd.to_numeric(df[metric], errors="coerce").dropna()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]

    evidence.summary_statistics = {
        "metric": metric,
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2),
        "iqr": round(float(iqr), 2),
        "lower_bound": round(float(lower), 2),
        "upper_bound": round(float(upper), 2),
        "outlier_count": len(outliers),
        "outlier_share": round(len(outliers) / max(len(series), 1) * 100, 1),
    }


def _exec_comparison_analysis(plan: SemanticPlan, df: pd.DataFrame, evidence: EvidencePackage) -> None:
    _exec_metric_aggregation(plan, df, evidence)


def _exec_strategic_synthesis(plan: SemanticPlan, df: pd.DataFrame, evidence: EvidencePackage) -> None:
    """Compute evidence for a strategic summary."""
    text_cols = [str(c) for c in df.columns if str(df[c].dtype) == "object"]
    numeric_cols = [str(c) for c in df.select_dtypes(include="number").columns]

    cat_col = plan.category_field or plan.dimension or _best_categorical_column(df)
    if cat_col and cat_col in df.columns:
        counts = df[cat_col].value_counts().head(10)
        evidence.computed_tables.append({
            "name": f"category_distribution_{cat_col}",
            "dimension": cat_col,
            "rows": [{"value": str(k), "count": int(v), "share": round(v / max(len(df), 1) * 100, 1)} for k, v in counts.items()],
        })

    # Add numeric summaries
    for col in numeric_cols[:3]:
        if not _is_year_column(col, df):
            desc = df[col].describe()
            evidence.summary_statistics[col] = {
                "mean": round(float(desc.get("mean", 0)), 2),
                "median": round(float(df[col].median()), 2),
                "min": round(float(desc.get("min", 0)), 2),
                "max": round(float(desc.get("max", 0)), 2),
            }

    # Add temporal dimension if available
    year_col = plan.time_field or _find_year_column(df)
    if year_col and cat_col:
        yearly = df.groupby(pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)).size()
        evidence.computed_tables.append({
            "name": "records_over_time",
            "time_field": year_col,
            "rows": [{"year": int(k), "count": int(v)} for k, v in yearly.tail(10).items()],
        })

    evidence.summary_statistics["dataset_overview"] = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_fields": numeric_cols,
        "text_fields": text_cols,
    }


def _exec_correlation_analysis(plan: SemanticPlan, df: pd.DataFrame, evidence: EvidencePackage) -> None:
    numeric_cols = [str(c) for c in df.select_dtypes(include="number").columns if not _is_year_column(str(c), df)]
    if len(numeric_cols) < 2:
        evidence.limitations.append("Need at least 2 numeric columns for correlation")
        return
    corr = df[numeric_cols].corr().round(3)
    table = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1:]:
            table.append({"field_1": c1, "field_2": c2, "correlation": float(corr.loc[c1, c2])})
    table.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    evidence.computed_tables.append({"name": "correlations", "rows": table[:10]})


def _exec_data_quality(plan: SemanticPlan, df: pd.DataFrame, evidence: EvidencePackage) -> None:
    missing = {str(c): int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    duplicates = int(df.duplicated().sum())
    evidence.summary_statistics = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": missing,
        "total_missing": sum(missing.values()),
        "duplicate_rows": duplicates,
    }


def _exec_dataset_overview(plan: SemanticPlan, df: pd.DataFrame, evidence: EvidencePackage) -> None:
    _exec_strategic_synthesis(plan, df, evidence)


# ── Utility helpers ─────────────────────────────────────────────────────────

def _metric_numeric_series(series: pd.Series) -> tuple[pd.Series | None, str]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0:
        return numeric, ""
    parsed = series.map(parse_ordered_range_value)
    if parsed.notna().sum() > 0:
        return parsed, f"Converted ordered/range values from '{series.name}' into approximate numeric midpoints."
    return None, ""


def _aggregation_title(aggregation: str) -> str:
    return {
        "sum": "Total",
        "mean": "Average",
        "median": "Median",
        "count": "Count",
        "min": "Minimum",
        "max": "Maximum",
    }.get(str(aggregation or "").lower(), str(aggregation or "Value").title())


def _normalize_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold().strip())

def _apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    result = df
    for f in filters:
        col = f.get("column")
        value = f.get("value")
        op = f.get("operator", "==")
        if not col or col not in df.columns or value is None:
            continue
        try:
            if op == "==":
                result = result[result[col] == value]
            elif op == "!=":
                result = result[result[col] != value]
            elif op == ">":
                result = result[pd.to_numeric(result[col], errors="coerce") > float(value)]
            elif op == "<":
                result = result[pd.to_numeric(result[col], errors="coerce") < float(value)]
            elif op == ">=":
                result = result[pd.to_numeric(result[col], errors="coerce") >= float(value)]
            elif op == "<=":
                result = result[pd.to_numeric(result[col], errors="coerce") <= float(value)]
        except Exception:
            continue
    return result


def _best_categorical_column(
    df: pd.DataFrame,
    exclude: set[str] | None = None,
    max_unique: int = 50,
) -> str | None:
    exclude = exclude or set()
    best, best_score = None, 0
    for col in df.columns:
        col_name = str(col)
        if col_name in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        nunique = df[col].nunique()
        if nunique < 2 or nunique > max_unique:
            continue
        if _is_identifier_col_name(col_name) and nunique > len(df) * 0.5:
            continue
        if _is_duration_column_values(df[col]):
            continue
        if nunique > best_score:
            best_score = nunique
            best = col_name
    return best


def _find_year_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        col_name = str(col)
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            desc = df[col].describe()
            smin, smax = float(desc.get("min", 0)), float(desc.get("max", 0))
            if 1900 <= smin <= 2100 and 1900 <= smax <= 2100:
                return col_name
        except (TypeError, ValueError):
            continue
    return None


def _find_duration_column(df: pd.DataFrame) -> str | None:
    dur_pat = re.compile(r"\d+\s*(min|hour|hr|sec|season|ep)", re.IGNORECASE)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        sample = [str(v) for v in df[col].dropna().head(10)]
        if sample and sum(1 for v in sample if dur_pat.search(v)) >= len(sample) * 0.4:
            return str(col)
    return None


def _is_year_column(col: str, df: pd.DataFrame) -> bool:
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return False
    try:
        desc = df[col].describe()
        return 1900 <= float(desc.get("min", 0)) <= 2100 and 1900 <= float(desc.get("max", 0)) <= 2100
    except (TypeError, ValueError):
        return False


def _is_identifier_col_name(name: str) -> bool:
    norm = name.lower().replace(" ", "_")
    return any(m in norm for m in ("_id", "id_", "uuid", "guid", "key", "index", "code"))


def _is_multi_label_column(series: pd.Series) -> bool:
    sample = [str(v) for v in series.dropna().head(20)]
    return len(sample) >= 3 and sum(1 for v in sample if "," in v) >= len(sample) * 0.3


def _explode_multi_label(df: pd.DataFrame, col: str) -> pd.DataFrame:
    working = df.copy()
    working[col] = working[col].astype(str).str.split(",")
    working = working.explode(col)
    working[col] = working[col].str.strip()
    working = working[working[col].str.len() > 0]
    return working


def _is_duration_column_values(series: pd.Series) -> bool:
    dur_pat = re.compile(r"\d+\s*(min|hour|hr|sec|season|ep)", re.IGNORECASE)
    sample = [str(v) for v in series.dropna().head(10)]
    return len(sample) >= 2 and sum(1 for v in sample if dur_pat.search(v)) >= len(sample) * 0.4


def _parse_duration_value(value: Any) -> float | None:
    text = str(value or "").strip().lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(min|hour|hr|sec|season|ep)", text)
    if match:
        num = float(match.group(1))
        unit = match.group(2)
        if unit in ("hour", "hr"):
            return num * 60
        if unit == "sec":
            return num / 60
        return num
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None
