from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd


class BranchType(StrEnum):
    DATASET_OVERVIEW = "dataset_overview"
    GROUPED_COMPARISON = "grouped_comparison"
    TREND_ANALYSIS = "trend_analysis"
    TEMPORAL_DECOMPOSITION = "temporal_decomposition"
    ANOMALY_INVESTIGATION = "anomaly_investigation"
    TRANSFORMATION_ANALYSIS = "transformation_analysis"
    VOLUME_EXPLANATION = "volume_explanation"
    HYPOTHESIS_VALIDATION = "hypothesis_validation"
    DATA_QUALITY = "data_quality"
    REPORT_GENERATION = "report_generation"


@dataclass(frozen=True)
class AnalyticalBranch:
    branch_type: BranchType
    objective: str = ""
    active_metric: str = ""
    active_dimension: str = ""
    active_time_axis: str = ""
    active_chart: str = ""
    active_transformation: str = ""
    active_hypothesis: str = ""
    active_validation_target: str = ""
    decomposition_dimensions: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    last_answer_summary: str = ""
    confidence: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "branch_type": self.branch_type.value,
            "objective": self.objective,
            "active_metric": self.active_metric,
            "active_dimension": self.active_dimension,
            "active_time_axis": self.active_time_axis,
            "active_chart": self.active_chart,
            "active_transformation": self.active_transformation,
            "active_hypothesis": self.active_hypothesis,
            "active_validation_target": self.active_validation_target,
            "decomposition_dimensions": self.decomposition_dimensions,
            "unresolved_questions": self.unresolved_questions,
            "last_answer_summary": self.last_answer_summary,
            "confidence": self.confidence,
        }


def route_branch_intent(question: str, state: dict[str, Any] | None = None) -> str:
    text = _normalize(question)
    branch = str((state or {}).get("active_branch_type") or (state or {}).get("branch_type") or "")
    temporal_context = branch in {BranchType.TREND_ANALYSIS.value, BranchType.TEMPORAL_DECOMPOSITION.value} or bool(
        (state or {}).get("active_time_axis")
    )
    if temporal_context and any(marker in text for marker in ("shipping", "delivery", "logistics", "fulfillment", "достав")):
        return "temporal_behavior_decomposition"
    if temporal_context and any(marker in text for marker in ("categor", "segment", "category", "категор", "сегмент", "объясняют рост", "explain growth")):
        return "temporal_decomposition"
    if temporal_context and any(marker in text for marker in ("season", "seasonality", "сезон", "сезонность")):
        return "seasonality_check"
    if temporal_context and any(marker in text for marker in ("anomalous period", "anomalous periods", "period anomalies", "аномальн период", "аномальные период", "скачк", "провал")):
        return "temporal_anomalies"
    if temporal_context and any(marker in text for marker in ("strongest growth", "growth", "рост", "прирост", "where", "где")):
        return "strongest_growth_periods"
    if any(marker in text for marker in ("trend", "over time", "time series", "динамик", "во времени", "по времени")):
        return "trend_summary"
    return "unknown"


def trend_summary_response(question: str, df: pd.DataFrame, metric_col: str, timestamp_col: str) -> dict[str, Any]:
    trend = _monthly_trend(df, metric_col, timestamp_col)
    if trend.empty:
        return _clarify_time_response(question, metric_col)
    first = trend.iloc[0]
    last = trend.iloc[-1]
    change = float(last["mean"] - first["mean"])
    direction = "increased" if change > 0 else "decreased" if change < 0 else "stayed roughly flat"
    strongest = _strongest_period_rows(trend, positive=True).head(1)
    strongest_text = _format_period_change(strongest.iloc[0]) if not strongest.empty else "no period-to-period movement"
    rows = _trend_rows(trend)
    summary = (
        f"`{metric_col}` {direction} over time using `{timestamp_col}`. "
        f"The monthly average moved from {float(first['mean']):.2f} to {float(last['mean']):.2f}; "
        f"the strongest positive period change is {strongest_text}. "
        "This branch is now a temporal analysis, so follow-ups will stay anchored to the time trend unless you switch topics."
    )
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Aggregated `{metric_col}` by month using `{timestamp_col}` over {int(trend['count'].sum())} valid rows."],
        limitations=["Monthly aggregation can hide shorter spikes and sparse periods can exaggerate movement."],
        artifacts=_trend_artifacts(metric_col, timestamp_col, rows, branch_type=BranchType.TREND_ANALYSIS.value),
        trace_metadata={"fallback": "analytical_branch", "analysis_type": "trend_check", "branch_type": BranchType.TREND_ANALYSIS.value, "metric": metric_col, "timestamp": timestamp_col},
    )


def strongest_growth_response(question: str, df: pd.DataFrame, metric_col: str, timestamp_col: str) -> dict[str, Any]:
    trend = _monthly_trend(df, metric_col, timestamp_col)
    growth = _strongest_period_rows(trend, positive=True).head(5)
    decline = _strongest_period_rows(trend, positive=False).head(3)
    summary = (
        f"Using monthly average `{metric_col}` and month-over-month differences from `{timestamp_col}`, "
        f"Strongest growth in `{metric_col}` over `{timestamp_col}` occurs in: {_format_period_rows(growth)}. "
        f"The sharpest declines are: {_format_period_rows(decline)}. "
        "Read these as period-to-period movements; sparse months or one-off large records can exaggerate growth."
    )
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Computed month-to-month change in average and total `{metric_col}`."],
        limitations=["Growth periods are descriptive and should be checked by volume and mix."],
        artifacts=[
            {"artifact_type": "table", "title": f"{metric_col} growth periods", "content": _records(growth), "visibility": "user", "metadata": {"branch_type": BranchType.TREND_ANALYSIS.value, "metric": metric_col, "timestamp": timestamp_col}},
            *_trend_artifacts(metric_col, timestamp_col, _trend_rows(trend), branch_type=BranchType.TREND_ANALYSIS.value),
        ],
        trace_metadata={"fallback": "analytical_branch", "analysis_type": "strongest_growth_periods", "branch_type": BranchType.TREND_ANALYSIS.value, "metric": metric_col, "timestamp": timestamp_col},
    )


def temporal_decomposition_response(
    question: str,
    df: pd.DataFrame,
    metric_col: str,
    timestamp_col: str,
    dimension_col: str,
) -> dict[str, Any]:
    monthly = _monthly_by_dimension(df, metric_col, timestamp_col, dimension_col)
    if monthly.empty:
        return trend_summary_response(question, df, metric_col, timestamp_col)
    first_period = monthly["period"].min()
    last_period = monthly["period"].max()
    first = monthly[monthly["period"] == first_period][[dimension_col, "sum", "mean", "count"]].rename(columns={"sum": "first_sum", "mean": "first_mean", "count": "first_count"})
    last = monthly[monthly["period"] == last_period][[dimension_col, "sum", "mean", "count"]].rename(columns={"sum": "last_sum", "mean": "last_mean", "count": "last_count"})
    contribution = first.merge(last, on=dimension_col, how="outer").fillna(0)
    contribution["total_change"] = contribution["last_sum"] - contribution["first_sum"]
    contribution["avg_change"] = contribution["last_mean"] - contribution["first_mean"]
    contribution = contribution.sort_values("total_change", ascending=False)
    leaders = contribution.head(5)
    laggards = contribution.tail(3).sort_values("total_change")
    summary = (
        f"To explain the `{metric_col}` trend over `{timestamp_col}`, I decomposed growth by `{dimension_col}` over time rather than using a static average. "
        f"Largest positive contributors from first to last period: {_format_dimension_changes(leaders, dimension_col)}. "
        f"Largest negative contributors: {_format_dimension_changes(laggards, dimension_col)}. "
        "This says which groups explain the time movement, not just which groups have high average values."
    )
    rows = _records(contribution.head(25))
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Compared first and last monthly `{metric_col}` totals by `{dimension_col}`."],
        limitations=["First-vs-last decomposition is sensitive to endpoint months; validate with rolling or yearly windows."],
        artifacts=[
            {"artifact_type": "table", "title": f"{metric_col} trend decomposition by {dimension_col}", "content": rows, "visibility": "user", "metadata": {"branch_type": BranchType.TEMPORAL_DECOMPOSITION.value, "metric": metric_col, "timestamp": timestamp_col, "dimension": dimension_col}},
            {"artifact_type": "chart", "title": f"{metric_col} growth contribution by {dimension_col}", "content": {"chart_type": "bar", "x": dimension_col, "y": "total_change", "metric": metric_col, "timestamp": timestamp_col, "dimension": dimension_col, "rows": rows[:20]}, "visibility": "user", "pinned": True, "metadata": {"branch_type": BranchType.TEMPORAL_DECOMPOSITION.value, "metric": metric_col, "timestamp": timestamp_col, "dimension": dimension_col, "aggregation": "total_change"}},
        ],
        trace_metadata={"fallback": "analytical_branch", "analysis_type": "temporal_decomposition", "branch_type": BranchType.TEMPORAL_DECOMPOSITION.value, "metric": metric_col, "timestamp": timestamp_col, "dimension": dimension_col},
    )


def seasonality_response(question: str, df: pd.DataFrame, metric_col: str, timestamp_col: str) -> dict[str, Any]:
    working = _time_metric_frame(df, metric_col, timestamp_col)
    if working.empty:
        return _clarify_time_response(question, metric_col)
    working["month"] = working[timestamp_col].dt.month
    by_month = working.groupby("month")[metric_col].agg(count="count", mean="mean", total="sum").reset_index()
    by_month = by_month.sort_values("mean", ascending=False)
    spread = float(by_month["mean"].max() - by_month["mean"].min()) if not by_month.empty else 0.0
    top = by_month.head(3)
    bottom = by_month.tail(3).sort_values("mean")
    summary = (
        f"Seasonality check for `{metric_col}` over `{timestamp_col}`: strongest months by average are {_format_month_rows(top)}, "
        f"weakest months are {_format_month_rows(bottom)}. The month-of-year average spread is {spread:.2f}. "
        "This is a seasonality signal only if the same month pattern repeats across years; otherwise it may be driven by isolated spikes."
    )
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Grouped `{metric_col}` by month-of-year using `{timestamp_col}`."],
        limitations=["Seasonality is approximate without a formal multi-year decomposition and sparse-month checks."],
        artifacts=[
            {"artifact_type": "chart", "title": f"{metric_col} seasonality by month", "content": {"chart_type": "bar", "x": "month", "y": "mean", "metric": metric_col, "timestamp": timestamp_col, "rows": _records(by_month.sort_values("month"))}, "visibility": "user", "pinned": True, "metadata": {"branch_type": BranchType.TREND_ANALYSIS.value, "analysis_type": "seasonality", "metric": metric_col, "timestamp": timestamp_col}},
        ],
        trace_metadata={"fallback": "analytical_branch", "analysis_type": "seasonality_check", "branch_type": BranchType.TREND_ANALYSIS.value, "metric": metric_col, "timestamp": timestamp_col},
    )


def temporal_anomalies_response(question: str, df: pd.DataFrame, metric_col: str, timestamp_col: str) -> dict[str, Any]:
    trend = _monthly_trend(df, metric_col, timestamp_col)
    if trend.empty:
        return _clarify_time_response(question, metric_col)
    q1 = float(trend["mean"].quantile(0.25))
    q3 = float(trend["mean"].quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    anomalous = trend[(trend["mean"] < lower) | (trend["mean"] > upper)].copy()
    if anomalous.empty:
        trend["abs_delta"] = trend["mean"].diff().abs()
        anomalous = trend.sort_values("abs_delta", ascending=False).head(5)
    summary = (
        f"Anomalous periods for `{metric_col}` over `{timestamp_col}` are {_format_period_levels(anomalous.head(5))}. "
        f"I used monthly means with IQR bounds ({lower:.2f} to {upper:.2f}); if few periods breach the bounds, the largest jumps are treated as candidates. "
        "These are time-period anomalies, not segment-level outliers."
    )
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Scored monthly `{metric_col}` periods using IQR bounds and period-to-period movement."],
        limitations=["Sparse periods and large individual records can create apparent period anomalies."],
        artifacts=[
            {"artifact_type": "table", "title": f"Anomalous {metric_col} periods", "content": _records(anomalous.head(20)), "visibility": "user", "metadata": {"branch_type": BranchType.TREND_ANALYSIS.value, "analysis_type": "temporal_anomalies", "metric": metric_col, "timestamp": timestamp_col}},
        ],
        trace_metadata={"fallback": "analytical_branch", "analysis_type": "temporal_anomalies", "branch_type": BranchType.TREND_ANALYSIS.value, "metric": metric_col, "timestamp": timestamp_col},
    )


def temporal_behavior_response(
    question: str,
    df: pd.DataFrame,
    metric_col: str,
    timestamp_col: str,
    behavior_col: str,
) -> dict[str, Any]:
    order_date_col = _find_order_date_column(df, timestamp_col)
    ship_date_col = _find_ship_date_column(df)
    delay_summary = ""
    delay_rows: list[dict[str, Any]] = []
    if order_date_col and ship_date_col:
        delay_working = df[[order_date_col, ship_date_col, metric_col]].copy()
        delay_working[order_date_col] = pd.to_datetime(delay_working[order_date_col], errors="coerce")
        delay_working[ship_date_col] = pd.to_datetime(delay_working[ship_date_col], errors="coerce")
        delay_working[metric_col] = pd.to_numeric(delay_working[metric_col], errors="coerce")
        delay_working = delay_working.dropna(subset=[order_date_col, ship_date_col, metric_col])
        if not delay_working.empty:
            delay_working["delivery_delay_days"] = (delay_working[ship_date_col] - delay_working[order_date_col]).dt.days
            delay_working["delay_bucket"] = pd.cut(
                delay_working["delivery_delay_days"],
                bins=[-999, 2, 5, 999],
                labels=["0-2 days", "3-5 days", "6+ days"],
            ).astype(str)
            delay_grouped = (
                delay_working.groupby("delay_bucket", dropna=False)[metric_col]
                .agg(count="count", total="sum", mean="mean")
                .reset_index()
                .sort_values("total", ascending=False)
            )
            delay_rows = _records(delay_grouped)
            top_delay = delay_grouped.iloc[0]
            delay_summary = (
                f" Delivery delay adds an operational view: the largest delay bucket is `{top_delay['delay_bucket']}` "
                f"with total `{metric_col}` {float(top_delay['total']):.2f} across {int(top_delay['count'])} rows."
            )
    monthly = _monthly_by_dimension(df, metric_col, timestamp_col, behavior_col)
    if monthly.empty:
        return trend_summary_response(question, df, metric_col, timestamp_col)
    totals = monthly.groupby(behavior_col)["sum"].sum().sort_values(ascending=False)
    latest_period = monthly["period"].max()
    earliest_period = monthly["period"].min()
    mix = (
        monthly[monthly["period"].isin([earliest_period, latest_period])]
        .pivot_table(index=behavior_col, columns="period", values="sum", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    period_cols = [col for col in mix.columns if col != behavior_col]
    if len(period_cols) >= 2:
        mix["mix_change"] = mix[period_cols[-1]] - mix[period_cols[0]]
    else:
        mix["mix_change"] = 0.0
    mix = mix.sort_values("mix_change", ascending=False)
    summary = (
        f"To test whether the `{metric_col}` trend over time is related to shipping behavior, I used `{behavior_col}` as the shipping category rather than raw shipping timestamp values. "
        f"Largest overall `{behavior_col}` contributors are {', '.join(f'`{idx}` ({value:.2f})' for idx, value in totals.head(3).items())}. "
        f"Biggest positive mix shifts from first to last period: {_format_behavior_mix(mix.head(3), behavior_col)}. "
        "If these shifts line up with growth or decline periods, shipping behavior is a plausible partial explanation; otherwise it is mostly a parallel segmentation."
        f"{delay_summary} The shipping date field is treated as a timestamp for delay, not as a shipping-behavior category."
    )
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Compared monthly `{metric_col}` totals by `{behavior_col}` using `{timestamp_col}`."],
        limitations=["This is decomposition, not causal proof; shipping mix can be correlated with product, region, or customer mix."],
        artifacts=[
            {"artifact_type": "table", "title": f"{metric_col} over time by {behavior_col}", "content": _records(monthly.head(80)), "visibility": "user", "metadata": {"branch_type": BranchType.TEMPORAL_DECOMPOSITION.value, "metric": metric_col, "timestamp": timestamp_col, "dimension": behavior_col}},
            *([{"artifact_type": "table", "title": f"{metric_col} by delivery delay", "content": delay_rows, "visibility": "user", "metadata": {"branch_type": BranchType.TEMPORAL_DECOMPOSITION.value, "metric": metric_col, "timestamp": timestamp_col, "dimension": "delivery_delay_days", "evidence_role": "shipping_delay"}}] if delay_rows else []),
        ],
        trace_metadata={"fallback": "analytical_branch", "analysis_type": "temporal_behavior_decomposition", "branch_type": BranchType.TEMPORAL_DECOMPOSITION.value, "metric": metric_col, "timestamp": timestamp_col, "dimension": behavior_col},
    )


def choose_decomposition_dimension(df: pd.DataFrame, metric_col: str, timestamp_col: str, preferred: str | None = None) -> str | None:
    candidates = []
    for col in df.columns:
        name = str(col)
        if name in {metric_col, timestamp_col}:
            continue
        series = df[name]
        if pd.api.types.is_numeric_dtype(series):
            continue
        nunique = int(series.nunique(dropna=True))
        if 2 <= nunique <= max(20, min(80, int(len(df) * 0.5))):
            candidates.append(name)
    question_like = "category growth explain categories"
    ranked = sorted(candidates, key=lambda name: _semantic_dimension_priority(name, question_like, preferred))
    return ranked[0] if ranked else None


def _semantic_dimension_priority(name: str, question_text: str = "", preferred: str | None = None) -> tuple[int, int, str]:
    normalized = _normalize(name)
    preferred_norm = _normalize(preferred or "")
    if preferred and normalized == preferred_norm and not any(marker in normalized for marker in ("segment", "customer", "name")):
        return (0, 0, normalized)
    if any(marker in normalized for marker in ("product category", "category", "subcategory", "sub category", "department", "product line")):
        return (1, 0, normalized)
    if any(marker in normalized for marker in ("ship mode", "shipping mode", "delivery mode")):
        return (2, 0, normalized)
    if normalized == "segment" or "segment" in normalized:
        return (5, 0, normalized)
    if any(marker in normalized for marker in ("customer name", "name", "id", "code")):
        return (8, 0, normalized)
    return (3, 0, normalized)


def choose_behavior_dimension(df: pd.DataFrame, metric_col: str, timestamp_col: str, question: str) -> str | None:
    text = _normalize(question)
    behavior_markers = ("ship", "shipping", "delivery", "достав")
    for col in df.columns:
        name = str(col)
        normalized = _normalize(name)
        if name in {metric_col, timestamp_col}:
            continue
        if any(marker in text for marker in behavior_markers) and any(marker in normalized for marker in ("ship mode", "shipping mode", "delivery mode", "mode")):
            if not _looks_date_column(df[name], normalized):
                return name
    for col in df.columns:
        name = str(col)
        normalized = _normalize(name)
        if name in {metric_col, timestamp_col}:
            continue
        if any(marker in text for marker in behavior_markers) and any(marker in normalized for marker in ("ship", "delivery", "достав", "mode")):
            if not _looks_date_column(df[name], normalized):
                return name
    return choose_decomposition_dimension(df, metric_col, timestamp_col)


def _find_order_date_column(df: pd.DataFrame, timestamp_col: str) -> str | None:
    for col in df.columns:
        normalized = _normalize(str(col))
        if "order" in normalized and "date" in normalized:
            return str(col)
    return timestamp_col if timestamp_col in df.columns else None


def _find_ship_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        normalized = _normalize(str(col))
        if ("ship" in normalized or "delivery" in normalized) and "date" in normalized:
            return str(col)
    return None


def _looks_date_column(series: pd.Series, normalized_name: str) -> bool:
    if "date" in normalized_name or "time" in normalized_name:
        return True
    sample = series.dropna().head(30)
    if sample.empty:
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.75)


def _monthly_trend(df: pd.DataFrame, metric_col: str, timestamp_col: str) -> pd.DataFrame:
    working = _time_metric_frame(df, metric_col, timestamp_col)
    if working.empty:
        return pd.DataFrame()
    working["period"] = working[timestamp_col].dt.to_period("M").dt.to_timestamp()
    trend = working.groupby("period")[metric_col].agg(count="count", mean="mean", sum="sum").reset_index().sort_values("period")
    trend["mean_delta"] = trend["mean"].diff()
    trend["sum_delta"] = trend["sum"].diff()
    return trend


def _monthly_by_dimension(df: pd.DataFrame, metric_col: str, timestamp_col: str, dimension_col: str) -> pd.DataFrame:
    working = df[[timestamp_col, dimension_col, metric_col]].copy()
    working[timestamp_col] = pd.to_datetime(working[timestamp_col], errors="coerce")
    working[metric_col] = pd.to_numeric(working[metric_col], errors="coerce")
    working = working.dropna(subset=[timestamp_col, metric_col])
    if working.empty:
        return pd.DataFrame()
    working["period"] = working[timestamp_col].dt.to_period("M").dt.to_timestamp()
    return working.groupby(["period", dimension_col], dropna=False)[metric_col].agg(count="count", mean="mean", sum="sum").reset_index().sort_values(["period", dimension_col])


def _time_metric_frame(df: pd.DataFrame, metric_col: str, timestamp_col: str) -> pd.DataFrame:
    working = df[[timestamp_col, metric_col]].copy()
    working[timestamp_col] = pd.to_datetime(working[timestamp_col], errors="coerce")
    working[metric_col] = pd.to_numeric(working[metric_col], errors="coerce")
    return working.dropna(subset=[timestamp_col, metric_col])


def _strongest_period_rows(trend: pd.DataFrame, *, positive: bool) -> pd.DataFrame:
    if trend.empty or "mean_delta" not in trend:
        return pd.DataFrame()
    data = trend.dropna(subset=["mean_delta"]).copy()
    if positive:
        return data.sort_values("mean_delta", ascending=False)
    return data.sort_values("mean_delta")


def _trend_rows(trend: pd.DataFrame) -> list[dict[str, Any]]:
    data = trend.copy()
    if "period" in data:
        data["period"] = data["period"].dt.strftime("%Y-%m-%d")
    return _records(data)


def _trend_artifacts(metric_col: str, timestamp_col: str, rows: list[dict[str, Any]], *, branch_type: str) -> list[dict[str, Any]]:
    return [
        {
            "artifact_type": "chart",
            "title": f"{metric_col} trend over time",
            "content": {"chart_type": "line", "x": "period", "y": "mean", "metric": metric_col, "timestamp": timestamp_col, "rows": rows},
            "visibility": "user",
            "pinned": True,
            "metadata": {"metric": metric_col, "timestamp": timestamp_col, "branch_type": branch_type, "analysis_type": "trend_summary"},
        },
        {
            "artifact_type": "table",
            "title": f"{metric_col} monthly trend",
            "content": rows,
            "visibility": "user",
            "metadata": {"metric": metric_col, "timestamp": timestamp_col, "branch_type": branch_type},
        },
    ]


def _format_period_rows(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "no clear periods"
    return ", ".join(_format_period_change(row) for _, row in rows.iterrows())


def _format_period_change(row: pd.Series) -> str:
    return f"{row['period'].strftime('%Y-%m-%d')} ({float(row.get('mean_delta') or 0):+.2f} avg change, n={int(row.get('count') or 0)})"


def _format_period_levels(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "no clear anomalous periods"
    return ", ".join(f"{row['period'].strftime('%Y-%m-%d')} (avg {float(row.get('mean') or 0):.2f}, n={int(row.get('count') or 0)})" for _, row in rows.iterrows())


def _format_dimension_changes(rows: pd.DataFrame, dimension_col: str) -> str:
    if rows.empty:
        return "no clear groups"
    return ", ".join(f"`{row[dimension_col]}` ({float(row.get('total_change') or 0):+.2f} total change)" for _, row in rows.iterrows())


def _format_month_rows(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "no months"
    return ", ".join(f"{int(row['month'])} (avg {float(row['mean']):.2f}, n={int(row['count'])})" for _, row in rows.iterrows())


def _format_behavior_mix(rows: pd.DataFrame, dimension_col: str) -> str:
    if rows.empty:
        return "no clear mix shifts"
    return ", ".join(f"`{row[dimension_col]}` ({float(row.get('mix_change') or 0):+.2f})" for _, row in rows.iterrows())


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    clean = frame.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%d")
    clean = clean.astype(object).where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


def _clarify_time_response(question: str, metric_col: str) -> dict[str, Any]:
    return _output(
        question=question,
        summary=f"I need a valid time column to analyze `{metric_col}` over time.",
        findings=[],
        evidence=[],
        limitations=["No usable time axis was available."],
        artifacts=[],
        trace_metadata={"fallback": "analytical_branch", "analysis_type": "clarification_needed"},
    )


def _output(
    *,
    question: str,
    summary: str,
    findings: list[str],
    evidence: list[str],
    limitations: list[str],
    artifacts: list[dict[str, Any]],
    trace_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "final_answer": summary,
        "summary": summary,
        "code": "",
        "result_preview": "",
        "result_base64": "",
        "exec_error": None,
        "engine": "pandas",
        "needs_data": True,
        "use_case": "data_analytics",
        "loaded_skills": ["data-analysis", "temporal-analysis", "analytical-branches"],
        "tool_timeline": [{"tool": trace_metadata.get("analysis_type", "analytical_branch"), "status": "ok"}],
        "sql_metadata": {},
        "structured_report": {
            "question": question,
            "summary": summary,
            "key_findings": findings,
            "evidence": evidence,
            "limitations": limitations,
            "artifacts": [],
            "next_steps": [],
            "generated_code": "",
            "loaded_skills": ["data-analysis", "temporal-analysis", "analytical-branches"],
            "tool_timeline": [{"tool": trace_metadata.get("analysis_type", "analytical_branch"), "status": "ok"}],
            "sql_metadata": {},
        },
        "trace_metadata": trace_metadata,
        "critic_verdict": "",
        "critic_feedback": "",
        "artifacts": artifacts,
    }


def _normalize(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").lower().split())
