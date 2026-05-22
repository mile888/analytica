from __future__ import annotations

import re
from typing import Any

import pandas as pd

from source.product.fallbacks.narration import _output, _timeline
from source.product.fallbacks.semantic_resolution import _looks_identifier_like
from source.product.fallbacks.validation import _is_distribution_question


def _correlation_strength(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 0.7:
        return "strong"
    if magnitude >= 0.4:
        return "moderate"
    if magnitude >= 0.2:
        return "weak"
    return "very weak"


def _outlier_response(question: str, df: pd.DataFrame, metric_col: str) -> dict[str, Any]:
    series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if series.empty:
        return _single_metric_response(question, df, metric_col)
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = pd.to_numeric(df[metric_col], errors="coerce").lt(lower) | pd.to_numeric(df[metric_col], errors="coerce").gt(upper)
    outliers = df.loc[mask].copy()
    outlier_count = int(mask.sum())
    share = outlier_count / max(int(series.count()), 1)
    preview_cols = [str(col) for col in df.columns[:8]]
    result_rows = outliers.sort_values(metric_col, ascending=False).head(25)[preview_cols].to_dict(orient="records")
    summary = (
        f"`{metric_col}` has {outlier_count} possible outlier{'' if outlier_count == 1 else 's'} "
        f"({share:.1%} of non-null values). Values above {upper:.2f} or below {lower:.2f} deserve review. "
        "These extremes may represent rare meaningful cases, data entry issues, or mixed populations rather than ordinary variation."
    )
    timeline = _timeline("outlier_check", metric_col=metric_col, outliers=outlier_count, rows=len(df))
    return _output(
        question=question,
        summary=summary,
        findings=[
            summary,
            f"The central range for `{metric_col}` is roughly {q1:.2f} to {q3:.2f}; extreme values outside the IQR fence may drive averages.",
            f"The evidence is stronger if the same outlier pattern appears inside a meaningful segment instead of only in isolated rows.",
        ],
        evidence=[f"Used the IQR rule on `{metric_col}` across {int(series.count()):,} non-null values."],
        limitations=["IQR flags unusual values statistically; domain review is needed before treating them as errors."],
        next_steps=[
            f"Break down `{metric_col}` outliers by an important category to see where they concentrate.",
            "Compare mean and median to understand whether outliers are skewing the metric.",
            "Review the largest extreme rows before using average-based conclusions in a report.",
        ],
        code=f"q1, q3 = df[{metric_col!r}].quantile([0.25, 0.75]); iqr = q3 - q1",
        result_preview=pd.DataFrame(result_rows).to_string(index=False) if result_rows else "",
        timeline=timeline,
        artifacts=[
            {
                "artifact_type": "table",
                "title": f"{metric_col} outlier candidates",
                "content": result_rows,
                "visibility": "user",
                "pinned": True,
                "metadata": {"metric": metric_col, "lower_bound": lower, "upper_bound": upper, "outlier_count": outlier_count},
            }
        ] if result_rows else [],
        trace_metadata={"fallback": "outlier_check", "metric": metric_col, "outlier_count": outlier_count},
    )


def _correlation_response(question: str, df: pd.DataFrame, metric_col: str) -> dict[str, Any]:
    specific = _specific_relationship_response(question, df)
    if specific:
        return specific
    numeric = df.select_dtypes(include="number").copy()
    for col in list(numeric.columns):
        if _looks_identifier_like(numeric[col], str(col)):
            numeric = numeric.drop(columns=[col])
    if metric_col not in numeric.columns or len(numeric.columns) < 2:
        return _single_metric_response(question, df, metric_col)
    correlations = numeric.corr(numeric_only=True)[metric_col].drop(labels=[metric_col], errors="ignore").dropna()
    if correlations.empty:
        return _single_metric_response(question, df, metric_col)
    rows = (
        correlations.abs()
        .sort_values(ascending=False)
        .rename("abs_correlation")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    rows["correlation"] = rows["column"].map(correlations.to_dict())
    result_rows = rows.head(12).to_dict(orient="records")
    strongest = result_rows[0]
    strongest_column = str(strongest["column"])
    direction = "positive" if float(strongest["correlation"]) > 0 else "negative"
    strength = _correlation_strength(float(strongest["correlation"]))
    summary = (
        f"`{strongest_column}` has the strongest {direction} relationship with `{metric_col}` "
        f"(correlation {float(strongest['correlation']):.2f}, {strength}). "
        "This is an association, not a causal result; it may reflect segment mix, outliers, or another hidden driver."
    )
    timeline = _timeline("correlation_check", metric_col=metric_col, compared_columns=len(numeric.columns) - 1)
    scatter_rows = (
        df[[strongest_column, metric_col]]
        .dropna()
        .head(80)
        .rename(columns={strongest_column: "x", metric_col: "y"})
        .to_dict(orient="records")
    )
    return _output(
        question=question,
        summary=summary,
        findings=[
            summary,
            f"The relationship is strongest among the available numeric fields, but it needs a segment check before treating `{strongest_column}` as a stable driver.",
        ],
        evidence=[f"Computed pairwise Pearson correlations among {len(numeric.columns)} numeric fields."],
        limitations=["Correlation describes association, not causality; missing values and outliers can affect the result."],
        next_steps=[
            f"Use the `{metric_col}` vs `{strongest['column']}` scatter plot to inspect whether a few extreme points dominate the relationship.",
            "Compare the relationship inside important segments to see whether the signal remains stable.",
            f"Check whether another field explains both `{metric_col}` and `{strongest_column}` before making a decision.",
        ],
        code=f"result = df.select_dtypes(include='number').corr()[{metric_col!r}].sort_values(key=abs, ascending=False)",
        result_preview=pd.DataFrame(result_rows).to_string(index=False),
        timeline=timeline,
        artifacts=[
            {
                "artifact_type": "table",
                "title": f"Correlations with {metric_col}",
                "content": result_rows,
                "visibility": "user",
                "pinned": True,
                "metadata": {"metric": metric_col},
            },
            {
                "artifact_type": "chart",
                "title": f"{metric_col} vs {strongest_column}",
                "content": {
                    "chart_type": "scatter",
                    "x": "x",
                    "y": "y",
                    "metric": metric_col,
                    "comparison": strongest_column,
                    "rows": scatter_rows,
                },
                "visibility": "user",
                "pinned": True,
                "metadata": {"metric": metric_col, "comparison": strongest_column},
            }
        ],
        trace_metadata={"fallback": "correlation_check", "metric": metric_col},
    )


def _specific_relationship_response(question: str, df: pd.DataFrame) -> dict[str, Any] | None:
    targets = _relationship_targets(question)
    if not targets:
        return None
    left_label, right_label = targets
    left_column, left_growth = _resolve_relationship_target(left_label, df)
    right_column, right_growth = _resolve_relationship_target(right_label, df)
    alternatives = _numeric_alternatives(df)
    if left_growth or right_growth:
        growth_label = left_label if left_growth else right_label
        growth_metric = left_column if left_growth else right_column
        companion = right_column if left_growth else left_column
        if not growth_metric:
            return _relationship_refusal(question, f"I cannot compute `{growth_label}` because that metric-like target is not present.", alternatives)
        if not companion:
            missing = right_label if left_growth else left_label
            return _relationship_refusal(question, f"I cannot test `{growth_label}` against `{missing}` because `{missing}` does not resolve to a numeric field.", alternatives)
        time_column = _relationship_time_column(df)
        if not time_column:
            return _relationship_refusal(
                question,
                f"I can analyze `{growth_metric}` against `{companion}`, but not `{growth_metric}` growth, because no reliable time field is available.",
                alternatives,
            )
        return _growth_relationship_response(question, df, growth_metric, companion, time_column)
    if not left_column:
        return _relationship_refusal(question, f"I cannot test the requested relationship because `{left_label}` does not resolve to a numeric field.", alternatives)
    if not right_column:
        return _relationship_refusal(question, f"I cannot test the requested relationship because `{right_label}` does not resolve to a numeric field.", alternatives)
    if left_column == right_column:
        return None
    return _locked_pair_correlation_response(question, df, left_column, right_column)


def _relationship_targets(question: str) -> tuple[str, str] | None:
    text = _normalize_for_match(question)
    between = re.search(r"\brelationship\s+between\s+(.+?)\s+and\s+(.+)$", text)
    if between:
        return _clean_target_label(between.group(1)), _clean_target_label(between.group(2))
    related = re.search(r"^(?:is|are|does|do|can)?\s*(.+?)\s+(?:related\s+to|correlated\s+with|correlate\s+with)\s+(.+)$", text)
    if related:
        return _clean_target_label(related.group(1)), _clean_target_label(related.group(2))
    return None


def _clean_target_label(value: str) -> str:
    words = [
        word
        for word in re.sub(r"[^\w\s]", " ", value).split()
        if word not in {"the", "a", "an", "is", "are", "does", "do", "metric"}
    ]
    return " ".join(words)


def _resolve_relationship_target(label: str, df: pd.DataFrame) -> tuple[str | None, bool]:
    needs_growth = "growth" in _normalize_for_match(label)
    cleaned = _clean_target_label(label.replace("growth", ""))
    columns = [str(col) for col in df.select_dtypes(include="number").columns if not _looks_identifier_like(df[col], str(col))]
    terms = _target_terms(cleaned)
    for column in columns:
        if _target_terms(column) & terms:
            return column, needs_growth
    return None, needs_growth


def _locked_pair_correlation_response(question: str, df: pd.DataFrame, left: str, right: str) -> dict[str, Any]:
    working = df[[left, right]].copy()
    working[left] = pd.to_numeric(working[left], errors="coerce")
    working[right] = pd.to_numeric(working[right], errors="coerce")
    working = working.dropna()
    if len(working) < 2:
        return _relationship_refusal(question, f"I cannot compute `{left}` vs `{right}` because fewer than two paired numeric rows are available.", _numeric_alternatives(df))
    corr = float(working[left].corr(working[right]))
    if pd.isna(corr):
        corr = 0.0
    direction = "positive" if corr > 0 else "negative" if corr < 0 else "flat"
    strength = _correlation_strength(corr)
    summary = (
        f"`{left}` and `{right}` show a {strength} {direction} relationship "
        f"(Pearson correlation {corr:.2f}) across {len(working):,} paired rows. "
        "This tests only the requested pair; it is an association, not a causal result."
    )
    rows = working.head(80).rename(columns={left: "x", right: "y"}).to_dict(orient="records")
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Computed Pearson correlation for the requested fields `{left}` and `{right}` only."],
        limitations=["Correlation describes association, not causality; missing values and outliers can affect the result."],
        next_steps=[f"Inspect the `{left}` vs `{right}` scatter plot for non-linear patterns and outliers."],
        code=f"df[[{left!r}, {right!r}]].corr().iloc[0, 1]",
        result_preview=pd.DataFrame([{"left": left, "right": right, "correlation": corr, "n": len(working)}]).to_string(index=False),
        timeline=_timeline("correlation_check", metric_col=left, compared_columns=1),
        artifacts=[
            {"artifact_type": "table", "title": f"{left} vs {right} correlation", "content": [{"left": left, "right": right, "correlation": corr, "n": len(working)}], "visibility": "user", "pinned": True, "metadata": {"metric": left, "comparison": right, "analysis_type": "specific_relationship"}},
            {"artifact_type": "chart", "title": f"{left} vs {right}", "content": {"chart_type": "scatter", "x": "x", "y": "y", "metric": left, "comparison": right, "rows": rows}, "visibility": "user", "pinned": True, "metadata": {"metric": left, "comparison": right, "analysis_type": "specific_relationship"}},
        ],
        trace_metadata={"fallback": "correlation_check", "analysis_type": "specific_relationship", "metric": left, "comparison": right},
    )


def _growth_relationship_response(question: str, df: pd.DataFrame, metric: str, companion: str, time_column: str) -> dict[str, Any]:
    working = df[[metric, companion, time_column]].copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    working[companion] = pd.to_numeric(working[companion], errors="coerce")
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    working = working.dropna()
    if working.empty:
        return _relationship_refusal(question, f"I cannot compute `{metric}` growth vs `{companion}` because paired numeric/time rows are unavailable.", _numeric_alternatives(df))
    working["period"] = working[time_column].dt.to_period("M").astype(str)
    grouped = working.groupby("period", dropna=False).agg(metric_value=(metric, "mean"), companion_value=(companion, "mean"), n=(metric, "count")).reset_index()
    grouped = grouped.sort_values("period")
    grouped["metric_growth"] = grouped["metric_value"].diff()
    comparable = grouped.dropna(subset=["metric_growth", "companion_value"])
    if len(comparable) < 2:
        return _relationship_refusal(question, f"I found `{time_column}`, but there are not enough time periods to compute `{metric}` growth vs `{companion}`.", _numeric_alternatives(df))
    corr = float(comparable["metric_growth"].corr(comparable["companion_value"]))
    if pd.isna(corr):
        corr = 0.0
    strength = _correlation_strength(corr)
    direction = "positive" if corr > 0 else "negative" if corr < 0 else "flat"
    rows = comparable.to_dict(orient="records")
    summary = (
        f"`{metric}` growth is tested against `{companion}` using `{time_column}` periods only. "
        f"The relationship is {strength} and {direction} (Pearson correlation {corr:.2f}) across {len(comparable):,} comparable periods."
    )
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Computed period-to-period `{metric}` growth, then correlated it with period average `{companion}`."],
        limitations=["Period-level growth can be unstable when period row counts are small."],
        next_steps=[f"Check whether the `{metric}` growth relationship with `{companion}` holds at a different time grain."],
        code=f"grouped = df.groupby(period)[[{metric!r}, {companion!r}]].mean(); grouped[{metric!r}].diff().corr(grouped[{companion!r}])",
        result_preview=pd.DataFrame(rows).to_string(index=False),
        timeline=_timeline("correlation_check", metric_col=metric, compared_columns=1),
        artifacts=[{"artifact_type": "table", "title": f"{metric} growth vs {companion}", "content": rows, "visibility": "user", "pinned": True, "metadata": {"metric": metric, "comparison": companion, "time_axis": time_column, "analysis_type": "specific_growth_relationship"}}],
        trace_metadata={"fallback": "correlation_check", "analysis_type": "specific_growth_relationship", "metric": metric, "comparison": companion, "time_axis": time_column},
    )


def _relationship_refusal(question: str, summary: str, alternatives: list[str]) -> dict[str, Any]:
    alt_text = ", ".join(f"`{item}`" for item in alternatives[:5])
    if alt_text:
        summary = f"{summary} Available numeric alternatives include {alt_text}."
    return _output(
        question=question,
        summary=summary,
        findings=[],
        evidence=[],
        limitations=[summary],
        next_steps=["Choose one of the available numeric fields if you want a different relationship test."],
        code="",
        result_preview="",
        timeline=_timeline("relationship_target_unavailable"),
        artifacts=[],
        trace_metadata={"fallback": "relationship_target_unavailable", "analysis_type": "relationship_target_unavailable", "suppress_key_findings": True},
    )


def _relationship_time_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            return str(column)
    for column in df.columns:
        normalized = _normalize_for_match(str(column))
        if any(marker in normalized for marker in ("date", "time", "timestamp", "period")):
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().mean() >= 0.7:
                return str(column)
    return None


def _numeric_alternatives(df: pd.DataFrame) -> list[str]:
    return [str(col) for col in df.select_dtypes(include="number").columns if not _looks_identifier_like(df[col], str(col))]


def _target_terms(value: str) -> set[str]:
    terms: set[str] = set()
    for raw in _normalize_for_match(value).split():
        if len(raw) < 3:
            continue
        terms.add(raw)
        if raw.endswith("ies") and len(raw) > 4:
            terms.add(raw[:-3] + "y")
        if raw.endswith("s") and len(raw) > 3:
            terms.add(raw[:-1])
    return terms


def _normalize_for_match(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").casefold().split())


def _trend_response(
    question: str,
    df: pd.DataFrame,
    metric_col: str,
    timestamp_col: str,
    *,
    chart_requested: bool = False,
) -> dict[str, Any]:
    working = df[[timestamp_col, metric_col]].copy()
    working[timestamp_col] = pd.to_datetime(working[timestamp_col], errors="coerce")
    working[metric_col] = pd.to_numeric(working[metric_col], errors="coerce")
    working = working.dropna()
    if working.empty:
        return _single_metric_response(question, df, metric_col)
    working["period"] = working[timestamp_col].dt.to_period("M").dt.to_timestamp()
    trend = working.groupby("period")[metric_col].agg(["count", "mean", "sum"]).reset_index()
    trend = trend.sort_values("period")
    first = trend.iloc[0]
    last = trend.iloc[-1]
    change = float(last["mean"] - first["mean"])
    direction = "increased" if change > 0 else "decreased" if change < 0 else "stayed roughly flat"
    trend["mean_delta"] = trend["mean"].diff()
    largest_shift = trend.dropna(subset=["mean_delta"]).assign(abs_delta=lambda data: data["mean_delta"].abs())
    largest_shift_row = largest_shift.sort_values("abs_delta", ascending=False).iloc[0] if not largest_shift.empty else None
    summary = (
        f"`{metric_col}` {direction} over time using `{timestamp_col}`. "
        f"The average moved from {first['mean']:.2f} to {last['mean']:.2f} across the observed period. "
        "This may indicate a real directional shift, but the pattern should be checked for sparse periods, seasonality, and one-off spikes."
    )
    if largest_shift_row is not None:
        summary += (
            f" The sharpest period change occurs around {largest_shift_row['period'].strftime('%Y-%m-%d')} "
            f"({float(largest_shift_row['mean_delta']):+.2f})."
        )
    rows = trend.assign(period=trend["period"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
    timeline = _timeline("trend_check", metric_col=metric_col, timestamp_col=timestamp_col, periods=len(rows))
    return _output(
        question=question,
        summary=summary,
        findings=[
            summary,
            f"The trend evidence is stronger when period counts are stable; sparse periods can exaggerate movement in `{metric_col}`.",
        ],
        evidence=[f"Aggregated `{metric_col}` by monthly periods from `{timestamp_col}` over {len(working):,} valid rows."],
        limitations=["Monthly aggregation can hide shorter spikes; use a finer period if the data supports it."],
        next_steps=[
            "Inspect the largest period-to-period change as a possible turning point.",
            "Compare the time trend across a key category if one is available.",
            "Check whether the pattern persists after excluding sparse periods or extreme rows.",
        ],
        code=f"result = df.assign({timestamp_col}=pd.to_datetime(df[{timestamp_col!r}], errors='coerce')).groupby(pd.Grouper(key={timestamp_col!r}, freq='M'))[{metric_col!r}].agg(['count','mean','sum'])",
        result_preview=pd.DataFrame(rows).to_string(index=False),
        timeline=timeline,
        artifacts=[
            {
                "artifact_type": "chart",
                "title": f"{metric_col} trend over time",
                "content": {
                    "chart_type": "line",
                    "x": "period",
                    "y": "mean",
                    "metric": metric_col,
                    "timestamp": timestamp_col,
                    "rows": rows,
                },
                "visibility": "user",
                "pinned": True,
                "metadata": {"metric": metric_col, "timestamp": timestamp_col, "chart_requested": chart_requested},
            },
            {
                "artifact_type": "table",
                "title": f"{metric_col} monthly trend",
                "content": rows,
                "visibility": "user",
                "metadata": {"metric": metric_col, "timestamp": timestamp_col},
            },
        ],
        trace_metadata={"fallback": "trend_check", "metric": metric_col, "timestamp": timestamp_col},
    )


def _single_metric_response(
    question: str,
    df: pd.DataFrame,
    metric_col: str,
    *,
    chart_requested: bool = False,
) -> dict[str, Any]:
    series = df[metric_col].dropna()
    summary_table = pd.DataFrame(
        [
            {
                "metric": metric_col,
                "count": int(series.count()),
                "sum": float(series.sum()) if len(series) else 0.0,
                "mean": float(series.mean()) if len(series) else 0.0,
                "median": float(series.median()) if len(series) else 0.0,
                "min": float(series.min()) if len(series) else 0.0,
                "max": float(series.max()) if len(series) else 0.0,
            }
        ]
    )
    result_preview = summary_table.to_string(index=False)
    summary = (
        f"`{metric_col}` has an average of {summary_table.iloc[0]['mean']:.2f}, "
        f"a median of {summary_table.iloc[0]['median']:.2f}, and ranges from "
        f"{summary_table.iloc[0]['min']:.2f} to {summary_table.iloc[0]['max']:.2f}. "
        "The gap between mean, median, and extremes shows whether the metric is stable or skewed by unusual records."
    )
    timeline = _timeline("deterministic_pandas_fallback", metric_col=metric_col, rows=len(df))
    histogram_rows = _histogram_rows(series, bins=10)
    artifacts = [
        {
            "artifact_type": "table",
            "title": f"{metric_col} summary",
            "content": summary_table.to_dict(orient="records"),
            "visibility": "user",
            "pinned": True,
            "metadata": {"preview": result_preview},
        }
    ]
    if chart_requested or _is_distribution_question(question):
        artifacts.append(
            {
                "artifact_type": "chart",
                "title": f"{metric_col} distribution",
                "content": {
                    "chart_type": "histogram",
                    "x": "bin",
                    "y": "count",
                    "metric": metric_col,
                    "rows": histogram_rows,
                },
                "visibility": "user",
                "pinned": True,
                "metadata": {"metric": metric_col, "recommended": True},
            }
        )
    return _output(
        question=question,
        summary=summary,
        findings=[
            f"`{metric_col}` has mean {summary_table.iloc[0]['mean']:.2f} and total {summary_table.iloc[0]['sum']:.2f}.",
            f"The evidence is descriptive until `{metric_col}` is segmented by a category or time field.",
        ],
        evidence=[f"Summarized `{metric_col}` over {len(series)} non-null rows."],
        limitations=["No suitable categorical/date dimension was found for comparison."],
        next_steps=["Add a categorical or time dimension to explain what drives this metric."],
        code=f"result = df[{metric_col!r}].describe()",
        result_preview=result_preview,
        timeline=timeline,
        artifacts=artifacts,
        trace_metadata={"fallback": "deterministic_pandas", "metric": metric_col},
    )


def _histogram_rows(series: pd.Series, bins: int = 10) -> list[dict[str, Any]]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return []
    counts = pd.cut(clean, bins=min(bins, max(1, clean.nunique())), include_lowest=True).value_counts().sort_index()
    return [
        {"bin": f"{interval.left:.2f} to {interval.right:.2f}", "count": int(count)}
        for interval, count in counts.items()
    ]
