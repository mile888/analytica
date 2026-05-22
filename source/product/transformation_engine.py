from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from source.product.language_policy import ResponseLanguagePolicy


@dataclass(frozen=True)
class TransformationIntent:
    kind: str
    label: str


@dataclass(frozen=True)
class RankShiftAnalysis:
    dropped: list[dict[str, Any]] = field(default_factory=list)
    stable: list[dict[str, Any]] = field(default_factory=list)
    improved: list[dict[str, Any]] = field(default_factory=list)
    largest_shifts: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutlierDependenceAnalysis:
    outlier_driven: list[dict[str, Any]] = field(default_factory=list)
    robust_leaders: list[dict[str, Any]] = field(default_factory=list)
    fragile_leaders: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransformationImpactAnalysis:
    metric: str
    dimension: str
    transformation_type: str
    removed_rows: int = 0
    original_rows: int = 0
    filtered_rows: int = 0
    threshold: dict[str, float] = field(default_factory=dict)
    raw_ranking: list[dict[str, Any]] = field(default_factory=list)
    adjusted_ranking: list[dict[str, Any]] = field(default_factory=list)
    comparison_rows: list[dict[str, Any]] = field(default_factory=list)
    rank_shift: RankShiftAnalysis = field(default_factory=RankShiftAnalysis)
    outlier_dependence: OutlierDependenceAnalysis = field(default_factory=OutlierDependenceAnalysis)
    weaker_conclusions: list[str] = field(default_factory=list)
    stronger_conclusions: list[str] = field(default_factory=list)
    confidence_change: str = ""

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def detect_transformation_intent(question: str) -> TransformationIntent | None:
    text = _normalize(question)
    outlier_markers = (
        "remove outlier", "remove outliers", "without outlier", "without outliers", "excluding outlier",
        "exclude outlier", "extreme order", "extreme orders", "extreme value", "extreme values",
        "убрать выброс", "убрать выбросы", "без выброс", "исключить выброс", "экстремальн",
        "убрать extreme", "без extreme",
    )
    median_markers = ("median instead", "use median", "by median", "по медиан", "медиан")
    normalize_markers = ("normalize by volume", "normalized by volume", "per order", "per record", "на заказ", "на запись", "нормализ")
    sparse_markers = (
        "exclude sparse", "without sparse", "minimum sample", "min sample", "small sample",
        "маленькие выборки", "малые выборки", "убрать маленькие", "без маленьких", "минимальный размер",
    )
    stability_markers = ("stable", "stability", "rank shift", "ranking stay", "останутся", "стабил", "сохранится", "изменится рейтинг")
    if any(marker in text for marker in outlier_markers):
        return TransformationIntent("remove_outliers", "remove extreme values")
    if any(marker in text for marker in median_markers):
        return TransformationIntent("median_instead_of_mean", "use median instead of mean")
    if any(marker in text for marker in normalize_markers):
        return TransformationIntent("normalize_by_volume", "normalize by record volume")
    if any(marker in text for marker in sparse_markers):
        return TransformationIntent("exclude_sparse_groups", "exclude sparse groups")
    if any(marker in text for marker in stability_markers):
        return TransformationIntent("stability_check", "check ranking stability")
    if "а если" in text and any(marker in text for marker in ("убрать", "без", "исключ", "median", "медиан", "нормализ", "объем", "объём")):
        return TransformationIntent("remove_outliers", "conditional recomputation")
    return None


def is_transformation_question(question: str) -> bool:
    return detect_transformation_intent(question) is not None


def transformation_clarification_response(question: str) -> dict[str, Any]:
    language = ResponseLanguagePolicy.from_message(question)
    summary = (
        "Какую метрику и группировку использовать для условной проверки? Укажите поля из текущей таблицы."
        if language.is_russian
        else "Which metric and grouping should I use for this conditional check? Name fields from the current table."
    )
    return _output(
        question=question,
        summary=summary,
        findings=[],
        evidence=[],
        limitations=["No active metric/dimension context was available for this transformation."],
        next_steps=[],
        code="",
        result_preview="",
        timeline=[{"tool": "analytical_transformation_clarification", "status": "needs_context"}],
        artifacts=[],
        trace_metadata={"fallback": "analytical_transformation", "analysis_type": "clarification_needed"},
    )


def run_transformation_analysis(
    *,
    question: str,
    df: pd.DataFrame,
    metric_col: str,
    dimension_col: str,
) -> dict[str, Any] | None:
    intent = detect_transformation_intent(question)
    if intent is None:
        return None
    if metric_col not in df.columns or dimension_col not in df.columns:
        return transformation_clarification_response(question)
    if intent.kind == "remove_outliers":
        return _remove_outliers_response(question, df, metric_col, dimension_col)
    if intent.kind == "median_instead_of_mean":
        return _median_response(question, df, metric_col, dimension_col)
    if intent.kind == "normalize_by_volume":
        return _normalize_by_volume_response(question, df, metric_col, dimension_col)
    if intent.kind == "exclude_sparse_groups":
        return _exclude_sparse_groups_response(question, df, metric_col, dimension_col)
    if intent.kind == "stability_check":
        return _stability_response(question, df, metric_col, dimension_col)
    return None


def _remove_outliers_response(question: str, df: pd.DataFrame, metric_col: str, dimension_col: str) -> dict[str, Any]:
    working = _working_frame(df, metric_col, dimension_col)
    if working.empty:
        return transformation_clarification_response(question)
    original = _group_ranking(working, metric_col, dimension_col, aggregation="mean")
    threshold = _iqr_threshold(working[metric_col])
    filtered = working[(working[metric_col] >= threshold["lower"]) & (working[metric_col] <= threshold["upper"])].copy()
    if filtered.empty or len(filtered) == len(working):
        threshold = _top_percent_threshold(working[metric_col])
        filtered = working[working[metric_col] <= threshold["upper"]].copy()
    adjusted = _stabilize_adjusted_ranking(_group_ranking(filtered, metric_col, dimension_col, aggregation="mean"))
    comparison = _compare_rankings(original, adjusted, dimension_col)
    removed = len(working) - len(filtered)
    impact = _transformation_impact(
        metric_col=metric_col,
        dimension_col=dimension_col,
        transformation_type="remove_outliers",
        original=original,
        adjusted=adjusted,
        comparison=comparison,
        removed_rows=removed,
        original_rows=len(working),
        filtered_rows=len(filtered),
        threshold=threshold,
    )
    summary = _impact_summary(impact)
    result = comparison.head(30)
    return _output(
        question=question,
        summary=summary,
        findings=[
            f"Removed {removed} extreme `{metric_col}` rows and recomputed `{metric_col}` by `{dimension_col}`.",
            _first_finding(impact),
            "Rank shifts identify which original conclusions became weaker and which adjusted leaders are more credible.",
        ],
        evidence=[
            f"Original rows: {len(working):,}; filtered rows: {len(filtered):,}; removed rows: {removed:,}.",
            f"Threshold rule: IQR bounds {threshold['lower']:.2f} to {threshold['upper']:.2f}.",
        ],
        limitations=[
            "This removes row-level extremes in the active metric; it does not prove those records are errors.",
            "Sparse groups can still move a lot after filtering because one row changes their average.",
        ],
        next_steps=[
            f"Compare median `{metric_col}` by `{dimension_col}` to check whether the adjusted ranking is robust.",
            f"Exclude sparse `{dimension_col}` groups and see whether the leaders remain the same.",
        ],
        code=(
            f"q1 = df[{metric_col!r}].quantile(0.25); q3 = df[{metric_col!r}].quantile(0.75)\n"
            "iqr = q3 - q1\n"
            f"filtered = df[(df[{metric_col!r}] >= q1 - 1.5 * iqr) & (df[{metric_col!r}] <= q3 + 1.5 * iqr)]\n"
            f"result = filtered.groupby({dimension_col!r})[{metric_col!r}].agg(['count', 'mean', 'median', 'sum'])"
        ),
        result_preview=result.to_string(index=False),
        timeline=[
            {
                "tool": "analytical_transformation_remove_outliers",
                "status": "ok",
                "metric_col": metric_col,
                "dimension_col": dimension_col,
                "removed_rows": removed,
                "remaining_rows": len(filtered),
            }
        ],
        artifacts=[
            {
                "artifact_type": "table",
                "title": f"Before/after {metric_col} by {dimension_col}",
                "content": _records(result),
                "visibility": "user",
                "pinned": True,
                "metadata": {
                    "analysis_type": "remove_outliers",
                    "metric": metric_col,
                    "dimension": dimension_col,
                    "aggregation": "mean",
                    "base_metric": metric_col,
                    "base_dimension": dimension_col,
                    "base_aggregation": "mean",
                    "transformation_type": "remove_outliers",
                    "transformation_params": {"threshold": threshold, "removed_rows": removed},
                    "parent_artifact_id": "",
                    "base_artifact_id": "",
                },
            },
            {
                "artifact_type": "chart",
                "title": f"Adjusted average {metric_col} by {dimension_col}",
                "content": {
                    "chart_type": "bar",
                    "x": dimension_col,
                    "y": "adjusted_mean",
                    "metric": metric_col,
                    "rows": adjusted.head(20)[[dimension_col, "mean", "median", "count"]]
                    .rename(columns={"mean": "adjusted_mean"})
                    .pipe(_records),
                    "transformation_impact": impact.to_payload(),
                    "base_metric": metric_col,
                    "base_dimension": dimension_col,
                    "base_aggregation": "mean",
                    "transformation_type": "remove_outliers",
                    "transformation_params": {"threshold": threshold, "removed_rows": removed},
                    "ranking_scope": f"Top {min(len(adjusted), 20)} stable `{dimension_col}` groups by confidence-adjusted average `{metric_col}` after removing extremes.",
                    "row_count": int(len(filtered)),
                },
                "visibility": "user",
                "pinned": True,
                "metadata": {
                    "metric": metric_col,
                    "dimension": dimension_col,
                    "aggregation": "adjusted_mean",
                    "base_metric": metric_col,
                    "base_dimension": dimension_col,
                    "base_aggregation": "mean",
                    "transformation_type": "remove_outliers",
                    "transformation_params": {"threshold": threshold, "removed_rows": removed},
                    "parent_artifact_id": "",
                    "base_artifact_id": "",
                    "analysis_type": "remove_outliers",
                    "row_count": int(len(filtered)),
                    "recommended": True,
                    "transformation_impact": impact.to_payload(),
                },
            },
        ],
        trace_metadata={
            "fallback": "analytical_transformation",
            "analysis_type": "remove_outliers",
            "metric": metric_col,
            "dimension": dimension_col,
            "removed_rows": removed,
            "active_transformation_result": impact.to_payload(),
        },
    )


def _transformation_impact(
    *,
    metric_col: str,
    dimension_col: str,
    transformation_type: str,
    original: pd.DataFrame,
    adjusted: pd.DataFrame,
    comparison: pd.DataFrame,
    removed_rows: int,
    original_rows: int,
    filtered_rows: int,
    threshold: dict[str, float],
) -> TransformationImpactAnalysis:
    raw_records = _records(original.head(30))
    adjusted_records = _records(adjusted.head(30))
    comp = comparison.copy()
    comp["abs_rank_change"] = comp["rank_change"].abs()
    raw_top = comp[comp["original_rank"] <= 10].copy()
    collapsed = raw_top[(raw_top["adjusted_rank"] > 20) | (raw_top["rank_change"] >= 5) | (raw_top["adjusted_count"] == 0)]
    dropped = collapsed.sort_values(["rank_change", "original_rank"], ascending=[False, True]).head(8)
    stable = raw_top[(raw_top["adjusted_rank"] <= 10) & (raw_top["rank_change"].abs() <= 3)].sort_values(["adjusted_rank", "original_rank"]).head(8)
    improved = comp[(comp["adjusted_rank"] <= 10) & (comp["rank_change"] < 0)].sort_values(["adjusted_rank", "rank_change"]).head(8)
    shifts = comp.sort_values(["abs_rank_change", "original_rank"], ascending=[False, True]).head(8)
    outlier_driven = dropped.copy()
    if not outlier_driven.empty:
        outlier_driven["outlier_dependence"] = outlier_driven.apply(_dependence_label, axis=1)
    robust = adjusted.head(10).copy()
    robust_names = set(stable[dimension_col].astype(str).tolist())
    robust = robust[robust[dimension_col].astype(str).isin(robust_names)] if robust_names else robust.head(0)
    fragile = adjusted.head(10)[adjusted.head(10)["count"] <= 2].copy()
    weaker = []
    stronger = []
    if not dropped.empty:
        weaker.append(
            "The raw average-leader conclusion is weaker because top-ranked groups collapsed after extreme records were removed."
        )
    if not stable.empty:
        stronger.append(
            "Groups that stayed near the top after filtering became stronger candidates for real group strength."
        )
    if not fragile.empty:
        weaker.append(
            "Some adjusted leaders are still weakly supported because their sample sizes are very small."
        )
    confidence = f"{removed_rows:,} of {original_rows:,} rows were removed before recomputing the adjusted ranking."
    return TransformationImpactAnalysis(
        metric=metric_col,
        dimension=dimension_col,
        transformation_type=transformation_type,
        removed_rows=removed_rows,
        original_rows=original_rows,
        filtered_rows=filtered_rows,
        threshold=threshold,
        raw_ranking=raw_records,
        adjusted_ranking=adjusted_records,
        comparison_rows=_records(_authoritative_comparison_rows(comp, dropped=dropped, stable=stable, shifts=shifts)),
        rank_shift=RankShiftAnalysis(
            dropped=_records(dropped),
            stable=_records(stable),
            improved=_records(improved),
            largest_shifts=_records(shifts),
        ),
        outlier_dependence=OutlierDependenceAnalysis(
            outlier_driven=_records(outlier_driven),
            robust_leaders=_records(robust),
            fragile_leaders=_records(fragile),
        ),
        weaker_conclusions=weaker,
        stronger_conclusions=stronger,
        confidence_change=confidence,
    )


def _impact_summary(impact: TransformationImpactAnalysis) -> str:
    leaders = _format_adjusted_impact_rows(impact.adjusted_ranking[:5], impact.dimension)
    dropped = _format_impact_shift_rows(impact.rank_shift.dropped[:4], impact.dimension)
    stable = _format_impact_shift_rows(impact.rank_shift.stable[:4], impact.dimension)
    outlier_driven = _format_impact_shift_rows(impact.outlier_dependence.outlier_driven[:4], impact.dimension)
    interpretation = (
        "the original leaders mostly disappear, suggesting the raw ranking was driven by a few unusually large records"
        if impact.rank_shift.dropped
        else "the leading groups remain broadly similar after filtering, suggesting the ranking is less dependent on extreme records"
    )
    return (
        f"After removing extreme `{impact.metric}` records, the `{impact.dimension}` story changes: {interpretation}. "
        f"Removed {impact.removed_rows:,} of {impact.original_rows:,} rows using a robust threshold "
        f"({impact.threshold.get('lower', 0):.2f} to {impact.threshold.get('upper', 0):.2f}). "
        f"Among groups with enough remaining records, adjusted leaders are {leaders}. "
        f"Raw leaders that dropped: {dropped}. Stable leaders: {stable}. "
        f"Most outlier-dependent groups: {outlier_driven}."
    )


def _authoritative_comparison_rows(
    comparison: pd.DataFrame,
    *,
    dropped: pd.DataFrame,
    stable: pd.DataFrame,
    shifts: pd.DataFrame,
) -> pd.DataFrame:
    frames = [
        comparison.sort_values("original_rank").head(25),
        comparison.sort_values("adjusted_rank").head(25),
        dropped,
        stable,
        shifts,
    ]
    combined = pd.concat([frame for frame in frames if frame is not None and not frame.empty], ignore_index=True)
    if combined.empty:
        return comparison.head(50)
    return combined.drop_duplicates(subset=[comparison.columns[0]], keep="first").sort_values(["original_rank", "adjusted_rank"]).head(120)


def _first_finding(impact: TransformationImpactAnalysis) -> str:
    if impact.rank_shift.dropped:
        row = impact.rank_shift.dropped[0]
        return (
            f"`{row.get(impact.dimension)}` fell from rank #{int(float(row.get('original_rank') or 0))} "
            f"to #{int(float(row.get('adjusted_rank') or 0))}, so its raw leadership was outlier-sensitive."
        )
    if impact.rank_shift.stable:
        row = impact.rank_shift.stable[0]
        return (
            f"`{row.get(impact.dimension)}` stayed near the top after filtering, strengthening that group-level conclusion."
        )
    return "The transformation produced an adjusted ranking, but no material rank shift was detected among raw leaders."


def _dependence_label(row: pd.Series) -> str:
    pct = row.get("mean_change_pct")
    try:
        pct_value = abs(float(pct))
    except Exception:
        pct_value = 0.0
    if pct_value >= 0.75 or float(row.get("adjusted_count") or 0) == 0:
        return "high"
    if pct_value >= 0.35:
        return "medium"
    return "low"


def _median_response(question: str, df: pd.DataFrame, metric_col: str, dimension_col: str) -> dict[str, Any]:
    working = _working_frame(df, metric_col, dimension_col)
    ranking = _group_ranking(working, metric_col, dimension_col, aggregation="median")
    language = ResponseLanguagePolicy.from_message(question, protected_terms=df.columns)
    summary = (
        f"Если использовать median `{metric_col}` вместо mean, самые сильные группы `{dimension_col}`: {_format_rows_ru(ranking.head(5), dimension_col)}. Median меньше зависит от extreme records, поэтому группы, которые остаются наверху, устойчивее."
        if language.is_russian
        else f"Using median `{metric_col}` instead of mean, the strongest `{dimension_col}` groups are {_format_rows_en(ranking.head(5), dimension_col)}. Median is less sensitive to extreme records, so groups that stay high here are more stable."
    )
    return _simple_transformation_output(question, summary, ranking, metric_col, dimension_col, "median_instead_of_mean", "median")


def _normalize_by_volume_response(question: str, df: pd.DataFrame, metric_col: str, dimension_col: str) -> dict[str, Any]:
    working = _working_frame(df, metric_col, dimension_col)
    ranking = _group_ranking(working, metric_col, dimension_col, aggregation="mean")
    language = ResponseLanguagePolicy.from_message(question, protected_terms=df.columns)
    summary = (
        f"Нормализация по record volume означает читать average `{metric_col}` на строку, а не total `{metric_col}`. По этой логике сильнее всего группы `{dimension_col}`: {_format_rows_ru(ranking.head(5), dimension_col)}. Это отделяет высокий total из-за большого числа строк от реально высокого per-record value."
        if language.is_russian
        else f"Normalizing by record volume means reading average `{metric_col}` per row, not total `{metric_col}`. The strongest `{dimension_col}` groups on that basis are {_format_rows_en(ranking.head(5), dimension_col)}. This separates high totals caused by many records from groups with genuinely high per-record value."
    )
    return _simple_transformation_output(question, summary, ranking, metric_col, dimension_col, "normalize_by_volume", "mean")


def _exclude_sparse_groups_response(question: str, df: pd.DataFrame, metric_col: str, dimension_col: str) -> dict[str, Any]:
    working = _working_frame(df, metric_col, dimension_col)
    ranking = _group_ranking(working, metric_col, dimension_col, aggregation="mean")
    min_count = max(3, int(ranking["count"].median())) if not ranking.empty else 3
    filtered = ranking[ranking["count"] >= min_count]
    language = ResponseLanguagePolicy.from_message(question, protected_terms=df.columns)
    summary = (
        f"После исключения sparse групп `{dimension_col}` с менее чем {min_count} строками сильнее всего по `{metric_col}`: {_format_rows_ru(filtered.head(5), dimension_col)}. Рейтинг становится менее эффектным, но надежнее."
        if language.is_russian
        else f"After excluding sparse `{dimension_col}` groups below {min_count} rows, the strongest groups by `{metric_col}` are {_format_rows_en(filtered.head(5), dimension_col)}. This makes the ranking less flashy but more reliable."
    )
    return _simple_transformation_output(question, summary, filtered, metric_col, dimension_col, "exclude_sparse_groups", "mean")


def _stability_response(question: str, df: pd.DataFrame, metric_col: str, dimension_col: str) -> dict[str, Any]:
    working = _working_frame(df, metric_col, dimension_col)
    mean_rank = _group_ranking(working, metric_col, dimension_col, aggregation="mean")
    median_rank = _group_ranking(working, metric_col, dimension_col, aggregation="median")
    comparison = _compare_rankings(mean_rank, median_rank, dimension_col)
    stable = comparison[comparison["rank_change"].abs() <= 2].head(5)
    language = ResponseLanguagePolicy.from_message(question, protected_terms=df.columns)
    summary = (
        f"Проверка стабильности рейтинга: ближе всего между mean и median остаются {_format_rank_shift_rows(stable, dimension_col)}. Большие сдвиги означают, что исходный рейтинг чувствителен к extremes или sparse samples."
        if language.is_russian
        else f"Ranking stability check: the groups that stay closest between mean and median are {_format_rank_shift_rows(stable, dimension_col)}. Large shifts mean the original ranking is sensitive to extremes or sparse samples."
    )
    return _simple_transformation_output(question, summary, comparison, metric_col, dimension_col, "stability_check", "rank_change")


def _simple_transformation_output(
    question: str,
    summary: str,
    table: pd.DataFrame,
    metric_col: str,
    dimension_col: str,
    analysis_type: str,
    aggregation: str,
) -> dict[str, Any]:
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Recomputed `{metric_col}` by `{dimension_col}` using `{aggregation}`."],
        limitations=["This is a deterministic descriptive check over the current dataframe."],
        next_steps=[],
        code="",
        result_preview=table.head(30).to_string(index=False),
        timeline=[{"tool": f"analytical_transformation_{analysis_type}", "status": "ok", "metric_col": metric_col, "dimension_col": dimension_col}],
        artifacts=[
            {
                "artifact_type": "table",
                "title": f"{analysis_type}: {metric_col} by {dimension_col}",
                "content": _records(table.head(50)),
                "visibility": "user",
                "pinned": True,
                "metadata": {"analysis_type": analysis_type, "metric": metric_col, "dimension": dimension_col},
            }
        ],
        trace_metadata={"fallback": "analytical_transformation", "analysis_type": analysis_type, "metric": metric_col, "dimension": dimension_col},
    )


def _working_frame(df: pd.DataFrame, metric_col: str, dimension_col: str) -> pd.DataFrame:
    working = df[[dimension_col, metric_col]].copy()
    working[metric_col] = pd.to_numeric(working[metric_col], errors="coerce")
    return working.dropna(subset=[metric_col])


def _group_ranking(df: pd.DataFrame, metric_col: str, dimension_col: str, *, aggregation: str) -> pd.DataFrame:
    grouped = (
        df.groupby(dimension_col, dropna=False)[metric_col]
        .agg(count="count", mean="mean", median="median", total="sum", min="min", max="max")
        .reset_index()
    )
    grouped["spread"] = grouped["max"] - grouped["min"]
    grouped = grouped.sort_values([aggregation, "count"], ascending=[False, False]).reset_index(drop=True)
    grouped["rank"] = grouped.index + 1
    return grouped


def _stabilize_adjusted_ranking(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty or "count" not in ranking.columns or "mean" not in ranking.columns:
        return ranking
    min_group_n = 5
    stable = ranking[ranking["count"] >= min_group_n].copy()
    if stable.empty:
        fallback_n = 3 if (ranking["count"] >= 3).any() else 2
        stable = ranking[ranking["count"] >= fallback_n].copy()
    if stable.empty:
        stable = ranking.copy()
    stable["confidence_weight"] = (stable["count"].clip(upper=min_group_n) / float(min_group_n)) ** 2
    stable["adjusted_score"] = stable["mean"] * stable["confidence_weight"]
    stable = stable.sort_values(["adjusted_score", "count", "mean"], ascending=[False, False, False]).reset_index(drop=True)
    stable["rank"] = stable.index + 1
    return stable


def _compare_rankings(original: pd.DataFrame, adjusted: pd.DataFrame, dimension_col: str) -> pd.DataFrame:
    before = original[[dimension_col, "rank", "mean", "median", "count", "total"]].rename(
        columns={"rank": "original_rank", "mean": "original_mean", "median": "original_median", "count": "original_count", "total": "original_total"}
    )
    after = adjusted[[dimension_col, "rank", "mean", "median", "count", "total"]].rename(
        columns={"rank": "adjusted_rank", "mean": "adjusted_mean", "median": "adjusted_median", "count": "adjusted_count", "total": "adjusted_total"}
    )
    merged = before.merge(after, on=dimension_col, how="left")
    missing_rank = int(len(original) + 1)
    merged["adjusted_rank"] = merged["adjusted_rank"].fillna(missing_rank)
    merged["adjusted_count"] = merged["adjusted_count"].fillna(0)
    for column in ("adjusted_mean", "adjusted_median", "adjusted_total"):
        merged[column] = merged[column].fillna(0.0)
    merged["rank_change"] = merged["adjusted_rank"] - merged["original_rank"]
    merged["mean_change"] = merged["adjusted_mean"] - merged["original_mean"]
    merged["mean_change_pct"] = merged["mean_change"] / merged["original_mean"].replace(0, pd.NA)
    merged = merged.where(pd.notna(merged), None)
    return merged.sort_values(["adjusted_rank", "original_rank"]).reset_index(drop=True)


def _iqr_threshold(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return _top_percent_threshold(clean)
    return {"lower": q1 - 1.5 * iqr, "upper": q3 + 1.5 * iqr}


def _top_percent_threshold(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return {"lower": float(clean.min()), "upper": float(clean.quantile(0.99))}


def _format_rows_en(rows: pd.DataFrame, dimension_col: str) -> str:
    if rows.empty:
        return "no groups after filtering"
    return ", ".join(
        f"`{row[dimension_col]}` (avg {float(row['mean']):.2f}, median {float(row['median']):.2f}, n={int(row['count'])})"
        for _, row in rows.iterrows()
    )


def _format_rows_ru(rows: pd.DataFrame, dimension_col: str) -> str:
    if rows.empty:
        return "нет групп после фильтрации"
    return ", ".join(
        f"`{row[dimension_col]}` (avg {float(row['mean']):.2f}, median {float(row['median']):.2f}, n={int(row['count'])})"
        for _, row in rows.iterrows()
    )


def _format_change_rows(rows: pd.DataFrame, dimension_col: str) -> str:
    if rows.empty:
        return "no material changes"
    return ", ".join(
        f"`{row[dimension_col]}` ({float(row['original_mean']):.2f} -> {float(row['adjusted_mean']):.2f})"
        for _, row in rows.iterrows()
    )


def _format_rank_shift_rows(rows: pd.DataFrame, dimension_col: str) -> str:
    if rows.empty:
        return "no material shifts"
    return ", ".join(
        f"`{row[dimension_col]}` (rank {int(row['original_rank'])} -> {int(row['adjusted_rank'])})"
        for _, row in rows.iterrows()
    )


def _format_adjusted_impact_rows(rows: list[dict[str, Any]], dimension_col: str) -> str:
    if not rows:
        return "no groups after filtering"
    parts = []
    for row in rows:
        value = row.get(dimension_col)
        mean = row.get("mean", row.get("adjusted_mean", 0))
        count = row.get("count", row.get("adjusted_count", 0))
        parts.append(f"`{value}` (avg {float(mean or 0):.2f}, n={int(float(count or 0))})")
    return ", ".join(parts)


def _format_impact_shift_rows(rows: list[dict[str, Any]], dimension_col: str) -> str:
    if not rows:
        return "none among the top groups"
    parts = []
    for row in rows:
        value = row.get(dimension_col)
        original_rank = int(float(row.get("original_rank") or 0))
        adjusted_rank = int(float(row.get("adjusted_rank") or 0))
        original_mean = float(row.get("original_mean") or 0)
        adjusted_mean = float(row.get("adjusted_mean") or 0)
        count = int(float(row.get("adjusted_count") or row.get("original_count") or 0))
        parts.append(f"`{value}` (rank #{original_rank} -> #{adjusted_rank}, avg {original_mean:.2f} -> {adjusted_mean:.2f}, n={count})")
    return ", ".join(parts)


def _normalize(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").lower().split())


def _contains_cyrillic(value: str) -> bool:
    return any("а" <= char.lower() <= "я" or char.lower() == "ё" for char in str(value or ""))


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _output(
    *,
    question: str,
    summary: str,
    findings: list[str],
    evidence: list[str],
    limitations: list[str],
    next_steps: list[str],
    code: str,
    result_preview: str,
    timeline: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    trace_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "final_answer": summary,
        "summary": summary,
        "code": code,
        "result_preview": result_preview,
        "result_base64": "",
        "exec_error": None,
        "engine": "pandas",
        "needs_data": True,
        "use_case": "data_analytics",
        "loaded_skills": ["data-analysis", "csv-dataframe-analysis", "analytical-transformation"],
        "tool_timeline": timeline,
        "sql_metadata": {},
        "structured_report": {
            "question": question,
            "summary": summary,
            "key_findings": findings,
            "evidence": evidence,
            "limitations": limitations,
            "artifacts": [],
            "next_steps": next_steps,
            "generated_code": code,
            "loaded_skills": ["data-analysis", "csv-dataframe-analysis", "analytical-transformation"],
            "tool_timeline": timeline,
            "sql_metadata": {},
        },
        "trace_metadata": trace_metadata,
        "critic_verdict": "",
        "critic_feedback": "",
        "artifacts": artifacts,
    }
