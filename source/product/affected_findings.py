from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from source.product.evidence_resolution import ActiveAnalyticalTarget, active_target_from_payload, transformation_state_from_payload
from source.product.language_policy import DetectedLanguage, ResponseLanguagePolicy


@dataclass(frozen=True)
class AffectedFinding:
    finding_type: str
    impact: str
    severity: str = "medium"
    reason: str = ""


@dataclass(frozen=True)
class AffectedFindingsResult:
    summary: str
    findings: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)


class AffectedFindingsAnalyzer:
    @classmethod
    def analyze(
        cls,
        *,
        question: str,
        dataframe: pd.DataFrame,
        active_target: Any,
        branch_state: dict[str, Any] | None = None,
        recent_findings: list[str] | None = None,
    ) -> AffectedFindingsResult | None:
        if not is_affected_findings_question(question):
            return None
        target = active_target_from_payload(active_target) or ActiveAnalyticalTarget()
        state = branch_state if isinstance(branch_state, dict) else {}
        issue = _quality_issue(target, state, question)
        metric = _metric(target, state, dataframe, question)
        language = ResponseLanguagePolicy.from_message(
            question,
            protected_terms=[item for item in (metric, target.dimension, target.time_axis) if item],
            fallback=_language_fallback(target.response_language),
        )
        if issue == "duplicates":
            return _duplicate_affected_findings(question, dataframe, metric, target, language, recent_findings or [])
        if issue == "missingness":
            return _missingness_affected_findings(dataframe, metric, target, language)
        if issue == "outliers":
            return _outlier_affected_findings(dataframe, metric, target, language, state)
        return _generic_quality_affected_findings(dataframe, metric, target, language)


def is_affected_findings_question(question: str) -> bool:
    text = " ".join(str(question or "").casefold().split())
    asks_findings = any(marker in text for marker in ("finding", "findings", "вывод", "выводы", "инсайт", "insight"))
    asks_reliability = any(marker in text for marker in ("unreliable", "reliable", "weaken", "invalid", "ненадеж", "ненадёж", "слабее", "ломает", "искаж"))
    return asks_findings and asks_reliability


def _duplicate_affected_findings(
    question: str,
    df: pd.DataFrame,
    metric: str,
    target: ActiveAnalyticalTarget,
    language: ResponseLanguagePolicy,
    recent_findings: list[str],
) -> AffectedFindingsResult:
    duplicate_rows = int(df.duplicated(keep=False).sum())
    exact_excess = int(df.duplicated(keep="first").sum())
    row_rate = duplicate_rows / max(len(df), 1)
    inflation = 0.0
    original_total = dedup_total = 0.0
    if metric and metric in df.columns:
        original_total = float(pd.to_numeric(df[metric], errors="coerce").sum())
        dedup_total = float(pd.to_numeric(df.drop_duplicates()[metric], errors="coerce").sum())
        inflation = original_total - dedup_total
    inflation_rate = abs(inflation) / max(abs(original_total), 1.0)
    severity = "low" if inflation_rate < 0.01 and row_rate < 0.01 else "medium" if inflation_rate < 0.05 else "high"
    rows = [
        {
            "quality_issue": "exact_duplicates",
            "affected_rows": duplicate_rows,
            "removable_duplicate_rows": exact_excess,
            "metric": metric,
            "total_before_dedup": original_total,
            "total_after_dedup": dedup_total,
            "estimated_inflation": inflation,
            "inflation_rate": inflation_rate,
            "severity": severity,
        }
    ]
    metric_label = f"`{metric}`" if metric else "основной метрики"
    if language.is_russian:
        if duplicate_rows:
            summary = (
                f"Из-за duplicates слабее становятся выводы, которые опираются на total {metric_label} и record counts. "
                f"Exact duplicates затрагивают {duplicate_rows:,} строк; estimated inflation для {metric_label}: {inflation:.2f} "
                f"({inflation_rate:.1%} от total). Поэтому average/median rankings обычно менее уязвимы, чем totals, если дубли не сконцентрированы в одной группе."
            )
        else:
            summary = (
                f"Exact duplicates не найдены, поэтому выводы по total {metric_label} почти не ослабляются именно duplicates. "
                "Более важными рисками могут быть repeated order-like IDs, выбросы или missingness."
            )
        findings = [
            f"Total {metric_label} и count-based выводы чувствительнее всего к exact duplicates.",
            "Group rankings by total могут измениться сильнее, чем rankings by average/median.",
            "Repeated `Order ID` или похожие business IDs нельзя автоматически считать дублями: это могут быть line items.",
        ]
        evidence = [f"Сравнил total {metric_label} до и после exact-row deduplication.", f"Duplicate row rate: {row_rate:.1%}."]
        limitations = ["Near-duplicates по business key не удалялись; для них нужны customer/product/date/amount правила."]
        next_steps = [f"Пересчитать top groups по total {metric_label} после exact-row deduplication и сравнить ранги."]
    else:
        if duplicate_rows:
            summary = (
                f"Duplicates mainly weaken conclusions based on total {metric_label} and record counts. "
                f"Exact duplicates affect {duplicate_rows:,} rows; estimated inflation for {metric_label} is {inflation:.2f} "
                f"({inflation_rate:.1%} of total). Average and median rankings are usually less vulnerable unless duplicates concentrate in one group."
            )
        else:
            summary = (
                f"Exact duplicates were not found, so total {metric_label} conclusions are not materially weakened by duplicate rows. "
                "Repeated order-like IDs, outliers, or missingness may still matter."
            )
        findings = [
            f"Total {metric_label} and count-based findings are most sensitive to exact duplicates.",
            "Group rankings by total can move more than rankings by average or median.",
            "Repeated `Order ID`-like business IDs should not be treated as duplicates automatically because they may be line items.",
        ]
        if target.dimension:
            findings.append(
                f"For `{target.dimension}` comparisons, duplicate concentration by group matters more than the global duplicate rate."
            )
        evidence = [f"Compared total {metric_label} before and after exact-row deduplication.", f"Duplicate row rate: {row_rate:.1%}."]
        limitations = ["Near-duplicates by business key were not removed; they need customer/product/date/amount rules."]
        next_steps = [f"Recompute top groups by total {metric_label} after exact-row deduplication and compare ranks."]
    if recent_findings:
        findings.append(("Relevant current finding: " if language.is_english else "Связанный текущий вывод: ") + recent_findings[-1])
    return AffectedFindingsResult(summary, findings, evidence, limitations, next_steps, rows)


def _missingness_affected_findings(
    df: pd.DataFrame,
    metric: str,
    target: ActiveAnalyticalTarget,
    language: ResponseLanguagePolicy,
) -> AffectedFindingsResult:
    missing = df.isna().sum().sort_values(ascending=False)
    top_col = str(missing.index[0]) if len(missing) else ""
    top_count = int(missing.iloc[0]) if len(missing) else 0
    rate = top_count / max(len(df), 1)
    if language.is_russian:
        summary = (
            f"Missingness сильнее всего ослабляет сравнения и тренды, которые используют `{top_col}`: там {top_count:,} missing rows ({rate:.1%}). "
            "Если missingness не случайная, rankings и segment comparisons могут быть смещены."
        )
        findings = [f"Сравнения по `{top_col}` становятся менее надежными.", "Totals и averages могут быть biased, если missing rows сконцентрированы в отдельных группах."]
        evidence = [f"Посчитана missingness по всем колонкам; максимум у `{top_col}`."]
        limitations = ["Нужно проверить, является ли missingness случайной или связана с группами/периодами."]
        next_steps = [f"Сравнить missing rate `{top_col}` по ключевым группам и периодам."]
    else:
        summary = (
            f"Missingness mainly weakens comparisons and trends using `{top_col}`: {top_count:,} rows are missing ({rate:.1%}). "
            "If missingness is not random, rankings and segment comparisons can be biased."
        )
        findings = [f"Comparisons involving `{top_col}` become less reliable.", "Totals and averages can be biased if missing rows concentrate in specific groups."]
        evidence = [f"Computed missingness by column; `{top_col}` has the highest missing count."]
        limitations = ["Check whether missingness is random or concentrated by group/time."]
        next_steps = [f"Compare `{top_col}` missing rate by key groups and periods."]
    return AffectedFindingsResult(summary, findings, evidence, limitations, next_steps, [{"column": top_col, "missing_rows": top_count, "missing_rate": rate}])


def _outlier_affected_findings(
    df: pd.DataFrame,
    metric: str,
    target: ActiveAnalyticalTarget,
    language: ResponseLanguagePolicy,
    state: dict[str, Any],
) -> AffectedFindingsResult:
    metric = metric if metric in df.columns else _first_numeric(df)
    series = pd.to_numeric(df[metric], errors="coerce").dropna() if metric else pd.Series(dtype=float)
    if series.empty:
        return AffectedFindingsResult("No numeric metric is available for outlier impact analysis.")
    q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = int(((series < lower) | (series > upper)).sum())
    rate = outliers / max(len(series), 1)
    metric_label = f"`{metric}`"
    transformed = transformation_state_from_payload(state.get("active_transformation_result"))
    transform_sentence = ""
    rows: list[dict[str, Any]] = [{"metric": metric, "outliers": outliers, "outlier_rate": rate, "lower_bound": lower, "upper_bound": upper}]
    if transformed and transformed.comparison_rows and transformed.dimension:
        dropped = _dropped_transform_rows(transformed.comparison_rows, transformed.dimension)
        stable = _stable_transform_rows(transformed.comparison_rows, transformed.dimension)
        if dropped:
            transform_sentence += (
                f" The original `{transformed.dimension}` ranking becomes unreliable where raw leaders collapsed: "
                f"{_shift_text(dropped[:3], transformed.dimension)}."
            )
        if stable:
            transform_sentence += (
                f" Conclusions are stronger for groups that stayed high after filtering: "
                f"{_shift_text(stable[:3], transformed.dimension)}."
            )
        rows.extend(transformed.comparison_rows[:12])
    summary = (
        f"Outliers weaken average-ranking findings for {metric_label}: {outliers:,} IQR outliers were found ({rate:.1%}). "
        "Median and trimmed or adjusted rankings are more reliable for these conclusions."
        f"{transform_sentence}"
    )
    findings = [f"Average-based rankings on {metric_label} are most vulnerable.", "Totals can also be inflated by a few large records."]
    evidence = [f"IQR bounds for {metric_label}: {lower:.2f} to {upper:.2f}."]
    limitations = ["The IQR rule is statistical; business-valid large values are not necessarily errors."]
    next_steps = [f"Compare raw mean ranking with median and outlier-filtered ranking for {metric_label}."]
    return AffectedFindingsResult(summary, findings, evidence, limitations, next_steps, rows)


def _dropped_transform_rows(rows: list[dict[str, Any]], dimension: str) -> list[dict[str, Any]]:
    return sorted(
        [
            row for row in rows
            if isinstance(row, dict) and (_num(row.get("rank_change")) >= 5 or _num(row.get("adjusted_rank")) > 20)
        ],
        key=lambda row: (-_num(row.get("rank_change")), _num(row.get("original_rank"))),
    )


def _stable_transform_rows(rows: list[dict[str, Any]], dimension: str) -> list[dict[str, Any]]:
    return sorted(
        [
            row for row in rows
            if isinstance(row, dict)
            and _num(row.get("original_rank")) <= 10
            and _num(row.get("adjusted_rank")) <= 10
            and abs(_num(row.get("rank_change"))) <= 3
        ],
        key=lambda row: _num(row.get("adjusted_rank")),
    )


def _shift_text(rows: list[dict[str, Any]], dimension: str) -> str:
    parts = []
    for row in rows:
        parts.append(
            f"`{row.get(dimension)}` rank #{int(_num(row.get('original_rank')))} -> #{int(_num(row.get('adjusted_rank')))} "
            f"(mean {_num(row.get('original_mean')):.2f} -> {_num(row.get('adjusted_mean')):.2f})"
        )
    return ", ".join(parts)


def _num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _generic_quality_affected_findings(
    df: pd.DataFrame,
    metric: str,
    target: ActiveAnalyticalTarget,
    language: ResponseLanguagePolicy,
) -> AffectedFindingsResult:
    duplicate_rows = int(df.duplicated(keep=False).sum())
    missing_total = int(df.isna().sum().sum())
    if language.is_russian:
        summary = (
            "Больше всего ослабляются выводы, которые используют поля с missingness, exact duplicates, sparse groups или outlier-heavy averages. "
            f"В текущих данных: duplicate rows={duplicate_rows:,}, total missing cells={missing_total:,}."
        )
    else:
        summary = (
            "The most affected findings are those using fields with missingness, exact duplicates, sparse groups, or outlier-heavy averages. "
            f"Current data: duplicate rows={duplicate_rows:,}, total missing cells={missing_total:,}."
        )
    return AffectedFindingsResult(
        summary=summary,
        findings=[summary],
        evidence=["Checked duplicate rows and missing cells."],
        limitations=["Use a focused quality issue for a more precise affected-findings map."],
        next_steps=["Map each important finding to its metric, dimension, and quality risk."],
        rows=[{"duplicate_rows": duplicate_rows, "missing_cells": missing_total}],
    )


def _quality_issue(target: ActiveAnalyticalTarget, state: dict[str, Any], question: str) -> str:
    text = " ".join(str(item or "").casefold() for item in (target.active_mechanism, target.branch_type, state.get("active_quality_issue"), question))
    if "duplicate" in text or "дублик" in text:
        return "duplicates"
    if "missing" in text or "null" in text or "пропуск" in text:
        return "missingness"
    if "outlier" in text or "выброс" in text:
        return "outliers"
    if str(state.get("active_transformation") or "").strip() == "remove_outliers" or state.get("active_transformation_result"):
        return "outliers"
    return "quality"


def _metric(target: ActiveAnalyticalTarget, state: dict[str, Any], df: pd.DataFrame, question: str) -> str:
    for value in (target.metric, state.get("active_metric")):
        text = str(value or "").strip()
        if text in df.columns:
            return text
    normalized = str(question or "").casefold()
    for col in df.select_dtypes(include="number").columns:
        if str(col).casefold() in normalized:
            return str(col)
    for col in df.select_dtypes(include="number").columns:
        name = str(col)
        if any(marker in name.casefold() for marker in ("sales", "revenue", "profit", "amount", "value")):
            return name
    return _first_numeric(df)


def _first_numeric(df: pd.DataFrame) -> str:
    numeric = [str(col) for col in df.select_dtypes(include="number").columns]
    return numeric[0] if numeric else ""


def _language_fallback(value: str) -> DetectedLanguage:
    return DetectedLanguage.RUSSIAN if str(value).lower().startswith("ru") else DetectedLanguage.ENGLISH
