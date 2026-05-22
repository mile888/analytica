from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from source.product.semantic_layer import (
    CategoryValueMatch,
    SemanticDatasetProfile,
    SemanticRole,
    match_entity_or_value,
)
from source.product.language_policy import ResponseLanguagePolicy
from source.product.evidence_resolution import transformation_state_from_payload
from source.product.analytical_graph import (
    evaluate_large_record_hypothesis,
    synthesize_contradiction,
    synthesize_support,
    transformation_from_payload,
)


class MechanismType(StrEnum):
    VOLUME_EFFECT = "volume_effect"
    OUTLIER_EFFECT = "outlier_effect"
    OUTLIER_CONCENTRATION = "outlier_concentration"
    SPARSE_GROUP_INSTABILITY = "sparse_group_instability"
    SUBGROUP_MIX = "subgroup_mix"
    TEMPORAL_SHIFT = "temporal_shift"
    SEASONALITY = "seasonality"
    CONCENTRATION = "concentration"
    OPERATIONAL_EFFECT = "operational_effect"
    CORRELATION_EFFECT = "correlation_effect"
    QUALITY_DISTORTION = "quality_distortion"
    DUPLICATE_INFLATION = "duplicate_inflation"
    MISSINGNESS_BIAS = "missingness_bias"
    NORMALIZATION_EFFECT = "normalization_effect"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HypothesisTarget:
    entity: str = ""
    column: str = ""
    value: str = ""
    semantic_role: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class HypothesisFrame:
    raw_text: str
    target_metric: str = ""
    target: HypothesisTarget = field(default_factory=HypothesisTarget)
    mechanism: MechanismType = MechanismType.UNKNOWN
    secondary_mechanisms: list[MechanismType] = field(default_factory=list)
    comparison_type: str = ""
    validation_goal: str = ""
    time_axis: str = ""


@dataclass(frozen=True)
class HypothesisEvidence:
    rows: list[dict[str, Any]] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    code: str = ""
    result_preview: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class HypothesisConclusion:
    summary: str
    confidence: str = "medium"
    supported: bool | None = None


@dataclass(frozen=True)
class ConfidenceAssessment:
    level: str
    reason: str
    evidence_strength: str
    limiting_factors: list[str] = field(default_factory=list)
    what_would_change_confidence: str = ""


@dataclass(frozen=True)
class HypothesisValidationPlan:
    mechanism: MechanismType
    operator_name: str
    metric: str
    dimension: str = ""
    value: str = ""
    time_axis: str = ""


def is_hypothesis_question(text: str) -> bool:
    normalized = _normalize(text)
    if any(marker in normalized for marker in ("formulate hypothesis", "create hypothesis", "сформируй hypothesis", "сформулируй hypothesis")):
        return False
    if any(
        marker in normalized
        for marker in (
            "what supports",
            "what contradicts",
            "contradict",
            "counterevidence",
            "counter evidence",
            "evidence supports",
            "supports this",
            "поддерж",
            "противореч",
            "опроверг",
        )
    ):
        return False
    if normalized.startswith(("hypothesis:", "hypothesis -", "hypothesis ")):
        return True
    if "hypothesis:" in normalized or "гипотеза:" in normalized or "гипотез:" in normalized:
        return True
    return normalized.startswith(("test whether", "validate whether", "check whether", "test hypothesis", "check hypothesis", "validate hypothesis"))


def resolve_entity_or_value(
    user_text: str,
    semantic_profile: SemanticDatasetProfile,
    dataframe: pd.DataFrame | None = None,
) -> CategoryValueMatch | None:
    return match_entity_or_value(user_text, semantic_profile, df=dataframe)


class HypothesisEngine:
    @classmethod
    def validate(
        cls,
        *,
        question: str,
        dataframe: pd.DataFrame,
        branch_state: dict[str, Any] | None,
        semantic_profile: SemanticDatasetProfile,
    ) -> dict[str, Any] | None:
        if not is_hypothesis_question(question) or not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            return None
        frame = cls.decompose(question, dataframe, branch_state or {}, semantic_profile)
        if not frame.target_metric:
            return None
        plan = HypothesisValidationPlan(
            mechanism=frame.mechanism,
            operator_name=_operator_name(frame.mechanism),
            metric=frame.target_metric,
            dimension="delivery_delay_days" if frame.mechanism == MechanismType.OPERATIONAL_EFFECT and _mentions_shipping_delay(question) else frame.target.column,
            value=frame.target.value,
            time_axis=frame.time_axis,
        )
        transformed = transformation_state_from_payload((branch_state or {}).get("active_transformation_result"))
        if transformed and frame.mechanism in {MechanismType.OUTLIER_EFFECT, MechanismType.CONCENTRATION, MechanismType.OUTLIER_CONCENTRATION}:
            evidence, conclusion = _validate_from_transformation(frame, transformed)
            if conclusion:
                return _output(
                    question=question,
                    summary=conclusion.summary,
                    findings=evidence.findings or [conclusion.summary],
                    evidence=evidence.evidence,
                    limitations=evidence.limitations,
                    next_steps=evidence.next_steps,
                    code=evidence.code,
                    result_preview=evidence.result_preview,
                    timeline=[{"tool": "hypothesis_engine", "status": "ok", "mechanism": frame.mechanism.value, "source": "active_transformation"}],
                    artifacts=evidence.artifacts,
                    trace_metadata={
                        "fallback": "hypothesis_engine",
                        "analysis_type": "hypothesis_validation",
                        "active_branch_type": "hypothesis_validation",
                        "metric": frame.target_metric,
                        "dimension": transformed.dimension or plan.dimension,
                        "matched_category_value": frame.target.value,
                        "hypothesis_mechanism": frame.mechanism.value,
                        "hypothesis_confidence": conclusion.confidence,
                        "hypothesis_supported": conclusion.supported,
                    },
                )
        evidence, conclusion = cls._run_operator(frame, dataframe, plan)
        if not conclusion:
            return None
        return _output(
            question=question,
            summary=conclusion.summary,
            findings=evidence.findings or [conclusion.summary],
            evidence=evidence.evidence,
            limitations=evidence.limitations,
            next_steps=evidence.next_steps,
            code=evidence.code,
            result_preview=evidence.result_preview,
            timeline=[{"tool": "hypothesis_engine", "status": "ok", "mechanism": frame.mechanism.value}],
            artifacts=evidence.artifacts,
            trace_metadata={
                "fallback": "hypothesis_engine",
                "analysis_type": "hypothesis_validation",
                "active_branch_type": "hypothesis_validation",
                "metric": frame.target_metric,
                "dimension": plan.dimension,
                "matched_category_value": frame.target.value,
                "hypothesis_mechanism": frame.mechanism.value,
                "hypothesis_confidence": conclusion.confidence,
                "hypothesis_supported": conclusion.supported,
            },
        )

    @classmethod
    def decompose(
        cls,
        question: str,
        df: pd.DataFrame,
        branch_state: dict[str, Any],
        semantic_profile: SemanticDatasetProfile,
    ) -> HypothesisFrame:
        mechanism, secondary = _infer_mechanisms(question)
        metric = _resolve_metric(question, df, branch_state, semantic_profile)
        value_match = resolve_entity_or_value(question, semantic_profile, df)
        target = _resolve_target(question, df, branch_state, semantic_profile, value_match)
        time_axis = _resolve_time_axis(question, df, branch_state, semantic_profile)
        comparison, goal = _comparison_and_goal(mechanism)
        return HypothesisFrame(
            raw_text=question,
            target_metric=metric,
            target=target,
            mechanism=mechanism,
            secondary_mechanisms=secondary,
            comparison_type=comparison,
            validation_goal=goal,
            time_axis=time_axis,
        )

    @classmethod
    def _run_operator(
        cls,
        frame: HypothesisFrame,
        df: pd.DataFrame,
        plan: HypothesisValidationPlan,
    ) -> tuple[HypothesisEvidence, HypothesisConclusion | None]:
        if plan.mechanism == MechanismType.VOLUME_EFFECT:
            return _validate_volume_effect(frame, df, plan)
        if plan.mechanism in {MechanismType.OUTLIER_CONCENTRATION, MechanismType.OUTLIER_EFFECT}:
            if frame.target.value:
                return _validate_outlier_concentration(frame, df, plan)
            return _validate_concentration(frame, df, plan)
        if plan.mechanism in {MechanismType.SPARSE_GROUP_INSTABILITY, MechanismType.NORMALIZATION_EFFECT}:
            return _validate_sparse_group_instability(frame, df, plan)
        if plan.mechanism == MechanismType.DUPLICATE_INFLATION:
            return _validate_duplicate_inflation(frame, df, plan)
        if plan.mechanism == MechanismType.SEASONALITY:
            return _validate_seasonality(frame, df, plan)
        if plan.mechanism in {MechanismType.OPERATIONAL_EFFECT, MechanismType.TEMPORAL_SHIFT} and _mentions_shipping_delay(frame.raw_text):
            return _validate_shipping_delay_effect(frame, df, plan)
        if plan.mechanism == MechanismType.CONCENTRATION:
            return _validate_concentration(frame, df, plan)
        if plan.dimension:
            return _validate_volume_effect(frame, df, plan)
        return HypothesisEvidence(limitations=["The hypothesis did not resolve to a testable metric and dimension."]), None


def _validate_volume_effect(
    frame: HypothesisFrame,
    df: pd.DataFrame,
    plan: HypothesisValidationPlan,
) -> tuple[HypothesisEvidence, HypothesisConclusion | None]:
    language = ResponseLanguagePolicy.from_message(frame.raw_text, protected_terms=df.columns)
    if not plan.dimension or plan.dimension not in df.columns:
        return HypothesisEvidence(limitations=["No grouping dimension was resolved for the volume-effect hypothesis."]), None
    working = _metric_dimension_frame(df, plan.metric, plan.dimension)
    if working.empty:
        return HypothesisEvidence(limitations=[f"`{plan.metric}` has no valid numeric values for this hypothesis."]), None
    grouped = (
        working.groupby(plan.dimension, dropna=False)[plan.metric]
        .agg(count="count", total="sum", mean="mean", median="median")
        .reset_index()
    )
    grouped["share_of_rows"] = grouped["count"] / max(float(grouped["count"].sum()), 1.0)
    grouped["share_of_total"] = grouped["total"] / max(float(grouped["total"].sum()), 1.0)
    grouped["total_rank"] = grouped["total"].rank(method="min", ascending=False).astype(int)
    grouped["mean_rank"] = grouped["mean"].rank(method="min", ascending=False).astype(int)
    grouped["count_rank"] = grouped["count"].rank(method="min", ascending=False).astype(int)
    grouped = grouped.sort_values("total", ascending=False)
    target_row = _target_row(grouped, plan.dimension, plan.value)
    if target_row is None:
        target_row = grouped.iloc[0]
    target_name = str(target_row[plan.dimension])
    mean_leader = grouped.sort_values("mean", ascending=False).iloc[0]
    count_leader = grouped.sort_values("count", ascending=False).iloc[0]
    total_leader = grouped.iloc[0]
    volume_supported = (
        str(target_name) == str(total_leader[plan.dimension])
        and int(target_row["count_rank"]) <= int(target_row["mean_rank"])
        and float(target_row["share_of_rows"]) >= float(target_row["share_of_total"]) * 0.75
    )
    if language.is_russian:
        if volume_supported:
            summary = (
                f"Гипотеза поддерживается для `{target_name}` в `{plan.dimension}`: доминирование по `{plan.metric}` в основном связано с объемом. "
                f"Группа занимает #{int(target_row['total_rank'])} по total, #{int(target_row['count_rank'])} по числу записей и #{int(target_row['mean_rank'])} по среднему значению. "
                f"Доля строк: {float(target_row['share_of_rows']):.1%}, доля total `{plan.metric}`: {float(target_row['share_of_total']):.1%}; "
                f"лидер по average value: `{mean_leader[plan.dimension]}` ({float(mean_leader['mean']):.2f})."
            )
        else:
            summary = (
                f"Гипотеза поддерживается только частично для `{target_name}` в `{plan.dimension}`: эффект не выглядит чисто volume-driven. "
                f"Группа занимает #{int(target_row['total_rank'])} по total `{plan.metric}`, #{int(target_row['count_rank'])} по count и #{int(target_row['mean_rank'])} по average value. "
                f"Лидер по count: `{count_leader[plan.dimension]}`, лидер по average value: `{mean_leader[plan.dimension]}`."
            )
        evidence = [f"Сравнил total, mean, median и record count для `{plan.metric}` по `{plan.dimension}`."]
        limitations = ["Record count является прокси для объема; если есть order ID, лучше считать distinct orders."]
        next_steps = [f"Повторить проверку по distinct order IDs внутри `{plan.dimension}`, если есть идентификатор заказа."]
    elif volume_supported:
        summary = (
            f"The hypothesis is supported for `{target_name}` in `{plan.dimension}`: it dominates `{plan.metric}` mainly through volume. "
            f"It ranks #{int(target_row['total_rank'])} by total, #{int(target_row['count_rank'])} by record count, and #{int(target_row['mean_rank'])} by average value. "
            f"Its row share is {float(target_row['share_of_rows']):.1%}, while its total `{plan.metric}` share is {float(target_row['share_of_total']):.1%}; "
            f"the average-value leader is `{mean_leader[plan.dimension]}` ({float(mean_leader['mean']):.2f})."
        )
    else:
        summary = (
            f"The hypothesis is only partially supported for `{target_name}` in `{plan.dimension}`: it does not look purely volume-driven. "
            f"It ranks #{int(target_row['total_rank'])} by total `{plan.metric}`, #{int(target_row['count_rank'])} by count, and #{int(target_row['mean_rank'])} by average value. "
            f"The count leader is `{count_leader[plan.dimension]}` and the average-value leader is `{mean_leader[plan.dimension]}`."
        )
        evidence = [f"Compared total, mean, median, and record count for `{plan.metric}` by `{plan.dimension}`."]
        limitations = ["Record count is a proxy for volume; distinct order count is better when order IDs are available."]
        next_steps = [f"Repeat with distinct order IDs by `{plan.dimension}` if an order identifier exists."]
    if language.is_english and volume_supported:
        evidence = [f"Compared total, mean, median, and record count for `{plan.metric}` by `{plan.dimension}`."]
        limitations = ["Record count is a proxy for volume; distinct order count is better when order IDs are available."]
        next_steps = [f"Repeat with distinct order IDs by `{plan.dimension}` if an order identifier exists."]
    rows = grouped.to_dict(orient="records")
    confidence = _confidence_assessment(
        supported=volume_supported,
        rows=len(working),
        groups=len(grouped),
        evidence_strength="medium",
        limiting_factors=limitations,
        what_would_change_confidence=next_steps[0] if next_steps else "",
    )
    summary = _finalize_hypothesis_summary(summary, confidence, language)
    return (
        HypothesisEvidence(
            rows=rows,
            findings=[summary],
            evidence=evidence,
            limitations=limitations,
            next_steps=next_steps,
            code="df.groupby(dimension)[metric].agg(count='count', total='sum', mean='mean', median='median')",
            result_preview=grouped.to_string(index=False),
            artifacts=[_table_artifact(f"{plan.metric} volume vs value by {plan.dimension}", rows, frame, {"operator": "volume_effect"})],
        ),
        HypothesisConclusion(summary=summary, confidence=confidence.level, supported=volume_supported),
    )


def _validate_from_transformation(frame: HypothesisFrame, transformed: Any) -> tuple[HypothesisEvidence, HypothesisConclusion | None]:
    payload = transformed.to_payload() if hasattr(transformed, "to_payload") else getattr(transformed, "__dict__", {})
    execution = transformation_from_payload(payload)
    if not execution:
        return HypothesisEvidence(limitations=["The active transformation did not contain rank-shift evidence."]), None
    metric = execution.metric or frame.target_metric
    dimension = execution.dimension or frame.target.column
    if not execution.rank_shifts or not metric or not dimension:
        return HypothesisEvidence(limitations=["The active transformation did not contain rank-shift evidence."]), None
    evaluation = evaluate_large_record_hypothesis(hypothesis=frame.raw_text, transformation=execution)
    support_text = synthesize_support(evaluation)
    contradiction_text = synthesize_contradiction(evaluation)
    rows_text = (
        f"Rows removed: {execution.removed_row_count:,} of {execution.original_row_count:,}; "
        f"remaining rows: {execution.filtered_row_count:,}; threshold {execution.threshold.get('lower', 0):.2f} to {execution.threshold.get('upper', 0):.2f}. "
        f"The rank-shift evidence covers {len(execution.rank_shifts):,} displayed groups, not the full dimension cardinality."
        if execution.is_valid
        else ""
    )
    summary = " ".join(part for part in (support_text, contradiction_text, rows_text) if part)
    confidence = _confidence_assessment(
        supported=evaluation.supported,
        rows=execution.original_row_count,
        groups=0,
        evidence_strength="high" if evaluation.supporting_evidence and execution.is_valid else "medium",
        limiting_factors=evaluation.limitations,
        what_would_change_confidence=evaluation.validation_paths[0] if evaluation.validation_paths else "",
    )
    summary = _finalize_hypothesis_summary(summary, confidence, ResponseLanguagePolicy.from_message(frame.raw_text))
    return (
        HypothesisEvidence(
            rows=execution.rank_shifts,
            findings=[summary],
            evidence=[
                *evaluation.supporting_evidence,
                *evaluation.contradicting_evidence,
            ],
            limitations=evaluation.limitations,
            next_steps=evaluation.validation_paths,
            code="compare raw group ranks with ranks after removing IQR-extreme metric records",
            result_preview=pd.DataFrame(execution.rank_shifts[:25]).to_string(index=False),
            artifacts=[_table_artifact(f"{metric} transformation evidence by {dimension}", execution.rank_shifts[:25], frame, {"operator": "active_transformation_rank_shift", "dimension": dimension})],
        ),
        HypothesisConclusion(summary=summary, confidence=confidence.level, supported=evaluation.supported),
    )


def _hypothesis_shift_text(rows: list[dict[str, Any]], dimension: str) -> str:
    parts = []
    for row in rows:
        parts.append(
            f"`{row.get(dimension)}` rank #{int(_number(row.get('original_rank')))} -> #{int(_number(row.get('adjusted_rank')))} "
            f"(mean {_number(row.get('original_mean')):.2f} -> {_number(row.get('adjusted_mean')):.2f}, n={int(_number(row.get('adjusted_count') or row.get('original_count')))})"
        )
    return ", ".join(parts)


def _number(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _validate_outlier_concentration(
    frame: HypothesisFrame,
    df: pd.DataFrame,
    plan: HypothesisValidationPlan,
) -> tuple[HypothesisEvidence, HypothesisConclusion | None]:
    language = ResponseLanguagePolicy.from_message(frame.raw_text, protected_terms=df.columns)
    if not plan.dimension or plan.dimension not in df.columns:
        return HypothesisEvidence(limitations=["No category dimension was resolved for the outlier hypothesis."]), None
    working = _metric_dimension_frame(df, plan.metric, plan.dimension)
    if working.empty:
        return HypothesisEvidence(limitations=[f"`{plan.metric}` has no valid numeric values for outlier scoring."]), None
    scored, lower, upper = _iqr_outliers(working, plan.metric)
    grouped = (
        scored.groupby(plan.dimension, dropna=False)
        .agg(count=(plan.metric, "count"), outlier_count=("is_outlier", "sum"), total=(plan.metric, "sum"), mean=(plan.metric, "mean"), median=(plan.metric, "median"))
        .reset_index()
    )
    total_outliers = max(float(grouped["outlier_count"].sum()), 1.0)
    grouped["outlier_share"] = grouped["outlier_count"] / total_outliers
    grouped["outlier_rate"] = grouped["outlier_count"] / grouped["count"].clip(lower=1)
    grouped = grouped.sort_values(["outlier_count", "outlier_rate"], ascending=False)
    target_row = _target_row(grouped, plan.dimension, plan.value)
    leader = grouped.iloc[0]
    target_name = str((target_row if target_row is not None else leader)[plan.dimension])
    supported = target_row is not None and str(target_row[plan.dimension]) == str(leader[plan.dimension]) and int(target_row["outlier_count"]) > 0
    if language.is_russian:
        if supported:
            summary = (
                f"Гипотеза поддерживается: `{target_name}` в `{plan.dimension}` содержит наибольшую концентрацию outliers по `{plan.metric}`. "
                f"Там {int(target_row['outlier_count'])} outlier rows, {float(target_row['outlier_share']):.1%} всех outliers "
                f"и outlier rate {float(target_row['outlier_rate']):.1%}. IQR bounds: {lower:.2f}–{upper:.2f}."
            )
        elif target_row is not None:
            summary = (
                f"Гипотеза не подтверждается в этой формулировке: у `{target_name}` {int(target_row['outlier_count'])} outliers по `{plan.metric}` "
                f"({float(target_row['outlier_share']):.1%} всех outliers), а максимум у `{leader[plan.dimension]}` "
                f"({int(leader['outlier_count'])}). IQR bounds: {lower:.2f}–{upper:.2f}."
            )
        else:
            summary = (
                f"Самая сильная концентрация outliers: `{leader[plan.dimension]}` в `{plan.dimension}` с {int(leader['outlier_count'])} outliers по `{plan.metric}`. "
                f"IQR bounds: {lower:.2f}–{upper:.2f}."
            )
        evidence = [f"Отметил outliers по `{plan.metric}` через IQR bounds и посчитал их по `{plan.dimension}`."]
        limitations = ["IQR outliers могут быть реальными business cases; перед удалением нужно смотреть raw records."]
        next_steps = [f"Проверить raw outlier rows для `{target_name}`."]
    elif supported:
        summary = (
            f"The hypothesis is supported: `{target_name}` in `{plan.dimension}` contains the largest concentration of `{plan.metric}` outliers. "
            f"It has {int(target_row['outlier_count'])} outlier rows, {float(target_row['outlier_share']):.1%} of all outliers, "
            f"and an outlier rate of {float(target_row['outlier_rate']):.1%}. IQR bounds were {lower:.2f} to {upper:.2f}."
        )
    elif target_row is not None:
        summary = (
            f"The hypothesis is not supported as stated for `{target_name}` in `{plan.dimension}`: it has {int(target_row['outlier_count'])} `{plan.metric}` outliers "
            f"({float(target_row['outlier_share']):.1%} of all outliers), while `{leader[plan.dimension]}` in `{plan.dimension}` has the most "
            f"({int(leader['outlier_count'])}). IQR bounds were {lower:.2f} to {upper:.2f}."
        )
    else:
        summary = (
            f"The strongest outlier concentration is `{leader[plan.dimension]}` in `{plan.dimension}` with {int(leader['outlier_count'])} `{plan.metric}` outliers. "
            f"IQR bounds were {lower:.2f} to {upper:.2f}."
        )
    if language.is_english:
        evidence = [f"Flagged `{plan.metric}` outliers using IQR bounds and counted them by `{plan.dimension}`."]
        limitations = ["IQR outliers can be valid extreme business cases; inspect raw records before removing them."]
        next_steps = [f"Inspect raw outlier rows for `{target_name}`."]
    rows = grouped.to_dict(orient="records")
    confidence = _confidence_assessment(
        supported=supported,
        rows=len(working),
        groups=len(grouped),
        evidence_strength="medium",
        limiting_factors=limitations,
        what_would_change_confidence=next_steps[0] if next_steps else "",
    )
    summary = _finalize_hypothesis_summary(summary, confidence, language)
    return (
        HypothesisEvidence(
            rows=rows,
            findings=[summary],
            evidence=evidence,
            limitations=limitations,
            next_steps=next_steps,
            code="q1, q3 = metric.quantile([.25, .75]); outliers = metric > q3 + 1.5 * (q3 - q1)",
            result_preview=grouped.to_string(index=False),
            artifacts=[_table_artifact(f"{plan.metric} outlier concentration by {plan.dimension}", rows, frame, {"operator": "outlier_concentration"})],
        ),
        HypothesisConclusion(summary=summary, confidence=confidence.level, supported=supported),
    )


def _validate_concentration(
    frame: HypothesisFrame,
    df: pd.DataFrame,
    plan: HypothesisValidationPlan,
) -> tuple[HypothesisEvidence, HypothesisConclusion | None]:
    language = ResponseLanguagePolicy.from_message(frame.raw_text, protected_terms=df.columns)
    if not plan.dimension or plan.dimension not in df.columns:
        return HypothesisEvidence(limitations=["No grouping dimension was resolved for the concentration hypothesis."]), None
    working = _metric_dimension_frame(df, plan.metric, plan.dimension)
    if working.empty:
        return HypothesisEvidence(limitations=[f"`{plan.metric}` has no valid numeric values for concentration analysis."]), None
    rows = []
    for value, group in working.groupby(plan.dimension, dropna=False):
        total = float(group[plan.metric].sum())
        top_value = float(group[plan.metric].max()) if len(group) else 0.0
        rows.append(
            {
                plan.dimension: value,
                "count": int(len(group)),
                "total": total,
                "mean": float(group[plan.metric].mean()),
                "median": float(group[plan.metric].median()),
                "top_record_value": top_value,
                "top_record_share": top_value / total if total else 0.0,
                "mean_median_gap": float(group[plan.metric].mean() - group[plan.metric].median()),
            }
        )
    result = pd.DataFrame(rows).sort_values(["total", "top_record_share"], ascending=[False, False])
    top = result.head(5)
    concentrated = top[top["top_record_share"] >= 0.5]
    supported = not concentrated.empty
    if supported:
        detail = ", ".join(
            f"`{row[plan.dimension]}` ({float(row['top_record_share']):.1%} from top record, n={int(row['count'])})"
            for _, row in concentrated.head(3).iterrows()
        )
        summary = (
            f"Проверка гипотезы по `{plan.metric}` by `{plan.dimension}`: top-order concentration test поддерживает механизм; сильные группы хрупкие для {detail}."
            if language.is_russian
            else f"Hypothesis check for `{plan.metric}` by `{plan.dimension}`: the top-order concentration test supports the mechanism; high groups are fragile for {detail}."
        )
    else:
        summary = (
            f"Проверка гипотезы по `{plan.metric}` by `{plan.dimension}`: top-order concentration test не показывает, что лидеров двигает одна крупная запись `{plan.metric}`. "
            "Я сравнил total, mean, median, count и top-record share для каждой группы."
            if language.is_russian
            else f"Hypothesis check for `{plan.metric}` by `{plan.dimension}`: the top-order concentration test does not show a single large `{plan.metric}` record driving the leading groups. "
            "I compared total, mean, median, count, and top-record share for each group."
        )
    rows_out = result.to_dict(orient="records")
    limitations = ["Строки трактуются как records; если датасет line-item, сначала лучше агрегировать до order level."] if language.is_russian else ["Rows are treated as records; aggregate to true order level first if the dataset is line-item based."]
    next_steps = [f"Пересчитать рейтинг `{plan.dimension}` после удаления крупнейшей записи `{plan.metric}` в каждой группе."] if language.is_russian else [f"Re-rank `{plan.dimension}` after removing each group's largest `{plan.metric}` record."]
    confidence = _confidence_assessment(
        supported=supported,
        rows=len(working),
        groups=len(result),
        evidence_strength="medium",
        limiting_factors=limitations,
        what_would_change_confidence=next_steps[0],
    )
    summary = _finalize_hypothesis_summary(summary, confidence, language)
    return (
        HypothesisEvidence(
            rows=rows_out,
            findings=[summary],
            evidence=[f"Посчитал top-record concentration для `{plan.metric}` по `{plan.dimension}`."] if language.is_russian else [f"Computed top-record concentration for `{plan.metric}` across `{plan.dimension}`."],
            limitations=limitations,
            next_steps=next_steps,
            code="grouped = df.groupby(dimension)[metric].agg(['count', 'sum', 'mean', 'median', 'max'])",
            result_preview=result.head(25).to_string(index=False),
            artifacts=[_table_artifact(f"{plan.metric} concentration by {plan.dimension}", rows_out[:25], frame, {"operator": "concentration"})],
        ),
        HypothesisConclusion(summary=summary, confidence=confidence.level, supported=supported),
    )


def _validate_sparse_group_instability(
    frame: HypothesisFrame,
    df: pd.DataFrame,
    plan: HypothesisValidationPlan,
) -> tuple[HypothesisEvidence, HypothesisConclusion | None]:
    language = ResponseLanguagePolicy.from_message(frame.raw_text, protected_terms=df.columns)
    if not plan.dimension or plan.dimension not in df.columns:
        return HypothesisEvidence(limitations=["No grouping dimension was resolved for the sparse-group hypothesis."]), None
    working = _metric_dimension_frame(df, plan.metric, plan.dimension)
    if working.empty:
        return HypothesisEvidence(limitations=[f"`{plan.metric}` has no valid numeric values for sparse-group analysis."]), None
    grouped = (
        working.groupby(plan.dimension, dropna=False)[plan.metric]
        .agg(count="count", total="sum", mean="mean", median="median")
        .reset_index()
    )
    threshold = max(3, int(grouped["count"].median()))
    grouped["is_sparse"] = grouped["count"] < threshold
    grouped["raw_mean_rank"] = grouped["mean"].rank(method="min", ascending=False).astype(int)
    stable = grouped[grouped["count"] >= threshold].copy().sort_values("mean", ascending=False)
    stable_names = set(stable.head(10)[plan.dimension].astype(str).tolist())
    raw_top = grouped.sort_values("mean", ascending=False).head(10).copy()
    raw_top["survives_stable_filter"] = raw_top[plan.dimension].astype(str).isin(stable_names)
    sparse_top = raw_top[raw_top["is_sparse"]]
    supported = not sparse_top.empty
    if supported:
        examples = ", ".join(f"`{row[plan.dimension]}` (n={int(row['count'])}, mean={float(row['mean']):.2f})" for _, row in sparse_top.head(4).iterrows())
        summary = (
            f"Гипотеза поддерживается: часть сильных на вид групп `{plan.dimension}` имеет sparse data. "
            f"При фильтре стабильности n >= {threshold} среди sparse raw leaders есть {examples}. "
            "Такие рейтинги нужно считать хрупкими, пока они не проходят minimum-sample или median check."
            if language.is_russian
            else f"The hypothesis is supported: some strong-looking `{plan.dimension}` groups are sparse. "
            f"Using n >= {threshold} as a stability filter, sparse raw leaders include {examples}. "
            "Those rankings should be treated as fragile until they survive a minimum-sample or median check."
        )
    else:
        summary = (
            f"Гипотеза слабо поддерживается: лидирующие группы `{plan.dimension}` по среднему `{plan.metric}` не являются sparse при фильтре n >= {threshold}. "
            "Рейтинг меньше похож на артефакт маленькой выборки."
            if language.is_russian
            else f"The hypothesis is not strongly supported: the leading `{plan.dimension}` groups by average `{plan.metric}` are not sparse under the n >= {threshold} filter. "
            "The ranking looks less likely to be just a small-sample artifact."
        )
    result = raw_top.sort_values(["is_sparse", "mean"], ascending=[False, False])
    rows = result.to_dict(orient="records")
    limitations = ["Порог sample size эвристический; для финального отчета нужен domain-specific minimum."] if language.is_russian else ["The sample-size threshold is heuristic; choose a domain-specific minimum for final reporting."]
    next_steps = [f"Использовать median `{plan.metric}` и n >= {threshold}, прежде чем называть группы `{plan.dimension}` сильными."] if language.is_russian else [f"Use median `{plan.metric}` and n >= {threshold} before calling `{plan.dimension}` groups strong."]
    confidence = _confidence_assessment(
        supported=supported,
        rows=len(working),
        groups=len(grouped),
        evidence_strength="medium" if len(working) >= 20 else "low",
        limiting_factors=limitations,
        what_would_change_confidence=next_steps[0],
    )
    summary = _finalize_hypothesis_summary(summary, confidence, language)
    return (
        HypothesisEvidence(
            rows=rows,
            findings=[summary],
            evidence=[f"Сравнил raw mean ranking с minimum-sample filter для `{plan.metric}` по `{plan.dimension}`."] if language.is_russian else [f"Compared raw mean ranking with a minimum-sample filter for `{plan.metric}` by `{plan.dimension}`."],
            limitations=limitations,
            next_steps=next_steps,
            code="grouped = df.groupby(dimension)[metric].agg(count='count', mean='mean', median='median')",
            result_preview=result.to_string(index=False),
            artifacts=[_table_artifact(f"{plan.metric} sparse-group stability by {plan.dimension}", rows, frame, {"operator": "sparse_group_instability", "threshold": threshold})],
        ),
        HypothesisConclusion(summary=summary, confidence=confidence.level, supported=supported),
    )


def _validate_duplicate_inflation(
    frame: HypothesisFrame,
    df: pd.DataFrame,
    plan: HypothesisValidationPlan,
) -> tuple[HypothesisEvidence, HypothesisConclusion | None]:
    language = ResponseLanguagePolicy.from_message(frame.raw_text, protected_terms=df.columns)
    duplicate_rows = int(df.duplicated(keep=False).sum())
    exact_excess = int(df.duplicated(keep="first").sum())
    original_total = float(pd.to_numeric(df[plan.metric], errors="coerce").sum()) if plan.metric in df.columns else 0.0
    dedup_total = float(pd.to_numeric(df.drop_duplicates()[plan.metric], errors="coerce").sum()) if plan.metric in df.columns else 0.0
    inflation = original_total - dedup_total
    supported = inflation > 0
    summary = (
        f"Гипотеза duplicate inflation {'поддерживается' if supported else 'не поддерживается'} для `{plan.metric}`. "
        f"Exact duplicate rows затрагивают {duplicate_rows:,} строк; removable duplicate rows: {exact_excess:,}. "
        f"Total `{plan.metric}` до exact-row deduplication: {original_total:.2f}, после: {dedup_total:.2f}; estimated inflation: {inflation:.2f}."
        if language.is_russian
        else f"The duplicate-inflation hypothesis is {'supported' if supported else 'not supported'} for `{plan.metric}`. "
        f"Exact duplicate rows affect {duplicate_rows:,} rows, with {exact_excess:,} removable duplicate rows. "
        f"Total `{plan.metric}` is {original_total:.2f} before exact-row deduplication and {dedup_total:.2f} after, so estimated inflation is {inflation:.2f}."
    )
    rows = [{
        "metric": plan.metric,
        "duplicate_rows": duplicate_rows,
        "exact_duplicate_excess_rows": exact_excess,
        "total_before_dedup": original_total,
        "total_after_exact_row_dedup": dedup_total,
        "estimated_duplicate_inflation": inflation,
    }]
    limitations = ["Повторяющиеся business IDs не удаляются автоматически: они могут быть валидными line items."] if language.is_russian else ["This does not remove repeated business IDs because those can represent valid line items."]
    next_steps = ["Проверить business-key duplicates отдельно перед удалением неидентичных строк."] if language.is_russian else ["Inspect business-key duplicates separately before deleting non-identical rows."]
    confidence = _confidence_assessment(
        supported=supported,
        rows=len(df),
        groups=1,
        evidence_strength="high" if duplicate_rows else "medium",
        limiting_factors=limitations,
        what_would_change_confidence=next_steps[0],
    )
    summary = _finalize_hypothesis_summary(summary, confidence, language)
    return (
        HypothesisEvidence(
            rows=rows,
            findings=[summary],
            evidence=[f"Сравнил total `{plan.metric}` до и после exact-row deduplication."] if language.is_russian else [f"Compared `{plan.metric}` totals before and after exact-row deduplication."],
            limitations=limitations,
            next_steps=next_steps,
            code="deduped = df.drop_duplicates(); inflation = df[metric].sum() - deduped[metric].sum()",
            result_preview=pd.DataFrame(rows).to_string(index=False),
            artifacts=[_table_artifact("Duplicate inflation evidence", rows, frame, {"operator": "duplicate_inflation"})],
        ),
        HypothesisConclusion(summary=summary, confidence=confidence.level, supported=supported),
    )


def _validate_seasonality(
    frame: HypothesisFrame,
    df: pd.DataFrame,
    plan: HypothesisValidationPlan,
) -> tuple[HypothesisEvidence, HypothesisConclusion | None]:
    language = ResponseLanguagePolicy.from_message(frame.raw_text, protected_terms=df.columns)
    if not plan.time_axis or plan.time_axis not in df.columns:
        return HypothesisEvidence(limitations=["No time axis was resolved for the seasonality hypothesis."]), None
    working = df[[plan.time_axis, plan.metric]].copy()
    working[plan.time_axis] = pd.to_datetime(working[plan.time_axis], errors="coerce")
    working[plan.metric] = pd.to_numeric(working[plan.metric], errors="coerce")
    working = working.dropna(subset=[plan.time_axis, plan.metric])
    if working.empty:
        return HypothesisEvidence(limitations=[f"`{plan.time_axis}` and `{plan.metric}` do not have enough valid rows."]), None
    working["month"] = working[plan.time_axis].dt.month
    by_month = working.groupby("month")[plan.metric].agg(count="count", total="sum", mean="mean", median="median").reset_index()
    spread = float(by_month["mean"].max() - by_month["mean"].min()) if not by_month.empty else 0.0
    mean_level = float(by_month["mean"].mean()) if not by_month.empty else 0.0
    supported = bool(mean_level and spread / abs(mean_level) >= 0.25)
    top = by_month.sort_values("mean", ascending=False).head(3)
    top_text = ", ".join(f"{int(row['month'])} (mean {float(row['mean']):.2f}, n={int(row['count'])})" for _, row in top.iterrows())
    summary = (
        f"Гипотеза seasonality {'выглядит правдоподобно' if supported else 'слабая'} для `{plan.metric}` over `{plan.time_axis}`. "
        f"Разброс month-of-year mean: {spread:.2f}; strongest months: {top_text}. "
        "Seasonal spikes убедительны только если паттерн повторяется по годам."
        if language.is_russian
        else f"The seasonality hypothesis is {'plausible' if supported else 'weak'} for `{plan.metric}` over `{plan.time_axis}`. "
        f"The month-of-year mean spread is {spread:.2f}; strongest months are {top_text}. "
        "Seasonal spikes are convincing only if the same month pattern repeats across years."
    )
    rows = by_month.sort_values("month").to_dict(orient="records")
    limitations = ["Средние по месяцам могут искажаться one-off spikes или sparse months."] if language.is_russian else ["Month averages can be distorted by one-off spikes or sparse months."]
    next_steps = ["Проверить повторяемость strongest months по годам."] if language.is_russian else ["Check year-by-year repeatability of the strongest months."]
    confidence = _confidence_assessment(
        supported=supported,
        rows=len(working),
        groups=len(by_month),
        evidence_strength="medium",
        limiting_factors=limitations,
        what_would_change_confidence=next_steps[0],
    )
    summary = _finalize_hypothesis_summary(summary, confidence, language)
    return (
        HypothesisEvidence(
            rows=rows,
            findings=[summary],
            evidence=[f"Сгруппировал `{plan.metric}` по month-of-year через `{plan.time_axis}`."] if language.is_russian else [f"Grouped `{plan.metric}` by month-of-year using `{plan.time_axis}`."],
            limitations=limitations,
            next_steps=next_steps,
            code="df.assign(month=date.dt.month).groupby('month')[metric].agg(['count', 'mean', 'sum'])",
            result_preview=by_month.sort_values("month").to_string(index=False),
            artifacts=[_table_artifact(f"{plan.metric} seasonality evidence", rows, frame, {"operator": "seasonality"})],
        ),
        HypothesisConclusion(summary=summary, confidence=confidence.level, supported=supported),
    )


def _validate_shipping_delay_effect(
    frame: HypothesisFrame,
    df: pd.DataFrame,
    plan: HypothesisValidationPlan,
) -> tuple[HypothesisEvidence, HypothesisConclusion | None]:
    metric = plan.metric
    order_col = _find_order_date_column(df, plan.time_axis)
    ship_col = _find_ship_date_column(df)
    if metric not in df.columns or not order_col or not ship_col:
        return HypothesisEvidence(limitations=["The shipping-delay hypothesis needs a metric plus order and ship date columns."]), None
    working = df[[metric, order_col, ship_col]].copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    working[order_col] = pd.to_datetime(working[order_col], errors="coerce")
    working[ship_col] = pd.to_datetime(working[ship_col], errors="coerce")
    working = working.dropna(subset=[metric, order_col, ship_col])
    if working.empty:
        return HypothesisEvidence(limitations=[f"`{metric}`, `{order_col}`, and `{ship_col}` do not have enough valid rows."]), None
    working["delivery_delay_days"] = (working[ship_col] - working[order_col]).dt.days
    working = working[working["delivery_delay_days"].notna()]
    if working.empty:
        return HypothesisEvidence(limitations=["Delivery delay could not be computed from the available date columns."]), None
    working["period"] = working[order_col].dt.to_period("M").dt.to_timestamp()
    working["delay_bucket"] = pd.cut(
        working["delivery_delay_days"],
        bins=[-10_000, 1, 3, 5, 10_000],
        labels=["0-1 days", "2-3 days", "4-5 days", "6+ days"],
    ).astype(str)
    by_period = working.groupby("period")[metric].agg(total="sum", mean="mean", count="count").reset_index()
    by_period["total_delta"] = by_period["total"].diff()
    spike_periods = by_period.sort_values("total_delta", ascending=False).head(max(1, min(3, len(by_period))))
    spike_keys = set(spike_periods["period"].dropna().tolist())
    working["is_spike_period"] = working["period"].isin(spike_keys)
    delay_summary = (
        working.groupby("delay_bucket", dropna=False)[metric]
        .agg(count="count", total="sum", mean="mean", median="median")
        .reset_index()
        .sort_values("total", ascending=False)
    )
    spike_delay = float(working.loc[working["is_spike_period"], "delivery_delay_days"].mean()) if working["is_spike_period"].any() else 0.0
    normal_delay = float(working.loc[~working["is_spike_period"], "delivery_delay_days"].mean()) if (~working["is_spike_period"]).any() else 0.0
    gap = spike_delay - normal_delay
    supported = abs(gap) >= 1.0
    leader = delay_summary.iloc[0]
    spike_text = ", ".join(str(period.date()) for period in spike_periods["period"].dropna().head(3))
    summary = (
        f"The shipping-delay hypothesis is {'partially supported' if supported else 'not supported strongly'} for `{metric}` spikes. "
        f"I computed delivery delay as `{ship_col}` minus `{order_col}` and compared delay behavior in spike periods versus other periods. "
        f"Spike-period average delay is {spike_delay:.2f} days versus {normal_delay:.2f} days outside spikes; the gap is {gap:+.2f} days. "
        f"The largest delay bucket by total `{metric}` is `{leader['delay_bucket']}` with total {float(leader['total']):.2f} across {int(leader['count'])} rows. "
        f"Spike periods checked: {spike_text or 'not enough period movement'}."
    )
    limitations = [
        "This is temporal association, not causal proof; shipping delay may be correlated with product, region, customer, or ship-mode mix.",
        "Rows are treated as records; order-level aggregation is better if the dataset is line-item based.",
    ]
    next_steps = [
        "Compare spike-period delay mix after controlling for the shipping-method field, and repeat at distinct order-like ID level if available."
    ]
    confidence = _confidence_assessment(
        supported=supported,
        rows=len(working),
        groups=int(working["delay_bucket"].nunique(dropna=True)),
        evidence_strength="medium",
        limiting_factors=limitations,
        what_would_change_confidence=next_steps[0],
    )
    summary = _finalize_hypothesis_summary(summary, confidence, ResponseLanguagePolicy.from_message(frame.raw_text, protected_terms=df.columns))
    rows = delay_summary.to_dict(orient="records")
    return (
        HypothesisEvidence(
            rows=rows,
            findings=[summary],
            evidence=[
                f"Computed delivery delay from `{ship_col}` minus `{order_col}`.",
                f"Compared average delay in strongest `{metric}` spike periods against non-spike periods.",
            ],
            limitations=limitations,
            next_steps=next_steps,
            code="delay = ship_date - order_date; compare delay buckets in spike periods",
            result_preview=delay_summary.to_string(index=False),
            artifacts=[_table_artifact(f"{metric} by delivery delay bucket", rows, frame, {"operator": "shipping_delay_effect", "dimension": "delivery_delay_days"})],
        ),
        HypothesisConclusion(summary=summary, confidence=confidence.level, supported=supported),
    )


def _infer_mechanisms(text: str) -> tuple[MechanismType, list[MechanismType]]:
    normalized = _normalize(text)
    mechanisms: list[MechanismType] = []
    if any(marker in normalized for marker in ("shipping delay", "shipping delays", "delivery delay", "delivery delays", "ship delay", "ship delays")):
        mechanisms.append(MechanismType.OPERATIONAL_EFFECT)
    if any(marker in normalized for marker in ("duplicate", "duplicates", "duplicat", "дублик")):
        mechanisms.append(MechanismType.DUPLICATE_INFLATION)
    if any(marker in normalized for marker in ("missing", "null", "пропуск")):
        mechanisms.append(MechanismType.MISSINGNESS_BIAS)
    if any(marker in normalized for marker in ("season", "seasonal", "сезон")):
        mechanisms.append(MechanismType.SEASONALITY)
    if any(marker in normalized for marker in ("sparse", "small sample", "few rows", "маленьк", "редк")):
        mechanisms.append(MechanismType.SPARSE_GROUP_INSTABILITY)
    if any(marker in normalized for marker in ("outlier", "outliers", "extreme", "large order", "large orders", "крупн", "выброс")):
        if any(marker in normalized for marker in ("most outliers", "creates most", "concentrat", "сконцентр")):
            mechanisms.append(MechanismType.OUTLIER_CONCENTRATION)
        else:
            mechanisms.append(MechanismType.OUTLIER_EFFECT)
    if any(marker in normalized for marker in ("few large", "single large", "top order", "only because of", "concentrat")):
        mechanisms.append(MechanismType.CONCENTRATION)
    if any(marker in normalized for marker in ("volume", "count", "record count", "order volume", "объем", "объём", "колич")):
        mechanisms.append(MechanismType.VOLUME_EFFECT)
    if any(marker in normalized for marker in ("growth", "shift", "change over time", "spike", "spikes", "рост", "сдвиг")):
        mechanisms.append(MechanismType.TEMPORAL_SHIFT)
    if any(marker in normalized for marker in ("correl", "relationship", "related", "связ")):
        mechanisms.append(MechanismType.CORRELATION_EFFECT)
    if not mechanisms:
        mechanisms.append(MechanismType.UNKNOWN)
    primary = mechanisms[0]
    if MechanismType.DUPLICATE_INFLATION in mechanisms:
        primary = MechanismType.DUPLICATE_INFLATION
    elif MechanismType.SEASONALITY in mechanisms:
        primary = MechanismType.SEASONALITY
    elif MechanismType.SPARSE_GROUP_INSTABILITY in mechanisms:
        primary = MechanismType.SPARSE_GROUP_INSTABILITY
    elif MechanismType.OUTLIER_CONCENTRATION in mechanisms:
        primary = MechanismType.OUTLIER_CONCENTRATION
    elif MechanismType.VOLUME_EFFECT in mechanisms:
        primary = MechanismType.VOLUME_EFFECT
    elif MechanismType.CONCENTRATION in mechanisms:
        primary = MechanismType.CONCENTRATION
    elif MechanismType.OPERATIONAL_EFFECT in mechanisms:
        primary = MechanismType.OPERATIONAL_EFFECT
    secondary = [item for item in mechanisms if item != primary]
    return primary, secondary


def _mentions_shipping_delay(text: str) -> bool:
    normalized = _normalize(text)
    return any(marker in normalized for marker in ("shipping delay", "shipping delays", "delivery delay", "delivery delays", "ship delay", "ship delays", "shipping", "delivery"))


def _find_order_date_column(df: pd.DataFrame, fallback: str = "") -> str | None:
    for col in df.columns:
        normalized = _normalize(col)
        if "order" in normalized and "date" in normalized:
            return str(col)
    return fallback if fallback in df.columns else None


def _find_ship_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        normalized = _normalize(col)
        if ("ship" in normalized or "delivery" in normalized) and "date" in normalized:
            return str(col)
    return None


def _confidence_assessment(
    *,
    supported: bool | None,
    rows: int,
    groups: int,
    evidence_strength: str,
    limiting_factors: list[str],
    what_would_change_confidence: str,
) -> ConfidenceAssessment:
    if rows >= 100 and groups >= 3 and evidence_strength == "high" and not limiting_factors:
        level = "high"
    elif rows >= 30 and groups >= 2 and evidence_strength in {"high", "medium"}:
        level = "medium"
    else:
        level = "low"
    if supported is False and evidence_strength == "medium" and rows >= 30:
        level = "medium"
    reason = (
        f"based on {rows:,} valid rows across {groups:,} group{'s' if groups != 1 else ''}"
        if groups
        else f"based on {rows:,} valid rows"
    )
    return ConfidenceAssessment(
        level=level,
        reason=reason,
        evidence_strength=evidence_strength,
        limiting_factors=limiting_factors[:3],
        what_would_change_confidence=what_would_change_confidence,
    )


def _finalize_hypothesis_summary(
    summary: str,
    confidence: ConfidenceAssessment,
    language: ResponseLanguagePolicy,
) -> str:
    if language.is_russian:
        level = {"high": "высокая", "medium": "средняя", "low": "низкая"}.get(confidence.level, confidence.level)
        factors = "; ".join(confidence.limiting_factors)
        parts = [summary, f"Уверенность: {level}, {confidence.reason}."]
        if factors:
            parts.append(f"Ограничение: {factors}")
        if confidence.what_would_change_confidence:
            parts.append(f"Следующая проверка: {confidence.what_would_change_confidence}")
        return " ".join(parts)
    factors = "; ".join(confidence.limiting_factors)
    parts = [summary, f"Confidence is {confidence.level}: {confidence.reason}."]
    if factors:
        parts.append(f"Limitation: {factors}")
    if confidence.what_would_change_confidence:
        parts.append(f"Next validation: {confidence.what_would_change_confidence}")
    return " ".join(parts)


def _resolve_metric(question: str, df: pd.DataFrame, branch_state: dict[str, Any], semantic_profile: SemanticDatasetProfile) -> str:
    normalized = _normalize(question)
    numeric = [str(col) for col in df.select_dtypes(include="number").columns if not _looks_identifier_name(str(col))]
    for col in numeric:
        if _normalize(col) in normalized:
            return col
    money_markers = ("sales", "revenue", "profit", "amount", "value", "cost", "price", "продаж", "выруч", "доход")
    if any(marker in normalized for marker in money_markers):
        for col in numeric:
            col_norm = _normalize(col)
            if any(marker in col_norm for marker in money_markers):
                return col
    active_metric = str(branch_state.get("active_metric") or "").strip()
    if active_metric in df.columns and active_metric in numeric:
        return active_metric
    metrics = [column.name for column in semantic_profile.metrics if column.name in numeric]
    return metrics[0] if metrics else (numeric[0] if numeric else "")


def _resolve_target(
    question: str,
    df: pd.DataFrame,
    branch_state: dict[str, Any],
    semantic_profile: SemanticDatasetProfile,
    value_match: CategoryValueMatch | None,
) -> HypothesisTarget:
    if value_match:
        return HypothesisTarget(
            entity=value_match.matched_value,
            column=value_match.matched_column,
            value=value_match.matched_value,
            semantic_role=value_match.semantic_role,
            confidence=value_match.confidence,
        )
    normalized = _normalize(question)
    for column in semantic_profile.columns:
        if column.role not in {SemanticRole.DIMENSION, SemanticRole.TEXT}:
            continue
        col_norm = _normalize(column.name)
        if col_norm and col_norm in normalized:
            return HypothesisTarget(entity=column.name, column=column.name, semantic_role=column.role.value, confidence=0.9)
        if _entity_priority(normalized, col_norm):
            return HypothesisTarget(entity=column.name, column=column.name, semantic_role=column.role.value, confidence=0.86)
    active_dimension = str(branch_state.get("active_dimension") or "").strip()
    if active_dimension in df.columns:
        return HypothesisTarget(entity=active_dimension, column=active_dimension, semantic_role="dimension", confidence=0.72)
    dimensions = [column for column in semantic_profile.dimensions if column.name in df.columns and (column.cardinality or 0) > 1]
    if dimensions:
        column = dimensions[0]
        return HypothesisTarget(entity=column.name, column=column.name, semantic_role=column.role.value, confidence=0.45)
    return HypothesisTarget()


def _resolve_time_axis(question: str, df: pd.DataFrame, branch_state: dict[str, Any], semantic_profile: SemanticDatasetProfile) -> str:
    active = str(branch_state.get("active_time_axis") or "").strip()
    if active in df.columns:
        return active
    normalized = _normalize(question)
    for column in semantic_profile.timestamps:
        if column.name in df.columns and (_normalize(column.name) in normalized or any(marker in normalized for marker in ("growth", "season", "time", "over time", "рост", "сезон"))):
            return column.name
    return semantic_profile.timestamps[0].name if semantic_profile.timestamps else ""


def _comparison_and_goal(mechanism: MechanismType) -> tuple[str, str]:
    mapping = {
        MechanismType.VOLUME_EFFECT: ("volume_vs_average_value", "determine whether dominance comes from count or order value"),
        MechanismType.OUTLIER_CONCENTRATION: ("outlier_concentration", "measure concentration of outliers inside the target category"),
        MechanismType.OUTLIER_EFFECT: ("mean_vs_median_and_top_record_share", "test whether large records distort group strength"),
        MechanismType.SPARSE_GROUP_INSTABILITY: ("raw_vs_min_sample_ranking", "check whether ranking collapses after sample-size filtering"),
        MechanismType.DUPLICATE_INFLATION: ("before_after_deduplication", "estimate metric inflation from exact duplicates"),
        MechanismType.SEASONALITY: ("month_of_year_spread", "test whether temporal movement is seasonal"),
        MechanismType.CONCENTRATION: ("top_record_share", "check whether concentration explains apparent strength"),
    }
    return mapping.get(mechanism, ("generic_evidence_check", "validate the stated analytical mechanism"))


def _operator_name(mechanism: MechanismType) -> str:
    return {
        MechanismType.VOLUME_EFFECT: "validate_volume_effect",
        MechanismType.OUTLIER_CONCENTRATION: "validate_outlier_concentration",
        MechanismType.OUTLIER_EFFECT: "validate_outlier_concentration",
        MechanismType.SPARSE_GROUP_INSTABILITY: "validate_sparse_group_instability",
        MechanismType.DUPLICATE_INFLATION: "validate_duplicate_inflation",
        MechanismType.SEASONALITY: "validate_seasonality",
        MechanismType.CONCENTRATION: "validate_concentration",
    }.get(mechanism, "validate_generic_hypothesis")


def _metric_dimension_frame(df: pd.DataFrame, metric: str, dimension: str) -> pd.DataFrame:
    if metric not in df.columns or dimension not in df.columns:
        return pd.DataFrame()
    working = df[[dimension, metric]].copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    return working.dropna(subset=[metric])


def _iqr_outliers(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, float, float]:
    q1 = float(df[metric].quantile(0.25))
    q3 = float(df[metric].quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    scored = df.copy()
    scored["is_outlier"] = (scored[metric] < lower) | (scored[metric] > upper)
    return scored, lower, upper


def _target_row(grouped: pd.DataFrame, dimension: str, value: str) -> pd.Series | None:
    if not value:
        return None
    rows = grouped[grouped[dimension].astype(str).str.casefold() == str(value).casefold()]
    return None if rows.empty else rows.iloc[0]


def _table_artifact(title: str, rows: list[dict[str, Any]], frame: HypothesisFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_type": "table",
        "title": title,
        "content": rows,
        "visibility": "user",
        "pinned": True,
        "metadata": {
            "branch_type": "hypothesis_validation",
            "analysis_type": "hypothesis_validation",
            "evidence_role": "supports_current_hypothesis",
            "supports_current_target": True,
            "claim_type": "hypothesis",
            "metric": frame.target_metric,
            "dimension": frame.target.column,
            "matched_value": frame.target.value,
            "mechanism": frame.mechanism.value,
            **metadata,
        },
    }


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
        "code": code,
        "result_preview": result_preview,
        "result_base64": "",
        "exec_error": None,
        "engine": "pandas",
        "needs_data": True,
        "use_case": "data_analytics",
        "loaded_skills": ["data-analysis", "csv-dataframe-analysis", "reporting"],
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
            "loaded_skills": ["data-analysis", "csv-dataframe-analysis", "reporting"],
            "tool_timeline": timeline,
            "sql_metadata": {},
        },
        "trace_metadata": trace_metadata,
        "critic_verdict": "",
        "critic_feedback": "",
        "artifacts": artifacts,
    }


def _entity_priority(question: str, column_name: str) -> bool:
    pairs = (
        (("city", "cities", "город"), ("city", "город")),
        (("category", "categories", "категор"), ("category", "категор")),
        (("segment", "segments", "сегмент"), ("segment", "сегмент")),
        (("ship", "shipping", "delivery", "class"), ("ship", "mode", "delivery", "class")),
        (("country", "страна"), ("country", "страна")),
        (("region", "регион"), ("region", "регион")),
    )
    return any(any(marker in question for marker in q) and any(marker in column_name for marker in c) for q, c in pairs)


def _looks_identifier_name(name: str) -> bool:
    normalized = _normalize(name)
    return any(marker in normalized for marker in (" id", "_id", "identifier", "uuid", "row id", "code", "postal", "zip"))


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())
