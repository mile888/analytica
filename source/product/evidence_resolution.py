from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from source.product.analytical_graph import (
    evaluate_large_record_hypothesis,
    synthesize_contradiction,
    synthesize_support,
    synthesize_validation,
    transformation_from_rows,
)
from source.product.language_policy import DetectedLanguage, ResponseLanguagePolicy


@dataclass(frozen=True)
class ActiveAnalyticalTarget:
    branch_id: str = ""
    branch_type: str = ""
    metric: str = ""
    dimension: str = ""
    time_axis: str = ""
    hypothesis: str = ""
    finding_id: str = ""
    chart_id: str = ""
    evidence_subject: str = ""
    active_entities: list[str] = field(default_factory=list)
    active_mechanism: str = ""
    active_transformation: str = ""
    response_language: str = ""
    recency_score: float = 1.0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceResolution:
    target: ActiveAnalyticalTarget | None
    score: float
    reason: str


@dataclass(frozen=True)
class EvidenceTarget:
    target_id: str
    target_type: str
    metric: str = ""
    dimension: str = ""
    mechanism: str = ""
    text: str = ""


@dataclass(frozen=True)
class ClaimEvidence:
    claim_id: str
    evidence_id: str
    evidence_type: str
    role: str = "supports"
    metric: str = ""
    dimension: str = ""
    mechanism: str = ""
    strength: str = "medium"
    text: str = ""


@dataclass(frozen=True)
class EvidenceGraph:
    active_target: EvidenceTarget | None = None
    claim_evidence: list[ClaimEvidence] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BranchIdentity:
    metric: str = ""
    dimension: str = ""
    branch_type: str = ""
    active_entities: list[str] = field(default_factory=list)
    active_hypothesis: str = ""
    active_transformation: str = ""

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActiveTransformationState:
    metric: str = ""
    dimension: str = ""
    transformation_type: str = ""
    adjusted_ranking: list[dict[str, Any]] = field(default_factory=list)
    raw_ranking: list[dict[str, Any]] = field(default_factory=list)
    removed_rows: int = 0
    original_rows: int = 0
    filtered_rows: int = 0
    threshold: dict[str, Any] = field(default_factory=dict)
    ranking_scope: str = ""
    interpretation_summary: str = ""
    comparison_rows: list[dict[str, Any]] = field(default_factory=list)
    rank_shift: dict[str, Any] = field(default_factory=dict)
    outlier_dependence: dict[str, Any] = field(default_factory=dict)
    weaker_conclusions: list[str] = field(default_factory=list)
    stronger_conclusions: list[str] = field(default_factory=list)
    confidence_change: str = ""

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def transformation_state_from_payload(value: Any) -> ActiveTransformationState | None:
    if isinstance(value, ActiveTransformationState):
        return value
    if not isinstance(value, dict):
        return None
    ranking = value.get("adjusted_ranking")
    if ranking is None:
        ranking = value.get("filtered_ranking")
    if not isinstance(ranking, list):
        ranking = []
    raw = value.get("raw_ranking")
    if not isinstance(raw, list):
        raw = []
    threshold = value.get("threshold")
    if not isinstance(threshold, dict):
        threshold = {}
    comparison_rows = value.get("comparison_rows")
    if not isinstance(comparison_rows, list):
        comparison_rows = []
    rank_shift = value.get("rank_shift")
    if not isinstance(rank_shift, dict):
        rank_shift = {}
    outlier_dependence = value.get("outlier_dependence")
    if not isinstance(outlier_dependence, dict):
        outlier_dependence = {}
    return ActiveTransformationState(
        metric=str(value.get("metric") or ""),
        dimension=str(value.get("dimension") or ""),
        transformation_type=str(value.get("transformation_type") or value.get("analysis_type") or ""),
        adjusted_ranking=[row for row in ranking if isinstance(row, dict)],
        raw_ranking=[row for row in raw if isinstance(row, dict)],
        removed_rows=int(value.get("removed_rows") or 0),
        original_rows=int(value.get("original_rows") or value.get("original_row_count") or 0),
        filtered_rows=int(value.get("filtered_rows") or value.get("filtered_row_count") or value.get("row_count") or 0),
        threshold=threshold,
        ranking_scope=str(value.get("ranking_scope") or ""),
        interpretation_summary=str(value.get("interpretation_summary") or ""),
        comparison_rows=[row for row in comparison_rows if isinstance(row, dict)],
        rank_shift=rank_shift,
        outlier_dependence=outlier_dependence,
        weaker_conclusions=[str(item) for item in value.get("weaker_conclusions") or [] if str(item).strip()] if isinstance(value.get("weaker_conclusions"), list) else [],
        stronger_conclusions=[str(item) for item in value.get("stronger_conclusions") or [] if str(item).strip()] if isinstance(value.get("stronger_conclusions"), list) else [],
        confidence_change=str(value.get("confidence_change") or ""),
    )


def active_target_from_payload(value: Any) -> ActiveAnalyticalTarget | None:
    if isinstance(value, ActiveAnalyticalTarget):
        return value
    if not isinstance(value, dict):
        return None
    keys = set(ActiveAnalyticalTarget.__dataclass_fields__)
    payload = {key: value.get(key) for key in keys}
    entities = payload.get("active_entities")
    if not isinstance(entities, list):
        payload["active_entities"] = [str(item) for item in entities or [] if str(item).strip()]
    return ActiveAnalyticalTarget(**payload)


def build_active_target(
    *,
    question: str,
    state: dict[str, Any] | None = None,
    trace_metadata: dict[str, Any] | None = None,
    finding: Any = None,
    chart: dict[str, Any] | None = None,
    report_summary: str = "",
    run_id: str = "",
) -> ActiveAnalyticalTarget:
    state = state if isinstance(state, dict) else {}
    trace = trace_metadata if isinstance(trace_metadata, dict) else {}
    chart = chart if isinstance(chart, dict) else {}
    metric = _first(trace.get("metric"), state.get("active_metric"), chart.get("metric"))
    dimension = _first(trace.get("dimension"), state.get("active_dimension"), chart.get("dimension"))
    time_axis = _first(trace.get("time_axis"), trace.get("timestamp"), state.get("active_time_axis"), chart.get("time_axis"))
    branch_type = _first(trace.get("active_branch_type"), trace.get("analysis_type"), state.get("active_branch_type"), chart.get("branch_type"))
    mechanism = _first(trace.get("hypothesis_mechanism"), trace.get("mechanism"))
    matched_value = _first(trace.get("matched_category_value"), trace.get("matched_value"))
    hypothesis = _first(state.get("active_hypothesis"), report_summary, getattr(finding, "text", ""))
    subject = _first(report_summary, getattr(finding, "text", ""), state.get("last_successful_answer_summary"), hypothesis)
    language = ResponseLanguagePolicy.from_message(
        question,
        protected_terms=[item for item in (metric, dimension, time_axis, matched_value) if item],
    )
    entities = [item for item in (metric, dimension, time_axis, matched_value) if item]
    return ActiveAnalyticalTarget(
        branch_id=run_id,
        branch_type=branch_type,
        metric=metric,
        dimension=dimension,
        time_axis=time_axis,
        hypothesis=hypothesis,
        finding_id=str(getattr(finding, "finding_id", "") or ""),
        chart_id=str(chart.get("artifact_id") or ""),
        evidence_subject=subject,
        active_entities=list(dict.fromkeys(str(item) for item in entities if str(item).strip())),
        active_mechanism=mechanism,
        active_transformation=_first(trace.get("active_transformation"), state.get("active_transformation"), chart.get("active_transformation")),
        response_language=language.language.value,
        recency_score=1.0,
    )


def is_evidence_followup(question: str) -> bool:
    text = _normalize(question)
    return any(
        marker in text
        for marker in (
            "evidence",
            "support",
            "supports",
            "supported",
            "proof",
            "prove",
            "why",
            "reliable",
            "confidence",
            "limitation",
            "validate",
            "validation",
            "how would you validate",
            "how can we validate",
            "contradict",
            "contradicts",
            "contradiction",
            "counterevidence",
            "counter evidence",
            "refute",
            "against the hypothesis",
            "подтверж",
            "доказ",
            "почему",
            "надеж",
            "надёж",
            "уверен",
            "огранич",
            "противореч",
            "опроверг",
        )
    )


def resolve_evidence_subject(
    user_message: str,
    active_target: Any,
    branch_state: dict[str, Any] | None = None,
    recent_findings: list[Any] | None = None,
) -> EvidenceResolution:
    target = active_target_from_payload(active_target)
    state = branch_state if isinstance(branch_state, dict) else {}
    candidates: list[tuple[ActiveAnalyticalTarget, str]] = []
    if target:
        candidates.append((target, "active_target"))
    state_target = active_target_from_payload(state.get("active_analytical_target"))
    if state_target:
        candidates.append((state_target, "state_active_target"))
    if not candidates:
        for index, finding in enumerate(reversed(recent_findings or [])):
            text = getattr(finding, "text", finding)
            candidates.append(
                (
                    ActiveAnalyticalTarget(
                        branch_type="finding_memory",
                        evidence_subject=str(text or ""),
                        recency_score=max(0.1, 0.6 - index * 0.08),
                    ),
                    "recent_finding",
                )
            )
    if not candidates:
        return EvidenceResolution(None, 0.0, "no_target")
    scored = [(target_item, _target_score(user_message, target_item, state), reason) for target_item, reason in candidates]
    scored.sort(key=lambda item: item[1], reverse=True)
    best, score, reason = scored[0]
    return EvidenceResolution(best, score, reason)


def evidence_response_text(question: str, target: ActiveAnalyticalTarget, *, artifact_rows: list[dict[str, Any]] | None = None) -> str:
    language = ResponseLanguagePolicy.from_message(
        question,
        protected_terms=target.active_entities,
        fallback=_language_fallback(target.response_language),
    )
    rows = artifact_rows or []
    transformed = transformation_from_rows(metric=target.metric, dimension=target.dimension, rows=rows)
    if transformed and target.active_mechanism in {"outlier_concentration", "outlier_effect", "concentration", "large_record_dependence"}:
        evaluation = evaluate_large_record_hypothesis(hypothesis=target.hypothesis or target.evidence_subject, transformation=transformed)
        if _is_validation_question(question):
            return synthesize_validation(evaluation)
        if _is_contradiction_question(question):
            return synthesize_contradiction(evaluation)
        return synthesize_support(evaluation)
    if _is_validation_question(question):
        return _validation_response_en(target, rows)
    if _is_contradiction_question(question):
        return _contradiction_response_en(target, rows)
    if language.is_russian:
        return _evidence_response_ru(target, rows)
    return _evidence_response_en(target, rows)


def target_quality_next_steps(question: str, target: ActiveAnalyticalTarget) -> str:
    language = ResponseLanguagePolicy.from_message(
        question,
        protected_terms=target.active_entities,
        fallback=_language_fallback(target.response_language),
    )
    metric = f"`{target.metric}`" if target.metric else "активную метрику"
    if language.is_russian:
        return (
            "Продолжаем ветку качества про duplicates. Дальше стоит проверить exact-row duplicates по источнику или дате загрузки, "
            f"оценить инфляцию для {metric}, посмотреть, сконцентрированы ли duplicates в отдельных группах, "
            "и отдельно разобрать repeated order-like IDs как возможные строки одного заказа, а не автоматически удаляемые дубли."
        )
    metric_en = f"`{target.metric}`" if target.metric else "the active metric"
    return (
        "In the duplicate-quality branch, extend the picture with source and concentration checks: compare exact-row duplicates by load/source fields, "
        f"estimate inflation for {metric_en}, inspect whether duplicates cluster in specific groups, "
        "and treat repeated order-like IDs as possible line items rather than automatically removable duplicates."
    )


def _evidence_response_ru(target: ActiveAnalyticalTarget, rows: list[dict[str, Any]]) -> str:
    metric = f"`{target.metric}`" if target.metric else "активной метрике"
    dimension = f"`{target.dimension}`" if target.dimension else "активной группировке"
    mechanism = target.active_mechanism
    examples = _row_examples(target, rows).replace("Examples:", "Примеры:")
    if mechanism == "sparse_group_instability":
        return (
            f"Поддерживающие данные относятся именно к гипотезе про разреженные группы: проверка сравнила исходный рейтинг среднего {metric} по {dimension} "
            "с фильтром минимальной выборки и отметила, какие лидеры не проходят порог n. "
            f"{examples} Поэтому вывод держится на механизме нестабильности малой выборки: сильный средний показатель при малом n хрупок и должен проверяться медианой и устойчивостью ранга."
        )
    if mechanism == "volume_effect":
        return (
            f"Поддерживающие данные здесь — разложение {metric} по {dimension} на total, mean/median и count. "
            f"{examples} Если цель лидирует по total и count, но не по average/median, это поддерживает механизм доминирования за счет объема."
        )
    if mechanism in {"outlier_concentration", "outlier_effect", "concentration"}:
        return (
            f"Поддерживающие данные — концентрационная проверка для {metric} по {dimension}: считались outliers или доля крупнейшей записи внутри групп. "
            f"{examples} Гипотеза сильнее, если большая доля эффекта сидит в одной группе или нескольких крупных записях."
        )
    if mechanism == "duplicate_inflation" or target.branch_type == "data_quality":
        return (
            f"Поддерживающие данные — проверка duplicates и сравнение total {metric} до и после exact-row deduplication. "
            "Повторяющиеся order-like IDs не считаются автоматической инфляцией метрики, потому что они могут быть строками одного заказа."
        )
    subject = target.evidence_subject or target.hypothesis or f"{metric} по {dimension}"
    return (
        f"Поддерживающие данные относятся к текущей активной цели: {subject}. "
        f"Проверять нужно ту же цель: {metric} по {dimension}, без перехода к старым выводам или другой ветке."
    )


def _evidence_response_en(target: ActiveAnalyticalTarget, rows: list[dict[str, Any]]) -> str:
    metric = f"`{target.metric}`" if target.metric else "the active metric"
    dimension = f"`{target.dimension}`" if target.dimension else "the active grouping"
    mechanism = target.active_mechanism
    examples = _row_examples(target, rows)
    if mechanism == "sparse_group_instability":
        concrete = _sparse_support_sentence(target, rows)
        if concrete:
            return concrete
        return f"The evidence gap for {metric} by {dimension} is concrete: no small-sample leader rows are stored yet, so the next check should save raw rank, mean, and n for the top groups."
    if mechanism == "volume_effect":
        return f"For {dimension}, {examples} Volume pressure is volume-driven when the same group leads on total and record count but not on average or median {metric}."
    if mechanism in {"outlier_concentration", "outlier_effect", "concentration"}:
        concrete = _transformation_evidence_sentence(target, rows)
        if concrete:
            return concrete
        if "rank #" in (target.evidence_subject or ""):
            return _subject_without_system_narration(target.evidence_subject)
        return (
            f"{examples} Large-record dependence is supported only when the stored groups lose rank or mean after extreme {metric} records are removed."
        )
    if mechanism in {"operational_effect", "temporal_shift"}:
        operational = _operational_evidence_sentence(target, rows)
        if operational:
            return operational
        if target.evidence_subject:
            return _subject_without_system_narration(target.evidence_subject)
        return (
            f"No concrete delay bucket is stored for {metric} over `{target.time_axis}` yet; the next evidence answer needs spike-period delay, normal-period delay, and shipping-method mix."
        )
    if mechanism == "duplicate_inflation" or target.branch_type == "data_quality":
        return (
            f"Duplicate impact is grounded in the before/after total {metric} comparison after exact-row deduplication. "
            "Repeated order-like IDs remain separate because they may be line items rather than duplicate orders."
        )
    subject = target.evidence_subject or target.hypothesis or f"{metric} by {dimension}"
    return f"{subject} The next useful check is to validate {metric} by {dimension} with a concrete chart, table, or quality check."


def _contradiction_response_en(target: ActiveAnalyticalTarget, rows: list[dict[str, Any]]) -> str:
    metric = f"`{target.metric}`" if target.metric else "the active metric"
    dimension = f"`{target.dimension}`" if target.dimension else "the active grouping"
    mechanism = target.active_mechanism
    examples = _row_examples(target, rows)
    if mechanism in {"outlier_concentration", "outlier_effect", "concentration"}:
        stable = _stable_counterevidence_sentence(target, rows)
        if stable:
            return stable
        if "Counterevidence:" in (target.evidence_subject or ""):
            return "Stable counterevidence: " + _subject_without_system_narration(target.evidence_subject.split("Counterevidence:", 1)[1].strip())
        return (
            f"No strong concrete counterevidence is visible for `{target.dimension}` in the current robustness checks. "
            f"The stored `{target.metric}` by `{target.dimension}` results mainly show large-order fragility; contradiction still needs a stable median or n-filtered leader."
        )
    if mechanism == "sparse_group_instability":
        sparse = _sparse_counterevidence_sentence(target, rows)
        if sparse:
            return sparse
        stable = _stable_counterevidence_sentence(target, rows)
        if stable:
            return stable
        return (
            f"No strong concrete counterevidence is visible for the sparse-data hypothesis on {metric} by {dimension}: "
            f"the current checks do not contain a high-n leader that stayed near the top. {examples}"
        )
    if mechanism == "volume_effect":
        return (
            f"No strong concrete counterevidence is visible for the volume-effect hypothesis on {metric} by {dimension}: "
            f"the current checks do not show a group leading by average or median after controlling for record count. {examples}"
        )
    if mechanism in {"operational_effect", "temporal_shift"}:
        subject = _subject_without_system_narration(target.evidence_subject)
        if subject:
            gap = _sentence_containing(subject, "spike-period average delay")
            factual = f" {gap}" if gap else ""
            return (
                f"Concrete counterevidence is limited in the current operational shipping-delay check.{factual} "
                "The small spike-versus-normal delay gap weakens a strong causal reading, and shipping-method mix still has to be controlled."
            )
        return (
            f"Concrete counterevidence is limited for the operational hypothesis on {metric} over `{target.time_axis}`: the saved state has no controlled shipping-method comparison yet. "
            "That leaves the delay finding as association, not proof that delays drive spikes."
        )
    return (
        f"No strong concrete counterevidence is visible for the current target, {metric} by {dimension}. "
        "The current state does not contain a grounded counterexample or controlled stress-test result yet."
    )


def _validation_response_en(target: ActiveAnalyticalTarget, rows: list[dict[str, Any]]) -> str:
    metric = f"`{target.metric}`" if target.metric else "the active metric"
    dimension = f"`{target.dimension}`" if target.dimension else "the active grouping"
    mechanism = target.active_mechanism
    examples = _row_examples(target, rows)
    if mechanism in {"outlier_concentration", "outlier_effect", "concentration"}:
        examples = _rank_shift_examples(target, rows)
        return (
            f"Validate the active large-record hypothesis on {metric} by {dimension} as one chain: "
            "1. compare raw mean leaders with median leaders; 2. remove each group's largest record and re-rank; "
            "3. inspect top-record share for the leading groups; 4. require a minimum sample size before calling a group strong. "
            f"{examples} The hypothesis is stronger if raw leaders fall after these stress tests."
        )
    if mechanism == "sparse_group_instability":
        return (
            f"Validate the sparse-data hypothesis on {metric} by {dimension} by comparing raw mean rank, median rank, and rank after an n-threshold filter. "
            f"{examples} The hypothesis is supported if small-n leaders disappear while sufficiently large groups remain stable."
        )
    if mechanism == "volume_effect":
        return (
            f"Validate the volume hypothesis on {metric} by {dimension} by comparing total, mean, median, record count, and distinct order count if an order ID exists. "
            f"{examples} The hypothesis is stronger when total rank follows count rank more closely than average-value rank."
        )
    if mechanism in {"operational_effect", "temporal_shift"}:
        axis = f"`{target.time_axis}`" if target.time_axis else "the active time axis"
        return (
            f"Validate the operational hypothesis on {metric} over {axis}: 1. compute delivery delay from order and ship dates; "
            "2. bucket delays and compare total/average metric by bucket; 3. compare delay mix during spike periods versus normal periods; "
            "4. control for shipping-method mix so delay is not just a proxy for shipping class. "
            "The hypothesis is stronger if spike periods have materially higher delay-bucket share or delay-adjusted totals."
        )
    return (
        f"Validate the active target {metric} by {dimension} with a direct stress test tied to the same branch: compare raw average ranking versus adjusted ranking, median ranking, sample-size filtering, and any relevant quality checks."
    )


def _transformation_evidence_sentence(target: ActiveAnalyticalTarget, rows: list[dict[str, Any]]) -> str:
    shifted = [row for row in rows if "original_rank" in row and "adjusted_rank" in row]
    if not shifted:
        return ""
    dimension = target.dimension
    dropped = sorted(
        [row for row in shifted if _num(row.get("rank_change")) >= 5 or _num(row.get("adjusted_rank")) > 20],
        key=lambda row: (-_num(row.get("rank_change")), _num(row.get("original_rank"))),
    )[:3]
    stable = sorted(
        [row for row in shifted if _num(row.get("original_rank")) <= 10 and _num(row.get("adjusted_rank")) <= 10 and abs(_num(row.get("rank_change"))) <= 3],
        key=lambda row: _num(row.get("adjusted_rank")),
    )[:3]
    parts = []
    if dropped:
        parts.append("Dropped raw leaders: " + _rank_shift_phrase(dropped, dimension) + ".")
    if stable:
        parts.append("Stable leaders: " + _rank_shift_phrase(stable, dimension) + ".")
    if not parts:
        return ""
    return (
        " ".join(parts)
        + f" That is the concrete support for large-record dependence in `{target.metric}` by `{target.dimension}`."
    )


def _stable_counterevidence_sentence(target: ActiveAnalyticalTarget, rows: list[dict[str, Any]]) -> str:
    shifted = [row for row in rows if "original_rank" in row and "adjusted_rank" in row]
    if not shifted:
        return ""
    stable = sorted(
        [row for row in shifted if _num(row.get("original_rank")) <= 10 and _num(row.get("adjusted_rank")) <= 10 and abs(_num(row.get("rank_change"))) <= 3],
        key=lambda row: _num(row.get("adjusted_rank")),
    )[:4]
    if not stable:
        return ""
    return (
        f"The strongest counterevidence is that some `{target.dimension}` groups stayed near the top after the stress test: "
        f"{_rank_shift_phrase(stable, target.dimension)}. "
        f"That weakens an overbroad claim that all high `{target.metric}` cities are only artifacts of isolated large records."
    )


def _sparse_support_sentence(target: ActiveAnalyticalTarget, rows: list[dict[str, Any]]) -> str:
    dimension = target.dimension
    if not dimension or not rows:
        subject = _subject_without_system_narration(target.evidence_subject)
        return subject if subject and "n=" in subject else ""
    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get(dimension)
        count = _num(row.get("count") or row.get("n") or row.get("record_count") or row.get("adjusted_count") or row.get("original_count"))
        mean = _num(row.get("mean") or row.get("original_mean") or row.get("adjusted_mean"))
        rank = _num(row.get("rank") or row.get("raw_mean_rank") or row.get("original_rank") or row.get("adjusted_rank"))
        if value is not None and 0 < count <= 5:
            candidates.append((rank or 9999, str(value), count, mean))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[0], item[2]))
    facts = [
        f"`{value}` has n={int(count)}" + (f" with mean `{target.metric}` {mean:.2f}" if mean else "")
        for _, value, count, mean in candidates[:4]
    ]
    return (
        ", ".join(facts)
        + f". Those top-ranked `{dimension}` groups are too small to treat as stable leaders without a median and minimum-n check."
        + " This is sample-size instability in the actual leaders, not a broad warning detached from the ranking."
    )


def _sparse_counterevidence_sentence(target: ActiveAnalyticalTarget, rows: list[dict[str, Any]]) -> str:
    dimension = target.dimension
    if not dimension or not rows:
        return ""
    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        count = _num(row.get("count") or row.get("n") or row.get("record_count") or row.get("adjusted_count") or row.get("original_count"))
        rank = _num(row.get("rank") or row.get("raw_mean_rank") or row.get("adjusted_rank") or row.get("original_rank"))
        value = row.get(dimension)
        if value is not None and count >= 6 and (rank == 0 or rank <= 10):
            candidates.append((rank or 9999, row, count))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0])
    examples = []
    for _, row, count in candidates[:3]:
        mean = row.get("mean", row.get("adjusted_mean", row.get("median", 0)))
        examples.append(f"`{row.get(dimension)}` (mean {_num(mean):.2f}, n={int(count)})")
    return (
        "The concrete counterevidence is that some sufficiently sampled groups remain strong: "
        + ", ".join(examples)
        + f". That weakens a blanket claim that the `{dimension}` ranking is only a sparse-sample artifact."
    )


def _operational_evidence_sentence(target: ActiveAnalyticalTarget, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    bucket_key = "delay_bucket" if any("delay_bucket" in row for row in rows) else target.dimension
    usable = [row for row in rows if isinstance(row, dict) and row.get(bucket_key) is not None]
    if not usable:
        return ""
    top = sorted(usable, key=lambda row: _num(row.get("total")), reverse=True)[0]
    parts = [
        f"`{top.get(bucket_key)}` has the largest total `{target.metric}` in the delivery delay evidence ({_num(top.get('total')):.2f} across {int(_num(top.get('count')))} rows)."
    ]
    if len(usable) > 1:
        next_row = sorted(usable, key=lambda row: _num(row.get("total")), reverse=True)[1]
        parts.append(
            f"The next bucket is `{next_row.get(bucket_key)}` ({_num(next_row.get('total')):.2f}, n={int(_num(next_row.get('count')))})."
        )
    subject = _subject_without_system_narration(target.evidence_subject)
    if subject and "spike-period average delay" in subject.lower():
        spike_sentence = _sentence_containing(subject, "spike-period average delay")
        if spike_sentence:
            parts.append(spike_sentence)
    return " ".join(parts)


def _sentence_containing(text: str, marker: str) -> str:
    for sentence in str(text or "").split(". "):
        if marker.casefold() in sentence.casefold():
            return sentence.strip().rstrip(".") + "."
    return ""


def _rank_shift_examples(target: ActiveAnalyticalTarget, rows: list[dict[str, Any]]) -> str:
    shifted = [row for row in rows if "original_rank" in row and "adjusted_rank" in row]
    if not shifted:
        return ""
    return " Current before/after examples: " + _rank_shift_phrase(shifted[:3], target.dimension) + "."


def _rank_shift_phrase(rows: list[dict[str, Any]], dimension: str) -> str:
    parts = []
    for row in rows:
        parts.append(
            f"`{row.get(dimension)}` rank #{int(_num(row.get('original_rank')))} -> #{int(_num(row.get('adjusted_rank')))} "
            f"(mean {_num(row.get('original_mean')):.2f} -> {_num(row.get('adjusted_mean')):.2f}, n={int(_num(row.get('adjusted_count') or row.get('original_count')))})"
        )
    return ", ".join(parts)


def _subject_without_system_narration(subject: str) -> str:
    text = str(subject or "").strip()
    replacements = (
        ("The hypothesis is supported by the active transformation for", "The rank shifts support the hypothesis for"),
        ("The hypothesis is not strongly supported by the active transformation for", "The rank shifts weakly support the hypothesis for"),
        ("Evidence for large-record dependence:", "Supporting rank shifts:"),
        ("Counterevidence:", "Stable counterexamples:"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    sentences = []
    for sentence in text.split(". "):
        lowered = sentence.lower()
        if "based on 0 valid rows" in lowered or "removed rows:" in lowered and " of 0" in lowered:
            continue
        if "active transformation" in lowered:
            continue
        sentences.append(sentence.strip())
    return ". ".join(item for item in sentences if item).strip()


def _num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _is_contradiction_question(question: str) -> bool:
    text = _normalize(question)
    return any(marker in text for marker in ("contradict", "contradiction", "counterevidence", "counter evidence", "refute", "against the hypothesis", "противореч", "опроверг"))


def _is_validation_question(question: str) -> bool:
    text = _normalize(question)
    return any(marker in text for marker in ("validate", "validation", "how would you validate", "how can we validate", "как проверить", "как это проверить"))


def _row_examples(target: ActiveAnalyticalTarget, rows: list[dict[str, Any]]) -> str:
    dimension = target.dimension
    if not dimension or not rows:
        return ""
    parts: list[str] = []
    for row in rows[:4]:
        value = row.get(dimension)
        if value is None:
            continue
        count = row.get("count") or row.get("record_count")
        mean = row.get("mean") or row.get("avg_value")
        share = row.get("top_order_share") or row.get("outlier_share")
        fields = []
        if count is not None:
            fields.append(f"n={_fmt_number(count, decimals=0)}")
        if mean is not None:
            fields.append(f"mean={_fmt_number(mean)}")
        if share is not None:
            fields.append(f"share={_fmt_number(float(share) * 100)}%")
        parts.append(f"`{value}`" + (f" ({', '.join(fields)})" if fields else ""))
    return "Examples: " + ", ".join(parts) + "." if parts else ""


def _target_score(user_message: str, target: ActiveAnalyticalTarget, state: dict[str, Any]) -> float:
    text = _normalize(user_message)
    score = float(target.recency_score or 0.0)
    if target.branch_id:
        score += 0.4
    if target.branch_type and str(state.get("active_branch_type") or "") == target.branch_type:
        score += 0.35
    if target.active_mechanism and target.active_mechanism in text:
        score += 0.3
    for entity in target.active_entities:
        if entity and _normalize(entity) in text:
            score += 0.25
    if target.branch_type in {"hypothesis_validation", "data_quality"}:
        score += 0.35
    if not target.metric and not target.dimension and target.branch_type == "finding_memory":
        score -= 0.35
    return score


def _fmt_number(value: Any, decimals: int = 2) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if decimals == 0:
        return str(int(number))
    return f"{number:.{decimals}f}"


def _first(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _language_fallback(value: str) -> DetectedLanguage:
    return DetectedLanguage.RUSSIAN if str(value).lower().startswith("ru") else DetectedLanguage.ENGLISH


def _normalize(value: str) -> str:
    return " ".join(str(value or "").casefold().split())
