from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ConversationIntent(StrEnum):
    OVERVIEW = "overview"
    DATASET_OVERVIEW = "dataset_overview"
    CHART = "chart"
    ANOMALY = "anomaly"
    VALIDATION = "validation"
    CAUSAL_HYPOTHESIS = "causal_hypothesis"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"
    FOLLOW_UP = "follow_up"
    CHALLENGE = "challenge"
    REPORT = "report"
    QUALITY_AUDIT = "quality_audit"
    ANALYTICAL_TRANSFORMATION = "analytical_transformation"
    EXPLORATORY_REASONING = "exploratory_reasoning"
    CONCEPTUAL_QUESTION = "conceptual_question"
    WORKFLOW_QUESTION = "workflow_question"
    CONTEXT_RESET = "context_reset"
    INVESTIGATION_BRANCH = "investigation_branch"
    CLARIFICATION_NEEDED = "clarification_needed"


class InvestigationPhase(StrEnum):
    ONBOARDING = "onboarding"
    EXPLORATORY_ANALYSIS = "exploratory_analysis"
    FOCUSED_INVESTIGATION = "focused_investigation"
    VALIDATION = "validation"
    ANOMALY_ANALYSIS = "anomaly_analysis"
    EVIDENCE_STRENGTHENING = "evidence_strengthening"
    REPORTING = "reporting"


@dataclass(frozen=True)
class ResolvedIntent:
    primary: ConversationIntent
    components: list[ConversationIntent] = field(default_factory=list)
    is_compound: bool = False
    requires_continuity: bool = False


@dataclass(frozen=True)
class InvestigationConversationState:
    active_topic: str = ""
    active_metric: str = ""
    active_dimension: str = ""
    active_time_axis: str = ""
    active_chart: dict[str, Any] = field(default_factory=dict)
    active_chart_aggregation: str = ""
    active_analysis_type: str = ""
    active_hypothesis: str = ""
    active_anomaly_target: str = ""
    unresolved_questions: list[str] = field(default_factory=list)
    recent_findings: list[str] = field(default_factory=list)
    recent_evidence: list[str] = field(default_factory=list)
    active_validation_target: str = ""
    active_comparison_focus: str = ""
    active_branch_type: str = ""
    active_objective: str = ""
    decomposition_dimensions: list[str] = field(default_factory=list)
    current_objective: str = ""
    active_analytical_target: dict[str, Any] = field(default_factory=dict)
    phase: InvestigationPhase = InvestigationPhase.ONBOARDING
    topic_confidence: float = 0.0
    latest_intent: str = ConversationIntent.EXPLORATORY_REASONING.value

    def to_payload(self) -> dict[str, Any]:
        return {
            "active_topic": self.active_topic,
            "active_metric": self.active_metric,
            "active_dimension": self.active_dimension,
            "active_time_axis": self.active_time_axis,
            "active_chart": self.active_chart,
            "active_chart_aggregation": self.active_chart_aggregation,
            "active_analysis_type": self.active_analysis_type,
            "active_hypothesis": self.active_hypothesis,
            "active_anomaly_target": self.active_anomaly_target,
            "unresolved_questions": self.unresolved_questions,
            "recent_findings": self.recent_findings,
            "recent_evidence": self.recent_evidence,
            "active_validation_target": self.active_validation_target,
            "active_comparison_focus": self.active_comparison_focus,
            "active_branch_type": self.active_branch_type,
            "active_objective": self.active_objective,
            "decomposition_dimensions": self.decomposition_dimensions,
            "current_objective": self.current_objective,
            "active_analytical_target": self.active_analytical_target,
            "phase": self.phase.value,
            "topic_confidence": self.topic_confidence,
            "latest_intent": self.latest_intent,
        }


ActiveInvestigationState = InvestigationConversationState


def resolve_user_intent(question: str, *, has_active_context: bool = False) -> ResolvedIntent:
    text = _normalize(question)
    components: list[ConversationIntent] = []
    checks: tuple[tuple[ConversationIntent, tuple[str, ...]], ...] = (
        (ConversationIntent.OVERVIEW, ("what can you say", "overview", "summarize", "describe data", "что ты можешь сказать", "что можешь сказать", "опиши данные", "опиши датасет")),
        (ConversationIntent.CONTEXT_RESET, ("reset", "start over", "new topic", "заново", "сброс", "новая тема")),
        (ConversationIntent.CHART, ("chart", "plot", "graph", "visual", "top ", "show top", "график", "диаграм", "построй", "построить", "визуал")),
        (ConversationIntent.ANOMALY, ("outlier", "anomal", "unusual", "extreme", "аномал", "выброс", "необыч")),
        (ConversationIntent.ANALYTICAL_TRANSFORMATION, ("without outlier", "remove outlier", "remove outliers", "extreme orders", "extreme values", "use median", "median instead", "normalize by volume", "exclude sparse", "minimum sample", "убрать выброс", "без выброс", "убрать extreme", "экстрем", "медиан", "нормализ", "маленькие выборки")),
        (ConversationIntent.VALIDATION, ("validate", "evidence", "prove", "support", "доказ", "подтверж", "проверь")),
        (ConversationIntent.CAUSAL_HYPOTHESIS, ("explain", "driven", "driver", "caus", "affect", "volume", "count", "record", "order volume", "влияет", "объяс", "объем", "объём", "колич", "заказ", "связано", "связана", "связан")),
        (ConversationIntent.COMPARISON, ("compare", "highest", "lowest", "strongest", "weakest", "top", "by ", "across", "сравн", "сильн", "слаб", "топ", " по ")),
        (ConversationIntent.EXPLANATION, ("explain this chart", "why", "почему", "объясни", "что значит")),
        (ConversationIntent.REPORT, ("report", "memo", "презентац", "отчет", "отчёт")),
        (ConversationIntent.CONCEPTUAL_QUESTION, ("what is", "how to interpret", "что такое", "как понимать", "что значит")),
        (ConversationIntent.WORKFLOW_QUESTION, ("how should we analyze", "plan", "workflow", "как анализировать", "план анализа")),
        (ConversationIntent.QUALITY_AUDIT, ("quality", "missing", "duplicate", "null", "качеств", "пропуск", "дублик")),
        (ConversationIntent.EXPLORATORY_REASONING, ("relationship", "relationships", "dependency", "dependencies", "correl", "business question", "important fields", "завис", "связ", "коррел", "бизнес вопрос", "можно исследовать", "важные поля", "важны")),
    )
    for intent, markers in checks:
        if any(marker in text for marker in markers):
            components.append(intent)
    if has_active_context and _is_contextual_reference(text):
        components.append(ConversationIntent.FOLLOW_UP)
    if not components:
        components.append(ConversationIntent.FOLLOW_UP if has_active_context else ConversationIntent.EXPLORATORY_REASONING)
    primary = _prioritize_intent(components, has_active_context=has_active_context)
    return ResolvedIntent(
        primary=primary,
        components=list(dict.fromkeys(components)),
        is_compound=len(set(components)) > 1,
        requires_continuity=has_active_context and primary not in {ConversationIntent.OVERVIEW, ConversationIntent.EXPLORATORY_REASONING, ConversationIntent.CONCEPTUAL_QUESTION, ConversationIntent.WORKFLOW_QUESTION, ConversationIntent.CONTEXT_RESET},
    )


def build_conversation_state(
    *,
    previous: dict[str, Any] | None = None,
    question: str = "",
    intent: ResolvedIntent | None = None,
    latest_chart_context: dict[str, Any] | None = None,
    latest_findings: list[str] | None = None,
    latest_evidence: list[str] | None = None,
    unresolved_questions: list[str] | None = None,
) -> InvestigationConversationState:
    prior = previous if isinstance(previous, dict) else {}
    resolved_intent = intent or resolve_user_intent(question, has_active_context=bool(prior or latest_chart_context or latest_findings))
    broad_reset = resolved_intent.primary in {
        ConversationIntent.EXPLORATORY_REASONING,
        ConversationIntent.CONCEPTUAL_QUESTION,
        ConversationIntent.WORKFLOW_QUESTION,
        ConversationIntent.CONTEXT_RESET,
        ConversationIntent.OVERVIEW,
    } and not resolved_intent.requires_continuity
    chart = {} if broad_reset else (latest_chart_context if isinstance(latest_chart_context, dict) else {})
    metric = "" if broad_reset else _first(chart.get("metric"), prior.get("active_metric"))
    dimension = "" if broad_reset else _first(chart.get("dimension"), prior.get("active_dimension"))
    time_axis = "" if broad_reset else _first(chart.get("time_axis"), prior.get("active_time_axis"))
    aggregation = "" if broad_reset else _first(chart.get("aggregation"), prior.get("active_chart_aggregation"))
    findings = [str(item).strip() for item in (latest_findings or prior.get("recent_findings") or []) if str(item).strip()][:8]
    evidence = [str(item).strip() for item in (latest_evidence or prior.get("recent_evidence") or []) if str(item).strip()][:8]
    unresolved = [str(item).strip() for item in (unresolved_questions or prior.get("unresolved_questions") or []) if str(item).strip()][:8]
    phase = _phase_for_intent(resolved_intent, has_evidence=bool(findings or evidence), has_chart=bool(chart))
    topic = _topic(metric, dimension, time_axis, chart, prior)
    hypothesis = _first(prior.get("active_hypothesis"), _hypothesis(metric, dimension, resolved_intent))
    objective = _objective_for_intent(resolved_intent, metric=metric, dimension=dimension, topic=topic)
    branch_type = _branch_type_from_context(resolved_intent, chart, metric, dimension, time_axis, prior)
    return InvestigationConversationState(
        active_topic=topic,
        active_metric=metric,
        active_dimension=dimension,
        active_time_axis=time_axis,
        active_chart=chart,
        active_chart_aggregation=aggregation,
        active_analysis_type=_first(_analysis_type_from_intent(resolved_intent, chart), prior.get("active_analysis_type")),
        active_hypothesis=hypothesis,
        active_anomaly_target=_first(prior.get("active_anomaly_target"), topic if resolved_intent.primary == ConversationIntent.ANOMALY else ""),
        unresolved_questions=unresolved,
        recent_findings=findings,
        recent_evidence=evidence,
        active_validation_target=_first(prior.get("active_validation_target"), findings[0] if findings else ""),
        active_comparison_focus=_first(prior.get("active_comparison_focus"), topic),
        active_branch_type=branch_type,
        active_objective=objective,
        decomposition_dimensions=list(prior.get("decomposition_dimensions") or []),
        current_objective=objective,
        active_analytical_target=dict(prior.get("active_analytical_target") or {}),
        phase=phase,
        topic_confidence=0.95 if metric and dimension else 0.75 if metric or chart else 0.35,
        latest_intent=resolved_intent.primary.value,
    )


def is_semantically_redundant_response(
    candidate_response: str,
    recent_assistant_messages: list[str],
    active_investigation_state: InvestigationConversationState | dict[str, Any] | None = None,
) -> bool:
    candidate = _semantic_signature(candidate_response)
    if not candidate:
        return False
    state_terms = set()
    if isinstance(active_investigation_state, InvestigationConversationState):
        state_payload = active_investigation_state.to_payload()
    elif isinstance(active_investigation_state, dict):
        state_payload = active_investigation_state
    else:
        state_payload = {}
    for key in ("active_metric", "active_dimension", "active_topic", "latest_intent"):
        state_terms.update(_semantic_signature(str(state_payload.get(key) or "")).split())
    for previous in recent_assistant_messages[-8:]:
        signature = _semantic_signature(previous)
        if not signature:
            continue
        left = set(candidate.split())
        right = set(signature.split())
        if candidate == signature:
            return True
        overlap = len(left & right) / max(len(left | right), 1)
        if overlap >= 0.84 and len(left & right) >= 10:
            return True
        generic_overlap = len((left - state_terms) & (right - state_terms)) / max(len((left - state_terms) | (right - state_terms)), 1)
        if generic_overlap >= 0.9 and len(left & right) >= 8:
            return True
    return False


def _prioritize_intent(components: list[ConversationIntent], *, has_active_context: bool) -> ConversationIntent:
    order = [
        ConversationIntent.REPORT,
        ConversationIntent.CONTEXT_RESET,
        ConversationIntent.QUALITY_AUDIT,
        ConversationIntent.ANALYTICAL_TRANSFORMATION,
        ConversationIntent.ANOMALY,
        ConversationIntent.VALIDATION,
        ConversationIntent.CAUSAL_HYPOTHESIS,
        ConversationIntent.CHART,
        ConversationIntent.COMPARISON,
        ConversationIntent.EXPLANATION,
        ConversationIntent.CONCEPTUAL_QUESTION,
        ConversationIntent.WORKFLOW_QUESTION,
        ConversationIntent.EXPLORATORY_REASONING,
        ConversationIntent.FOLLOW_UP,
        ConversationIntent.OVERVIEW,
    ]
    if ConversationIntent.OVERVIEW in components and len(set(components)) > 1:
        return next((item for item in order if item in components and item != ConversationIntent.OVERVIEW), ConversationIntent.OVERVIEW)
    if has_active_context and ConversationIntent.FOLLOW_UP in components and len(set(components)) == 1:
        return ConversationIntent.FOLLOW_UP
    return next((item for item in order if item in components), components[0])


def _phase_for_intent(intent: ResolvedIntent, *, has_evidence: bool, has_chart: bool) -> InvestigationPhase:
    if intent.primary in {ConversationIntent.OVERVIEW, ConversationIntent.CONTEXT_RESET} and not has_evidence and not has_chart:
        return InvestigationPhase.ONBOARDING
    if intent.primary in {ConversationIntent.EXPLORATORY_REASONING, ConversationIntent.CONCEPTUAL_QUESTION, ConversationIntent.WORKFLOW_QUESTION}:
        return InvestigationPhase.EXPLORATORY_ANALYSIS
    if intent.primary == ConversationIntent.ANOMALY:
        return InvestigationPhase.ANOMALY_ANALYSIS
    if intent.primary in {ConversationIntent.VALIDATION, ConversationIntent.CAUSAL_HYPOTHESIS, ConversationIntent.ANALYTICAL_TRANSFORMATION}:
        return InvestigationPhase.VALIDATION
    if intent.primary == ConversationIntent.REPORT:
        return InvestigationPhase.REPORTING
    if has_chart or has_evidence:
        return InvestigationPhase.FOCUSED_INVESTIGATION
    return InvestigationPhase.EXPLORATORY_ANALYSIS


def _analysis_type_from_intent(intent: ResolvedIntent, chart: dict[str, Any]) -> str:
    chart_type = str(chart.get("chart_type") or "").strip()
    if chart_type == "bar":
        return "grouped_ranking"
    if chart_type == "scatter":
        return "volume_validation"
    if intent.primary == ConversationIntent.ANOMALY:
        return "anomaly_analysis"
    if intent.primary == ConversationIntent.CAUSAL_HYPOTHESIS:
        return "causal_validation"
    if intent.primary == ConversationIntent.ANALYTICAL_TRANSFORMATION:
        return "analytical_transformation"
    if intent.primary == ConversationIntent.COMPARISON:
        return "grouped_comparison"
    if intent.primary == ConversationIntent.CHART:
        return "chart_continuation"
    return intent.primary.value


def _branch_type_from_context(
    intent: ResolvedIntent,
    chart: dict[str, Any],
    metric: str,
    dimension: str,
    time_axis: str,
    prior: dict[str, Any],
) -> str:
    if intent.primary == ConversationIntent.QUALITY_AUDIT:
        return "data_quality"
    branch_from_chart = str(chart.get("branch_type") or "").strip()
    if branch_from_chart:
        return branch_from_chart
    chart_type = str(chart.get("chart_type") or "").strip()
    if chart_type == "line" or (metric and time_axis and intent.primary in {ConversationIntent.CHART, ConversationIntent.FOLLOW_UP, ConversationIntent.COMPARISON, ConversationIntent.ANOMALY}):
        return "trend_analysis"
    if metric and dimension:
        return "grouped_comparison"
    return str(prior.get("active_branch_type") or "")


def _objective_for_intent(intent: ResolvedIntent, *, metric: str, dimension: str, topic: str) -> str:
    if intent.primary == ConversationIntent.ANOMALY:
        return f"Identify unusual values and fragile evidence inside {topic or 'the active comparison'}."
    if intent.primary == ConversationIntent.CAUSAL_HYPOTHESIS:
        return f"Test whether volume, mix, or outliers explain {topic or 'the active gap'}."
    if intent.primary == ConversationIntent.ANALYTICAL_TRANSFORMATION:
        return f"Recompute {topic or 'the active comparison'} after the requested analytical transformation."
    if intent.primary == ConversationIntent.CHART:
        return f"Create visual evidence for {topic or 'the requested metric comparison'}."
    if intent.primary == ConversationIntent.COMPARISON:
        return f"Compare strongest and weakest groups for {topic or metric or 'the active metric'}."
    if intent.primary == ConversationIntent.QUALITY_AUDIT:
        return f"Check data quality risks affecting {topic or 'the current conclusion'}."
    return f"Deepen the current analytical thread around {topic}." if topic else "Understand the dataset and strongest analytical directions."


def _topic(metric: str, dimension: str, time_axis: str, chart: dict[str, Any], prior: dict[str, Any]) -> str:
    if metric and dimension:
        return f"`{metric}` by `{dimension}`"
    if metric and time_axis:
        return f"`{metric}` over `{time_axis}`"
    if chart.get("title"):
        return str(chart.get("title"))
    return str(prior.get("active_topic") or "")


def _hypothesis(metric: str, dimension: str, intent: ResolvedIntent) -> str:
    if metric and dimension and intent.primary in {ConversationIntent.CAUSAL_HYPOTHESIS, ConversationIntent.VALIDATION, ConversationIntent.ANOMALY}:
        return f"The `{dimension}` gap in `{metric}` may be driven by volume, subgroup mix, sparse groups, or extreme observations."
    return ""


def _is_contextual_reference(text: str) -> bool:
    return any(marker in text for marker in ("this", "that", "these", "those", "it", "here", "same", "gap", "chart", "это", "этот", "эта", "эти", "здесь", "разрыв", "график"))


def _first(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalize(value: str) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").split())


def _semantic_signature(value: str) -> str:
    text = _normalize(value)
    cleaned = "".join(char if char.isalnum() or char.isspace() or char == "_" else " " for char in text)
    stopwords = {
        "the", "a", "an", "and", "or", "to", "of", "in", "by", "for", "with", "this", "that",
        "i", "would", "useful", "step", "next", "analysis", "аналит", "данных", "это", "нужно",
    }
    return " ".join(token for token in cleaned.split() if token not in stopwords)
