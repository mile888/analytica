from __future__ import annotations

from typing import Any

from source.product.fallbacks.semantic_resolution import _normalize


def _is_findings_evidence_question(question: str) -> bool:
    normalized = _normalize(question)
    markers = (
        "finding",
        "findings",
        "evidence",
        "support",
        "prove",
        "validate",
        "need more evidence",
        "needs more evidence",
        "доказ",
        "подтвержд",
        "вывод",
        "наход",
    )
    return any(marker in normalized for marker in markers)


def _is_overview_question(question: str) -> bool:
    normalized = _normalize(question)
    markers = (
        "what can you say",
        "summarize dataset",
        "summarize this dataset",
        "describe dataset",
        "describe data",
        "summarize",
        "schema",
        "overview",
        "суммариз",
        "суммари",
        "резюм",
        "что можешь сказать",
        "что ты можешь сказать",
        "какие тут данные",
        "опиши данные",
        "опиши датасет",
        "структур",
        "схем",
    )
    return any(marker in normalized for marker in markers)


def _is_chart_question(question: str) -> bool:
    normalized = _normalize(question)
    markers = (
        "chart",
        "plot",
        "graph",
        "visual",
        "histogram",
        "distribution",
        "show this",
        "show it",
        "build a chart",
        "build chart",
        "график",
        "диаграм",
        "визуал",
        "построй",
        "построить",
        "нарисуй",
        "изобрази",
        "покажи это",
    )
    return any(marker in normalized for marker in markers)


def _is_outlier_question(question: str) -> bool:
    normalized = _normalize(question)
    markers = ("outlier", "outliers", "anomal", "unusual", "extreme", "аномал", "выброс", "необыч")
    return any(marker in normalized for marker in markers)


def _is_trend_question(question: str) -> bool:
    normalized = _normalize(question)
    markers = (
        "trend",
        "over time",
        "time series",
        "timeline",
        "change over time",
        "growth",
        "monthly",
        "yearly",
        "daily",
        "динамик",
        "тренд",
        "во времени",
    )
    return any(marker in normalized for marker in markers)


def _is_correlation_question(question: str) -> bool:
    normalized = _normalize(question)
    markers = ("correlat", "relationship", "related to", "affect", "affects", "drives", "driver", "связ", "коррел", "влияет")
    return any(marker in normalized for marker in markers)


def _is_dependency_question(question: str) -> bool:
    normalized = _normalize(question)
    markers = (
        "dependency",
        "dependencies",
        "relationship",
        "relationships",
        "related",
        "affect",
        "affects",
        "drivers",
        "what drives",
        "завис",
        "связ",
        "влияет",
        "коррел",
    )
    return any(marker in normalized for marker in markers)


def _is_duplicate_impact_question(question: str) -> bool:
    normalized = _normalize(question)
    return any(marker in normalized for marker in ("duplicate", "duplicates", "дублик")) and any(
        marker in normalized
        for marker in ("inflate", "impact", "affect", "counts", "group counts", "искаж", "вли", "увелич", "счет", "счёт", "колич")
    )


def _is_duplicate_impact_followup(question: str, state: dict[str, Any] | None) -> bool:
    normalized = _normalize(question)
    if not isinstance(state, dict):
        return False
    quality_issue = str(state.get("active_quality_issue") or "").lower()
    branch = str(state.get("active_branch_type") or state.get("branch_type") or "").lower()
    recent = " ".join(str(item) for item in (state.get("recent_findings") or [])).lower()
    pronoun = any(marker in normalized for marker in ("they", "these", "those", "они", "их", "это"))
    impact = any(marker in normalized for marker in ("impact", "affect", "inflate", "вли", "искаж", "увелич"))
    duplicate_context = "duplicate" in quality_issue or "duplicate" in recent or "дублик" in recent
    return branch == "data_quality" and duplicate_context and pronoun and impact


def _is_quality_issue_impact_question(question: str) -> bool:
    normalized = _normalize(question)
    return any(marker in normalized for marker in ("quality issue", "quality issues", "missing", "duplicates", "пропуск", "дублик", "качест")) and any(
        marker in normalized for marker in ("strongest conclusion", "affects", "impact", "most", "главн", "сильн", "вли")
    )


def _is_volume_relationship_follow_up(question: str) -> bool:
    normalized = _normalize(question)
    relationship_markers = (
        "related",
        "relationship",
        "explain",
        "driven",
        "drives",
        "affect",
        "associated",
        "связ",
        "объяс",
        "влияет",
        "завис",
    )
    volume_markers = (
        "volume",
        "count",
        "record",
        "records",
        "order volume",
        "transaction",
        "quantity",
        "объем",
        "объём",
        "колич",
        "число",
        "заказ",
        "строк",
    )
    return any(marker in normalized for marker in relationship_markers) and any(marker in normalized for marker in volume_markers)


def _is_distribution_question(question: str) -> bool:
    normalized = _normalize(question)
    markers = ("distribution", "histogram", "spread", "range", "frequency", "density", "распредел", "гистограмм", "разброс")
    return any(marker in normalized for marker in markers)


def _is_ranking_or_group_comparison_question(question: str) -> bool:
    normalized = _normalize(question)
    markers = (
        "top",
        "highest",
        "lowest",
        "best",
        "worst",
        "ranking",
        "rank",
        "compare",
        "comparison",
        "by ",
        "across",
        "group",
        "groups",
        "category",
        "categories",
        "segment",
        "segments",
        "role",
        "roles",
        "profession",
        "professions",
        "type",
        "types",
        "breakdown by",
        "сравн",
        "топ",
        "лучшие",
        "худшие",
        "сильн",
        "групп",
        " по ",
        "разн",
        "разным",
        "разных",
    )
    return any(marker in normalized for marker in markers)


def _is_composition_question(question: str) -> bool:
    normalized = _normalize(question)
    markers = ("share", "composition", "proportion", "breakdown", "mix", "доля", "состав", "структура")
    return any(marker in normalized for marker in markers)


def _is_quality_question(question: str) -> bool:
    normalized = _normalize(question)
    markers = (
        "quality", "reliable", "reliability", "missing", "duplicates", "duplicate", "repeated records",
        "repeated orders", "duplicate orders", "duplicate transactions", "duplicated customers", "duplicated orders",
        "clean", "preprocess", "null", "invalid dates", "suspicious ids", "inconsistent categories",
        "пропуск", "дублик", "качеств", "повторяющиеся заказы", "повторные заказы",
    )
    return any(marker in normalized for marker in markers)


def _is_duplicate_question(question: str) -> bool:
    normalized = _normalize(question)
    return any(marker in normalized for marker in ("duplicate", "duplicates", "repeated", "duplicated", "дублик", "повтор"))


def _is_hypothesis_validation_question(question: str) -> bool:
    normalized = _normalize(question)
    if _is_hypothesis_followup_question(normalized):
        return False
    if any(marker in normalized for marker in ("hypothesis:", "hypothesis ", "гипотез")) and any(
        marker in normalized for marker in ("driven", "dominates", "because", "volume", "few large", "order value", "объем", "объём")
    ):
        return True
    return any(marker in normalized for marker in ("hypothesis", "гипотез")) and any(
        marker in normalized for marker in ("check", "test", "validate", "проверь", "проверить", "проверим")
    )


def _is_hypothesis_followup_question(normalized_question: str) -> bool:
    return any(
        marker in normalized_question
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
    )


def _is_large_order_concentration_hypothesis(question: str) -> bool:
    normalized = _normalize(question)
    return any(marker in normalized for marker in ("large order", "large orders", "few large", "outlier", "extreme", "крупн", "выброс"))


def _is_volume_hypothesis(question: str) -> bool:
    normalized = _normalize(question)
    return any(marker in normalized for marker in ("volume", "count", "record", "records", "order volume", "колич", "объем", "объём"))


def _metric_is_explicit_enough(question: str, metric_col: str | None) -> bool:
    return bool(metric_col and _normalize(str(metric_col)) in _normalize(question))


def _is_business_questions_request(question: str) -> bool:
    normalized = _normalize(question)
    return any(
        marker in normalized
        for marker in (
            "business question", "business questions", "what questions", "research questions",
            "бизнес вопрос", "бизнес вопросы", "какие вопросы", "что можно исследовать", "можно исследовать",
        )
    )


def _is_important_fields_request(question: str) -> bool:
    normalized = _normalize(question)
    return any(
        marker in normalized
        for marker in (
            "important fields", "fields are most important", "important columns", "columns are most important", "key fields", "key columns", "most important fields",
            "важные поля", "важны", "наиболее важ", "ключевые поля", "главные поля",
        )
    )


def _is_context_reset_question(question: str) -> bool:
    normalized = _normalize(question)
    return any(marker in normalized for marker in ("reset", "start over", "new topic", "заново", "сброс", "новая тема"))


def _is_analytical_follow_up(question: str) -> bool:
    if _is_business_questions_request(question) or _is_important_fields_request(question) or _is_context_reset_question(question):
        return False
    return any(
        (
            _is_outlier_question(question),
            _is_chart_question(question),
            _is_trend_question(question),
            _is_correlation_question(question),
            _is_distribution_question(question),
            _is_ranking_or_group_comparison_question(question),
            _is_composition_question(question),
            _is_quality_question(question),
            any(marker in _normalize(question) for marker in ("vary", "compare", "across", "by ", "group", "segment", "сравн", "групп")),
        )
    )


def _is_contextual_follow_up(question: str) -> bool:
    normalized = _normalize(question)
    if _is_business_questions_request(question) or _is_important_fields_request(question) or _is_context_reset_question(question):
        return False
    markers = (
        "this",
        "that",
        "these",
        "those",
        "here",
        "same",
        "strongest",
        "weakest",
        "это",
        "этот",
        "эта",
        "эти",
        "здесь",
        "самые",
        "сильн",
        "слаб",
        "аномал",
    )
    return any(marker in normalized for marker in markers)


def _should_continue_from_context(question: str, conversation_context: Any) -> bool:
    if _is_overview_question(question) or not isinstance(conversation_context, dict):
        return False
    if conversation_context.get("active_message_id"):
        return True
    messages = conversation_context.get("messages")
    if isinstance(messages, list) and len(messages) > 1:
        return True
    return bool(conversation_context.get("latest_findings") or conversation_context.get("latest_report_summary"))


def _is_next_check_question(question: str) -> bool:
    text = str(question or "").casefold()
    return any(marker in text for marker in ("что проверить", "проверить дополнительно", "дальше", "next check", "check next", "what should we check"))


def _is_investigation_reset_question(question: str) -> bool:
    normalized = _normalize(question)
    return any(marker in normalized for marker in ("reset", "start over", "new investigation", "сброс", "заново", "новое расслед"))
