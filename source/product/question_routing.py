from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class QuestionIntentType(StrEnum):
    NON_ANALYTICAL_QUERY = "non_analytical_query"
    CREATIVE_ANALOGY = "creative_analogy"
    SEMANTIC_ANALOGY = "semantic_analogy"
    CASUAL_CONVERSATION = "casual_conversation"
    META_PROJECT_QUESTION = "meta_project_question"
    OUT_OF_SCOPE_SAFE = "out_of_scope_safe"
    DIRECT_ANALYSIS = "direct_analysis"
    DISTRIBUTION_ANALYSIS = "distribution_analysis"
    TRANSFORMATION = "transformation"
    ARTIFACT_EXPLANATION = "artifact_explanation"
    DATA_QUALITY = "data_quality"
    DATAFRAME_OPERATION = "dataframe_operation"
    BUSINESS_RISK = "business_risk"
    CUSTOMER_BEHAVIOR = "customer_behavior"
    EXECUTIVE_SUMMARY = "executive_summary"
    STRATEGIC_RECOMMENDATION = "strategic_recommendation"
    BUSINESS_HYPOTHESIS = "business_hypothesis"
    ANOMALY_INVESTIGATION = "anomaly_investigation"
    MULTI_DATASET_SCHEMA_COMPARISON = "multi_dataset_schema_comparison"
    MULTI_DATASET_SEMANTIC_REASONING = "multi_dataset_semantic_reasoning"
    MULTI_DATASET_JOINABILITY = "multi_dataset_joinability"
    MULTI_DATASET_WAREHOUSE_DESIGN = "multi_dataset_warehouse_design"
    MULTI_DATASET_MISSING_LINKS = "multi_dataset_missing_links"
    MULTI_DATASET_PARALLEL_ANALYSIS = "multi_dataset_parallel_analysis"
    MULTI_DATASET_CAPABILITY_REASONING = "multi_dataset_capability_reasoning"
    MULTI_DATASET_EXECUTIVE_SYNTHESIS = "multi_dataset_executive_synthesis"
    MULTI_DATASET_DIVERSITY_COMPARISON = "multi_dataset_diversity_comparison"
    MULTI_DATASET_TEMPORAL_COMPARISON = "multi_dataset_temporal_comparison"
    CROSS_DATASET_COMPARISON = "cross_dataset_comparison"
    DATASET_CAPABILITY_REASONING = "dataset_capability_reasoning"
    PARALLEL_VISUAL_ANALYSIS = "parallel_visual_analysis"
    CROSS_DATASET_EXECUTIVE_SYNTHESIS = "cross_dataset_executive_synthesis"
    CROSS_DATASET_DIVERSITY_ANALYSIS = "cross_dataset_diversity_analysis"
    CROSS_DATASET_TEMPORAL_ANALYSIS = "cross_dataset_temporal_analysis"
    CROSS_DATASET_LIMITATION_ANALYSIS = "cross_dataset_limitation_analysis"
    REPORTING = "reporting"
    CLARIFICATION = "clarification"
    CASUAL_OR_META = "casual_or_meta"


class BranchAction(StrEnum):
    CONTINUE = "continue"
    SWITCH = "switch"
    CREATE = "create"
    GLOBAL = "global"
    NONE = "none"


class ContextPolicy(StrEnum):
    CONTINUE = "continue"
    RESET_ANALYTICAL = "reset_analytical"
    RESET_GLOBAL = "reset_global"
    CONVERSATIONAL = "conversational"


GLOBAL_INTENTS = {
    QuestionIntentType.MULTI_DATASET_SCHEMA_COMPARISON,
    QuestionIntentType.MULTI_DATASET_SEMANTIC_REASONING,
    QuestionIntentType.MULTI_DATASET_JOINABILITY,
    QuestionIntentType.MULTI_DATASET_WAREHOUSE_DESIGN,
    QuestionIntentType.MULTI_DATASET_MISSING_LINKS,
    QuestionIntentType.MULTI_DATASET_PARALLEL_ANALYSIS,
    QuestionIntentType.MULTI_DATASET_CAPABILITY_REASONING,
    QuestionIntentType.MULTI_DATASET_EXECUTIVE_SYNTHESIS,
    QuestionIntentType.MULTI_DATASET_DIVERSITY_COMPARISON,
    QuestionIntentType.MULTI_DATASET_TEMPORAL_COMPARISON,
    QuestionIntentType.CROSS_DATASET_COMPARISON,
    QuestionIntentType.DATASET_CAPABILITY_REASONING,
    QuestionIntentType.PARALLEL_VISUAL_ANALYSIS,
    QuestionIntentType.CROSS_DATASET_EXECUTIVE_SYNTHESIS,
    QuestionIntentType.CROSS_DATASET_DIVERSITY_ANALYSIS,
    QuestionIntentType.CROSS_DATASET_TEMPORAL_ANALYSIS,
    QuestionIntentType.CROSS_DATASET_LIMITATION_ANALYSIS,
}

# Strategic intents that should be computed from data, not bypassed.
# These were previously in GLOBAL_INTENTS but now flow through normal analysis.
SINGLE_DATASET_STRATEGIC_INTENTS = {
    QuestionIntentType.BUSINESS_RISK,
    QuestionIntentType.CUSTOMER_BEHAVIOR,
    QuestionIntentType.EXECUTIVE_SUMMARY,
    QuestionIntentType.STRATEGIC_RECOMMENDATION,
    QuestionIntentType.BUSINESS_HYPOTHESIS,
}

MULTI_DATASET_INTENTS = {
    QuestionIntentType.MULTI_DATASET_SCHEMA_COMPARISON,
    QuestionIntentType.MULTI_DATASET_SEMANTIC_REASONING,
    QuestionIntentType.MULTI_DATASET_JOINABILITY,
    QuestionIntentType.MULTI_DATASET_WAREHOUSE_DESIGN,
    QuestionIntentType.MULTI_DATASET_MISSING_LINKS,
    QuestionIntentType.MULTI_DATASET_PARALLEL_ANALYSIS,
    QuestionIntentType.MULTI_DATASET_CAPABILITY_REASONING,
    QuestionIntentType.MULTI_DATASET_EXECUTIVE_SYNTHESIS,
    QuestionIntentType.MULTI_DATASET_DIVERSITY_COMPARISON,
    QuestionIntentType.MULTI_DATASET_TEMPORAL_COMPARISON,
    QuestionIntentType.CROSS_DATASET_COMPARISON,
    QuestionIntentType.DATASET_CAPABILITY_REASONING,
    QuestionIntentType.PARALLEL_VISUAL_ANALYSIS,
    QuestionIntentType.CROSS_DATASET_EXECUTIVE_SYNTHESIS,
    QuestionIntentType.CROSS_DATASET_DIVERSITY_ANALYSIS,
    QuestionIntentType.CROSS_DATASET_TEMPORAL_ANALYSIS,
    QuestionIntentType.CROSS_DATASET_LIMITATION_ANALYSIS,
}

NON_ANALYTICAL_INTENTS = {
    QuestionIntentType.NON_ANALYTICAL_QUERY,
    QuestionIntentType.CREATIVE_ANALOGY,
    QuestionIntentType.SEMANTIC_ANALOGY,
    QuestionIntentType.CASUAL_CONVERSATION,
    QuestionIntentType.META_PROJECT_QUESTION,
    QuestionIntentType.OUT_OF_SCOPE_SAFE,
}


@dataclass(frozen=True)
class RoutingDecision:
    question_intent_type: QuestionIntentType
    branch_action: BranchAction
    execution_mode: str
    artifact_context_action: str
    reason: str
    context_policy: ContextPolicy = ContextPolicy.RESET_ANALYTICAL

    def to_payload(self) -> dict[str, str]:
        return {
            "question_intent_type": self.question_intent_type.value,
            "branch_action": self.branch_action.value,
            "execution_mode": self.execution_mode,
            "artifact_context_action": self.artifact_context_action,
            "reason": self.reason,
            "context_policy": self.context_policy.value,
        }


def classify_question_intent(question: str) -> QuestionIntentType:
    text = _norm(question)
    if not text:
        return QuestionIntentType.CLARIFICATION
    if _is_ambiguous_creative_comparison(text):
        return QuestionIntentType.SEMANTIC_ANALOGY
    if _is_creative_analogy_request(text):
        return QuestionIntentType.CREATIVE_ANALOGY
    if _has(text, "why did the previous answer", "why was the previous answer", "why did it answer", "previous answer look wrong", "как устроен агент", "что умеет агент", "почему ответ", "почему он ответил", "как это работает"):
        return QuestionIntentType.META_PROJECT_QUESTION
    if _has(text, "what should i test next", "how should i test", "how should i present", "what can this agent do", "is this a good analysis", "что тестировать", "как презентовать", "как представить", "что проверить дальше"):
        return QuestionIntentType.META_PROJECT_QUESTION
    if _has(text, "tell a joke", "joke", "шутк", "придумай", "напиши текст", "как дела", "what do you think about"):
        return QuestionIntentType.CASUAL_CONVERSATION
    if _has(text, "warehouse", "data warehouse", "unified analytics", "unified warehouse", "fact table", "dimension table", "shared entities", "shared entity", "единое хранилище", "витрин"):
        return QuestionIntentType.MULTI_DATASET_WAREHOUSE_DESIGN
    if _has(text, "missing links", "missing link", "cannot be answered", "can't be answered", "what questions cannot", "what important business questions cannot", "what fields are missing", "missing identifiers", "не хватает связ", "нельзя ответить"):
        return QuestionIntentType.MULTI_DATASET_MISSING_LINKS
    # --- New multi-dataset intents (must be checked BEFORE generic multi-dataset) ---
    if _is_parallel_analysis_request(text):
        return QuestionIntentType.PARALLEL_VISUAL_ANALYSIS
    if _has(text, "comparative visual", "visual analysis showing", "build comparative visual", "separate charts"):
        return QuestionIntentType.PARALLEL_VISUAL_ANALYSIS
    if _is_capability_reasoning_request(text):
        return QuestionIntentType.DATASET_CAPABILITY_REASONING
    if _has(text, "prioritize one dataset", "which dataset would you choose", "best dataset for"):
        return QuestionIntentType.DATASET_CAPABILITY_REASONING
    if _is_cross_executive_synthesis(text):
        return QuestionIntentType.CROSS_DATASET_EXECUTIVE_SYNTHESIS
    if _is_cross_diversity_request(text):
        return QuestionIntentType.CROSS_DATASET_DIVERSITY_ANALYSIS
    if _is_cross_temporal_request(text):
        return QuestionIntentType.CROSS_DATASET_TEMPORAL_ANALYSIS
    if _has(text, "join", "joined", "merge", "merged", "key", "connect these datasets", "fields could connect", "trustworthy", "reliably", "соедин", "джойн", "объедин", "ключ"):
        if _has(text, "dataset", "datasets", "these datasets", "both", "between", "merge", "join", "оба датасет", "между датасет"):
            return QuestionIntentType.MULTI_DATASET_JOINABILITY
    if _has(text, "compare columns", "same columns", "shared columns", "schema", "columns in both", "колонки", "схем"):
        if _has(text, "both", "dataset", "datasets", "двух датасет", "обоих датасет"):
            return QuestionIntentType.MULTI_DATASET_SCHEMA_COMPARISON
    if _has(
        text,
        "both datasets",
        "these datasets",
        "using both datasets",
        "combining these datasets",
        "datasets together",
        "analysis together",
        "across datasets",
        "between datasets",
        "all datasets",
        "which dataset contains",
        "which dataset tracks",
        "related from a business",
        "business themes",
        "common patterns",
        "common business",
        "cross dataset",
        "cross dataset insights",
        "shared business",
        "shared business entities",
        "concepts appear in both",
        "different ways",
        "оба датасет",
        "между датасет",
        "общие паттерн",
    ):
        return QuestionIntentType.MULTI_DATASET_SEMANTIC_REASONING
    if _has(text, "risk", "risks", "hidden business", "problems might", "business problems", "threat", "риск", "проблем"):
        return QuestionIntentType.BUSINESS_RISK
    if _has(text, "customer behavior", "customers behave", "segmentation", "retention", "loyalty", "customer patterns", "поведен", "сегмент", "удержан"):
        return QuestionIntentType.CUSTOMER_BEHAVIOR
    if _has(text, "executive", "summary", "3 most important", "three most important", "key insights", "management", "report conclusion", "вывод", "руководств"):
        return QuestionIntentType.EXECUTIVE_SUMMARY
    if _has(text, "product manager", "what would you investigate first", "recommend", "recommendation", "strategy", "strategic", "management do", "приоритет", "стратег"):
        return QuestionIntentType.STRATEGIC_RECOMMENDATION
    if text.startswith("hypothesis ") or _has(text, "сформируй hypothesis", "supports the hypothesis", "support the hypothesis", "contradicts the hypothesis", "contradict the hypothesis", "как это проверить"):
        return QuestionIntentType.DIRECT_ANALYSIS
    if _has(text, "business hypothesis", "business hypotheses", "what hypotheses", "why might", "could explain", "business question", "гипотез"):
        return QuestionIntentType.BUSINESS_HYPOTHESIS
    if _has(text, "explain this chart", "this chart", "этот график", "объясни график"):
        return QuestionIntentType.ARTIFACT_EXPLANATION
    if _has(text, "remove outlier", "remove outliers", "extreme orders", "without outlier", "median instead", "show median", "exclude sparse", "normalize", "убрать выброс", "убрать extreme", "медиан"):
        return QuestionIntentType.TRANSFORMATION
    if _has(text, "histogram", "distribution", "распредел", "гистограмм"):
        return QuestionIntentType.DISTRIBUTION_ANALYSIS
    if _has(text, "quality", "missing", "duplicate", "null", "clean", "preprocess", "качество", "пропуск", "дублик"):
        return QuestionIntentType.DATA_QUALITY
    if _has(text, "outlier", "anomaly", "unusual", "extreme", "аномал", "необыч"):
        return QuestionIntentType.ANOMALY_INVESTIGATION
    if _has(text, "report", "pdf", "use in report", "отчет", "отчёт"):
        return QuestionIntentType.REPORTING
    if _has(text, "hello", "thanks", "thank you", "how are you", "привет", "спасибо"):
        return QuestionIntentType.CASUAL_OR_META
    return QuestionIntentType.DIRECT_ANALYSIS


def decide_routing(question: str, *, has_active_context: bool = False) -> RoutingDecision:
    intent = classify_question_intent(question)
    text = _norm(question)
    if intent in NON_ANALYTICAL_INTENTS:
        return RoutingDecision(intent, BranchAction.NONE, "conversational", "ignore_active_branch", "Non-analytical or creative intent escapes dataframe branch continuation.", ContextPolicy.CONVERSATIONAL)
    if intent in GLOBAL_INTENTS:
        return RoutingDecision(intent, BranchAction.GLOBAL, "conceptual" if intent in MULTI_DATASET_INTENTS else "reasoning", "ignore_active_branch", "Global or cross-dataset intent overrides active branch.", ContextPolicy.RESET_GLOBAL)
    if intent in {QuestionIntentType.CASUAL_OR_META, QuestionIntentType.CLARIFICATION}:
        return RoutingDecision(intent, BranchAction.NONE, "none", "none", "No analytical branch needed.", ContextPolicy.CONVERSATIONAL)
    if _is_clear_followup(text) and has_active_context:
        return RoutingDecision(intent, BranchAction.CONTINUE, "compute" if intent != QuestionIntentType.ARTIFACT_EXPLANATION else "reasoning", "use_active_artifact", "Question is a dependent follow-up.", ContextPolicy.CONTINUE)
    if intent in {QuestionIntentType.TRANSFORMATION, QuestionIntentType.ARTIFACT_EXPLANATION} and has_active_context:
        return RoutingDecision(intent, BranchAction.CONTINUE, "compute_or_explain", "use_active_artifact", "Transformation or chart explanation continues active artifact context.", ContextPolicy.CONTINUE)
    return RoutingDecision(intent, BranchAction.CREATE, "compute_or_reason", "select_relevant_context", "New analytical intent creates an independent branch.", ContextPolicy.RESET_ANALYTICAL)


def should_ignore_active_branch(intent: QuestionIntentType | str) -> bool:
    try:
        value = QuestionIntentType(str(intent))
    except ValueError:
        return False
    return value in GLOBAL_INTENTS or value in NON_ANALYTICAL_INTENTS


def is_multi_dataset_intent(intent: QuestionIntentType | str) -> bool:
    try:
        value = QuestionIntentType(str(intent))
    except ValueError:
        return False
    return value in MULTI_DATASET_INTENTS


def is_non_analytical_intent(intent: QuestionIntentType | str) -> bool:
    try:
        value = QuestionIntentType(str(intent))
    except ValueError:
        return False
    return value in NON_ANALYTICAL_INTENTS


def branch_type_for_intent(intent: QuestionIntentType | str) -> str:
    try:
        value = QuestionIntentType(str(intent))
    except ValueError:
        return "analysis"
    mapping = {
        QuestionIntentType.BUSINESS_RISK: "business",
        QuestionIntentType.CUSTOMER_BEHAVIOR: "customer_behavior",
        QuestionIntentType.EXECUTIVE_SUMMARY: "business",
        QuestionIntentType.STRATEGIC_RECOMMENDATION: "business",
        QuestionIntentType.BUSINESS_HYPOTHESIS: "business",
        QuestionIntentType.ANOMALY_INVESTIGATION: "anomaly",
        QuestionIntentType.MULTI_DATASET_SCHEMA_COMPARISON: "multi_dataset",
        QuestionIntentType.MULTI_DATASET_SEMANTIC_REASONING: "multi_dataset",
        QuestionIntentType.MULTI_DATASET_JOINABILITY: "joinability",
        QuestionIntentType.MULTI_DATASET_WAREHOUSE_DESIGN: "warehouse_design",
        QuestionIntentType.MULTI_DATASET_MISSING_LINKS: "multi_dataset",
        QuestionIntentType.MULTI_DATASET_PARALLEL_ANALYSIS: "multi_dataset",
        QuestionIntentType.MULTI_DATASET_CAPABILITY_REASONING: "multi_dataset",
        QuestionIntentType.MULTI_DATASET_EXECUTIVE_SYNTHESIS: "multi_dataset",
        QuestionIntentType.MULTI_DATASET_DIVERSITY_COMPARISON: "multi_dataset",
        QuestionIntentType.MULTI_DATASET_TEMPORAL_COMPARISON: "multi_dataset",
        QuestionIntentType.CROSS_DATASET_COMPARISON: "multi_dataset",
        QuestionIntentType.DATASET_CAPABILITY_REASONING: "multi_dataset",
        QuestionIntentType.PARALLEL_VISUAL_ANALYSIS: "multi_dataset",
        QuestionIntentType.CROSS_DATASET_EXECUTIVE_SYNTHESIS: "multi_dataset",
        QuestionIntentType.CROSS_DATASET_DIVERSITY_ANALYSIS: "multi_dataset",
        QuestionIntentType.CROSS_DATASET_TEMPORAL_ANALYSIS: "multi_dataset",
        QuestionIntentType.CROSS_DATASET_LIMITATION_ANALYSIS: "multi_dataset",
        QuestionIntentType.DISTRIBUTION_ANALYSIS: "distribution",
        QuestionIntentType.TRANSFORMATION: "transformation",
        QuestionIntentType.DATA_QUALITY: "quality",
        QuestionIntentType.REPORTING: "report",
    }
    return mapping.get(value, "analysis")


def title_for_intent(intent: QuestionIntentType | str) -> str:
    try:
        value = QuestionIntentType(str(intent))
    except ValueError:
        return "Analysis"
    titles = {
        QuestionIntentType.NON_ANALYTICAL_QUERY: "Conversation",
        QuestionIntentType.CREATIVE_ANALOGY: "Creative analogy",
        QuestionIntentType.SEMANTIC_ANALOGY: "Semantic analogy",
        QuestionIntentType.CASUAL_CONVERSATION: "Conversation",
        QuestionIntentType.META_PROJECT_QUESTION: "Project question",
        QuestionIntentType.OUT_OF_SCOPE_SAFE: "Conversation",
        QuestionIntentType.BUSINESS_RISK: "Business risks",
        QuestionIntentType.CUSTOMER_BEHAVIOR: "Customer behavior",
        QuestionIntentType.EXECUTIVE_SUMMARY: "Executive summary",
        QuestionIntentType.STRATEGIC_RECOMMENDATION: "Strategic recommendation",
        QuestionIntentType.BUSINESS_HYPOTHESIS: "Business hypotheses",
        QuestionIntentType.ANOMALY_INVESTIGATION: "Anomaly investigation",
        QuestionIntentType.MULTI_DATASET_SCHEMA_COMPARISON: "Dataset schema comparison",
        QuestionIntentType.MULTI_DATASET_SEMANTIC_REASONING: "Cross-dataset relationship",
        QuestionIntentType.MULTI_DATASET_JOINABILITY: "Dataset joinability",
        QuestionIntentType.MULTI_DATASET_WAREHOUSE_DESIGN: "Warehouse design",
        QuestionIntentType.MULTI_DATASET_MISSING_LINKS: "Missing dataset links",
        QuestionIntentType.MULTI_DATASET_PARALLEL_ANALYSIS: "Parallel dataset analysis",
        QuestionIntentType.MULTI_DATASET_CAPABILITY_REASONING: "Dataset capability assessment",
        QuestionIntentType.MULTI_DATASET_EXECUTIVE_SYNTHESIS: "Cross-dataset executive summary",
        QuestionIntentType.MULTI_DATASET_DIVERSITY_COMPARISON: "Cross-dataset diversity",
        QuestionIntentType.MULTI_DATASET_TEMPORAL_COMPARISON: "Cross-dataset temporal analysis",
        QuestionIntentType.CROSS_DATASET_COMPARISON: "Cross-dataset comparison",
        QuestionIntentType.DATASET_CAPABILITY_REASONING: "Dataset capability assessment",
        QuestionIntentType.PARALLEL_VISUAL_ANALYSIS: "Parallel visual analysis",
        QuestionIntentType.CROSS_DATASET_EXECUTIVE_SYNTHESIS: "Cross-dataset executive summary",
        QuestionIntentType.CROSS_DATASET_DIVERSITY_ANALYSIS: "Cross-dataset diversity",
        QuestionIntentType.CROSS_DATASET_TEMPORAL_ANALYSIS: "Cross-dataset temporal analysis",
        QuestionIntentType.CROSS_DATASET_LIMITATION_ANALYSIS: "Cross-dataset limitations",
        QuestionIntentType.REPORTING: "Report",
    }
    return titles.get(value, value.value.replace("_", " ").title())


def _is_clear_followup(text: str) -> bool:
    return _has(
        text,
        "remove outlier",
        "remove outliers",
        "extreme orders",
        "do not use",
        "don't use",
        "dont use",
        "instead",
        "use total",
        "use average",
        "use mean",
        "use median",
        "use count",
        "switch to",
        "change to",
        "same but",
        "compare against",
        "explain this chart",
        "this chart",
        "which remain",
        "remain leaders",
        "show median instead",
        "same",
        "again",
        "now by",
        "now compare",
        "теперь по",
        "а теперь по",
        "strongest",
        "leader",
        "leaders",
        "explain growth",
        "explain the growth",
        "categories explain",
        "seasonality",
        "anomalous period",
        "anomalous periods",
        "related to",
        "is it related",
        "does volume",
        "record volume",
        "order volume",
        "hypothesis",
        "supports the hypothesis",
        "support the hypothesis",
        "supports this",
        "contradicts the hypothesis",
        "contradict the hypothesis",
        "contradicts it",
        "what contradicts",
        "validate it",
        "which bins",
        "bins are sparse",
        "sparse bins",
        "what changed",
        "which findings",
        "become unreliable",
        "became unreliable",
        "how do they affect",
        "affect ",
        "without them",
        "after filtering",
        "how to validate",
        "убери",
        "убрать",
        "этот график",
        "остались",
        "остаются",
        "сравни с",
        "связано",
        "объясняют рост",
        "самые сильные",
        "объем",
        "объём",
        "заказ",
        "как это проверить",
    )


def _has(text: str, *markers: str) -> bool:
    return any(marker in text for marker in markers)


def _is_ambiguous_creative_comparison(text: str) -> bool:
    if not _has_named_creative_entity(text):
        return False
    return _has(text, "compare this", "compare it", "сравни это", "сравнить это", "сравни с", "похоже на")


def _is_creative_analogy_request(text: str) -> bool:
    """Detect genuine creative/metaphorical analogy requests.

    Must distinguish 'find similarity with La La Land' (creative) from
    'Compare movie and TV show distributions' (analytical about dataset).
    """
    if _has(text, "метафор", "metaphor", "analogy", "аналог"):
        return True
    if _has_named_creative_entity(text) and _has(
        text, "сходство с", "similarity", "similarities",
        "похож", "remind", "напомина", "as a metaphor",
    ):
        return True
    if _has(text, "book", "song", "песн", "книг") and not _has_analytical_intent_markers(text):
        return True
    return False


def _has_named_creative_entity(text: str) -> bool:
    """Check for named creative works (La La Land, specific films/books)."""
    return _has(
        text,
        "la la land", "lalaland", "лалалэнд", "лала лэнд",
        "фильмом", "фильму",
        "с фильм",
    )


def _has_analytical_intent_markers(text: str) -> bool:
    """Check for markers that indicate a genuine analytical/data question."""
    return _has(
        text,
        "distribution", "compare", "correlation", "trend", "average",
        "count", "how many", "top", "rank", "group by", "by country",
        "by genre", "by type", "by rating", "histogram", "chart",
        "graph", "plot", "data", "dataset", "column",
        "распредел", "сравни по", "по жанр", "по стран", "по типу",
        "сколько", "построй", "график", "данных", "датасет",
    )


def _norm(value: Any) -> str:
    cleaned = [char if char.isalnum() or char.isspace() else " " for char in str(value or "").casefold().replace("_", " ")]
    return " ".join("".join(cleaned).split())


def _is_parallel_analysis_request(text: str) -> bool:
    """Detect requests for separate analyses across multiple datasets."""
    return _has(
        text,
        "build separate", "separate visual", "separate analysis", "separate analyses",
        "visual analysis for", "visual analyses for",
        "for each dataset", "for all datasets", "for every dataset",
        "chart for each", "charts for each",
        "analysis of each", "analyze each dataset",
        "для каждого датасет", "отдельный анализ",
    )


def _is_capability_reasoning_request(text: str) -> bool:
    """Detect questions about which dataset is best suited for a task."""
    return (
        _has(text, "which dataset", "какой датасет")
        and _has(
            text,
            "best suited", "most suitable", "support", "capable",
            "best for", "good for", "suited for",
            "anomaly detection", "forecasting", "trend",
            "time series", "time-series",
            "лучше подходит", "подходит для",
        )
    ) or _has(
        text,
        "datasets support", "datasets are best",
        "analytical strengths", "analytical capabilities",
        "prioritize one dataset", "which dataset would you choose",
        "датасеты подходят", "аналитические возможности",
    )


def _is_cross_executive_synthesis(text: str) -> bool:
    """Detect cross-dataset executive summary/synthesis requests."""
    return (
        _has(text, "executive summary", "strengths and limitations",
             "analytical strengths", "compare strengths",
             "strongest analytical", "strongest dimensions",
             "сильные стороны", "слабые стороны")
        and _has(text, "all datasets", "across", "datasets", "each dataset",
                 "всех датасет", "каждого датасет")
    )


def _is_cross_diversity_request(text: str) -> bool:
    """Detect cross-dataset diversity comparison requests."""
    return (
        _has(text, "diversity", "разнообразие", "разнообразия")
        and (
            _has(text, "datasets", "all datasets", "each dataset", "every dataset", "between datasets", "across datasets", "датасет")
            or "diversity patterns across" in text
        )
    )


def _is_cross_temporal_request(text: str) -> bool:
    """Detect cross-dataset temporal/time-series comparison."""
    return (
        _has(text, "time series", "time-series", "temporal", "trend", "trends",
             "временной", "тренд")
        and _has(text, "across", "datasets", "all datasets", "each dataset",
                 "compare", "which datasets",
                 "датасеты", "каждого датасет")
    )
