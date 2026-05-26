"""Question semantic extraction for compatibility checking.

Extracts structured semantic information from user questions:
domain, entities, operations, required concepts, and whether
the question is generic (domain-agnostic).

Uses LLM for primary extraction with a fast deterministic fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QuestionSemantics:
    """Structured semantic profile of a user question."""

    domain: str = "general"
    entities: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    required_concepts: list[str] = field(default_factory=list)
    is_generic: bool = False
    confidence: float = 0.0


# ── Main entry point ───────────────────────────────────────────────────────

def extract_question_semantics(question: str) -> QuestionSemantics:
    """Extract semantic profile from a user question.

    Tries LLM extraction first; falls back to deterministic on failure.
    """
    if not question or not question.strip():
        return QuestionSemantics(is_generic=True, confidence=1.0)

    # Check for generic questions first (fast, no LLM needed)
    if _is_generic_question(question):
        return QuestionSemantics(
            domain="general",
            operations=_detect_generic_operations(question),
            is_generic=True,
            confidence=0.9,
        )

    # Try LLM extraction
    try:
        result = _llm_extract(question)
        if result:
            return result
    except Exception:
        pass

    # Deterministic fallback
    return _deterministic_extract(question)


# ── Generic question detection ─────────────────────────────────────────────

_GENERIC_MARKERS = (
    "outlier", "outliers", "anomal", "missing value", "missing data",
    "null", "data quality", "distribution", "histogram", "summary",
    "overview", "describe", "statistics", "statistic", "explore",
    "structure", "schema", "columns", "fields", "dtypes", "data type",
    "cluster", "correlation", "variance", "standard deviation",
    "percentile", "quartile", "skew", "kurtosis", "entropy",
    "diversity", "variety", "unique values", "cardinality",
    "duplicate", "sparse", "density", "completeness",
    "выброс", "аномал", "пропущенн", "пропуск", "качество данных",
    "распределени", "статистик", "обзор", "структур", "описа",
    "корреляц", "дублик", "разнообраз",
)

_GENERIC_OPERATION_MAP = {
    "outlier": "outlier_analysis", "outliers": "outlier_analysis", "anomal": "outlier_analysis",
    "выброс": "outlier_analysis", "аномал": "outlier_analysis",
    "missing": "data_quality", "null": "data_quality", "пропущенн": "data_quality",
    "пропуск": "data_quality", "duplicate": "data_quality", "дублик": "data_quality",
    "quality": "data_quality", "качество": "data_quality",
    "distribution": "distribution_analysis", "histogram": "distribution_analysis",
    "распределени": "distribution_analysis",
    "overview": "dataset_overview", "describe": "dataset_overview",
    "summary": "dataset_overview", "обзор": "dataset_overview",
    "structure": "dataset_overview", "schema": "dataset_overview",
    "cluster": "clustering", "correlation": "correlation_analysis",
    "корреляц": "correlation_analysis",
    "diversity": "diversity_analysis", "variety": "diversity_analysis",
    "разнообраз": "diversity_analysis",
}

# Domain-specific qualifier phrases that make a generic marker domain-specific.
# E.g. "summarize health risks" is NOT generic; "summarize this dataset" IS generic.
_DOMAIN_QUALIFIER_PHRASES = (
    # Healthcare
    "health risk", "public health", "health condition", "health indicator",
    "heart disease", "disease risk", "patient group", "patient outcome",
    "clinical", "clinically", "medical", "diagnosis", "symptom",
    "prevalence", "morbidity", "mortality", "treatment", "therapy",
    "bmi", "blood pressure", "smoking", "smoker",
    # Financial
    "financial risk", "credit risk", "loan default", "investment return",
    "profit margin", "revenue growth", "stock price",
    # Scientific
    "species diversity", "habitat", "ecosystem", "gene expression",
    # Education
    "student performance", "graduation rate", "enrollment",
    # Retail (when on non-retail data)
    "product return", "customer churn", "sales trend",
    # Russian
    "здоровь", "болезн", "пациент", "диагноз", "лечени",
    "клинич", "медицин", "симптом",
)


def _is_generic_question(question: str) -> bool:
    """Check if the question is domain-agnostic (outliers, quality, etc.).

    A question like "summarize this dataset" is generic.
    A question like "summarize the main public health risks" is NOT generic —
    it combines a generic operation with domain-specific content.
    """
    normalized = _normalize(question)
    # Must have a generic marker
    has_generic = any(marker in normalized for marker in _GENERIC_MARKERS)
    if not has_generic:
        return False

    # If domain-specific qualifier phrases are present, NOT generic
    if any(phrase in normalized for phrase in _DOMAIN_QUALIFIER_PHRASES):
        return False

    # Check that domain-specific content is minimal
    domain_scores = _domain_keyword_scores(normalized)
    max_domain_score = max(domain_scores.values()) if domain_scores else 0
    # Even a single strong domain hit (score >= 2) makes it non-generic
    return max_domain_score < 2


def _detect_generic_operations(question: str) -> list[str]:
    """Detect which generic operations the question requests."""
    normalized = _normalize(question)
    ops: set[str] = set()
    for marker, op in _GENERIC_OPERATION_MAP.items():
        if marker in normalized:
            ops.add(op)
    return sorted(ops) or ["dataset_overview"]


# ── LLM extraction ─────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are a semantic analyzer. Extract structured semantic information from user questions.

Analyze the question and output ONLY valid JSON matching this schema:
{
  "domain": "healthcare | entertainment | retail | financial | scientific | education | hr | ecology | logistics | geographic | operational | survey | general",
  "entities": ["list of specific domain entities mentioned"],
  "operations": ["comparison | trend | aggregation | prevalence_analysis | count_distribution | correlation | ranking | prediction | clustering"],
  "required_concepts": ["list of abstract concepts needed to answer this question"],
  "confidence": 0.0
}

RULES:
1. "domain" = the knowledge domain the question belongs to.
2. "entities" = specific things mentioned (e.g. "heart disease", "smoker", "genre", "profit").
3. "operations" = what analytical operations are requested.
4. "required_concepts" = abstract data concepts needed (e.g. "disease", "smoking_status", "category").
5. If the question is about general statistics (outliers, distributions, missing values), set domain="general".
6. Confidence = how certain you are about the domain classification (0.0-1.0).

OUTPUT JSON ONLY. No markdown, no prose.

QUESTION: {question}
"""


def _llm_extract(question: str) -> QuestionSemantics | None:
    """Use LLM to extract question semantics."""
    from source.llm.factory import make_llm

    llm = make_llm("semantic_compatibility")
    prompt = _EXTRACTION_PROMPT.format(question=question)
    response = llm.invoke(prompt)

    text = _response_text(response)
    return _parse_llm_output(text)


def _parse_llm_output(text: str) -> QuestionSemantics | None:
    """Parse LLM response into QuestionSemantics."""
    json_str = _extract_json(text)
    if not json_str:
        return None
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    domain = str(data.get("domain", "general")).lower().strip()
    entities = data.get("entities", [])
    if not isinstance(entities, list):
        entities = []
    entities = [str(e).strip() for e in entities if str(e).strip()]

    operations = data.get("operations", [])
    if not isinstance(operations, list):
        operations = []
    operations = [str(o).strip() for o in operations if str(o).strip()]

    required_concepts = data.get("required_concepts", [])
    if not isinstance(required_concepts, list):
        required_concepts = []
    required_concepts = [str(c).strip() for c in required_concepts if str(c).strip()]

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5

    is_generic = domain == "general"

    return QuestionSemantics(
        domain=domain,
        entities=entities,
        operations=operations,
        required_concepts=required_concepts,
        is_generic=is_generic,
        confidence=confidence,
    )


# ── Deterministic fallback extraction ──────────────────────────────────────

# Domain evidence for question text (matches against question words)
_QUESTION_DOMAIN_EVIDENCE: dict[str, tuple[str, ...]] = {
    "healthcare": (
        "heart disease", "disease", "diagnosis", "patient", "treatment",
        "clinical", "hospital", "symptom", "medical", "health",
        "prescription", "blood", "bmi", "therapy", "admission",
        "smoker", "smoking", "prevalence", "mortality", "morbidity",
        "surgery", "chronic", "acute", "infection", "vaccine",
        "диагноз", "лечение", "пациент", "болезн", "здоровь",
    ),
    "entertainment": (
        "genre", "movie", "film", "tv show", "show", "series",
        "director", "actor", "cast", "rating", "imdb",
        "streaming", "platform", "studio", "season", "episode",
        "жанр", "фильм", "сериал", "режиссер", "рейтинг",
    ),
    "retail": (
        "sales", "revenue", "profit", "discount", "order",
        "product", "customer", "shipping", "delivery", "purchase",
        "cart", "store", "inventory", "sku", "return",
        "продаж", "выручк", "прибыл", "скидк", "заказ", "товар",
    ),
    "financial": (
        "income", "salary", "compensation", "loan", "credit",
        "investment", "portfolio", "interest", "mortgage", "tax",
        "stock", "bond", "dividend", "equity", "debt",
        "доход", "зарплат", "кредит", "инвестиц",
    ),
    "scientific": (
        "species", "habitat", "organism", "ecosystem", "gene",
        "protein", "molecule", "experiment", "hypothesis",
        "observation", "specimen", "taxonomy", "evolution",
        "вид", "среда обитания", "организм", "экосистем",
    ),
    "education": (
        "student", "grade", "course", "teacher", "school",
        "university", "enrollment", "gpa", "exam", "curriculum",
        "студент", "оценк", "курс", "учител",
    ),
    "hr": (
        "employee", "hire", "attrition", "department",
        "performance review", "tenure", "headcount", "turnover",
        "сотрудник", "наём", "отдел",
    ),
    "ecology": (
        "species", "habitat", "population", "biodiversity",
        "conservation", "ecosystem", "endangered", "migration",
    ),
    "logistics": (
        "shipment", "warehouse", "route", "fleet", "cargo",
        "freight", "supply chain", "logistics", "delivery time",
    ),
}


def _deterministic_extract(question: str) -> QuestionSemantics:
    """Fast keyword-based extraction when LLM is unavailable."""
    normalized = _normalize(question)
    scores = _domain_keyword_scores(normalized)

    # Boost domain score if domain-specific qualifier phrases are present
    for domain, markers in _QUESTION_DOMAIN_EVIDENCE.items():
        qualifier_boost = 0
        for phrase in _DOMAIN_QUALIFIER_PHRASES:
            if phrase in normalized and any(m in phrase or phrase in m for m in markers):
                qualifier_boost += 1
        if qualifier_boost and domain in scores:
            scores[domain] = scores[domain] + qualifier_boost
        elif qualifier_boost:
            scores[domain] = qualifier_boost

    if not any(scores.values()):
        return QuestionSemantics(domain="general", is_generic=True, confidence=0.3)

    best_domain = max(scores, key=lambda d: scores[d])
    best_score = scores[best_domain]
    second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0

    if best_score < 1.5 or (second_best and best_score - second_best < 1 and best_score < 2):
        return QuestionSemantics(domain="general", is_generic=True, confidence=0.3)

    # Extract matched entities
    entities = []
    for marker in _QUESTION_DOMAIN_EVIDENCE.get(best_domain, ()):
        if marker in normalized:
            entities.append(marker)

    confidence = min(1.0, best_score / 5.0)

    return QuestionSemantics(
        domain=best_domain,
        entities=entities,
        required_concepts=entities[:5],
        is_generic=False,
        confidence=round(confidence, 2),
    )


def _domain_keyword_scores(normalized: str) -> dict[str, float]:
    """Score each domain based on keyword matches in normalized text."""
    scores: dict[str, float] = {}
    for domain, markers in _QUESTION_DOMAIN_EVIDENCE.items():
        hits = sum(1 for marker in markers if marker in normalized)
        if hits:
            scores[domain] = hits
    return scores


# ── Utilities ──────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return " ".join(str(text or "").casefold().replace("_", " ").split())


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return " ".join(str(item.get("text") if isinstance(item, dict) else item) for item in content)
    return str(content or "")


def _extract_json(text: str) -> str | None:
    """Extract JSON object from response text."""
    stripped = text.strip()
    if stripped.startswith("{"):
        return stripped
    match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*\n?```", stripped, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", stripped, re.DOTALL)
    if match:
        return match.group(0)
    return None
