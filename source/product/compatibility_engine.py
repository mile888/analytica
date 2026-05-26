"""Check whether a question matches the selected dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from source.product.question_semantics import QuestionSemantics, extract_question_semantics
from source.product.dataset_semantics import DatasetSemanticProfile, build_dataset_semantic_profile


INCOMPATIBILITY_THRESHOLD = 0.3
DOMAIN_MATCH_BONUS = 0.4
ENTITY_MATCH_BONUS = 0.1
CONCEPT_MATCH_BONUS = 0.15

@dataclass(frozen=True)
class CompatibilityResult:
    """Stores the compatibility score and missing semantic parts."""

    compatible: bool
    compatibility_score: float
    matched_entities: list[str] = field(default_factory=list)
    missing_entities: list[str] = field(default_factory=list)
    missing_concepts: list[str] = field(default_factory=list)
    reason: str = ""
    dataset_domain: str = "general"
    question_domain: str = "general"


def check_compatibility(
    question_semantics: QuestionSemantics,
    dataset_profile: DatasetSemanticProfile,
) -> CompatibilityResult:
    """Score a question against one dataset profile."""
    if question_semantics.is_generic:
        return CompatibilityResult(
            compatible=True,
            compatibility_score=1.0,
            reason="Generic analytical question compatible with any dataset.",
            dataset_domain=dataset_profile.domain,
            question_domain="general",
        )

    if question_semantics.domain == "general":
        return CompatibilityResult(
            compatible=True,
            compatibility_score=0.5,
            reason="Question domain could not be classified. Proceeding with analysis.",
            dataset_domain=dataset_profile.domain,
            question_domain="general",
        )

    score = 0.0

    domain_match = _domains_match(question_semantics.domain, dataset_profile.domain)
    if domain_match:
        score += DOMAIN_MATCH_BONUS

    dataset_text = _build_dataset_text(dataset_profile)
    matched_entities: list[str] = []
    missing_entities: list[str] = []
    for entity in question_semantics.entities:
        if _entity_in_dataset(entity, dataset_text, dataset_profile):
            matched_entities.append(entity)
            score += ENTITY_MATCH_BONUS
        else:
            missing_entities.append(entity)

    matched_concepts: list[str] = []
    missing_concepts: list[str] = []
    for concept in question_semantics.required_concepts:
        if _concept_in_dataset(concept, dataset_text, dataset_profile):
            matched_concepts.append(concept)
            score += CONCEPT_MATCH_BONUS
        else:
            missing_concepts.append(concept)

    score = min(1.0, score)

    question_has_specific_domain = (
        question_semantics.domain != "general"
        and question_semantics.confidence >= 0.4
    )
    dataset_has_specific_domain = (
        dataset_profile.domain != "general"
        and dataset_profile.domain_confidence >= 0.3
    )
    domains_clearly_different = (
        question_has_specific_domain
        and dataset_has_specific_domain
        and not domain_match
    )

    if domains_clearly_different and score < INCOMPATIBILITY_THRESHOLD:
        reason = _build_incompatibility_reason(
            question_semantics, dataset_profile,
            missing_entities, missing_concepts,
        )
        return CompatibilityResult(
            compatible=False,
            compatibility_score=round(score, 2),
            matched_entities=matched_entities,
            missing_entities=missing_entities,
            missing_concepts=missing_concepts,
            reason=reason,
            dataset_domain=dataset_profile.domain,
            question_domain=question_semantics.domain,
        )

    if (
        not domain_match
        and not matched_entities
        and not matched_concepts
        and len(question_semantics.entities) >= 2
        and question_semantics.confidence >= 0.5
    ):
        reason = _build_incompatibility_reason(
            question_semantics, dataset_profile,
            missing_entities, missing_concepts,
        )
        return CompatibilityResult(
            compatible=False,
            compatibility_score=round(score, 2),
            matched_entities=matched_entities,
            missing_entities=missing_entities,
            missing_concepts=missing_concepts,
            reason=reason,
            dataset_domain=dataset_profile.domain,
            question_domain=question_semantics.domain,
        )

    return CompatibilityResult(
        compatible=True,
        compatibility_score=round(score, 2),
        matched_entities=matched_entities,
        missing_entities=missing_entities,
        missing_concepts=missing_concepts,
        reason="Question is compatible with the dataset.",
        dataset_domain=dataset_profile.domain,
        question_domain=question_semantics.domain,
    )


def check_question_dataset_compatibility(
    question: str,
    df: pd.DataFrame | None,
) -> CompatibilityResult | None:
    """Check compatibility between a question and a DataFrame."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if not question or not question.strip():
        return None

    question_semantics = extract_question_semantics(question)
    dataset_profile = build_dataset_semantic_profile(df)

    return check_compatibility(question_semantics, dataset_profile)


@dataclass(frozen=True)
class CompatibilityDecision:
    """Tells callers whether analysis may continue."""

    allowed: bool
    reason: str = ""
    is_generic_analysis: bool = False
    is_overview_request: bool = False
    incompatibility_response: str = ""
    missing_concepts: list[str] = field(default_factory=list)
    matched_concepts: list[str] = field(default_factory=list)
    dataset_domain: str = ""
    question_domain: str = ""
    confidence: float = 0.0


def enforce_question_dataset_compatibility(
    question: str,
    df: pd.DataFrame | None,
    context: dict[str, Any] | None = None,
) -> CompatibilityDecision:
    """Return the gate decision before planning or execution."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return CompatibilityDecision(
            allowed=True,
            reason="No DataFrame available; cannot perform compatibility check.",
            is_generic_analysis=True,
        )

    if not question or not question.strip():
        return CompatibilityDecision(allowed=True, reason="Empty question.", is_generic_analysis=True)

    result = check_question_dataset_compatibility(question, df)
    if result is None:
        return CompatibilityDecision(allowed=True, reason="Check could not run.", is_generic_analysis=True)

    q_semantics = extract_question_semantics(question)
    is_generic = q_semantics.is_generic
    is_overview = _is_pure_overview_request(question)

    if result.compatible:
        return CompatibilityDecision(
            allowed=True,
            reason=result.reason,
            is_generic_analysis=is_generic,
            is_overview_request=is_overview,
            matched_concepts=result.matched_entities,
            missing_concepts=result.missing_entities,
            dataset_domain=result.dataset_domain,
            question_domain=result.question_domain,
            confidence=result.compatibility_score,
        )

    return CompatibilityDecision(
        allowed=False,
        reason=result.reason,
        is_generic_analysis=False,
        is_overview_request=False,
        incompatibility_response=result.reason,
        missing_concepts=result.missing_concepts,
        matched_concepts=result.matched_entities,
        dataset_domain=result.dataset_domain,
        question_domain=result.question_domain,
        confidence=result.compatibility_score,
    )


_PURE_OVERVIEW_PATTERNS = (
    "summarize this dataset", "describe this dataset", "what is this dataset",
    "overview of this dataset", "what can be analyzed", "what columns",
    "what fields", "show the columns", "show the fields", "show data types",
    "show missing", "missing values", "data quality", "find outliers",
    "find anomalies", "show duplicates", "data profile",
    "обзор данных", "опиши данные", "опиши датасет", "что ты можешь сказать о данных",
    "какие колонки", "какие поля", "покажи пропуски", "качество данных",
    "summarize this data", "what does this dataset contain",
    "summarize the dataset", "describe the data",
)


def _is_pure_overview_request(question: str) -> bool:
    """Check if the question is a genuine dataset overview request.

    'Summarize this dataset' → True (pure overview)
    'Summarize the main public health risks' → False (domain-specific request)
    """
    normalized = _normalize(question)
    return any(pattern in normalized for pattern in _PURE_OVERVIEW_PATTERNS)


# ── Domain matching ────────────────────────────────────────────────────────

_DOMAIN_ALIASES: dict[str, set[str]] = {
    "entertainment": {"entertainment", "media", "content"},
    "healthcare": {"healthcare", "medical", "health", "clinical"},
    "retail": {"retail", "ecommerce", "commerce", "sales"},
    "financial": {"financial", "finance", "banking"},
    "scientific": {"scientific", "science", "research", "ecology"},
    "education": {"education", "academic", "school"},
    "hr": {"hr", "human resources", "workforce"},
    "logistics": {"logistics", "supply chain", "transportation"},
    "geographic": {"geographic", "geospatial", "geo"},
    "operational": {"operational", "operations", "service management"},
    "survey": {"survey", "feedback", "polling"},
}


def _domains_match(question_domain: str, dataset_domain: str) -> bool:
    """Check if two domains match, considering aliases."""
    if question_domain == dataset_domain:
        return True
    q_aliases = _DOMAIN_ALIASES.get(question_domain, {question_domain})
    d_aliases = _DOMAIN_ALIASES.get(dataset_domain, {dataset_domain})
    return bool(q_aliases & d_aliases)


# ── Entity and concept matching ────────────────────────────────────────────

def _build_dataset_text(profile: DatasetSemanticProfile) -> str:
    """Build searchable text from dataset profile."""
    parts = [_normalize(c) for c in profile.column_names]
    parts.extend(_normalize(c) for c in profile.semantic_concepts)
    parts.extend(_normalize(e) for e in profile.supported_entities[:200])
    return " ".join(parts)


def _entity_in_dataset(entity: str, dataset_text: str, profile: DatasetSemanticProfile) -> bool:
    """Check if a question entity can be found in the dataset."""
    normalized_entity = _normalize(entity)
    if not normalized_entity or len(normalized_entity) < 2:
        return False

    # Direct substring match in dataset text (word-boundary aware for short entities)
    if len(normalized_entity) >= 4 and normalized_entity in dataset_text:
        return True
    elif len(normalized_entity) < 4:
        # Short entities need word-boundary matching
        import re
        if re.search(r'\b' + re.escape(normalized_entity) + r'\b', dataset_text):
            return True

    # Check individual words (for multi-word entities like "heart disease")
    words = normalized_entity.split()
    if len(words) > 1:
        # Require word-boundary matches for each word, only count words >= 3 chars
        significant_words = [w for w in words if len(w) >= 3]
        if significant_words:
            import re
            word_matches = sum(
                1 for w in significant_words
                if re.search(r'\b' + re.escape(w) + r'\b', dataset_text)
            )
            if word_matches >= len(significant_words) * 0.6:
                return True

    # Check column concepts (sample values) — require minimum length match
    for col_values in profile.column_concepts.values():
        for val in col_values:
            nv = _normalize(val)
            if len(nv) < 3:
                continue  # Skip single-character/very short values
            if normalized_entity in nv or nv in normalized_entity:
                return True

    return False


def _concept_in_dataset(concept: str, dataset_text: str, profile: DatasetSemanticProfile) -> bool:
    """Check if a required concept exists in the dataset."""
    return _entity_in_dataset(concept, dataset_text, profile)


# ── Incompatibility response builders ──────────────────────────────────────

def _build_incompatibility_reason(
    question_semantics: QuestionSemantics,
    dataset_profile: DatasetSemanticProfile,
    missing_entities: list[str],
    missing_concepts: list[str],
) -> str:
    """Build a human-readable incompatibility explanation."""
    # Describe what the dataset contains
    dataset_desc = _describe_dataset(dataset_profile)

    # Describe what's missing
    missing_parts: list[str] = []
    if missing_entities:
        missing_parts.append(
            f"fields related to {', '.join(missing_entities[:5])}"
        )
    if missing_concepts:
        missing_parts.append(
            f"concepts like {', '.join(missing_concepts[:5])}"
        )

    missing_desc = " or ".join(missing_parts) if missing_parts else "the required data fields"

    return (
        f"This dataset contains {dataset_desc}, "
        f"but it does not contain {missing_desc}. "
        f"The requested analysis cannot be performed from the available data."
    )


def _describe_dataset(profile: DatasetSemanticProfile) -> str:
    """Generate a human-readable description of what the dataset contains."""
    domain_desc = _DOMAIN_DESCRIPTIONS.get(profile.domain, "general data")

    # Add some specific concepts
    concepts = profile.semantic_concepts[:8]
    if concepts:
        concept_str = ", ".join(concepts[:6])
        return f"{domain_desc} such as {concept_str}"
    return domain_desc


_DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "retail": "retail and sales data",
    "entertainment": "entertainment and media metadata",
    "healthcare": "healthcare and clinical data",
    "financial": "financial and monetary data",
    "scientific": "scientific research data",
    "education": "educational data",
    "hr": "human resources and workforce data",
    "geographic": "geographic and demographic data",
    "operational": "operational and service management data",
    "logistics": "logistics and supply chain data",
    "survey": "survey and feedback data",
    "general": "general data",
}


# ── Incompatibility output builder ─────────────────────────────────────────

def build_incompatibility_output(
    question: str,
    result: CompatibilityResult,
) -> dict[str, Any]:
    """Build a standard output dict for an incompatible question."""
    summary = result.reason

    # Suggest compatible analyses
    suggestions = [
        "Ask a question about the fields and concepts available in this dataset.",
        "Use generic analytical questions like outlier analysis, distributions, or data quality checks.",
    ]

    return {
        "summary": summary,
        "final_answer": summary,
        "structured_report": {
            "question": question,
            "summary": summary,
            "key_findings": [],
            "evidence": [
                f"Dataset domain: {result.dataset_domain}",
                f"Question domain: {result.question_domain}",
                f"Compatibility score: {result.compatibility_score}",
            ],
            "limitations": [
                f"Missing entities: {', '.join(result.missing_entities[:5])}" if result.missing_entities else "",
                f"Missing concepts: {', '.join(result.missing_concepts[:5])}" if result.missing_concepts else "",
            ],
            "next_steps": suggestions,
            "tool_timeline": [{"tool": "semantic_compatibility_check", "status": "incompatible"}],
        },
        "tool_timeline": [{"tool": "semantic_compatibility_check", "status": "incompatible"}],
        "artifacts": [],
        "key_findings": [],
        "limitations": [
            f"Missing entities: {', '.join(result.missing_entities[:5])}" if result.missing_entities else "",
        ],
        "trace_metadata": {
            "analysis_type": "semantic_incompatibility",
            "suppress_key_findings": True,
            "disable_grouped_analysis_fallback": True,
            "compatibility_score": result.compatibility_score,
            "dataset_domain": result.dataset_domain,
            "question_domain": result.question_domain,
        },
    }


# ── Utilities ──────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return " ".join(str(text or "").casefold().replace("_", " ").split())
