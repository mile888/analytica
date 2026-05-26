from __future__ import annotations

from collections.abc import Iterable

from source.product.data_sources import (
    ColumnSemanticNote,
    DataSourceProfile,
    DataSourceSemanticNotes,
    DataSourceUsageContext,
)
from source.product.cross_investigation import CrossInvestigationPattern, ReusableHypothesis, pattern_aware_suggestions
from source.product.language_policy import DetectedLanguage
from source.product.semantic_layer import build_semantic_dataset_profile
from source.product.semantic_layer import InvestigationThreadState


def build_generic_question_suggestions(language: str | DetectedLanguage = DetectedLanguage.ENGLISH) -> list[str]:
    """Universal prompts that are safe for any dataset domain."""
    return [
        "Summarize this dataset.",
        "What are the most important patterns in this dataset?",
        "Which columns seem most useful for analysis?",
        "Are there missing values or data quality issues?",
        "Which groups or categories differ the most?",
        "Are there unusual values or anomalies?",
        "What should I investigate first?",
    ]


def build_data_aware_question_suggestions(
    profile: DataSourceProfile | None = None,
    usage_context: DataSourceUsageContext | None = None,
    semantic_notes: DataSourceSemanticNotes | None = None,
    limit: int = 6,
    response_language: str | DetectedLanguage | None = None,
) -> list[str]:
    """Build deterministic, dataset-agnostic prompts from profile/context metadata."""
    suggestions: list[str] = []
    language = DetectedLanguage.ENGLISH
    labels = _semantic_labels(semantic_notes)
    semantic_profile = build_semantic_dataset_profile(
        profile=profile,
        usage_context=usage_context,
        semantic_notes=semantic_notes,
    )

    metrics = [column.name for column in semantic_profile.metrics]
    dimensions = [column.name for column in semantic_profile.dimensions]
    timestamps = [column.name for column in semantic_profile.timestamps]
    identifiers = [column.name for column in semantic_profile.identifiers]

    metric = metrics[0] if metrics else None
    dimension = dimensions[0] if dimensions else None
    timestamp = timestamps[0] if timestamps else None
    identifier = identifiers[0] if identifiers else None

    if metric and dimension:
        suggestions.append(f"How does {_label(metric, labels)} vary across {_label(dimension, labels)}?")
        suggestions.append(f"Which {_label(dimension, labels)} groups have unusual {_label(metric, labels)} values?")
        suggestions.append(f"What segments explain the largest differences in {_label(metric, labels)}?")
    if identifier:
        suggestions.append(f"How many unique entities are there by {_label(identifier, labels)}?")
    if metric:
        suggestions.append(f"What are the highest and lowest values of {_label(metric, labels)}?")
        suggestions.append(f"Are there outliers in {_label(metric, labels)}?")
        suggestions.append(f"Which columns appear most correlated with {_label(metric, labels)}?")
    if dimension:
        suggestions.append(f"How are records distributed by {_label(dimension, labels)}?")
        suggestions.append(f"Which values in {_label(dimension, labels)} look unusual?")
    if timestamp:
        suggestions.append(f"How does the dataset change over time using {_label(timestamp, labels)}?")
        suggestions.append("Are there trends or spikes over time?")
        if metric:
            suggestions.append(f"Can {_label(metric, labels)} be forecast from the time pattern?")
            suggestions.append(f"Are there seasonal or recurring changes in {_label(metric, labels)}?")
    if len(metrics) >= 2:
        suggestions.append(f"How are {_label(metrics[0], labels)} and {_label(metrics[1], labels)} related?")

    if _has_missing_values(profile, usage_context):
        suggestions.append("Are there missing values or data quality issues?")

    suggestions.extend(build_generic_question_suggestions(language))
    return _dedupe(suggestions)[: max(1, limit)]


def build_thread_aware_question_suggestions(
    thread_state: InvestigationThreadState | dict | None,
    limit: int = 6,
    response_language: str | DetectedLanguage | None = None,
) -> list[str]:
    """Continue the current analytical thread with specific, non-generic prompts."""

    if isinstance(thread_state, InvestigationThreadState):
        metric = thread_state.active_metric
        dimension = thread_state.active_dimension
        time_axis = thread_state.active_time_axis
        findings = thread_state.recent_findings
        seeded = list(thread_state.suggested_next_questions or [])
        source_question = thread_state.active_business_question or ""
    elif isinstance(thread_state, dict):
        metric = _clean(thread_state.get("active_metric"))
        dimension = _clean(thread_state.get("active_dimension"))
        time_axis = _clean(thread_state.get("active_time_axis"))
        findings = [str(item) for item in thread_state.get("recent_findings", []) if str(item).strip()]
        seeded = [str(item) for item in thread_state.get("suggested_next_questions", []) if str(item).strip()]
        source_question = str(thread_state.get("active_business_question") or "")
    else:
        return []

    language = DetectedLanguage.ENGLISH
    suggestions: list[str] = []
    suggestions.extend(seeded)
    if metric and dimension:
        suggestions.extend(
            [
                f"Does record volume explain the `{metric}` differences across `{dimension}`?",
                f"Which `{dimension}` groups are statistical outliers for `{metric}`?",
                f"Which `{dimension}` groups are strongest by `{metric}`?",
                f"What explains the `{metric}` gap across `{dimension}`?",
                f"Could small sample size distort the `{dimension}` ranking?",
                f"Are high-`{metric}` `{dimension}` groups driven by average value or volume?",
            ]
        )
        if time_axis:
            suggestions.append(f"Does the `{metric}` by `{dimension}` pattern change over time?")
        else:
            suggestions.append(f"Does the `{metric}` pattern persist across another segment?")
    elif metric:
        suggestions.extend(
            [
                f"Which groups have unusual `{metric}` values?",
                f"What appears most associated with `{metric}`?",
                f"Which variables explain most variance in `{metric}`?",
            ]
        )
    if findings:
        suggestions.append("Which current findings need stronger evidence?")
    return _dedupe(suggestions)[: max(1, limit)]


def build_cross_investigation_question_suggestions(
    patterns: list[CrossInvestigationPattern],
    reusable_hypotheses: list[ReusableHypothesis] | None = None,
    limit: int = 6,
) -> list[str]:
    """Build suggestions from reusable analytical memory across investigations."""

    return pattern_aware_suggestions(patterns, reusable_hypotheses, limit=limit)


def _semantic_labels(notes: DataSourceSemanticNotes | None) -> dict[str, str]:
    if not notes:
        return {}
    labels: dict[str, str] = {}
    for note in notes.column_notes:
        labels[note.column_name] = _label_from_note(note)
    return {key: value for key, value in labels.items() if value}


def _label_from_note(note: ColumnSemanticNote) -> str:
    if note.display_name:
        return f"`{note.display_name}` (`{note.column_name}`)"
    if note.business_meaning:
        return f"`{note.column_name}` ({note.business_meaning})"
    if note.description:
        return f"`{note.column_name}` ({note.description})"
    return f"`{note.column_name}`"


def _label(column_name: str, labels: dict[str, str]) -> str:
    return labels.get(column_name) or f"`{column_name}`"


def _has_missing_values(profile: DataSourceProfile | None, context: DataSourceUsageContext | None) -> bool:
    missing = context.missing_summary if context else profile.missing_summary if profile else {}
    return any(int(value or 0) > 0 for value in missing.values())


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = " ".join(item.split()).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(item)
    return deduped


def _clean(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None
