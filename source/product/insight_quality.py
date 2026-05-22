from __future__ import annotations

from typing import Any

from source.product.hypothesis_engine import build_hypothesis_branches, hypothesis_metadata


def build_insight_metadata(
    text: str,
    *,
    evidence: list[str] | None = None,
    limitations: list[str] | None = None,
    next_steps: list[str] | None = None,
    artifact_ids: list[str] | None = None,
    kind: str = "insight",
    analysis_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return lightweight analyst-facing quality metadata for a finding."""

    evidence_items = [item for item in (evidence or []) if str(item).strip()]
    limitation_items = [item for item in (limitations or []) if str(item).strip()]
    next_step_items = [item for item in (next_steps or []) if str(item).strip()]
    output_count = len([item for item in (artifact_ids or []) if item])
    confidence = _confidence_level(evidence_items, limitation_items, output_count, kind)
    normalized_evidence = _normalized_evidence_strength(evidence_items, output_count)
    recommended = _recommended_next_step(text, next_step_items)
    context = analysis_context if isinstance(analysis_context, dict) else {}
    analysis_type = _normalized_analysis_type(
        str(context.get("analysis_type") or context.get("fallback") or _analysis_type(text, evidence_items))
    )
    branches = build_hypothesis_branches(
        conclusion=_clean_sentence(text),
        analysis_type=analysis_type,
        evidence=evidence_items,
        limitations=limitation_items,
        next_steps=next_step_items,
        metric=_context_text(context, "metric"),
        dimension=_context_text(context, "dimension"),
        time_axis=_context_text(context, "timestamp") or _context_text(context, "time_axis"),
    )
    metadata = {
        "kind": kind,
        "conclusion": _clean_sentence(text),
        "confidence": confidence,
        "confidence_level": confidence,
        "confidence_reason": _confidence_reason(confidence, evidence_items, limitation_items, output_count, kind),
        "evidence": evidence_items,
        "evidence_strength": normalized_evidence,
        "evidence_reason": _evidence_reason(evidence_items, output_count),
        "business_impact": _business_impact(text, kind),
        "business_implication": _business_implication(text, kind),
        "limitation": _first_or_default(limitation_items, _default_limitation(text, kind)),
        "recommended_validation": recommended,
        "recommended_next_step": recommended,
        "supporting_artifact_ids": list(artifact_ids or []),
        "related_outputs": list(artifact_ids or []),
        "supporting_dimensions": _supporting_dimensions(text, evidence_items),
        "supporting_evidence_count": len(evidence_items) + output_count,
        "analysis_type": analysis_type,
    }
    for key in (
        "metric",
        "dimension",
        "time_axis",
        "timestamp",
        "active_branch_type",
        "matched_category_value",
        "hypothesis_mechanism",
        "hypothesis_confidence",
        "hypothesis_supported",
    ):
        value = context.get(key)
        if value not in (None, ""):
            metadata[key] = value
    metadata.update(hypothesis_metadata(branches))
    return metadata


def _confidence_level(
    evidence: list[str],
    limitations: list[str],
    output_count: int,
    kind: str,
) -> str:
    if kind == "limitation":
        return "Needs validation"
    score = len(evidence) + output_count
    if limitations:
        score -= 1
    if score >= 3:
        return "High"
    if score >= 1:
        return "Medium"
    return "Low"


def _evidence_strength(evidence: list[str], output_count: int) -> str:
    score = len(evidence) + output_count
    if score >= 3:
        return "Strong"
    if score >= 1:
        return "Moderate"
    return "Needs evidence"


def _normalized_evidence_strength(evidence: list[str], output_count: int) -> str:
    score = len(evidence) + output_count
    if score >= 3:
        return "high"
    if score >= 1:
        return "medium"
    return "low"


def _confidence_reason(
    confidence: str,
    evidence: list[str],
    limitations: list[str],
    output_count: int,
    kind: str,
) -> str:
    if kind == "limitation":
        return "This item describes a risk or caveat that needs validation before final conclusions."
    if confidence == "High":
        return "Confidence is higher because the conclusion has multiple evidence references or supporting outputs."
    if limitations:
        return "Confidence is moderated by explicit limitations that should be checked before using the conclusion."
    if evidence or output_count:
        return "Confidence is moderate because the conclusion has some supporting evidence but still needs validation."
    return "Confidence is low until supporting evidence is attached."


def _evidence_reason(evidence: list[str], output_count: int) -> str:
    parts = []
    if evidence:
        parts.append(f"{len(evidence)} evidence note{'' if len(evidence) == 1 else 's'}")
    if output_count:
        parts.append(f"{output_count} linked output{'' if output_count == 1 else 's'}")
    return "Supported by " + " and ".join(parts) + "." if parts else "No supporting output is linked yet."


def _business_impact(text: str, kind: str) -> str:
    if kind == "limitation":
        return "Risk control"
    normalized = text.lower()
    high_markers = ("outlier", "anomal", "risk", "highest", "lowest", "significant", "strong", "variance", "spread")
    if any(marker in normalized for marker in high_markers):
        return "High"
    if any(marker in normalized for marker in ("trend", "segment", "group", "compare", "correlat")):
        return "Medium"
    return "Medium"


def _business_implication(text: str, kind: str) -> str:
    if kind == "limitation":
        return "This caveat affects whether the conclusion is ready for decision-making."
    normalized = text.lower()
    if any(marker in normalized for marker in ("outlier", "anomal", "extreme")):
        return "Unusual records may signal meaningful edge cases, data quality issues, or concentrated operational risk."
    if any(marker in normalized for marker in ("variance", "spread", "varies", "differs", "highest", "lowest")):
        return "Differences across groups may reveal inconsistent segment behavior and guide prioritization."
    if any(marker in normalized for marker in ("correlation", "relationship", "associated")):
        return "The relationship is worth deeper validation before using it as a decision driver."
    if any(marker in normalized for marker in ("trend", "increased", "decreased", "period")):
        return "The movement may indicate a change in underlying behavior that deserves monitoring."
    if any(marker in normalized for marker in ("missing", "duplicate", "quality")):
        return "Data reliability affects how much weight analysts should place on downstream conclusions."
    return "This conclusion can help focus the next analytical step or report narrative."


def _recommended_next_step(text: str, next_steps: list[str] | None) -> str:
    for step in next_steps or []:
        cleaned = " ".join(str(step).split())
        if cleaned:
            return cleaned
    normalized = text.lower()
    if "outlier" in normalized or "anomal" in normalized:
        return "Validate whether the unusual values are data issues or meaningful edge cases."
    if "chart" in normalized or "trend" in normalized:
        return "Use the visualization as evidence and inspect the strongest movement."
    if "group" in normalized or "segment" in normalized or "varies" in normalized:
        return "Compare the strongest and weakest groups with sample-size checks."
    return "Attach supporting evidence and decide whether this should move into the report."


def _analysis_type(text: str, evidence: list[str]) -> str:
    normalized = " ".join([text, *evidence]).lower()
    if any(marker in normalized for marker in ("outlier", "anomal", "extreme", "iqr")):
        return "outlier"
    if any(marker in normalized for marker in ("correlation", "relationship", "scatter")):
        return "correlation"
    if any(marker in normalized for marker in ("trend", "monthly", "period", "over time")):
        return "trend"
    if any(marker in normalized for marker in ("missing", "duplicate", "quality")):
        return "data_quality"
    if any(marker in normalized for marker in ("grouped", "by `", "across", "group", "segment")):
        return "grouped_metric"
    return "overview"


def _normalized_analysis_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    mapping = {
        "deterministic_pandas": "grouped_metric",
        "group_unusual_values_check": "outlier",
        "outlier_check": "outlier",
        "correlation_check": "correlation",
        "trend_check": "trend",
        "data_quality_check": "data_quality",
        "thread_count_relationship_check": "volume_relationship",
    }
    return mapping.get(normalized, normalized or "overview")


def _supporting_dimensions(text: str, evidence: list[str]) -> list[str]:
    normalized = " ".join([text, *evidence])
    dimensions: list[str] = []
    markers = (" by `", " across `", " group `", " segment `")
    for marker in markers:
        start = normalized.lower().find(marker)
        if start == -1:
            continue
        after = normalized[start + len(marker):]
        end = after.find("`")
        if end > 0:
            dimensions.append(after[:end])
    return list(dict.fromkeys(item for item in dimensions if item))


def _default_limitation(text: str, kind: str) -> str:
    if kind == "limitation":
        return _clean_sentence(text)
    normalized = text.lower()
    if "correlation" in normalized or "relationship" in normalized:
        return "Association does not imply causality and may reflect segment mix or outliers."
    if "outlier" in normalized or "anomal" in normalized:
        return "Outliers may reflect data errors or true rare cases."
    if "spread" in normalized or "variance" in normalized or "differs" in normalized:
        return "Small groups and extreme records can distort rankings."
    return "The conclusion should be validated with supporting evidence before being treated as final."


def _first_or_default(items: list[str], default: str) -> str:
    return _clean_sentence(items[0]) if items else default


def _clean_sentence(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _context_text(context: dict[str, Any], key: str) -> str | None:
    value = context.get(key)
    text = str(value or "").strip()
    return text or None
