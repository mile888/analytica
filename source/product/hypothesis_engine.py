from __future__ import annotations

from dataclasses import asdict
from typing import Any

from source.product.semantic_layer import HypothesisBranch


def build_hypothesis_branches(
    *,
    conclusion: str,
    analysis_type: str,
    evidence: list[str] | None = None,
    limitations: list[str] | None = None,
    next_steps: list[str] | None = None,
    metric: str | None = None,
    dimension: str | None = None,
    time_axis: str | None = None,
) -> list[HypothesisBranch]:
    """Create cautious, analyst-facing investigation branches from real analytical signals."""

    clean_type = str(analysis_type or "").strip().lower()
    evidence_items = _clean_items(evidence)
    limitation_items = _clean_items(limitations)
    next_step_items = _clean_items(next_steps)
    if clean_type in {"overview", "profile", "suggestion"}:
        return []
    if clean_type in {"grouped_metric", "deterministic_pandas", "volume_relationship"}:
        return [_grouped_branch(conclusion, evidence_items, limitation_items, next_step_items, metric, dimension)]
    if clean_type == "outlier":
        return [_outlier_branch(conclusion, evidence_items, limitation_items, next_step_items, metric, dimension)]
    if clean_type == "correlation":
        return [_correlation_branch(conclusion, evidence_items, limitation_items, next_step_items, metric)]
    if clean_type == "trend":
        return [_trend_branch(conclusion, evidence_items, limitation_items, next_step_items, metric, time_axis)]
    if clean_type == "data_quality":
        return [_quality_branch(conclusion, evidence_items, limitation_items, next_step_items)]
    return _generic_branches(conclusion, evidence_items, limitation_items, next_step_items)


def hypothesis_metadata(branches: list[HypothesisBranch]) -> dict[str, Any]:
    clean = [branch for branch in branches if branch.hypothesis.strip()]
    return {
        "hypotheses": [branch.hypothesis for branch in clean],
        "uncertainty_notes": [branch.uncertainty for branch in clean if branch.uncertainty],
        "validation_questions": [
            question
            for branch in clean
            for question in [branch.suggested_validation, *branch.next_questions]
            if question.strip()
        ],
        "possible_drivers": [
            driver
            for branch in clean
            for driver in branch.possible_drivers
            if driver.strip()
        ],
        "investigation_branches": [asdict(branch) for branch in clean],
    }


def _grouped_branch(
    conclusion: str,
    evidence: list[str],
    limitations: list[str],
    next_steps: list[str],
    metric: str | None,
    dimension: str | None,
) -> HypothesisBranch:
    metric_label = f"`{metric}`" if metric else "the metric"
    dimension_label = f"`{dimension}`" if dimension else "the grouping dimension"
    return HypothesisBranch(
        hypothesis=f"The {metric_label} gap across {dimension_label} may be driven by segment mix, volume effects, or outlier concentration.",
        supporting_evidence=evidence[:3],
        confidence=_confidence_from_evidence(evidence, limitations),
        uncertainty=_first(limitations, f"Group rankings need sample-size and outlier checks before treating the gap as stable."),
        possible_drivers=["subgroup mix", "record volume", "outlier concentration", "different average value per group"],
        conflicting_signals=["A high-total group may not have the highest average value."],
        suggested_validation=_first(next_steps, f"Compare average and total {metric_label} side by side for the strongest {dimension_label} groups."),
        next_questions=[
            f"Does record volume explain the {metric_label} gap across {dimension_label}?",
            f"Which {dimension_label} groups remain strong after checking outliers?",
            f"Does the pattern persist inside another relevant segment?",
        ],
        related_findings=[conclusion] if conclusion else [],
        branch_type="variance_explanation",
    )


def _outlier_branch(
    conclusion: str,
    evidence: list[str],
    limitations: list[str],
    next_steps: list[str],
    metric: str | None,
    dimension: str | None,
) -> HypothesisBranch:
    metric_label = f"`{metric}`" if metric else "the metric"
    dimension_label = f"`{dimension}`" if dimension else "specific groups"
    return HypothesisBranch(
        hypothesis=f"The unusual {metric_label} values may reflect a concentrated pocket of rare cases rather than broad behavior.",
        supporting_evidence=evidence[:3],
        confidence=_confidence_from_evidence(evidence, limitations),
        uncertainty=_first(limitations, "Outliers can be true edge cases, data errors, or small-sample artifacts."),
        possible_drivers=["rare high-value records", "data entry issues", "mixed populations", "small group sizes"],
        conflicting_signals=["A group can look anomalous because of one extreme record rather than a stable pattern."],
        suggested_validation=_first(next_steps, f"Inspect raw extreme records and compare mean vs median {metric_label} by {dimension_label}."),
        next_questions=[
            f"Are these {metric_label} outliers concentrated in one segment?",
            f"Do the same groups remain unusual by median {metric_label}?",
            "Do the extreme records look valid or like data quality issues?",
        ],
        related_findings=[conclusion] if conclusion else [],
        branch_type="anomaly",
    )


def _correlation_branch(
    conclusion: str,
    evidence: list[str],
    limitations: list[str],
    next_steps: list[str],
    metric: str | None,
) -> HypothesisBranch:
    metric_label = f"`{metric}`" if metric else "the target metric"
    return HypothesisBranch(
        hypothesis=f"The relationship around {metric_label} may be partly explained by a shared driver rather than a direct effect.",
        supporting_evidence=evidence[:3],
        confidence=_confidence_from_evidence(evidence, limitations),
        uncertainty=_correlation_uncertainty(limitations),
        possible_drivers=["segment mix", "record volume", "outliers", "shared dependence on another field"],
        conflicting_signals=["A visible correlation can weaken after segmenting the data."],
        suggested_validation=_first(next_steps, f"Break down the strongest relationship with {metric_label} by a relevant dimension."),
        next_questions=[
            f"Does the strongest relationship with {metric_label} persist inside key segments?",
            "Could outliers be driving the relationship?",
            "Which third variable might explain both measures?",
        ],
        related_findings=[conclusion] if conclusion else [],
        branch_type="causal_hypothesis",
    )


def _trend_branch(
    conclusion: str,
    evidence: list[str],
    limitations: list[str],
    next_steps: list[str],
    metric: str | None,
    time_axis: str | None,
) -> HypothesisBranch:
    metric_label = f"`{metric}`" if metric else "the metric"
    time_label = f"`{time_axis}`" if time_axis else "time"
    return HypothesisBranch(
        hypothesis=f"The movement in {metric_label} over {time_label} may reflect a real shift, seasonality, or sparse-period noise.",
        supporting_evidence=evidence[:3],
        confidence=_confidence_from_evidence(evidence, limitations),
        uncertainty=_first(limitations, "Sparse periods and one-off spikes can exaggerate trend movement."),
        possible_drivers=["seasonality", "one-off spike", "changing segment mix", "sparse observations"],
        conflicting_signals=["A sharp period change may disappear after smoothing or segmenting."],
        suggested_validation=_first(next_steps, f"Compare the {metric_label} trend across key segments and check period counts."),
        next_questions=[
            f"Does this {metric_label} trend persist inside the strongest segment?",
            "Is the movement seasonal or driven by one sparse period?",
            "Which period explains the largest change?",
        ],
        related_findings=[conclusion] if conclusion else [],
        branch_type="temporal_shift",
    )


def _quality_branch(
    conclusion: str,
    evidence: list[str],
    limitations: list[str],
    next_steps: list[str],
) -> HypothesisBranch:
    return HypothesisBranch(
        hypothesis="Data reliability issues may weaken downstream comparisons, rankings, and anomaly conclusions.",
        supporting_evidence=evidence[:3],
        confidence=_confidence_from_evidence(evidence, limitations),
        uncertainty=_first(limitations, "Quality checks are descriptive until domain rules confirm what is invalid."),
        possible_drivers=["missing values", "duplicate records", "inconsistent formatting", "extreme numeric values"],
        conflicting_signals=["Some apparent quality issues may be valid domain behavior."],
        suggested_validation=_first(next_steps, "Resolve the highest-impact missingness, duplicate, and extreme-value issues before final reporting."),
        next_questions=[
            "Which data quality issue affects the strongest conclusion most?",
            "Do duplicates inflate any important group counts?",
            "Which missing fields would change the analysis if filled?",
        ],
        related_findings=[conclusion] if conclusion else [],
        branch_type="data_quality_risk",
    )


def _generic_branches(
    conclusion: str,
    evidence: list[str],
    limitations: list[str],
    next_steps: list[str],
) -> list[HypothesisBranch]:
    if not conclusion:
        return []
    return [
        HypothesisBranch(
            hypothesis="This conclusion needs a supporting comparison, chart, or segment check before it becomes report-ready.",
            supporting_evidence=evidence[:3],
            confidence=_confidence_from_evidence(evidence, limitations),
            uncertainty=_first(limitations, "The evidence base is still limited."),
            possible_drivers=["segment differences", "outliers", "sample-size effects"],
            suggested_validation=_first(next_steps, "Attach supporting evidence and test the conclusion with a focused follow-up."),
            next_questions=["What evidence would strengthen this conclusion?"],
            related_findings=[conclusion],
            branch_type="driver_analysis",
        )
    ]


def _confidence_from_evidence(evidence: list[str], limitations: list[str]) -> str:
    if len(evidence) >= 2 and not limitations:
        return "high"
    if evidence:
        return "medium"
    return "low"


def _correlation_uncertainty(limitations: list[str]) -> str:
    base = "Association does not imply causality; segment mix and outliers can create apparent relationships."
    if not limitations:
        return base
    first = limitations[0]
    if "does not imply causality" in first.lower():
        return first
    return f"{base} {first}"


def _first(items: list[str], default: str) -> str:
    return items[0] if items else default


def _clean_items(items: list[str] | None) -> list[str]:
    return [" ".join(str(item).split()) for item in (items or []) if str(item).strip()]
