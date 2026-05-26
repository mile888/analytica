from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExecutionValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HypothesisExecutionResult:
    hypothesis: str
    metric: str = ""
    dimension: str = ""
    valid_row_count: int = 0
    group_count: int = 0
    verdict: str = ""
    evidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    validation_paths: list[str] = field(default_factory=list)
    confidence: str = "medium"
    derived_claims: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceExecutionResult:
    metric: str = ""
    dimension: str = ""
    evidence_target: str = ""
    supporting_facts: list[str] = field(default_factory=list)
    contradicting_facts: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    valid_row_count: int = 0
    group_count: int = 0
    derived_claims: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualityExecutionResult:
    quality_issue: str = ""
    metric: str = ""
    dimension: str = ""
    valid_row_count: int = 0
    group_count: int = 0
    affected_findings: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    derived_claims: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransformationExecutionResult:
    metric: str = ""
    dimension: str = ""
    transformation_type: str = ""
    original_row_count: int = 0
    filtered_row_count: int = 0
    removed_row_count: int = 0
    threshold: dict[str, float] = field(default_factory=dict)
    raw_rankings: list[dict[str, Any]] = field(default_factory=list)
    adjusted_rankings: list[dict[str, Any]] = field(default_factory=list)
    rank_shifts: list[dict[str, Any]] = field(default_factory=list)
    confidence_changes: list[str] = field(default_factory=list)
    weaker_conclusions: list[str] = field(default_factory=list)
    stronger_conclusions: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_valid(self) -> bool:
        return validate_execution_result(self).valid

    @property
    def dropped_leaders(self) -> list[dict[str, Any]]:
        rows = [
            row for row in self.rank_shifts
            if _num(row.get("rank_change")) >= 5 or _num(row.get("adjusted_rank")) > 20 or _num(row.get("adjusted_count")) == 0
        ]
        return sorted(rows, key=lambda row: (-_num(row.get("rank_change")), _num(row.get("original_rank"))))

    @property
    def stable_leaders(self) -> list[dict[str, Any]]:
        rows = [
            row for row in self.rank_shifts
            if _num(row.get("original_rank")) <= 10
            and _num(row.get("adjusted_rank")) <= 10
            and abs(_num(row.get("rank_change"))) <= 3
        ]
        return sorted(rows, key=lambda row: (_num(row.get("adjusted_rank")), _num(row.get("original_rank"))))

    @property
    def fragile_adjusted_leaders(self) -> list[dict[str, Any]]:
        rows = [row for row in self.adjusted_rankings[:10] if _num(row.get("count") or row.get("adjusted_count")) <= 2]
        return rows

    @property
    def adjusted_leaders(self) -> list[dict[str, Any]]:
        return self.adjusted_rankings[:5]


@dataclass(frozen=True)
class HypothesisEvaluation:
    hypothesis: str
    metric: str = ""
    dimension: str = ""
    mechanism: str = ""
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    validation_paths: list[str] = field(default_factory=list)
    confidence: str = "medium"
    supported: bool | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalyticalConversationGraph:
    active_branch: str = ""
    active_metric: str = ""
    active_dimension: str = ""
    active_time_axis: str = ""
    active_transformation: TransformationExecutionResult | None = None
    active_hypothesis: HypothesisEvaluation | HypothesisExecutionResult | None = None
    active_evidence_target: str = ""
    active_quality_issue: str = ""
    active_transformations: list[TransformationExecutionResult] = field(default_factory=list)
    active_findings: list[str] = field(default_factory=list)
    active_hypotheses: list[HypothesisEvaluation] = field(default_factory=list)
    active_quality_issues: list[str] = field(default_factory=list)
    active_artifacts: list[dict[str, Any]] = field(default_factory=list)
    active_validations: list[str] = field(default_factory=list)
    branch_history: list[str] = field(default_factory=list)
    claim_graph: dict[str, Any] = field(default_factory=dict)

    @property
    def latest_transformation(self) -> TransformationExecutionResult | None:
        if self.active_transformation is not None:
            return self.active_transformation
        return self.active_transformations[-1] if self.active_transformations else None


def validate_execution_result(
    result: TransformationExecutionResult,
    *,
    dataframe_row_count: int | None = None,
    dimension_cardinality: int | None = None,
) -> ExecutionValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if result.removed_row_count < 0 or result.filtered_row_count < 0 or result.original_row_count < 0:
        errors.append("row counts must be non-negative")
    if result.removed_row_count > result.original_row_count:
        errors.append("removed rows cannot exceed original rows")
    if result.filtered_row_count + result.removed_row_count != result.original_row_count:
        errors.append("filtered rows plus removed rows must equal original rows")
    if dataframe_row_count and result.original_row_count == 0:
        errors.append("original row count cannot be zero when dataframe rows are available")
    if result.threshold and result.original_row_count <= 0:
        errors.append("thresholded transformations require row counts")
    if result.adjusted_rankings and result.original_row_count <= 0 and result.removed_row_count > 0:
        errors.append("adjusted rankings require complete transformation metadata")
    if dimension_cardinality is not None:
        observed = _observed_group_count(result)
        if observed and observed > dimension_cardinality:
            errors.append("observed group count exceeds active dimension cardinality")
    if not result.raw_rankings and result.adjusted_rankings:
        warnings.append("adjusted rankings exist without raw ranking context")
    return ExecutionValidationResult(valid=not errors, errors=errors, warnings=warnings)


class ExecutionMetadataValidator:
    @staticmethod
    def validate(
        result: TransformationExecutionResult,
        *,
        dataframe_row_count: int | None = None,
        dimension_cardinality: int | None = None,
    ) -> ExecutionValidationResult:
        return validate_execution_result(
            result,
            dataframe_row_count=dataframe_row_count,
            dimension_cardinality=dimension_cardinality,
        )


def transformation_from_payload(value: Any) -> TransformationExecutionResult | None:
    if isinstance(value, TransformationExecutionResult):
        return value
    if not isinstance(value, dict):
        return None
    raw = _dict_rows(value.get("raw_rankings") or value.get("raw_ranking"))
    adjusted = _dict_rows(value.get("adjusted_rankings") or value.get("adjusted_ranking") or value.get("filtered_ranking"))
    shifts = _dict_rows(value.get("rank_shifts") or value.get("comparison_rows"))
    original = int(_num(value.get("original_row_count") or value.get("original_rows")))
    filtered = int(_num(value.get("filtered_row_count") or value.get("filtered_rows") or value.get("row_count")))
    removed = int(_num(value.get("removed_row_count") or value.get("removed_rows")))
    if original <= 0 and (filtered or removed):
        original = filtered + removed
    if filtered <= 0 and original >= removed:
        filtered = original - removed
    confidence = []
    if value.get("confidence_changes") and isinstance(value.get("confidence_changes"), list):
        confidence = [str(item) for item in value.get("confidence_changes") if str(item).strip()]
    elif str(value.get("confidence_change") or "").strip():
        confidence = [str(value.get("confidence_change"))]
    return TransformationExecutionResult(
        metric=str(value.get("metric") or ""),
        dimension=str(value.get("dimension") or ""),
        transformation_type=str(value.get("transformation_type") or value.get("analysis_type") or ""),
        original_row_count=original,
        filtered_row_count=filtered,
        removed_row_count=removed,
        threshold=value.get("threshold") if isinstance(value.get("threshold"), dict) else {},
        raw_rankings=raw,
        adjusted_rankings=adjusted,
        rank_shifts=shifts,
        confidence_changes=confidence,
        weaker_conclusions=[str(item) for item in value.get("weaker_conclusions") or [] if str(item).strip()] if isinstance(value.get("weaker_conclusions"), list) else [],
        stronger_conclusions=[str(item) for item in value.get("stronger_conclusions") or [] if str(item).strip()] if isinstance(value.get("stronger_conclusions"), list) else [],
    )


def transformation_from_rows(*, metric: str, dimension: str, rows: list[dict[str, Any]]) -> TransformationExecutionResult | None:
    shifts = _dict_rows(rows)
    if not shifts:
        return None
    return TransformationExecutionResult(metric=metric, dimension=dimension, rank_shifts=shifts)


class TransformationImpactAnalyzer:
    @staticmethod
    def analyze(payload: Any) -> TransformationExecutionResult | None:
        result = transformation_from_payload(payload)
        if result is None:
            return None
        validation = validate_execution_result(result)
        if not validation.valid and result.original_row_count > 0:
            return result
        return result


class EvidenceSynthesizer:
    @staticmethod
    def support(evaluation: HypothesisEvaluation) -> str:
        return synthesize_support(evaluation)

    @staticmethod
    def validation(evaluation: HypothesisEvaluation) -> str:
        return synthesize_validation(evaluation)


class ContradictionSynthesizer:
    @staticmethod
    def contradict(evaluation: HypothesisEvaluation) -> str:
        return synthesize_contradiction(evaluation)


class BranchTransitionSynthesizer:
    @staticmethod
    def synthesize(*, metric: str, dimension: str, leaders: list[dict[str, Any]] | None = None, caveat: str = "") -> str:
        leader_text = _leader_list(leaders[:3], dimension) if leaders else ""
        parts = [f"Back to the `{dimension}`-level `{metric}` view."]
        if leader_text:
            parts.append(f"The current leaders are {leader_text}.")
        if caveat:
            parts.append(caveat)
        return " ".join(parts)


def evaluate_large_record_hypothesis(
    *,
    hypothesis: str,
    transformation: TransformationExecutionResult,
) -> HypothesisEvaluation:
    dropped = transformation.dropped_leaders[:4]
    stable = transformation.stable_leaders[:4]
    supporting = []
    contradicting = []
    if dropped:
        supporting.append(
            f"{_shift_list(dropped, transformation.dimension)} collapsed after extreme `{transformation.metric}` records were removed."
        )
    if stable:
        contradicting.append(
            f"{_shift_list(stable, transformation.dimension)} stayed near the top after filtering."
        )
    if transformation.fragile_adjusted_leaders:
        supporting.append(
            f"Adjusted leaders remain sample-sensitive: {_leader_list(transformation.fragile_adjusted_leaders[:4], transformation.dimension)} have n<=2."
        )
    limitations = [
        "Row-level outlier filtering does not prove the removed records are data errors.",
        "Order-level aggregation is needed if rows are line items.",
    ]
    validation = [
        f"Remove each `{transformation.dimension}` group's largest `{transformation.metric}` record and re-rank.",
        f"Compare raw mean, median, and n-filtered `{transformation.dimension}` rankings.",
        "Repeat at distinct order level if an order identifier exists.",
    ]
    confidence = "medium" if transformation.is_valid and dropped else "low"
    return HypothesisEvaluation(
        hypothesis=hypothesis,
        metric=transformation.metric,
        dimension=transformation.dimension,
        mechanism="large_record_dependence",
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        limitations=limitations,
        validation_paths=validation,
        confidence=confidence,
        supported=bool(dropped),
    )


def synthesize_transformation_change(transformation: TransformationExecutionResult) -> str:
    parts = [
        f"Filtering changed `{transformation.metric}` by `{transformation.dimension}` from a raw average ranking into a robustness check."
    ]
    if transformation.dropped_leaders:
        parts.append(f"Collapsed raw leaders: {_shift_list(transformation.dropped_leaders[:4], transformation.dimension)}.")
    if transformation.stable_leaders:
        parts.append(f"Stable leaders: {_shift_list(transformation.stable_leaders[:4], transformation.dimension)}.")
    else:
        parts.append("No raw top leader stayed stable near the top after filtering.")
    if transformation.adjusted_leaders:
        parts.append(f"Adjusted leaders: {_leader_list(transformation.adjusted_leaders, transformation.dimension)}.")
    if transformation.fragile_adjusted_leaders:
        parts.append(f"Fragile adjusted leaders: {_leader_list(transformation.fragile_adjusted_leaders[:4], transformation.dimension)}.")
    parts.append(
        f"Rows removed: {transformation.removed_row_count:,} of {transformation.original_row_count:,}; "
        f"remaining rows: {transformation.filtered_row_count:,}."
    )
    if transformation.weaker_conclusions:
        parts.append("Weaker findings: " + " ".join(transformation.weaker_conclusions[:2]))
    if transformation.stronger_conclusions:
        parts.append("Stronger findings: " + " ".join(transformation.stronger_conclusions[:2]))
    return " ".join(parts)


def synthesize_support(evaluation: HypothesisEvaluation) -> str:
    if evaluation.supporting_evidence:
        return (
            " ".join(evaluation.supporting_evidence)
            + f" That is the strongest support in the current `{evaluation.metric}` by `{evaluation.dimension}` checks. Confidence is {evaluation.confidence}."
        )
    return f"No concrete supporting rank collapse is visible yet in the current `{evaluation.metric}` by `{evaluation.dimension}` checks."


def synthesize_contradiction(evaluation: HypothesisEvaluation) -> str:
    if evaluation.contradicting_evidence:
        return " ".join(evaluation.contradicting_evidence) + " That weakens an overbroad version of the hypothesis."
    return (
        f"No strong concrete counterexample is visible for `{evaluation.dimension}` in the current robustness checks. "
        "The raw leaders mostly fail the transformation or sample-size stress tests, so the data mainly supports fragility rather than contradicting it."
    )


def synthesize_validation(evaluation: HypothesisEvaluation) -> str:
    return "Validate it further by: " + "; ".join(evaluation.validation_paths) + "."


def has_impossible_counts(text: str) -> bool:
    normalized = str(text or "").replace(",", "")
    if " of 0" in normalized and any(marker in normalized.lower() for marker in ("removed", "rows", "based on 0 valid rows")):
        return True
    return "based on 0 valid rows" in normalized.lower()


def _observed_group_count(result: TransformationExecutionResult) -> int:
    dimension = result.dimension
    if not dimension:
        return 0
    values = set()
    for collection in (result.raw_rankings, result.adjusted_rankings, result.rank_shifts):
        for row in collection:
            value = row.get(dimension)
            if value is not None:
                values.add(str(value))
    return len(values)


def _shift_list(rows: list[dict[str, Any]], dimension: str) -> str:
    return ", ".join(
        f"`{row.get(dimension)}` rank #{int(_num(row.get('original_rank')))} -> #{int(_num(row.get('adjusted_rank')))} "
        f"(mean {_num(row.get('original_mean')):.2f} -> {_num(row.get('adjusted_mean')):.2f}, n={int(_num(row.get('adjusted_count') or row.get('original_count')))})"
        for row in rows
    )


def _leader_list(rows: list[dict[str, Any]], dimension: str) -> str:
    parts = []
    for row in rows:
        mean = row.get("adjusted_mean", row.get("mean", row.get("median", 0)))
        count = row.get("adjusted_count", row.get("count", 0))
        parts.append(f"`{row.get(dimension)}` (avg {_num(mean):.2f}, n={int(_num(count))})")
    return ", ".join(parts)


def _dict_rows(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
