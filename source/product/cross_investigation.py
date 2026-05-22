from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from source.product.investigation import ArtifactType, Finding, Investigation


@dataclass(frozen=True)
class CrossInvestigationPattern:
    pattern_type: str
    semantic_signature: str
    related_metrics: list[str] = field(default_factory=list)
    related_dimensions: list[str] = field(default_factory=list)
    confidence: str = "medium"
    supporting_investigations: list[str] = field(default_factory=list)
    typical_followups: list[str] = field(default_factory=list)
    common_validation_steps: list[str] = field(default_factory=list)
    business_interpretation: str = ""


@dataclass(frozen=True)
class AnalyticalPlaybook:
    playbook_id: str
    title: str
    pattern_types: list[str]
    common_flow: list[str]
    common_charts: list[str]
    common_followups: list[str]
    common_hypotheses: list[str]
    validation_questions: list[str]
    report_structure_hints: list[str]


@dataclass(frozen=True)
class ReusableHypothesis:
    hypothesis: str
    branch_type: str
    semantic_signature: str
    supporting_investigations: list[str] = field(default_factory=list)
    validation_questions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReusableInvestigationBranch:
    branch_type: str
    title: str
    suggested_question: str
    validation_step: str
    supporting_pattern: str


@dataclass(frozen=True)
class InvestigationSimilarity:
    investigation_id: str
    title: str
    score: float
    shared_patterns: list[str] = field(default_factory=list)


def extract_cross_investigation_patterns(
    investigations: list[Investigation],
    *,
    target_investigation_id: str | None = None,
) -> list[CrossInvestigationPattern]:
    """Extract deterministic reusable analytical patterns from existing investigations."""

    buckets: dict[str, dict[str, Any]] = {}
    for investigation in investigations:
        for item in _pattern_observations(investigation):
            key = f"{item['pattern_type']}::{item['semantic_signature']}"
            bucket = buckets.setdefault(
                key,
                {
                    "pattern_type": item["pattern_type"],
                    "semantic_signature": item["semantic_signature"],
                    "metrics": [],
                    "dimensions": [],
                    "investigations": [],
                },
            )
            bucket["metrics"].extend(item["metrics"])
            bucket["dimensions"].extend(item["dimensions"])
            bucket["investigations"].append(investigation.investigation_id)

    patterns = []
    for bucket in buckets.values():
        supporting = _dedupe(bucket["investigations"])
        if target_investigation_id and supporting == [target_investigation_id]:
            pass
        pattern_type = bucket["pattern_type"]
        patterns.append(
            CrossInvestigationPattern(
                pattern_type=pattern_type,
                semantic_signature=bucket["semantic_signature"],
                related_metrics=_dedupe(bucket["metrics"])[:6],
                related_dimensions=_dedupe(bucket["dimensions"])[:6],
                confidence="high" if len(set(supporting)) >= 2 else "medium",
                supporting_investigations=supporting[:8],
                typical_followups=_typical_followups(pattern_type),
                common_validation_steps=_validation_steps(pattern_type),
                business_interpretation=_business_interpretation(pattern_type),
            )
        )
    return sorted(patterns, key=lambda item: (len(item.supporting_investigations), item.confidence), reverse=True)


def build_analytical_playbooks(patterns: list[CrossInvestigationPattern]) -> list[AnalyticalPlaybook]:
    pattern_types = {pattern.pattern_type for pattern in patterns}
    playbooks = []
    if {"grouped_metric", "volume_relationship", "outlier"} & pattern_types:
        playbooks.append(metric_segmentation_playbook())
    if {"outlier", "data_quality"} & pattern_types:
        playbooks.append(anomaly_investigation_playbook())
    if {"trend"} & pattern_types:
        playbooks.append(demand_trend_playbook())
    if {"correlation", "volume_relationship"} & pattern_types:
        playbooks.append(driver_analysis_playbook())
    return playbooks or [metric_segmentation_playbook()]


def extract_reusable_hypotheses(investigations: list[Investigation]) -> list[ReusableHypothesis]:
    hypotheses: dict[str, ReusableHypothesis] = {}
    for investigation in investigations:
        for finding in investigation.findings:
            metadata = finding.metadata or {}
            branches = metadata.get("investigation_branches")
            if not isinstance(branches, list):
                continue
            for branch in branches:
                if not isinstance(branch, dict):
                    continue
                hypothesis = str(branch.get("hypothesis") or "").strip()
                if not hypothesis:
                    continue
                branch_type = str(branch.get("branch_type") or "driver_analysis")
                signature = _signature_from_finding(finding, branch_type)
                key = f"{branch_type}::{_normalize(hypothesis)}"
                existing = hypotheses.get(key)
                validation = [
                    str(item)
                    for item in branch.get("next_questions", [])
                    if str(item).strip()
                ] if isinstance(branch.get("next_questions"), list) else []
                if existing:
                    hypotheses[key] = ReusableHypothesis(
                        hypothesis=existing.hypothesis,
                        branch_type=existing.branch_type,
                        semantic_signature=existing.semantic_signature,
                        supporting_investigations=_dedupe([*existing.supporting_investigations, investigation.investigation_id]),
                        validation_questions=_dedupe([*existing.validation_questions, *validation]),
                    )
                else:
                    hypotheses[key] = ReusableHypothesis(
                        hypothesis=hypothesis,
                        branch_type=branch_type,
                        semantic_signature=signature,
                        supporting_investigations=[investigation.investigation_id],
                        validation_questions=validation,
                    )
    return sorted(hypotheses.values(), key=lambda item: len(item.supporting_investigations), reverse=True)


def build_reusable_branches(patterns: list[CrossInvestigationPattern]) -> list[ReusableInvestigationBranch]:
    branches = []
    for pattern in patterns:
        for question, validation in zip(pattern.typical_followups[:3], pattern.common_validation_steps[:3]):
            branches.append(
                ReusableInvestigationBranch(
                    branch_type=pattern.pattern_type,
                    title=_pattern_title(pattern.pattern_type),
                    suggested_question=question,
                    validation_step=validation,
                    supporting_pattern=pattern.semantic_signature,
                )
            )
    return branches[:8]


def find_similar_investigations(target: Investigation, candidates: list[Investigation], limit: int = 5) -> list[InvestigationSimilarity]:
    target_patterns = {_observation_key(item) for item in _pattern_observations(target)}
    similarities = []
    for candidate in candidates:
        if candidate.investigation_id == target.investigation_id:
            continue
        candidate_patterns = {_observation_key(item) for item in _pattern_observations(candidate)}
        if not target_patterns or not candidate_patterns:
            continue
        shared = sorted(target_patterns & candidate_patterns)
        score = len(shared) / max(len(target_patterns | candidate_patterns), 1)
        if score:
            similarities.append(
                InvestigationSimilarity(
                    investigation_id=candidate.investigation_id,
                    title=candidate.title,
                    score=round(score, 3),
                    shared_patterns=shared,
                )
            )
    return sorted(similarities, key=lambda item: item.score, reverse=True)[:limit]


def pattern_aware_suggestions(
    patterns: list[CrossInvestigationPattern],
    hypotheses: list[ReusableHypothesis] | None = None,
    limit: int = 6,
) -> list[str]:
    suggestions: list[str] = []
    for pattern in patterns:
        suggestions.extend(pattern.typical_followups)
    for hypothesis in hypotheses or []:
        suggestions.extend(hypothesis.validation_questions[:2])
    return _dedupe(suggestions)[: max(1, limit)]


def report_pattern_hints(patterns: list[CrossInvestigationPattern]) -> list[str]:
    hints: list[str] = []
    for pattern in patterns[:4]:
        if pattern.business_interpretation:
            hints.append(pattern.business_interpretation)
        hints.extend(pattern.common_validation_steps[:2])
    return _dedupe(hints)[:8]


def metric_segmentation_playbook() -> AnalyticalPlaybook:
    return AnalyticalPlaybook(
        playbook_id="metric_segmentation",
        title="Metric distribution and segmentation",
        pattern_types=["grouped_metric", "volume_relationship", "outlier"],
        common_flow=["compare metric by dimension", "check group spread", "normalize by volume", "inspect outliers"],
        common_charts=["grouped bar", "distribution", "volume scatter"],
        common_followups=[
            "Does volume explain this gap?",
            "Do the strongest groups remain strong by median?",
            "Are outliers concentrated in one subgroup?",
        ],
        common_hypotheses=["Group differences may reflect segment mix, volume concentration, or outlier effects."],
        validation_questions=["Could small sample size distort the result?", "Does the pattern persist across another segment?"],
        report_structure_hints=["State the strongest groups, evidence strength, volume caveat, and validation recommendation."],
    )


def anomaly_investigation_playbook() -> AnalyticalPlaybook:
    return AnalyticalPlaybook(
        playbook_id="anomaly_investigation",
        title="Anomaly and stability investigation",
        pattern_types=["outlier", "data_quality"],
        common_flow=["locate unusual groups", "check raw extreme records", "separate valid edge cases from data quality issues"],
        common_charts=["outlier table", "distribution", "grouped bar"],
        common_followups=["Are these outliers concentrated in one segment?", "Do extremes look valid or like data issues?"],
        common_hypotheses=["Unusual values may be rare valid cases, data errors, or sparse-group artifacts."],
        validation_questions=["Check row-level records behind the anomaly.", "Compare mean and median before using averages."],
        report_structure_hints=["Separate signal from data quality risk before making a recommendation."],
    )


def demand_trend_playbook() -> AnalyticalPlaybook:
    return AnalyticalPlaybook(
        playbook_id="trend_analysis",
        title="Trend and temporal shift analysis",
        pattern_types=["trend"],
        common_flow=["aggregate over time", "locate spikes/drops", "check sparse periods", "segment the trend"],
        common_charts=["line chart", "period table"],
        common_followups=["Does this pattern persist inside key segments?", "Is the movement seasonal or driven by one sparse period?"],
        common_hypotheses=["Temporal movement may reflect true change, seasonality, or sparse-period noise."],
        validation_questions=["Check period counts.", "Compare the trend across a relevant segment."],
        report_structure_hints=["Report movement, confidence, sparse-period caveat, and monitoring recommendation."],
    )


def driver_analysis_playbook() -> AnalyticalPlaybook:
    return AnalyticalPlaybook(
        playbook_id="driver_analysis",
        title="Driver and relationship analysis",
        pattern_types=["correlation", "volume_relationship"],
        common_flow=["rank relationships", "check volume effects", "segment the relationship", "test outlier sensitivity"],
        common_charts=["scatter", "volume scatter", "segmented comparison"],
        common_followups=["Does the relationship persist inside key segments?", "Could a third variable explain both measures?"],
        common_hypotheses=["Relationships may reflect shared drivers rather than direct causal effects."],
        validation_questions=["Avoid causal wording until segment and outlier checks pass."],
        report_structure_hints=["Frame relationships as associations with validation needs, not causal claims."],
    )


def _pattern_observations(investigation: Investigation) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for finding in investigation.findings:
        metadata = finding.metadata or {}
        pattern_type = _normalize_pattern_type(str(metadata.get("analysis_type") or ""))
        if pattern_type in {"", "overview", "profile", "suggestion"}:
            continue
        metrics = _metadata_list(metadata, "related_metrics") or _extract_metric_dimension_from_text(finding.text)[0]
        dimensions = _metadata_list(metadata, "supporting_dimensions") or _extract_metric_dimension_from_text(finding.text)[1]
        observations.append(
            {
                "pattern_type": pattern_type,
                "semantic_signature": _semantic_signature(pattern_type, metrics, dimensions),
                "metrics": metrics,
                "dimensions": dimensions,
            }
        )
    for artifact in investigation.artifacts:
        if artifact.artifact_type != ArtifactType.CHART:
            continue
        content = artifact.content if isinstance(artifact.content, dict) else {}
        metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
        chart_type = str(content.get("chart_type") or metadata.get("chart_type") or "")
        pattern_type = _pattern_from_chart(chart_type)
        metric = str(content.get("metric") or metadata.get("metric") or content.get("y") or "").strip()
        dimension = str(content.get("x") or metadata.get("dimension") or "").strip()
        observations.append(
            {
                "pattern_type": pattern_type,
                "semantic_signature": _semantic_signature(pattern_type, [metric], [dimension]),
                "metrics": [metric] if metric else [],
                "dimensions": [dimension] if dimension else [],
            }
        )
    return observations


def _observation_key(item: dict[str, Any]) -> str:
    return f"{item['pattern_type']}::{item['semantic_signature']}"


def _normalize_pattern_type(value: str) -> str:
    normalized = value.strip().lower()
    mapping = {
        "deterministic_pandas": "grouped_metric",
        "group_unusual_values_check": "outlier",
        "outlier_check": "outlier",
        "correlation_check": "correlation",
        "trend_check": "trend",
        "thread_count_relationship_check": "volume_relationship",
    }
    return mapping.get(normalized, normalized)


def _pattern_from_chart(chart_type: str) -> str:
    normalized = chart_type.strip().lower()
    if normalized == "bar":
        return "grouped_metric"
    if normalized == "line":
        return "trend"
    if normalized == "scatter":
        return "correlation"
    if normalized == "histogram":
        return "distribution"
    return "visual_analysis"


def _semantic_signature(pattern_type: str, metrics: list[str], dimensions: list[str]) -> str:
    metric_roles = sorted({_semantic_role(value, metric=True) for value in metrics if value})
    dimension_roles = sorted({_semantic_role(value, metric=False) for value in dimensions if value})
    return f"{pattern_type}|metric:{','.join(metric_roles) or 'generic'}|dimension:{','.join(dimension_roles) or 'generic'}"


def _semantic_role(value: str, *, metric: bool) -> str:
    text = _normalize(value)
    if metric:
        if any(token in text for token in ("revenue", "sales", "amount", "price", "cost", "profit", "salary", "income", "выруч", "продаж", "доход")):
            return "value_metric"
        if any(token in text for token in ("count", "quantity", "volume", "users", "orders", "records", "колич", "число")):
            return "volume_metric"
        if any(token in text for token in ("score", "rating", "rank", "оцен", "рейтинг")):
            return "quality_metric"
        return "numeric_metric"
    if any(token in text for token in ("city", "region", "country", "state", "location", "город", "регион", "страна")):
        return "geo_dimension"
    if any(token in text for token in ("segment", "category", "type", "class", "group", "сегмент", "категор", "тип")):
        return "segment_dimension"
    if any(token in text for token in ("role", "job", "title", "profession", "position", "професс", "должн")):
        return "role_dimension"
    if any(token in text for token in ("company", "customer", "product", "industry", "vendor", "компан", "клиент", "продукт", "отрасл")):
        return "entity_dimension"
    return "categorical_dimension"


def _typical_followups(pattern_type: str) -> list[str]:
    mapping = {
        "grouped_metric": [
            "Does volume explain this gap?",
            "Do the strongest groups remain strong by median?",
            "Are outliers concentrated in one subgroup?",
            "Does this pattern persist across another segment?",
        ],
        "volume_relationship": [
            "Which groups outperform their record volume?",
            "Which groups are high volume but low average value?",
            "Does average value explain the remaining gap?",
        ],
        "outlier": [
            "Are these outliers concentrated in one segment?",
            "Do the same groups remain unusual by median?",
            "Do the extreme records look valid or like data quality issues?",
        ],
        "correlation": [
            "Does the relationship persist inside key segments?",
            "Could outliers be driving the relationship?",
            "Could a third variable explain both measures?",
        ],
        "trend": [
            "Does this pattern persist inside key segments?",
            "Is the movement seasonal or driven by one sparse period?",
            "Which period explains the largest change?",
        ],
        "data_quality": [
            "Which quality issue affects the strongest conclusion most?",
            "Do duplicates inflate important group counts?",
        ],
    }
    return mapping.get(pattern_type, ["What evidence would strengthen this conclusion?"])


def _validation_steps(pattern_type: str) -> list[str]:
    mapping = {
        "grouped_metric": ["Check sample size by group.", "Compare mean and median.", "Normalize totals by record volume."],
        "volume_relationship": ["Compare totals against record counts.", "Inspect groups that outperform volume."],
        "outlier": ["Inspect raw extreme records.", "Check whether outliers remain after segmenting."],
        "correlation": ["Break down the relationship by a relevant dimension.", "Check outlier sensitivity.", "Avoid causal claims without validation."],
        "trend": ["Check period counts.", "Compare trend inside key segments."],
        "data_quality": ["Resolve high-impact missingness.", "Check duplicates before interpreting counts."],
    }
    return mapping.get(pattern_type, ["Attach supporting chart or table evidence."])


def _business_interpretation(pattern_type: str) -> str:
    mapping = {
        "grouped_metric": "Group differences usually need volume, sample-size, and outlier checks before becoming decision-ready.",
        "volume_relationship": "Volume normalization helps separate scale effects from true performance differences.",
        "outlier": "Outlier-heavy conclusions need row-level validation before they drive decisions.",
        "correlation": "Relationship patterns should be framed as associations until segment and confounder checks pass.",
        "trend": "Temporal shifts need sparse-period and seasonality checks before operational interpretation.",
        "data_quality": "Quality risks affect how much weight downstream conclusions should carry.",
    }
    return mapping.get(pattern_type, "")


def _pattern_title(pattern_type: str) -> str:
    return pattern_type.replace("_", " ").title()


def _signature_from_finding(finding: Finding, branch_type: str) -> str:
    metadata = finding.metadata or {}
    metrics = _metadata_list(metadata, "related_metrics")
    dimensions = _metadata_list(metadata, "supporting_dimensions")
    return _semantic_signature(branch_type, metrics, dimensions)


def _metadata_list(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata.get(key)
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _extract_metric_dimension_from_text(text: str) -> tuple[list[str], list[str]]:
    values = []
    raw = str(text or "")
    start = 0
    while True:
        first = raw.find("`", start)
        if first == -1:
            break
        second = raw.find("`", first + 1)
        if second == -1:
            break
        values.append(raw[first + 1:second])
        start = second + 1
    return values[:1], values[1:2]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        key = _normalize(text)
        if key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _normalize(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").lower().split())
