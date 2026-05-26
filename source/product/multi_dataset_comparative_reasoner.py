from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


COMPARATIVE_REASONING_OPERATIONS = {
    "CROSS_DATASET_DISTRIBUTION_COMPARISON",
    "CROSS_DATASET_DIVERSITY_ANALYSIS",
    "CROSS_DATASET_ANOMALY_COMPARISON",
    "CROSS_DATASET_EXECUTIVE_SYNTHESIS",
    "CROSS_DATASET_TEMPORAL_ANALYSIS",
    "CROSS_DATASET_PROFITABILITY_COMPARISON",
    "CROSS_DATASET_RISK_PERFORMANCE_COMPARISON",
    "CROSS_DATASET_RELATIONSHIP_COMPARISON",
    "CROSS_DATASET_IMBALANCE_COMPARISON",
    "CROSS_DATASET_CONCENTRATION_COMPARISON",
    "CROSS_DATASET_OPERATIONAL_OPTIMIZATION",
    "CROSS_DATASET_PREDICTABILITY_COMPARISON",
    "CROSS_DATASET_FEATURE_STRUCTURE_COMPARISON",
    "PARALLEL_VISUAL_ANALYSIS",
}


@dataclass
class ComparativeEvidencePackage:
    dataset_id: str
    dataset_name: str
    domain_type: str
    comparison_type: str
    subquestion: str = ""
    metrics: list[str] = field(default_factory=list)
    derived_kpis: list[str] = field(default_factory=list)
    grouping_fields: list[str] = field(default_factory=list)
    distribution_stats: dict[str, Any] = field(default_factory=dict)
    concentration_stats: dict[str, Any] = field(default_factory=dict)
    relationship_stats: dict[str, Any] = field(default_factory=dict)
    anomaly_stats: dict[str, Any] = field(default_factory=dict)
    trend_stats: dict[str, Any] = field(default_factory=dict)
    diversity_stats: dict[str, Any] = field(default_factory=dict)
    kpi_stats: dict[str, Any] = field(default_factory=dict)
    feature_structure_stats: dict[str, Any] = field(default_factory=dict)
    key_findings: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    computed_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def has_computed_evidence(self) -> bool:
        return any(
            bool(value)
            for value in (
                self.distribution_stats,
                self.concentration_stats,
                self.relationship_stats,
                self.anomaly_stats,
                self.trend_stats,
                self.diversity_stats,
                self.kpi_stats,
                self.feature_structure_stats,
            )
        )


def normalize_comparative_evidence(
    *,
    question: str,
    operation: str,
    branches: list[dict[str, Any]],
) -> list[ComparativeEvidencePackage]:
    comparison_type = _comparison_type_from_operation(operation)
    packages: list[ComparativeEvidencePackage] = []
    for branch in branches:
        if not isinstance(branch, dict):
            continue
        computed = [result for result in branch.get("computed_results") or [] if isinstance(result, dict)]
        package = ComparativeEvidencePackage(
            dataset_id=str(branch.get("dataset_id") or ""),
            dataset_name=str(branch.get("dataset_name") or branch.get("dataset_id") or "Dataset"),
            domain_type=_domain_type(branch),
            comparison_type=str(branch.get("comparison_type") or comparison_type),
            subquestion=str(branch.get("subquestion") or ""),
            metrics=_unique([str(item) for item in branch.get("metrics_used") or [] if str(item).strip()]),
            derived_kpis=_unique([str(item) for item in branch.get("derived_kpis") or [] if str(item).strip()]),
            grouping_fields=_unique([str(item) for item in branch.get("grouping_fields") or [] if str(item).strip()]),
            key_findings=[str(item) for item in branch.get("findings") or [] if str(item).strip()],
            artifacts=[artifact for artifact in branch.get("artifacts") or [] if isinstance(artifact, dict)],
            limitations=[str(item) for item in branch.get("limitations") or [] if str(item).strip()],
            computed_results=computed,
        )
        _fill_distribution(package, computed)
        _fill_anomaly(package, computed)
        _fill_concentration(package, computed)
        _fill_trend(package, computed)
        _fill_diversity(package, computed)
        _fill_relationship(package, computed)
        _fill_kpi(package, computed)
        _fill_feature_structure(package, branch)
        if not package.metrics:
            package.metrics = _infer_metrics_from_results(computed)
        if not package.grouping_fields:
            package.grouping_fields = _infer_groupings_from_results(computed)
        packages.append(package)
    return packages


def synthesize_comparative_evidence(
    *,
    question: str,
    operation: str,
    packages: list[ComparativeEvidencePackage],
) -> dict[str, Any]:
    active = [package for package in packages if package.has_computed_evidence]
    if len(active) < 2 and operation != "CROSS_DATASET_EXECUTIVE_SYNTHESIS":
        return _insufficient("comparative reasoning", packages)
    if operation == "CROSS_DATASET_DISTRIBUTION_COMPARISON":
        return _distribution_synthesis(active)
    if operation == "CROSS_DATASET_ANOMALY_COMPARISON":
        return _anomaly_synthesis(active)
    if operation in {"CROSS_DATASET_IMBALANCE_COMPARISON", "CROSS_DATASET_CONCENTRATION_COMPARISON"}:
        return _concentration_synthesis(active)
    if operation == "CROSS_DATASET_TEMPORAL_ANALYSIS":
        return _trend_synthesis(active)
    if operation == "CROSS_DATASET_DIVERSITY_ANALYSIS":
        return _diversity_synthesis(active)
    if operation in {"CROSS_DATASET_PROFITABILITY_COMPARISON", "CROSS_DATASET_RISK_PERFORMANCE_COMPARISON", "CROSS_DATASET_KPI_SYNTHESIS"}:
        return _kpi_synthesis(active)
    if operation == "CROSS_DATASET_RELATIONSHIP_COMPARISON":
        return _relationship_synthesis(active)
    if operation in {"CROSS_DATASET_PREDICTABILITY_COMPARISON", "CROSS_DATASET_FEATURE_STRUCTURE_COMPARISON"}:
        return _predictability_synthesis(active)
    if operation == "CROSS_DATASET_OPERATIONAL_OPTIMIZATION":
        return _operational_synthesis(active)
    if operation == "CROSS_DATASET_EXECUTIVE_SYNTHESIS":
        return _executive_synthesis(active or packages)
    if operation == "PARALLEL_VISUAL_ANALYSIS":
        return _visual_synthesis(active)
    return _generic_synthesis(active)


def critic_warnings_for_comparative_output(
    *,
    question: str,
    operation: str,
    answer: str,
    packages: list[ComparativeEvidencePackage],
    artifact_count: int = 0,
) -> list[str]:
    text = _norm(question)
    answer_text = _norm(answer)
    warnings: list[str] = []
    if operation in COMPARATIVE_REASONING_OPERATIONS:
        active = _packages_relevant_to_operation(operation, packages)
        if len(active) < 2:
            warnings.append("Comparative synthesis used fewer than two computed dataset branches.")
        omitted = [package.dataset_name for package in active if _norm(package.dataset_name) not in answer_text]
        if omitted and any(marker in text for marker in ("compare", "both", "each dataset", "between", "side by side", "side-by-side")):
            warnings.append("Comparative synthesis omitted one or more participating dataset names.")
        if "which dataset should i use" in answer_text:
            warnings.append("A compare-both request was answered as a dataset-choice prompt.")
        if operation not in {"DATASET_CAPABILITY_REASONING"} and "capability summary" in answer_text:
            warnings.append("Capability-summary language replaced analytical comparison.")
    if any(marker in text for marker in ("visual", "visualization", "chart", "plot", "graph", "side by side", "side-by-side")):
        expected = max(1, min(2, len(packages)))
        if artifact_count < expected:
            warnings.append("Comparative visualization request produced too few artifacts.")
    if operation == "CROSS_DATASET_RELATIONSHIP_COMPARISON":
        if sum(1 for package in packages if package.relationship_stats) < 2:
            warnings.append("Relationship comparison did not compute relationship evidence for at least two branches.")
    return warnings


def _packages_relevant_to_operation(operation: str, packages: list[ComparativeEvidencePackage]) -> list[ComparativeEvidencePackage]:
    if operation == "CROSS_DATASET_DISTRIBUTION_COMPARISON":
        return [package for package in packages if package.distribution_stats]
    if operation == "CROSS_DATASET_ANOMALY_COMPARISON":
        return [package for package in packages if package.anomaly_stats]
    if operation in {"CROSS_DATASET_IMBALANCE_COMPARISON", "CROSS_DATASET_CONCENTRATION_COMPARISON"}:
        return [package for package in packages if package.concentration_stats]
    if operation == "CROSS_DATASET_TEMPORAL_ANALYSIS":
        return [package for package in packages if package.trend_stats]
    if operation == "CROSS_DATASET_DIVERSITY_ANALYSIS":
        return [package for package in packages if package.diversity_stats]
    if operation in {"CROSS_DATASET_PROFITABILITY_COMPARISON", "CROSS_DATASET_RISK_PERFORMANCE_COMPARISON", "CROSS_DATASET_KPI_SYNTHESIS"}:
        return [package for package in packages if package.kpi_stats or package.concentration_stats or package.anomaly_stats]
    if operation == "CROSS_DATASET_RELATIONSHIP_COMPARISON":
        return [package for package in packages if package.relationship_stats]
    return [package for package in packages if package.has_computed_evidence]


def _distribution_synthesis(packages: list[ComparativeEvidencePackage]) -> dict[str, Any]:
    rows = [package for package in packages if package.distribution_stats]
    if len(rows) < 2:
        return _insufficient("distribution comparison", packages)
    widest = max(rows, key=lambda package: float(package.distribution_stats.get("range") or 0))
    highest_outliers = max(rows, key=lambda package: float(package.distribution_stats.get("outlier_rate") or 0))
    parts = [
        f"`{package.dataset_name}` uses `{package.distribution_stats.get('metric')}`: range {_fmt(package.distribution_stats.get('range'))}, median {_fmt(package.distribution_stats.get('median'))}, outlier rate {_fmt(package.distribution_stats.get('outlier_rate'))}%."
        for package in rows
    ]
    answer = " ".join(parts)
    answer += f" `{widest.dataset_name}` has the wider distribution, while `{highest_outliers.dataset_name}` has the higher outlier rate."
    return _result(answer, rows, [f"`{widest.dataset_name}` has the widest branch distribution.", f"`{highest_outliers.dataset_name}` has the highest distribution outlier rate."])


def _anomaly_synthesis(packages: list[ComparativeEvidencePackage]) -> dict[str, Any]:
    rows = [package for package in packages if package.anomaly_stats]
    if len(rows) < 2:
        return _insufficient("anomaly comparison", packages)
    strongest = max(rows, key=lambda package: float(package.anomaly_stats.get("outlier_rate") or 0))
    parts = [
        f"`{package.dataset_name}` has {int(package.anomaly_stats.get('outlier_count') or 0)} `{package.anomaly_stats.get('metric')}` outlier(s), outlier rate {_fmt(package.anomaly_stats.get('outlier_rate'))}%."
        for package in rows
    ]
    answer = " ".join(parts) + f" The stronger anomaly pattern is `{strongest.dataset_name}` by computed outlier rate."
    return _result(answer, rows, [f"`{strongest.dataset_name}` has the highest anomaly rate."])


def _concentration_synthesis(packages: list[ComparativeEvidencePackage]) -> dict[str, Any]:
    rows = [package for package in packages if package.concentration_stats]
    if len(rows) < 2:
        return _insufficient("concentration comparison", packages)
    strongest = max(rows, key=_concentration_strength)
    parts = []
    findings = []
    for package in rows:
        stats = package.concentration_stats
        leader = stats.get("leader")
        leader_text = f"; leader `{leader}`" if leader else ""
        if stats.get("metric"):
            parts.append(f"`{package.dataset_name}` concentration for `{stats.get('metric')}` is {float(stats.get('gini') or 0):.3f}{leader_text}.")
        else:
            parts.append(f"`{package.dataset_name}` top-share concentration for `{stats.get('dimension')}` is {_fmt(stats.get('top_share'))}%{leader_text}.")
        findings.append(f"`{package.dataset_name}` concentration score {_concentration_strength(package):.3f}.")
    answer = " ".join(parts) + f" `{strongest.dataset_name}` shows stronger inequality or concentration from the normalized branch evidence."
    return _result(answer, rows, findings)


def _trend_synthesis(packages: list[ComparativeEvidencePackage]) -> dict[str, Any]:
    rows = [package for package in packages if package.trend_stats]
    if len(rows) < 2:
        return _insufficient("trend comparison", packages)
    strongest = max(rows, key=lambda package: abs(float(package.trend_stats.get("percent_change") or 0)))
    parts = []
    findings = []
    for package in rows:
        stats = package.trend_stats
        parts.append(
            f"`{package.dataset_name}` `{stats.get('metric')}` moves from {_fmt(stats.get('first_value'))} in {stats.get('first_period')} to {_fmt(stats.get('last_value'))} in {stats.get('last_period')} ({_fmt(stats.get('percent_change'))}% change)."
        )
        findings.append(f"`{package.dataset_name}` trend change: {_fmt(stats.get('percent_change'))}%.")
    answer = " ".join(parts) + f" `{strongest.dataset_name}` has the stronger observed trend movement by absolute percent change."
    return _result(answer, rows, findings)


def _diversity_synthesis(packages: list[ComparativeEvidencePackage]) -> dict[str, Any]:
    rows = [package for package in packages if package.diversity_stats]
    if len(rows) < 2:
        return _insufficient("diversity comparison", packages)
    strongest = max(rows, key=lambda package: float(package.diversity_stats.get("max_unique") or 0))
    parts = [
        f"`{package.dataset_name}` has peak diversity {int(package.diversity_stats.get('max_unique') or 0)} unique `{package.diversity_stats.get('entity_column')}` values within `{package.diversity_stats.get('group_column')}`."
        for package in rows
    ]
    answer = " ".join(parts) + f" `{strongest.dataset_name}` shows greater observed diversity; strategically, it needs more segmented controls because its branch evidence has the broader category mix, without merging unrelated entities."
    return _result(answer, rows, [f"`{strongest.dataset_name}` ranks highest on branch diversity richness."])


def _kpi_synthesis(packages: list[ComparativeEvidencePackage]) -> dict[str, Any]:
    rows = [package for package in packages if package.kpi_stats or package.concentration_stats or package.anomaly_stats]
    if len(rows) < 2:
        return _insufficient("KPI comparison", packages)
    parts = []
    findings = []
    for package in rows:
        stats = package.kpi_stats or package.concentration_stats or package.anomaly_stats
        leader = stats.get("leader") or stats.get("metric") or "top evidence"
        value = stats.get("leader_value", stats.get("value", stats.get("outlier_rate", stats.get("gini"))))
        metric = stats.get("metric") or ", ".join(package.metrics[:2]) or ", ".join(package.derived_kpis[:2]) or "computed KPI"
        parts.append(f"`{package.dataset_name}` leading `{metric}` evidence is `{leader}` ({_fmt(value)}).")
        findings.append(f"`{package.dataset_name}` branch KPI leader: `{leader}`.")
    answer = " ".join(parts) + " The comparison is grounded in each branch's deterministic KPI rows, so metrics are not merged across incompatible schemas."
    return _result(answer, rows, findings)


def _relationship_synthesis(packages: list[ComparativeEvidencePackage]) -> dict[str, Any]:
    rows = [package for package in packages if package.relationship_stats]
    if len(rows) < 2:
        return _insufficient("relationship comparison", packages)
    strongest = max(rows, key=lambda package: abs(float(package.relationship_stats.get("correlation") or 0)))
    parts = [
        f"`{package.dataset_name}` compares `{package.relationship_stats.get('x_metric')}` with `{package.relationship_stats.get('y_metric')}`; correlation {_fmt(package.relationship_stats.get('correlation'))}."
        for package in rows
    ]
    answer = " ".join(parts) + f" `{strongest.dataset_name}` shows the stronger metric relationship by absolute correlation."
    return _result(answer, rows, [f"`{strongest.dataset_name}` has the strongest branch relationship signal."])


def _predictability_synthesis(packages: list[ComparativeEvidencePackage]) -> dict[str, Any]:
    rows = [package for package in packages if package.feature_structure_stats]
    if len(rows) < 2:
        return _insufficient("predictability comparison", packages)
    strongest = max(rows, key=lambda package: float(package.feature_structure_stats.get("predictability_score") or 0))
    parts = [
        f"`{package.dataset_name}` predictability score {_fmt(package.feature_structure_stats.get('predictability_score'))} from {int(package.feature_structure_stats.get('metric_count') or 0)} metrics, {int(package.feature_structure_stats.get('dimension_count') or 0)} dimensions, and {int(package.feature_structure_stats.get('computed_signal_count') or 0)} computed signals."
        for package in rows
    ]
    answer = " ".join(parts) + f" `{strongest.dataset_name}` appears more predictable from feature structure because it has the strongest mix of measurable signals and segmentation fields."
    return _result(answer, rows, [f"`{strongest.dataset_name}` ranks highest on deterministic feature-structure predictability."])


def _operational_synthesis(packages: list[ComparativeEvidencePackage]) -> dict[str, Any]:
    rows = [package for package in packages if package.kpi_stats or package.feature_structure_stats]
    if len(rows) < 2:
        return _insufficient("operational comparison", packages)
    scored = sorted(rows, key=_operational_score, reverse=True)
    parts = []
    for package in scored:
        levers = _unique([*package.derived_kpis, *package.metrics, *package.grouping_fields])[:4]
        parts.append(f"`{package.dataset_name}` operational score {_fmt(_operational_score(package))} using {', '.join(f'`{item}`' for item in levers) or 'computed branch evidence'}.")
    answer = f"`{scored[0].dataset_name}` is better suited for operational optimization from the computed branch evidence. " + " ".join(parts)
    return _result(answer, scored, [f"`{scored[0].dataset_name}` has the strongest operational evidence package."])


def _executive_synthesis(packages: list[ComparativeEvidencePackage]) -> dict[str, Any]:
    rows = [package for package in packages if package.has_computed_evidence]
    if len(rows) < 2:
        return _insufficient("executive comparison", packages)
    direct = _direct_contrast(rows)
    insights = [direct]
    for package in rows:
        stats = package.kpi_stats or package.concentration_stats or package.trend_stats or package.anomaly_stats or package.diversity_stats or package.distribution_stats
        leader = stats.get("leader") or stats.get("metric") or ", ".join(package.metrics[:2]) or "computed evidence"
        value = stats.get("leader_value", stats.get("value", stats.get("gini", stats.get("outlier_rate", stats.get("percent_change")))))
        insights.append(f"`{package.dataset_name}` executive evidence centers on `{leader}` ({_fmt(value)}), based on {', '.join(package.derived_kpis[:3] or package.metrics[:3] or package.grouping_fields[:3])}.")
    insights = [item for item in insights if item and item.strip()][:5]
    answer = " ".join(f"{idx}. {insight}" for idx, insight in enumerate(insights, start=1))
    return _result(answer, rows, insights)


def _visual_synthesis(packages: list[ComparativeEvidencePackage]) -> dict[str, Any]:
    rows = [package for package in packages if package.artifacts]
    if len(rows) < 2:
        return _insufficient("comparative visualization", packages)
    parts = [f"`{package.dataset_name}` produced {len(package.artifacts)} dataset-scoped artifact(s) using {', '.join(package.metrics[:2] or package.grouping_fields[:2]) or 'computed evidence'}." for package in rows]
    answer = " ".join(parts) + " The artifacts remain side-by-side rather than forcing unrelated row-level data into one chart."
    return _result(answer, rows, [f"`{package.dataset_name}` contributed comparative artifact evidence." for package in rows])


def _generic_synthesis(packages: list[ComparativeEvidencePackage]) -> dict[str, Any]:
    if len(packages) < 2:
        return _insufficient("comparison", packages)
    parts = []
    findings = []
    for package in packages:
        evidence = package.key_findings[0] if package.key_findings else ", ".join(package.metrics[:3] or package.grouping_fields[:3]) or "computed branch evidence"
        parts.append(f"`{package.dataset_name}`: {evidence}.")
        findings.append(f"`{package.dataset_name}` contributed normalized branch evidence.")
    return _result(" ".join(parts), packages, findings)


def _fill_distribution(package: ComparativeEvidencePackage, computed: list[dict[str, Any]]) -> None:
    result = _first_result(computed, "distribution_comparison")
    if not result:
        return
    minimum = _float(result.get("minimum"))
    maximum = _float(result.get("maximum"))
    package.distribution_stats = {
        "metric": result.get("metric"),
        "minimum": minimum,
        "maximum": maximum,
        "range": maximum - minimum,
        "mean": _float(result.get("mean")),
        "median": _float(result.get("median")),
        "outlier_rate": _float(result.get("outlier_rate")),
        "outlier_count": int(_float(result.get("outlier_count"))),
    }


def _fill_anomaly(package: ComparativeEvidencePackage, computed: list[dict[str, Any]]) -> None:
    result = _first_result(computed, "anomaly_comparison")
    if not result:
        return
    package.anomaly_stats = {
        "metric": result.get("metric"),
        "outlier_count": int(_float(result.get("outlier_count"))),
        "outlier_rate": _float(result.get("outlier_rate")),
        "minimum": _float(result.get("minimum")),
        "maximum": _float(result.get("maximum")),
    }


def _fill_concentration(package: ComparativeEvidencePackage, computed: list[dict[str, Any]]) -> None:
    metric_result = _first_result(computed, "imbalance_comparison")
    category_result = _first_result(computed, "category_concentration")
    grouped_result = _first_result(computed, "grouped_metric")
    if metric_result:
        package.concentration_stats = {
            "metric": metric_result.get("metric"),
            "gini": _float(metric_result.get("gini")),
            "minimum": _float(metric_result.get("minimum")),
            "maximum": _float(metric_result.get("maximum")),
        }
    if category_result:
        rows = [row for row in category_result.get("rows") or [] if isinstance(row, dict)]
        leader = _row_label(rows[0]) if rows else ""
        package.concentration_stats.update({
            "dimension": category_result.get("dimension"),
            "top_share": _float(category_result.get("top_share")),
            "leader": leader,
        })
    if grouped_result:
        rows = [row for row in grouped_result.get("rows") or [] if isinstance(row, dict)]
        leader = _row_label(rows[0]) if rows else ""
        grouped_stats = {
            "metric": grouped_result.get("metric"),
            "dimension": grouped_result.get("dimension"),
            "top_share": _float(grouped_result.get("top_share")),
            "leader": leader,
            "leader_value": _float(rows[0].get("value")) if rows else 0.0,
        }
        package.concentration_stats.update(grouped_stats)


def _fill_trend(package: ComparativeEvidencePackage, computed: list[dict[str, Any]]) -> None:
    result = _first_result(computed, "temporal_trend")
    if not result:
        return
    rows = [row for row in result.get("rows") or [] if isinstance(row, dict)]
    stats = {
        "time_field": result.get("time_field"),
        "metric": result.get("metric"),
        "aggregation": result.get("aggregation"),
        "first_period": result.get("first_period"),
        "last_period": result.get("last_period"),
        "first_value": _float(result.get("first_value")),
        "last_value": _float(result.get("last_value")),
        "absolute_change": _float(result.get("absolute_change")),
        "percent_change": _float(result.get("percent_change")),
    }
    if rows and not stats["first_period"]:
        first = rows[0]
        last = rows[-1]
        first_value = _float(first.get("value"))
        last_value = _float(last.get("value"))
        stats.update({
            "first_period": first.get("period"),
            "last_period": last.get("period"),
            "first_value": first_value,
            "last_value": last_value,
            "absolute_change": last_value - first_value,
            "percent_change": ((last_value - first_value) / abs(first_value) * 100) if first_value else 0.0,
        })
    package.trend_stats = stats


def _fill_diversity(package: ComparativeEvidencePackage, computed: list[dict[str, Any]]) -> None:
    result = _first_result(computed, "diversity_analysis")
    if not result:
        return
    rows = [row for row in result.get("rows") or [] if isinstance(row, dict)]
    counts = [_float(row.get("unique_count")) for row in rows]
    package.diversity_stats = {
        "group_column": result.get("group_column"),
        "entity_column": result.get("entity_column"),
        "max_unique": max(counts) if counts else 0.0,
        "mean_unique": sum(counts) / len(counts) if counts else 0.0,
        "leader": _row_label(rows[0]) if rows else "",
    }


def _fill_relationship(package: ComparativeEvidencePackage, computed: list[dict[str, Any]]) -> None:
    result = _first_result(computed, "relationship_comparison") or _first_result(computed, "relationship_analysis") or _first_result(computed, "discount_sensitivity")
    if not result:
        return
    package.relationship_stats = {
        "x_metric": result.get("x_metric") or result.get("metric_x") or result.get("metric"),
        "y_metric": result.get("y_metric") or result.get("metric_y"),
        "correlation": _float(result.get("correlation")),
        "rows": result.get("rows") or [],
    }


def _fill_kpi(package: ComparativeEvidencePackage, computed: list[dict[str, Any]]) -> None:
    preferred = (
        "grouped_metric",
        "profit_margin_analysis",
        "business_efficiency_analysis",
        "high_sales_low_profit",
        "executive_priority",
        "discount_sensitivity",
        "risk_prevalence",
        "indicator_prevalence",
        "business_kpi",
        "operational_optimization_suitability",
    )
    result = next((_first_result(computed, analysis_type) for analysis_type in preferred if _first_result(computed, analysis_type)), {})
    if not result:
        return
    rows = [row for row in result.get("rows") or [] if isinstance(row, dict)]
    leader = _row_label(rows[0]) if rows else str(result.get("metric") or result.get("analysis_type") or "")
    package.kpi_stats = {
        "analysis_type": result.get("analysis_type"),
        "metric": result.get("metric") or result.get("y_metric") or result.get("x_metric"),
        "dimension": result.get("dimension"),
        "aggregation": result.get("aggregation"),
        "leader": leader,
        "leader_value": _row_value(rows[0]) if rows else _float(result.get("score", result.get("correlation", result.get("outlier_rate")))),
        "summary": result.get("summary"),
    }


def _fill_feature_structure(package: ComparativeEvidencePackage, branch: dict[str, Any]) -> None:
    profile = branch.get("semantic_profile") if isinstance(branch.get("semantic_profile"), dict) else {}
    metrics = _unique([*package.metrics, *[str(item) for item in profile.get("metric_columns") or [] if str(item).strip()]])
    dimensions = _unique([*package.grouping_fields, *[str(item) for item in profile.get("dimension_columns") or [] if str(item).strip()]])
    temporal = [str(item) for item in profile.get("timestamp_columns") or [] if str(item).strip()]
    computed_signal_count = sum(1 for result in branch.get("computed_results") or [] if isinstance(result, dict))
    score = min(1.0, 0.25 + 0.2 * bool(metrics) + 0.2 * bool(dimensions) + 0.15 * bool(temporal) + 0.1 * min(len(metrics), 5) / 5 + 0.1 * min(computed_signal_count, 4) / 4)
    package.feature_structure_stats = {
        "predictability_score": round(score, 3),
        "metric_count": len(metrics),
        "dimension_count": len(dimensions),
        "temporal_count": len(temporal),
        "computed_signal_count": computed_signal_count,
    }


def _result(answer: str, packages: list[ComparativeEvidencePackage], findings: list[str]) -> dict[str, Any]:
    return {
        "answer": answer,
        "findings": _unique(findings),
        "limitations": _unique([limitation for package in packages for limitation in package.limitations]),
        "next_steps": [],
        "evidence_packages": [package.to_dict() for package in packages],
    }


def _insufficient(label: str, packages: list[ComparativeEvidencePackage]) -> dict[str, Any]:
    names = ", ".join(f"`{package.dataset_name}`" for package in packages) or "the selected datasets"
    return {
        "answer": f"The {label} could not be completed because at least two dataset branches did not return comparable computed evidence for {names}.",
        "findings": [],
        "limitations": _unique([f"Not enough normalized {label} evidence was available.", *[limitation for package in packages for limitation in package.limitations]]),
        "next_steps": ["Run dataset-specific branch calculations with comparable evidence before final synthesis."],
        "evidence_packages": [package.to_dict() for package in packages],
    }


def _direct_contrast(packages: list[ComparativeEvidencePackage]) -> str:
    for selector, label in (
        (lambda p: p.concentration_stats, "concentration"),
        (lambda p: p.anomaly_stats, "anomaly pressure"),
        (lambda p: p.trend_stats, "trend movement"),
        (lambda p: p.diversity_stats, "diversity"),
        (lambda p: p.distribution_stats, "distribution spread"),
    ):
        rows = [package for package in packages if selector(package)]
        if len(rows) >= 2:
            if label == "concentration":
                strongest = max(rows, key=_concentration_strength)
            elif label == "trend movement":
                strongest = max(rows, key=lambda package: abs(float(package.trend_stats.get("percent_change") or 0)))
            elif label == "diversity":
                strongest = max(rows, key=lambda package: float(package.diversity_stats.get("max_unique") or 0))
            elif label == "distribution spread":
                strongest = max(rows, key=lambda package: float(package.distribution_stats.get("range") or 0))
            else:
                strongest = max(rows, key=lambda package: float(package.anomaly_stats.get("outlier_rate") or 0))
            return f"`{strongest.dataset_name}` stands out on {label}, so the main executive contrast is not schema capability but the size of the computed branch signal."
    kpi_rows = [package for package in packages if package.kpi_stats]
    if len(kpi_rows) >= 2:
        domains = {package.domain_type for package in kpi_rows}
        if len(domains) > 1:
            return "The executive contrast is domain-specific: branches use different KPI families, so decisions should compare risk or performance structure rather than merge the raw metrics."
        strongest = max(kpi_rows, key=lambda package: float(package.kpi_stats.get("leader_value") or 0))
        return f"`{strongest.dataset_name}` has the strongest branch KPI value, but the comparison should still read each KPI in its own dataset context."
    return f"All {len(packages)} dataset branches produced evidence; executive comparison should keep their KPIs separate while contrasting the strongest computed signal from each branch."


def _comparison_type_from_operation(operation: str) -> str:
    mapping = {
        "CROSS_DATASET_DISTRIBUTION_COMPARISON": "DISTRIBUTION_COMPARISON",
        "CROSS_DATASET_DIVERSITY_ANALYSIS": "DIVERSITY_COMPARISON",
        "CROSS_DATASET_ANOMALY_COMPARISON": "ANOMALY_COMPARISON",
        "CROSS_DATASET_EXECUTIVE_SYNTHESIS": "EXECUTIVE_COMPARISON",
        "CROSS_DATASET_TEMPORAL_ANALYSIS": "TREND_COMPARISON",
        "CROSS_DATASET_KPI_SYNTHESIS": "KPI_COMPARISON",
        "CROSS_DATASET_PROFITABILITY_COMPARISON": "PROFITABILITY_COMPARISON",
        "CROSS_DATASET_RISK_PERFORMANCE_COMPARISON": "RISK_PATTERN_COMPARISON",
        "CROSS_DATASET_RELATIONSHIP_COMPARISON": "RELATIONSHIP_COMPARISON",
        "CROSS_DATASET_IMBALANCE_COMPARISON": "CONCENTRATION_COMPARISON",
        "CROSS_DATASET_CONCENTRATION_COMPARISON": "CONCENTRATION_COMPARISON",
        "CROSS_DATASET_OPERATIONAL_OPTIMIZATION": "OPERATIONAL_COMPARISON",
        "CROSS_DATASET_PREDICTABILITY_COMPARISON": "PREDICTABILITY_COMPARISON",
        "CROSS_DATASET_FEATURE_STRUCTURE_COMPARISON": "FEATURE_STRUCTURE_COMPARISON",
    }
    return mapping.get(operation, "COMPARISON")


def _domain_type(branch: dict[str, Any]) -> str:
    profile = branch.get("semantic_profile") if isinstance(branch.get("semantic_profile"), dict) else {}
    concepts = " ".join(str(item) for item in profile.get("concepts") or []).casefold()
    if any(marker in concepts for marker in ("monetary", "purchase", "business", "customer", "product")):
        return "business"
    if any(marker in concepts for marker in ("health", "risk", "patient", "disease")):
        return "health"
    if any(marker in concepts for marker in ("education", "student", "grade")):
        return "education"
    return "general"


def _first_result(results: list[dict[str, Any]], analysis_type: str) -> dict[str, Any]:
    for result in results:
        if result.get("analysis_type") == analysis_type:
            return result
    return {}


def _infer_metrics_from_results(results: list[dict[str, Any]]) -> list[str]:
    metrics: list[str] = []
    for result in results:
        for key in ("metric", "x_metric", "y_metric"):
            value = result.get(key)
            if value:
                metrics.append(str(value))
    return _unique(metrics)


def _infer_groupings_from_results(results: list[dict[str, Any]]) -> list[str]:
    groups: list[str] = []
    for result in results:
        for key in ("dimension", "group_column", "entity_column", "time_field"):
            value = result.get(key)
            if value:
                groups.append(str(value))
    return _unique(groups)


def _concentration_strength(package: ComparativeEvidencePackage) -> float:
    stats = package.concentration_stats
    return max(_float(stats.get("gini")), _float(stats.get("top_share")) / 100.0)


def _operational_score(package: ComparativeEvidencePackage) -> float:
    if package.kpi_stats and package.kpi_stats.get("analysis_type") == "operational_optimization_suitability":
        return _float(package.kpi_stats.get("leader_value"))
    return min(1.0, 0.35 + 0.2 * bool(package.derived_kpis) + 0.2 * bool(package.metrics) + 0.15 * bool(package.grouping_fields) + 0.1 * bool(package.artifacts))


def _row_label(row: dict[str, Any]) -> str:
    for key in ("group", "dimension", "label", "name", "category", "segment", "metric"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    for key, value in row.items():
        if key not in {"count", "rank", "score", "n"} and not isinstance(value, (int, float)):
            return str(value)
    return "top row"


def _row_value(row: dict[str, Any]) -> float:
    for key in ("value", "total_profit", "total_revenue", "profit_margin", "operational_inefficiency_score", "high_sales_low_profit_score", "prevalence", "count", "records"):
        if key in row:
            return _float(row.get(key))
    for value in row.values():
        if isinstance(value, (int, float)):
            return _float(value)
    return 0.0


def _float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any) -> str:
    number = _float(value)
    if abs(number) >= 100:
        return f"{number:,.2f}"
    return f"{number:.2f}"


def _norm(value: Any) -> str:
    return " ".join("".join(char if char.isalnum() or char.isspace() else " " for char in str(value or "").casefold()).split())


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
