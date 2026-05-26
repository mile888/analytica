from __future__ import annotations

from typing import Any


JOINABILITY_OPERATIONS = {
    "JOINABILITY_ANALYSIS",
    "WAREHOUSE_DESIGN",
    "CROSS_DATASET_LIMITATION_ANALYSIS",
}


def synthesize_cross_dataset_branches(
    *,
    question: str,
    operation: str,
    branches: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a cross-dataset answer from computed branch evidence."""

    normalized = _normalize_question(question)
    branch_views = [_branch_view(item) for item in branches if isinstance(item, dict)]
    if not branch_views:
        return {
            "answer": "No dataset branches were available for cross-dataset synthesis.",
            "findings": [],
            "limitations": ["No dataset-scoped evidence was available."],
            "next_steps": [],
        }

    if operation == "CROSS_DATASET_DISTRIBUTION_COMPARISON":
        return _distribution_comparison_synthesis(branch_views)
    if operation == "CROSS_DATASET_ANOMALY_COMPARISON":
        return _anomaly_comparison_synthesis(branch_views)
    if operation == "CROSS_DATASET_IMBALANCE_COMPARISON":
        return _imbalance_evidence_synthesis(branch_views)
    if operation == "CROSS_DATASET_RISK_PERFORMANCE_COMPARISON":
        return _risk_performance_synthesis(branch_views)
    if operation == "CROSS_DATASET_OPERATIONAL_OPTIMIZATION":
        return _operational_optimization_synthesis(branch_views)
    if operation == "CROSS_DATASET_EXECUTIVE_SYNTHESIS":
        return _executive_evidence_synthesis(branch_views)
    if _is_kpi_request(normalized):
        return _kpi_synthesis(branch_views)
    if operation == "CROSS_DATASET_DIVERSITY_ANALYSIS" or "diversity" in normalized:
        return _diversity_synthesis(branch_views)
    if _is_behavior_comparison(normalized):
        return _behavior_synthesis(branch_views)
    if "common patterns" in normalized or "common pattern" in normalized:
        return _common_patterns_synthesis(branch_views)
    if "risk concentration" in normalized or "imbalance" in normalized:
        return _risk_concentration_synthesis(branch_views)
    if _is_information_rich_category_request(normalized):
        return _categorical_synthesis(branch_views)
    if _is_prioritization_request(normalized):
        return _prioritization_synthesis(branch_views)
    if operation == "DATASET_CAPABILITY_REASONING" or _is_capability_request(normalized):
        return _capability_synthesis(normalized, branch_views)
    if operation == "PARALLEL_VISUAL_ANALYSIS":
        return _parallel_visual_synthesis(branch_views)
    return _comparative_synthesis(branch_views)


def critic_warnings_for_cross_dataset_output(
    *,
    question: str,
    operation: str,
    answer: str,
    branches: list[dict[str, Any]],
    artifact_count: int = 0,
) -> list[str]:
    warnings: list[str] = []
    normalized = _normalize_question(question)
    answer_norm = _normalize_question(answer)
    if operation not in JOINABILITY_OPERATIONS and not _is_joinability_request(normalized):
        if any(marker in answer_norm for marker in ("join", "joinable", "shared identifier", "warehouse", "row level")):
            warnings.append("Joinability language appeared in a non-joinability cross-dataset answer.")
    if _is_behavior_comparison(normalized) and any(marker in answer_norm for marker in ("average `", "higher average", "sum `")):
        warnings.append("Behavior comparison appears to use arbitrary metric aggregation.")
    if "time" in normalized or "forecast" in normalized or "trend" in normalized:
        for branch in branches:
            temporal = branch.get("temporal_fields") or []
            ordinal = branch.get("ordinal_fields") or []
            if any(_is_age_like(field) for field in temporal) or ("age" in answer_norm and "time series" in answer_norm and ordinal):
                warnings.append("Age or age-like ordinal fields were treated as time-series evidence.")
                break
    if len(branches) >= 2 and len({branch.get("dataset_id") for branch in branches}) < 2:
        warnings.append("Cross-dataset synthesis used fewer than two dataset branches.")
    if operation == "PARALLEL_VISUAL_ANALYSIS" and artifact_count < len(branches):
        warnings.append("Parallel visual analysis produced fewer artifacts than dataset branches.")
    if _is_kpi_request(normalized):
        missing = [branch.get("dataset_name") for branch in branches if branch.get("dataset_name") not in answer]
        if missing:
            warnings.append("KPI synthesis omitted one or more dataset names.")
    if "diversity" in normalized and "need numeric metric" in answer_norm:
        warnings.append("Diversity synthesis incorrectly required a numeric metric.")
    return warnings


def _branch_view(branch: dict[str, Any]) -> dict[str, Any]:
    roles = branch.get("semantic_roles") if isinstance(branch.get("semantic_roles"), dict) else {}
    profile = branch.get("semantic_profile") if isinstance(branch.get("semantic_profile"), dict) else {}
    computed = branch.get("computed_results") if isinstance(branch.get("computed_results"), list) else []
    columns = list(branch.get("columns") or roles.keys())
    metric_columns = _unique([*profile.get("metric_columns", []), *[col for col, role in roles.items() if role == "metric"]])
    dimension_columns = _unique([*profile.get("dimension_columns", []), *[col for col, role in roles.items() if role == "dimension"]])
    temporal_columns = _unique([col for col in profile.get("timestamp_columns", []) if _is_true_time_field(col)])
    ordinal_columns = _unique([col for col in columns if _is_ordinal_demographic_field(col)])
    concepts = list(profile.get("concepts") or [])
    capability_rows = []
    for result in computed:
        if isinstance(result, dict) and result.get("analysis_type") == "dataset_capability":
            capability_rows = [row for row in result.get("rows", []) if isinstance(row, dict)]
            break
    return {
        "dataset_id": str(branch.get("dataset_id") or ""),
        "dataset_name": str(branch.get("dataset_name") or branch.get("dataset_id") or "Dataset"),
        "operation": str(branch.get("operation") or ""),
        "compatibility_score": float(branch.get("compatibility_score") or 0.0),
        "roles": roles,
        "semantic_profile": profile,
        "concepts": concepts,
        "columns": columns,
        "metric_columns": [col for col in metric_columns if not _looks_identifier(col) and not _is_ordinal_demographic_field(col)],
        "dimension_columns": [col for col in dimension_columns if not _looks_identifier(col) and not _looks_entity_label(col)],
        "temporal_fields": temporal_columns,
        "ordinal_fields": ordinal_columns,
        "computed_results": computed,
        "findings": [str(item) for item in branch.get("findings", []) if str(item).strip()],
        "limitations": [str(item) for item in branch.get("limitations", []) if str(item).strip()],
        "artifact_count": len(branch.get("artifacts") or []),
        "capability_rows": capability_rows,
        "metrics_used": [str(item) for item in branch.get("metrics_used", []) if str(item).strip()],
        "derived_kpis": [str(item) for item in branch.get("derived_kpis", []) if str(item).strip()],
        "grouping_fields": [str(item) for item in branch.get("grouping_fields", []) if str(item).strip()],
        "subquestion": str(branch.get("subquestion") or ""),
        "comparison_type": str(branch.get("comparison_type") or ""),
    }


def _distribution_comparison_synthesis(branches: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in branches:
        result = _first_result(item, "distribution_comparison")
        if not result:
            continue
        rows.append((item, result))
    if not rows:
        return _insufficient_branch_evidence("distribution comparison", branches)
    widest = max(rows, key=lambda pair: float(pair[1].get("maximum", 0)) - float(pair[1].get("minimum", 0)))
    answer = " ".join(
        f"`{item['dataset_name']}` `{result.get('metric')}` ranges {float(result.get('minimum', 0)):.2f} to {float(result.get('maximum', 0)):.2f}, "
        f"median {float(result.get('median', 0)):.2f}, outlier rate {float(result.get('outlier_rate', 0)):.1f}%."
        for item, result in rows
    )
    answer += f" The wider distribution is `{widest[0]['dataset_name']}` based on observed range."
    return {
        "answer": answer,
        "findings": [f"`{item['dataset_name']}` distribution evidence uses `{result.get('metric')}` with outlier rate {float(result.get('outlier_rate', 0)):.1f}%." for item, result in rows],
        "limitations": _collect_limitations(branches),
        "next_steps": ["Inspect the dataset-scoped histogram artifacts before comparing operational implications."],
    }


def _anomaly_comparison_synthesis(branches: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [(item, result) for item in branches if (result := _first_result(item, "anomaly_comparison"))]
    if not rows:
        return _insufficient_branch_evidence("anomaly comparison", branches)
    strongest = max(rows, key=lambda pair: float(pair[1].get("outlier_rate", 0)))
    rates = [float(result.get("outlier_rate", 0)) for _, result in rows]
    answer = " ".join(
        f"`{item['dataset_name']}` has {int(result.get('outlier_count', 0))} `{result.get('metric')}` outlier(s), rate {float(result.get('outlier_rate', 0)):.1f}%."
        for item, result in rows
    )
    if max(rates, default=0.0) > min(rates, default=0.0):
        answer += f" The stronger anomaly pattern is `{strongest[0]['dataset_name']}` by outlier rate."
        finding = f"`{strongest[0]['dataset_name']}` has the highest branch outlier rate ({float(strongest[1].get('outlier_rate', 0)):.1f}%)."
    else:
        answer += " No branch shows a stronger anomaly pattern by outlier rate."
        finding = "No dataset branch has a higher outlier rate from the computed anomaly evidence."
    return {
        "answer": answer,
        "findings": [finding],
        "limitations": _collect_limitations(branches),
        "next_steps": ["Review row-level outlier tables for the highest-rate branch."],
    }


def _imbalance_evidence_synthesis(branches: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [(item, result) for item in branches if (result := _first_result(item, "imbalance_comparison"))]
    if not rows:
        return _risk_concentration_synthesis(branches)
    strongest = max(rows, key=lambda pair: float(pair[1].get("gini", 0)))
    answer = " ".join(
        f"`{item['dataset_name']}` imbalance for `{result.get('metric')}` is {float(result.get('gini', 0)):.3f}."
        for item, result in rows
    )
    answer += f" `{strongest[0]['dataset_name']}` shows stronger metric inequality by this branch evidence."
    return {
        "answer": answer,
        "findings": [f"`{item['dataset_name']}` gini-like imbalance: {float(result.get('gini', 0)):.3f}." for item, result in rows],
        "limitations": _collect_limitations(branches),
        "next_steps": ["Compare top-group concentration artifacts for categorical imbalance."],
    }


def _risk_performance_synthesis(branches: list[dict[str, Any]]) -> dict[str, Any]:
    parts = []
    findings = []
    for item in branches:
        result = _first_business_or_risk_result(item)
        if not result:
            parts.append(f"`{item['dataset_name']}` did not produce a compatible risk/performance branch.")
            continue
        if item["derived_kpis"]:
            parts.append(f"`{item['dataset_name']}` uses derived KPIs {', '.join(item['derived_kpis'][:4])}; leading evidence: {_result_leader(result)}.")
            findings.append(f"`{item['dataset_name']}` branch evidence: {_result_leader(result)}.")
        else:
            parts.append(f"`{item['dataset_name']}` uses risk evidence from {', '.join(item['metrics_used'][:4])}; leading evidence: {_result_leader(result)}.")
            findings.append(f"`{item['dataset_name']}` risk evidence: {_result_leader(result)}.")
    return {"answer": " ".join(parts), "findings": findings, "limitations": _collect_limitations(branches), "next_steps": ["Keep business KPIs and health-risk indicators separate; compare risk structure, not row-level records."]}


def _operational_optimization_synthesis(branches: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for item in branches:
        result = _first_result(item, "operational_optimization_suitability")
        score = float(result.get("score", 0)) if result else (0.45 + 0.1 * bool(item["metrics_used"]) + 0.1 * bool(item["grouping_fields"]))
        scored.append((score, item, result))
    scored.sort(key=lambda value: value[0], reverse=True)
    best = scored[0]
    answer = (
        f"`{best[1]['dataset_name']}` is better suited for operational optimization from the available evidence "
        f"(score {best[0]:.2f}) because it exposes {', '.join(best[1]['derived_kpis'][:4] or best[1]['metrics_used'][:4] or best[1]['grouping_fields'][:4])}. "
        + " ".join(f"`{item['dataset_name']}` score {score:.2f}: {(_result_leader(result) if result else 'limited branch evidence').rstrip('.')}." for score, item, result in scored)
    )
    return {"answer": answer, "findings": [f"`{item['dataset_name']}` operational suitability score {score:.2f}." for score, item, _ in scored], "limitations": _collect_limitations(branches), "next_steps": ["Prioritize the branch with computed controllable levers such as margin, discount, shipping burden, or risk-factor prevalence."]}


def _executive_evidence_synthesis(branches: list[dict[str, Any]]) -> dict[str, Any]:
    findings = []
    parts = []
    for item in branches:
        result = _first_business_or_risk_result(item) or (item["computed_results"][0] if item["computed_results"] else {})
        leader = _result_leader(result)
        parts.append(f"`{item['dataset_name']}`: {leader}.")
        findings.append(f"`{item['dataset_name']}` executive evidence: {leader}.")
    if len(branches) >= 2:
        findings.insert(0, f"Comparison used evidence from {len(branches)} dataset branches: {', '.join(item['dataset_name'] for item in branches)}.")
    return {
        "answer": " ".join(parts),
        "findings": findings[:5],
        "limitations": _collect_limitations(branches),
        "next_steps": ["Turn each branch insight into a dataset-scoped dashboard card before combining them in an executive view."],
    }


def _capability_synthesis(question: str, branches: list[dict[str, Any]]) -> dict[str, Any]:
    capability = _capability_kind(question)
    if capability == "time_series" and not any(item["temporal_fields"] for item in branches):
        answer = (
            "A true time-series comparison is not possible from the current branches because no dataset exposes a true observation date, timestamp, period, or event-year field. "
            + " ".join(
                f"`{item['dataset_name']}` has {', '.join(f'`{field}`' for field in item['ordinal_fields']) if item['ordinal_fields'] else 'no ordered time-like field'}; "
                f"{'these are ordinal demographic fields, not time-series fields' if item['ordinal_fields'] else 'add a date-like field for temporal analysis'}."
                for item in branches
            )
        )
        return {
            "answer": answer,
            "findings": ["Capability ranking: no dataset qualifies for true time-series analysis from the available fields."],
            "limitations": _collect_limitations(branches),
            "next_steps": _next_steps_for_capability(capability),
        }
    ranked = sorted(
        branches,
        key=lambda item: _capability_score(item, capability),
        reverse=True,
    )
    best = ranked[0]
    per_dataset = []
    findings = []
    for item in ranked:
        score = _capability_score(item, capability)
        evidence = _capability_evidence(item, capability)
        realistic = _realistic_analysis(item, capability)
        per_dataset.append(f"`{item['dataset_name']}` ({score:.2f}): {evidence}. Realistic analysis: {realistic}.")
        findings.append(f"`{item['dataset_name']}` {capability.replace('_', ' ')} suitability score {score:.2f}: {evidence}.")
    answer = (
        f"`{best['dataset_name']}` is the strongest for {capability.replace('_', ' ')} "
        f"because {_capability_evidence(best, capability)}. "
        + " ".join(per_dataset)
    )
    limitations = _collect_limitations(branches)
    if capability == "time_series":
        ordinal = [f"`{item['dataset_name']}` has ordinal fields ({', '.join(item['ordinal_fields'][:3])}) but no true time field" for item in branches if item["ordinal_fields"] and not item["temporal_fields"]]
        limitations.extend(ordinal)
    findings.insert(0, f"Capability ranking: `{best['dataset_name']}` ranks highest for {capability.replace('_', ' ')}.")
    return {"answer": answer, "findings": findings, "limitations": _unique(limitations), "next_steps": _next_steps_for_capability(capability)}


def _kpi_synthesis(branches: list[dict[str, Any]]) -> dict[str, Any]:
    parts = []
    findings = []
    for item in branches:
        kpis = _kpis_for_branch(item)
        parts.append(f"`{item['dataset_name']}`: " + "; ".join(f"{idx}. {kpi}" for idx, kpi in enumerate(kpis, start=1)) + ".")
        findings.append(f"`{item['dataset_name']}` dashboard KPIs should focus on {', '.join(kpi.split(' using ')[0] for kpi in kpis)}.")
    return {
        "answer": " ".join(parts),
        "findings": findings,
        "limitations": _collect_limitations(branches),
        "next_steps": ["Validate KPI definitions with stakeholders before turning them into dashboard cards."],
    }


def _behavior_synthesis(branches: list[dict[str, Any]]) -> dict[str, Any]:
    parts = []
    findings = []
    for item in branches:
        signals = _behavior_signals(item)
        parts.append(f"`{item['dataset_name']}` captures {signals}.")
        findings.append(f"`{item['dataset_name']}` behavior evidence is schema-grounded in {signals}.")
    answer = " ".join(parts) + " These behavior views should be compared as different analytical lenses, not merged into a shared behavioral metric."
    return {"answer": answer, "findings": findings, "limitations": _collect_limitations(branches), "next_steps": ["Choose one behavior lens per dataset and validate it with focused branch-level charts."]}


def _common_patterns_synthesis(branches: list[dict[str, Any]]) -> dict[str, Any]:
    parts = []
    findings = []
    for item in branches:
        strengths = _dataset_strengths(item)
        parts.append(f"`{item['dataset_name']}` contributes aggregate behavior evidence through {strengths}.")
        findings.append(f"`{item['dataset_name']}` common-pattern comparison should stay at aggregate behavior level.")
    answer = " ".join(parts) + " Compare these as aggregate behavior patterns rather than as a forced shared entity analysis."
    return {"answer": answer, "findings": findings, "limitations": _collect_limitations(branches), "next_steps": ["Validate each common pattern with a dataset-scoped chart or summary table."]}


def _diversity_synthesis(branches: list[dict[str, Any]]) -> dict[str, Any]:
    parts = []
    findings = []
    for item in branches:
        diversity = _diversity_definition(item)
        parts.append(f"`{item['dataset_name']}`: diversity means {diversity}.")
        findings.append(f"`{item['dataset_name']}` diversity can be studied through {diversity}.")
    answer = " ".join(parts) + " The comparison is about richness and concentration within each domain, without merging unrelated entities, so it does not require a shared numeric metric."
    return {"answer": answer, "findings": findings, "limitations": _collect_limitations(branches), "next_steps": ["Compare top-category concentration and long-tail richness within each dataset branch."]}


def _categorical_synthesis(branches: list[dict[str, Any]]) -> dict[str, Any]:
    parts = []
    findings = []
    for item in branches:
        fields = _information_rich_categories(item)
        parts.append(f"`{item['dataset_name']}`: {', '.join(f'`{field}`' for field in fields) if fields else 'no strong categorical field detected'} supports segmentation, concentration checks, and drill-down analysis.")
        if fields:
            findings.append(f"`{item['dataset_name']}` information-rich categorical fields: {', '.join(fields)}.")
    return {"answer": " ".join(parts), "findings": findings, "limitations": _collect_limitations(branches), "next_steps": ["Profile cardinality and top-category share for the shortlisted categorical fields."]}


def _prioritization_synthesis(branches: list[dict[str, Any]]) -> dict[str, Any]:
    priorities = {
        "business strategy": ("monetary_value", "purchase_volume", "customer_entity", "product_or_category"),
        "public policy": ("age_or_birth", "gender", "education_level", "health", "risk"),
        "operational optimization": ("operational_process", "time", "purchase_volume", "inventory"),
    }
    parts = []
    findings = []
    for label, concepts in priorities.items():
        chosen = max(branches, key=lambda item: _concept_score(item, concepts))
        evidence = _matching_concepts(chosen, concepts)
        parts.append(f"For {label}, choose `{chosen['dataset_name']}` because it exposes {evidence or 'the strongest relevant schema signals'}." )
        findings.append(f"`{chosen['dataset_name']}` is the strongest {label} candidate.")
    return {"answer": " ".join(parts), "findings": findings, "limitations": _collect_limitations(branches), "next_steps": ["Confirm the chosen dataset has enough row coverage and stable definitions for the target decision."]}


def _risk_concentration_synthesis(branches: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        branches,
        key=lambda item: _risk_signal_score(item) + 0.25 * bool(item["dimension_columns"]) + 0.15 * bool(item["metric_columns"]),
        reverse=True,
    )
    parts = []
    findings = []
    for item in ranked:
        score = round(min(1.0, _risk_signal_score(item) + 0.25 * bool(item["dimension_columns"]) + 0.15 * bool(item["metric_columns"])), 2)
        fields = _unique([*item["metric_columns"][:4], *item["dimension_columns"][:3]])
        parts.append(f"`{item['dataset_name']}` ({score:.2f}) has risk/imbalance evidence in {', '.join(f'`{field}`' for field in fields) or 'its available fields'}.")
        findings.append(f"`{item['dataset_name']}` risk concentration score {score:.2f}.")
    answer = f"`{ranked[0]['dataset_name']}` has the strongest evidence of risk concentration or imbalance. " + " ".join(parts)
    return {"answer": answer, "findings": findings, "limitations": _collect_limitations(branches), "next_steps": ["Run branch-level concentration checks: top-category share, binary prevalence gaps, and outlier scans."]}


def _parallel_visual_synthesis(branches: list[dict[str, Any]]) -> dict[str, Any]:
    parts = []
    findings = []
    for item in branches:
        result = _first_result(item)
        analysis_type = str(result.get("analysis_type") or "visual branch")
        parts.append(f"`{item['dataset_name']}` produced a {analysis_type.replace('_', ' ')} artifact scoped to that dataset.")
        findings.append(f"`{item['dataset_name']}` visual branch remained dataset-scoped.")
    return {"answer": " ".join(parts), "findings": findings, "limitations": _collect_limitations(branches), "next_steps": ["Review each chart separately, then compare patterns at the narrative level."]}


def _comparative_synthesis(branches: list[dict[str, Any]]) -> dict[str, Any]:
    parts = []
    findings = []
    for item in branches:
        strengths = _dataset_strengths(item)
        parts.append(f"`{item['dataset_name']}` is strongest for {strengths}.")
        findings.append(f"`{item['dataset_name']}` strongest analytical dimensions: {strengths}.")
    return {"answer": " ".join(parts), "findings": findings, "limitations": _collect_limitations(branches), "next_steps": ["Select a concrete question per dataset branch to deepen the comparison."]}


def _capability_kind(question: str) -> str:
    if any(marker in question for marker in ("forecast", "time series", "time-series", "temporal", "trend", "year", "years")):
        return "time_series"
    if "anomaly" in question or "outlier" in question or "imbalance" in question or "risk concentration" in question:
        return "anomaly_detection"
    if "dashboard" in question or "kpi" in question:
        return "dashboard_kpi"
    return "analytical_capability"


def _capability_score(item: dict[str, Any], capability: str) -> float:
    metrics = len(item["metric_columns"])
    dims = len(item["dimension_columns"])
    temporal = len(item["temporal_fields"])
    rows = _row_depth_score(item)
    if capability == "time_series":
        return round(min(1.0, 0.45 * bool(temporal) + 0.25 * bool(metrics) + 0.15 * bool(dims) + 0.15 * rows), 2)
    if capability == "anomaly_detection":
        return round(min(1.0, 0.4 * bool(metrics) + 0.25 * rows + 0.2 * bool(dims) + 0.15 * _risk_signal_score(item)), 2)
    if capability == "dashboard_kpi":
        return round(min(1.0, 0.35 * bool(metrics) + 0.25 * bool(dims) + 0.2 * bool(temporal) + 0.2 * _outcome_signal_score(item)), 2)
    return round(min(1.0, 0.35 * bool(metrics) + 0.35 * bool(dims) + 0.15 * bool(temporal) + 0.15 * rows), 2)


def _capability_evidence(item: dict[str, Any], capability: str) -> str:
    if capability == "time_series":
        if item["temporal_fields"]:
            return f"true time fields {', '.join(f'`{field}`' for field in item['temporal_fields'][:4])} plus metrics {', '.join(f'`{field}`' for field in item['metric_columns'][:4]) or 'record counts'}"
        if item["ordinal_fields"]:
            return f"ordinal demographic fields {', '.join(f'`{field}`' for field in item['ordinal_fields'][:4])}, which support ordered-group analysis but not longitudinal forecasting"
        return "no true date, timestamp, period, or event-year field was detected"
    if capability == "anomaly_detection":
        return f"numeric signals {', '.join(f'`{field}`' for field in item['metric_columns'][:5]) or 'none detected'} and segmentation fields {', '.join(f'`{field}`' for field in item['dimension_columns'][:4]) or 'none detected'}"
    if capability == "dashboard_kpi":
        return f"KPI candidates {', '.join(f'`{field}`' for field in item['metric_columns'][:5]) or 'record counts'} with dimensions {', '.join(f'`{field}`' for field in item['dimension_columns'][:4]) or 'limited segmentation'}"
    return _dataset_strengths(item)


def _realistic_analysis(item: dict[str, Any], capability: str) -> str:
    if capability == "time_series":
        if item["temporal_fields"]:
            metric = item["metric_columns"][0] if item["metric_columns"] else "record volume"
            return f"`{metric}` or record volume over `{item['temporal_fields'][0]}`"
        if item["ordinal_fields"]:
            return f"risk or outcome patterns by `{item['ordinal_fields'][0]}` as ordered groups, not time-series"
        return "metadata/profile comparison only unless a date-like field is added"
    if capability == "anomaly_detection":
        metric = item["metric_columns"][0] if item["metric_columns"] else "record frequency"
        segment = item["dimension_columns"][0] if item["dimension_columns"] else "available segments"
        return f"outliers in `{metric}` by `{segment}`"
    return _dataset_strengths(item)


def _kpis_for_branch(item: dict[str, Any]) -> list[str]:
    kpis: list[str] = []
    concepts = set(item["concepts"])
    metrics = item["metric_columns"]
    dims = item["dimension_columns"]
    temporal = item["temporal_fields"]
    if _outcome_signal_score(item) and metrics:
        kpis.append(f"Outcome or risk prevalence using `{_first_binary_like(metrics)}`")
    if "monetary_value" in concepts and metrics and not _risk_signal_score(item):
        kpis.append(f"Revenue or value performance using `{metrics[0]}`")
    if "purchase_volume" in concepts and metrics and not _risk_signal_score(item):
        kpis.append(f"Volume or activity intensity using `{metrics[min(len(metrics) - 1, 0)]}`")
    if temporal:
        kpis.append(f"Trend over time using `{temporal[0]}`")
    if dims:
        kpis.append(f"Segment/category concentration using `{dims[0]}`")
    if len(dims) >= 2:
        kpis.append(f"Diversity or mix breadth using `{dims[1]}`")
    if not kpis and metrics:
        kpis.append(f"Primary metric distribution using `{metrics[0]}`")
    while len(kpis) < 3:
        fallback = "Record coverage and data completeness" if len(kpis) == 0 else "Top-category share and long-tail coverage" if len(kpis) == 1 else "Segment-level change or imbalance"
        if fallback not in kpis:
            kpis.append(fallback)
        else:
            kpis.append("Branch-specific data quality guardrail")
    return kpis[:3]


def _behavior_signals(item: dict[str, Any]) -> str:
    concepts = set(item["concepts"])
    dims = item["dimension_columns"]
    metrics = item["metric_columns"]
    signals = []
    if concepts & {"customer_entity", "purchase_volume", "monetary_value", "product_or_category"}:
        signals.append("purchasing, segment, product, or value behavior")
    if concepts & {"survey_or_rating", "time", "geography"} and not concepts & {"purchase_volume", "monetary_value"}:
        signals.append("catalog or audience-proxy behavior through category, geography, rating, and timing fields")
    if concepts & {"age_or_birth", "gender", "education_level"} or _risk_signal_score(item):
        signals.append("demographic, lifestyle, or risk-indicator behavior")
    if not signals:
        signals.append("category and metric patterns")
    fields = _unique([*dims[:3], *metrics[:3]])
    return "; ".join(signals) + (f" grounded in {', '.join(f'`{field}`' for field in fields)}" if fields else "")


def _diversity_definition(item: dict[str, Any]) -> str:
    dims = item["dimension_columns"]
    metrics = item["metric_columns"]
    concepts = set(item["concepts"])
    if concepts & {"product_or_category", "customer_entity"} and dims:
        return f"customer/product/category breadth via `{dims[0]}`" + (f" and `{dims[1]}`" if len(dims) > 1 else "")
    if concepts & {"survey_or_rating", "geography", "time"} and dims:
        return f"content/category/geographic mix via `{dims[0]}`" + (f" and `{dims[1]}`" if len(dims) > 1 else "")
    if _risk_signal_score(item):
        return f"risk-factor or patient-segment spread via `{dims[0] if dims else metrics[0] if metrics else 'available indicator fields'}`"
    if dims:
        return f"categorical richness and concentration via `{dims[0]}`"
    return "available field variety; richer categorical fields would improve diversity analysis"


def _information_rich_categories(item: dict[str, Any]) -> list[str]:
    dims = [field for field in item["dimension_columns"] if not _looks_identifier(field)]
    ordinal = [field for field in item["ordinal_fields"] if field not in dims]
    return _unique([*dims, *ordinal])[:4]


def _dataset_strengths(item: dict[str, Any]) -> str:
    strengths = []
    concepts = set(item["concepts"])
    if item["temporal_fields"]:
        strengths.append("temporal trend analysis")
    if item["metric_columns"]:
        strengths.append("metric-based ranking and anomaly checks")
    if item["dimension_columns"]:
        strengths.append("segmentation and categorical comparison")
    if concepts & {"monetary_value", "purchase_volume"}:
        strengths.append("commercial performance analysis")
    if _risk_signal_score(item):
        strengths.append("risk or prevalence analysis")
    return ", ".join(_unique(strengths)[:4]) or "schema/profile reasoning"


def _first_result(item: dict[str, Any], analysis_type: str | None = None) -> dict[str, Any]:
    for result in item.get("computed_results") or []:
        if isinstance(result, dict) and (analysis_type is None or result.get("analysis_type") == analysis_type):
            return result
    return {}


def _first_business_or_risk_result(item: dict[str, Any]) -> dict[str, Any]:
    preferred = (
        "business_efficiency_analysis",
        "profit_margin_analysis",
        "high_sales_low_profit",
        "discount_sensitivity",
        "relationship_analysis",
        "executive_priority",
        "risk_prevalence",
        "indicator_prevalence",
        "business_kpi",
        "operational_optimization_suitability",
    )
    for analysis_type in preferred:
        result = _first_result(item, analysis_type)
        if result:
            return result
    return _first_result(item)


def _result_leader(result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict) or not result:
        return "limited computed evidence"
    rows = result.get("rows") if isinstance(result.get("rows"), list) else []
    if rows:
        row = next((item for item in rows if isinstance(item, dict)), {})
        if row:
            label = _row_label(row)
            facts = _row_facts(row)
            return f"{label} ({facts})" if facts else label
    evidence = result.get("evidence")
    if isinstance(evidence, list) and evidence:
        return str(evidence[0])
    metric = result.get("metric") or result.get("x_metric") or result.get("y_metric")
    if metric and "correlation" in result:
        return f"`{metric}` relationship correlation {float(result.get('correlation') or 0):.3f}"
    if metric and "outlier_rate" in result:
        return f"`{metric}` outlier rate {float(result.get('outlier_rate') or 0):.1f}%"
    if metric and "gini" in result:
        return f"`{metric}` imbalance {float(result.get('gini') or 0):.3f}"
    return str(result.get("summary") or result.get("analysis_type") or "computed branch evidence")


def _row_label(row: dict[str, Any]) -> str:
    for key in ("group", "dimension", "label", "name", "category", "segment", "metric"):
        value = row.get(key)
        if value not in (None, ""):
            return f"`{value}`"
    for key, value in row.items():
        if key not in {"count", "rank", "score", "n"} and not isinstance(value, (int, float)):
            return f"`{value}`"
    return "top row"


def _row_facts(row: dict[str, Any]) -> str:
    facts: list[str] = []
    priority = (
        "profit_margin",
        "total_profit",
        "total_revenue",
        "high_sales_low_profit_score",
        "operational_inefficiency_score",
        "prevalence",
        "outlier_rate",
        "gini",
        "records",
    )
    ordered_keys = [key for key in priority if key in row] + [key for key in row if key not in priority]
    for key in ordered_keys:
        value = row.get(key)
        if key in {"group", "dimension", "label", "name", "category", "segment", "rank"}:
            continue
        if isinstance(value, (int, float)):
            facts.append(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}")
        elif value not in (None, "") and len(facts) < 2:
            facts.append(f"{key}={value}")
        if len(facts) >= 3:
            break
    return ", ".join(facts)


def _insufficient_branch_evidence(label: str, branches: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_names = ", ".join(f"`{item['dataset_name']}`" for item in branches) or "the selected datasets"
    return {
        "answer": f"The {label} could not be completed because no branch returned compatible computed evidence for {dataset_names}.",
        "findings": [],
        "limitations": _unique([f"No compatible {label} evidence was available.", *_collect_limitations(branches)]),
        "next_steps": ["Run dataset-specific branch calculations with compatible metrics before synthesizing the comparison."],
    }


def _collect_limitations(branches: list[dict[str, Any]]) -> list[str]:
    limitations: list[str] = []
    for item in branches:
        limitations.extend(item.get("limitations") or [])
    return _unique(limitations)


def _next_steps_for_capability(capability: str) -> list[str]:
    if capability == "time_series":
        return ["Validate that date/year fields represent observation time, then plot branch-specific trends."]
    if capability == "anomaly_detection":
        return ["Run outlier checks on meaningful numeric metrics and compare anomalies by segment."]
    return ["Turn the strongest branch signals into focused dataset-specific analyses."]


def _row_depth_score(item: dict[str, Any]) -> float:
    for result in item.get("computed_results") or []:
        if isinstance(result, dict):
            for key in ("n", "row_count", "record_count"):
                try:
                    return min(float(result.get(key) or 0) / 1000.0, 1.0)
                except (TypeError, ValueError):
                    pass
    return min(float(item.get("compatibility_score") or 0), 1.0)


def _risk_signal_score(item: dict[str, Any]) -> float:
    text = _normalize_question(" ".join([*item["metric_columns"], *item["dimension_columns"], *item["concepts"]]))
    markers = ("risk", "outcome", "prevalence", "condition", "indicator", "disease", "health", "attack", "smoker", "activity", "bmi")
    return min(sum(1 for marker in markers if marker in text) / 3.0, 1.0)


def _outcome_signal_score(item: dict[str, Any]) -> float:
    text = _normalize_question(" ".join(item["metric_columns"]))
    return 1.0 if any(marker in text for marker in ("outcome", "target", "risk", "prevalence", "indicator", "flag", "attack", "disease")) else 0.0


def _concept_score(item: dict[str, Any], concepts: tuple[str, ...]) -> int:
    haystack = set(item["concepts"]) | set(_normalize_question(" ".join(item["columns"])).split())
    return sum(1 for concept in concepts if concept in haystack or concept.replace("_", " ") in " ".join(haystack))


def _matching_concepts(item: dict[str, Any], concepts: tuple[str, ...]) -> str:
    matched = []
    text = _normalize_question(" ".join(item["columns"]))
    for concept in concepts:
        if concept in set(item["concepts"]) or concept.replace("_", " ") in text or any(part in text for part in concept.split("_")):
            matched.append(concept.replace("_", " "))
    return ", ".join(_unique(matched)[:4])


def _first_binary_like(metrics: list[str]) -> str:
    for metric in metrics:
        text = _normalize_question(metric)
        if any(marker in text for marker in ("flag", "indicator", "outcome", "risk", "attack", "disease", "smoker")):
            return metric
    return metrics[0] if metrics else "available indicator"


def _is_true_time_field(field: Any) -> bool:
    text = _normalize_question(field)
    if _is_ordinal_demographic_field(text):
        return False
    return any(marker in text for marker in ("date", "datetime", "timestamp", "time", "year", "month", "period", "order date", "ship date", "date added", "release year"))


def _is_ordinal_demographic_field(field: Any) -> bool:
    text = _normalize_question(field)
    return any(marker in text for marker in ("age", "age group", "birth", "year birth", "income bracket", "education level", "risk score group"))


def _is_age_like(field: Any) -> bool:
    return "age" in _normalize_question(field).split()


def _looks_identifier(field: Any) -> bool:
    tokens = set(_normalize_question(field).split())
    return bool(tokens & {"id", "uuid", "key", "postal", "zip", "postcode"}) or str(field).lower().endswith("id")


def _looks_entity_label(field: Any) -> bool:
    tokens = set(_normalize_question(field).split())
    return bool(tokens & {"title", "name", "description", "comment", "text"})


def _is_joinability_request(text: str) -> bool:
    return any(marker in text for marker in ("join", "joinable", "merge", "shared key", "shared identifier", "warehouse", "relationship between datasets", "connect these datasets"))


def _is_capability_request(text: str) -> bool:
    return any(marker in text for marker in ("best suited", "most suitable", "support", "supports", "forecast", "time series", "anomaly detection", "prioritize one dataset"))


def _is_kpi_request(text: str) -> bool:
    return "kpi" in text or "dashboard" in text


def _is_behavior_comparison(text: str) -> bool:
    return "behavior" in text and any(marker in text for marker in ("compare", "differ", "across"))


def _is_information_rich_category_request(text: str) -> bool:
    return "categorical" in text or "information rich" in text


def _is_prioritization_request(text: str) -> bool:
    return "prioritize" in text and "dataset" in text


def _normalize_question(value: Any) -> str:
    cleaned = [char if char.isalnum() or char.isspace() else " " for char in str(value or "").casefold().replace("_", " ")]
    return " ".join("".join(cleaned).split())


def _unique(values: list[Any]) -> list[Any]:
    seen = set()
    out = []
    for value in values:
        key = str(value)
        if key and key not in seen:
            out.append(value)
            seen.add(key)
    return out
