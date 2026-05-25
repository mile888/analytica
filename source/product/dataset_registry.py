from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd


from source.product.data_sources import DataSourceProfile
from source.product.cross_dataset_synthesis import (
    JOINABILITY_OPERATIONS,
    critic_warnings_for_cross_dataset_output,
    synthesize_cross_dataset_branches,
)
from source.product.business_semantic_planner import execute_business_plan
from source.product.multi_dataset_comparative_reasoner import (
    COMPARATIVE_REASONING_OPERATIONS,
    critic_warnings_for_comparative_output,
    normalize_comparative_evidence,
    synthesize_comparative_evidence,
)
from source.product.multi_dataset_synthesis_planner import plan_multi_dataset_synthesis
from source.product.question_routing import QuestionIntentType, classify_question_intent, is_multi_dataset_intent


class DatasetScope(StrEnum):
    SINGLE = "single_dataset"
    MULTIPLE = "multiple_datasets"
    CROSS = "cross_dataset"
    AMBIGUOUS = "ambiguous"


@dataclass
class InvestigationDatasetEntry:
    dataset_id: str
    display_name: str
    source_name: str
    schema: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    row_count: int = 0
    column_names: list[str] = field(default_factory=list)
    semantic_roles: dict[str, str] = field(default_factory=dict)
    semantic_profile: dict[str, Any] = field(default_factory=dict)
    sample_values: dict[str, list[str]] = field(default_factory=dict)
    created_at: str = ""
    executable_available: bool = False
    runtime_reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetResolutionResult:
    scope: DatasetScope
    selected_dataset_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    missing_requirements: list[str] = field(default_factory=list)
    candidate_scores: dict[str, int] = field(default_factory=dict)
    clarification_options: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scope"] = self.scope.value
        return payload


@dataclass
class DatasetExecutionScope:
    dataset_id: str
    dataset_name: str
    question_id: str
    operation: str
    compatibility_score: float
    semantic_roles: dict[str, str] = field(default_factory=dict)
    semantic_profile: dict[str, Any] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    comparison_type: str = ""
    subquestion: str = ""
    metrics_used: list[str] = field(default_factory=list)
    derived_kpis: list[str] = field(default_factory=list)
    grouping_fields: list[str] = field(default_factory=list)
    computed_results: list[dict[str, Any]] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_investigation_dataset_registry(store: Any, investigation: Any, data_source_ids: list[str]) -> list[InvestigationDatasetEntry]:
    registry: list[InvestigationDatasetEntry] = []
    for dataset_id in _unique(data_source_ids):
        try:
            source = store.get_data_source(dataset_id)
        except KeyError:
            continue
        profile = _profile_or_none(store, dataset_id)
        metadata = source.metadata if isinstance(source.metadata, dict) else {}
        context = metadata.get("execution_context") if isinstance(metadata.get("execution_context"), dict) else metadata
        columns = list(profile.columns if profile else [])
        semantic_roles = {str(column.name): _semantic_role(column.name, column.dtype) for column in columns}
        semantic_profile = _dataset_semantic_profile(
            column_names=[str(column.name) for column in columns],
            semantic_roles=semantic_roles,
            profile=_compact_profile(profile),
            display_name=str(source.name or dataset_id),
        )
        registry.append(
            InvestigationDatasetEntry(
                dataset_id=dataset_id,
                display_name=str(source.name or dataset_id),
                source_name=str(source.name or dataset_id),
                schema={
                    "columns": [
                        {"name": column.name, "dtype": column.dtype, "nullable": column.nullable}
                        for column in columns
                    ],
                    "row_count": int(getattr(profile, "row_count", 0) or 0),
                    "column_count": int(getattr(profile, "column_count", len(columns)) or len(columns)),
                },
                profile=_compact_profile(profile),
                row_count=int(getattr(profile, "row_count", 0) or 0),
                column_names=[str(column.name) for column in columns],
                semantic_roles=semantic_roles,
                semantic_profile=semantic_profile,
                sample_values={str(column.name): [str(value) for value in list(column.sample_values or [])[:10]] for column in columns},
                created_at=getattr(source.created_at, "isoformat", lambda: "")(),
                executable_available=bool(context.get("executable_available")),
                runtime_reference=str(context.get("dataset_runtime_reference") or context.get("storage_reference") or ""),
            )
        )
    return registry


def resolve_dataset_scope(
    *,
    question: str,
    registry: list[InvestigationDatasetEntry],
    conversation_context: dict[str, Any] | None = None,
    user_selected_dataset_ids: list[str] | None = None,
) -> DatasetResolutionResult:
    if not registry:
        return DatasetResolutionResult(DatasetScope.AMBIGUOUS, confidence=0.0, missing_requirements=["No datasets are attached to this investigation."])
    if len(registry) == 1:
        entry = registry[0]
        return DatasetResolutionResult(DatasetScope.SINGLE, [entry.dataset_id], 1.0, ["Only one dataset is attached."])

    text = _norm(question)
    if _is_cross_dataset_question(text):
        return DatasetResolutionResult(
            DatasetScope.CROSS,
            [entry.dataset_id for entry in registry],
            0.9,
            ["Question asks for a dataset comparison."],
        )

    selected = [item for item in user_selected_dataset_ids or [] if item in {entry.dataset_id for entry in registry}]
    if selected:
        return DatasetResolutionResult(DatasetScope.SINGLE if len(selected) == 1 else DatasetScope.MULTIPLE, selected, 0.92, ["User-selected dataset scope."])

    lineage_id = _lineage_dataset_id(conversation_context)
    explicit_scores: dict[str, int] = {}
    reasons_by_dataset: dict[str, list[str]] = {}
    for entry in registry:
        score = 0
        reasons: list[str] = []
        names = {_norm(entry.display_name), _norm(entry.source_name), _norm(entry.dataset_id)}
        if any(name and name in text for name in names):
            score += 100
            reasons.append("explicit dataset mention")
        matched_columns = [column for column in entry.column_names if _column_mentioned(column, text)]
        if matched_columns:
            # Metric columns (explicitly requested numeric fields) get a higher score
            # than shared generic columns, so the dataset containing the user's
            # requested metric wins over one that merely shares a dimension.
            for column in matched_columns:
                role = entry.semantic_roles.get(column, "")
                if role == "metric":
                    score += 80
                    reasons.append(f"explicit metric match: {column}")
                else:
                    score += 30
            if not any("explicit metric match" in r for r in reasons):
                reasons.append("column match: " + ", ".join(matched_columns[:4]))
        matched_values = _matched_values(entry, text)
        if matched_values:
            score += 15 * len(matched_values)
            reasons.append("sample value match: " + ", ".join(matched_values[:4]))
        role_score = _semantic_role_score(entry, text)
        if role_score:
            score += role_score
            reasons.append("semantic role match")
        # Lineage followup: strong signal when the question is clearly a follow-up
        if lineage_id == entry.dataset_id and _looks_like_followup(text):
            score += 50
            reasons.append("active branch/artifact lineage (follow-up)")
        if entry.executable_available:
            score += 1
        explicit_scores[entry.dataset_id] = score
        reasons_by_dataset[entry.dataset_id] = reasons

    best = max(explicit_scores.values())
    if best <= 1:
        return _ambiguous(registry, explicit_scores, ["No dataset has a decisive schema, value, or lineage match."])
    winners = [dataset_id for dataset_id, score in explicit_scores.items() if score == best]
    if len(winners) == 1:
        reasons = reasons_by_dataset.get(winners[0]) or ["Highest deterministic dataset score."]
        confidence = 0.85 if best >= 20 else 0.65
        return DatasetResolutionResult(DatasetScope.SINGLE, winners, confidence, reasons, candidate_scores=explicit_scores)

    # Equal strong column matches are unsafe unless lineage explicitly breaks the tie.
    if lineage_id in winners and _looks_like_followup(text):
        return DatasetResolutionResult(
            DatasetScope.SINGLE,
            [lineage_id],
            0.78,
            ["Follow-up uses active branch/artifact dataset lineage."],
            candidate_scores=explicit_scores,
        )
    return _ambiguous(registry, explicit_scores, ["Multiple datasets match equally."])


def resolve_explicit_metric_dataset(
    question: str,
    registry: list[InvestigationDatasetEntry],
) -> tuple[str, str] | None:
    """Search ALL loaded datasets for an explicitly mentioned metric column.

    Returns ``(dataset_id, column_name)`` if exactly one dataset contains
    a metric column that the user explicitly named in *question*, or
    ``None`` if zero or multiple datasets match.
    """
    text = _norm(question)
    candidates: list[tuple[str, str]] = []
    for entry in registry:
        for column in entry.column_names:
            if entry.semantic_roles.get(column) != "metric":
                continue
            if _column_mentioned(column, text):
                candidates.append((entry.dataset_id, column))
    # Unique by dataset_id — if same dataset matches twice, still one winner
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for dataset_id, column in candidates:
        if dataset_id not in seen:
            unique.append((dataset_id, column))
            seen.add(dataset_id)
    if len(unique) == 1:
        return unique[0]
    return None


def clarification_output(question: str, result: DatasetResolutionResult) -> dict[str, Any]:
    names = ", ".join(option["label"] for option in result.clarification_options) or "the attached datasets"
    summary = f"Which dataset should I use? Options: {names}."
    return {
        "query": question,
        "summary": summary,
        "final_answer": summary,
        "key_findings": [],
        "evidence": [],
        "limitations": ["Dataset resolution is ambiguous; no row-level analysis was executed."],
        "next_steps": ["Choose one dataset, or ask for a cross-dataset comparison."],
        "artifacts": [],
        "trace_metadata": {
            "dataset_resolution": result.to_dict(),
            "dataset_scope": DatasetScope.AMBIGUOUS.value,
            "suppress_key_findings": True,
            "no_execution": True,
        },
        "tool_timeline": [{"tool": "dataset_resolver", "status": "needs_clarification", "metadata": result.to_dict()}],
    }

def _finding_signature(text: str) -> str:
    """Normalize a finding to a comparable semantic signature."""
    import re
    cleaned = re.sub(r"[^\w\s]", "", _norm(text))
    tokens = cleaned.split()
    stop = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "of",
            "in", "by", "for", "with", "not", "no", "it", "that", "this", "these",
            "can", "be", "has", "have", "but", "yet", "from", "on", "at"}
    return " ".join(t for t in tokens if t not in stop)


def _finding_overlap(sig_a: str, sig_b: str) -> float:
    """Jaccard overlap between two finding signatures."""
    tokens_a = set(sig_a.split())
    tokens_b = set(sig_b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / max(union, 1)


def _deduplicate_structural_findings(
    findings: list[str],
    prior_findings: list[str],
    *,
    threshold: float = 0.7,
) -> list[str]:
    """Remove findings that are near-duplicates of prior investigation findings."""
    if not prior_findings:
        return findings
    prior_sigs = [_finding_signature(f) for f in prior_findings]
    deduplicated: list[str] = []
    for finding in findings:
        sig = _finding_signature(finding)
        if any(_finding_overlap(sig, prior) >= threshold for prior in prior_sigs):
            continue
        deduplicated.append(finding)
    return deduplicated


def cross_dataset_output(
    *,
    question: str,
    registry: list[InvestigationDatasetEntry],
    frames: dict[str, pd.DataFrame],
    result: DatasetResolutionResult,
    prior_findings: list[str] | None = None,
) -> dict[str, Any]:
    text = _norm(question)
    selected_entries = [entry for entry in registry if entry.dataset_id in result.selected_dataset_ids]
    explicitly_named = [entry for entry in selected_entries if _dataset_entry_mentioned(entry, text)]
    if len(explicitly_named) >= 2:
        selected_entries = explicitly_named
    effective_dataset_ids = [entry.dataset_id for entry in selected_entries]
    relationship = analyze_dataset_relationships(selected_entries)
    synthesis_plan = plan_multi_dataset_synthesis(question, [entry.display_name for entry in selected_entries])
    operation = _cross_dataset_operation(text, synthesis_plan.operation)
    include_relationship_context = operation in JOINABILITY_OPERATIONS
    branch_scopes = _build_dataset_execution_scopes(
        question=question,
        operation=operation,
        entries=selected_entries,
        frames=frames,
    )
    rows: list[dict[str, Any]] = []
    for entry in selected_entries:
        df = frames.get(entry.dataset_id)
        rows.append(
            {
                "dataset_id": entry.dataset_id,
                "dataset": entry.display_name,
                "semantic_roles": ", ".join(entry.semantic_profile.get("roles", [])[:6]),
                "domain_concepts": ", ".join(entry.semantic_profile.get("concepts", [])[:8]),
                "rows": int(len(df)) if isinstance(df, pd.DataFrame) else entry.row_count,
                "columns": int(len(getattr(df, "columns", []))) if isinstance(df, pd.DataFrame) else len(entry.column_names),
                "column_names": ", ".join(entry.column_names[:20]),
            }
        )
    shared = sorted(set.intersection(*(set(entry.column_names) for entry in selected_entries))) if len(selected_entries) >= 2 else []
    artifacts: list[dict[str, Any]] = [
        {
            "artifact_type": "table",
            "title": "Dataset relationship summary",
            "content": rows,
            "visibility": "user",
            "pinned": True,
            "metadata": {"dataset_scope": DatasetScope.CROSS.value, "dataset_ids": effective_dataset_ids, "analysis_type": "cross_dataset_semantic_summary"},
        }
    ]
    if include_relationship_context:
        artifacts.extend([
            {
                "artifact_type": "table",
                "title": "Conceptual overlap assessment",
                "content": relationship["concept_rows"],
                "visibility": "user",
                "pinned": True,
                "metadata": {"dataset_scope": DatasetScope.CROSS.value, "dataset_ids": result.selected_dataset_ids, "analysis_type": "cross_dataset_concept_overlap"},
            },
            _compatibility_heatmap(selected_entries, relationship),
            _entity_coverage_heatmap(selected_entries),
        ])
        if len(selected_entries) >= 2:
            bar = _joinability_bar_chart(selected_entries, relationship)
            if bar:
                artifacts.append(bar)
    elif operation == "CROSS_DATASET_COMPARISON":
        artifacts.append(_entity_coverage_heatmap(selected_entries))
    for scope in branch_scopes:
        artifacts.extend(scope.artifacts)

    branch_dicts = [scope.to_dict() for scope in branch_scopes]
    fallback_synthesis = synthesize_cross_dataset_branches(
        question=question,
        operation=operation,
        branches=branch_dicts,
    )
    comparative_packages = []
    if operation in COMPARATIVE_REASONING_OPERATIONS:
        comparative_packages = normalize_comparative_evidence(
            question=question,
            operation=operation,
            branches=branch_dicts,
        )
        synthesis = synthesize_comparative_evidence(
            question=question,
            operation=operation,
            packages=comparative_packages,
        )
    else:
        synthesis = fallback_synthesis
    evidence_first_operations = {
        "CROSS_DATASET_DISTRIBUTION_COMPARISON",
        "CROSS_DATASET_ANOMALY_COMPARISON",
        "CROSS_DATASET_IMBALANCE_COMPARISON",
        "CROSS_DATASET_CONCENTRATION_COMPARISON",
        "CROSS_DATASET_PROFITABILITY_COMPARISON",
        "CROSS_DATASET_RISK_PERFORMANCE_COMPARISON",
        "CROSS_DATASET_OPERATIONAL_OPTIMIZATION",
        "CROSS_DATASET_EXECUTIVE_SYNTHESIS",
        "CROSS_DATASET_TEMPORAL_ANALYSIS",
        "CROSS_DATASET_DIVERSITY_ANALYSIS",
        "CROSS_DATASET_RELATIONSHIP_COMPARISON",
        "CROSS_DATASET_PREDICTABILITY_COMPARISON",
        "CROSS_DATASET_KPI_SYNTHESIS",
        "PARALLEL_VISUAL_ANALYSIS",
    }
    intent_answer = _intent_specific_cross_dataset_answer(text, selected_entries, relationship, shared) if include_relationship_context else _non_join_cross_dataset_answer(text, selected_entries, relationship, shared)
    if operation in evidence_first_operations:
        intent_answer = ""
    findings = [intent_answer or synthesis["answer"]]
    if include_relationship_context:
        visual_findings = _findings_from_visuals(selected_entries, relationship)
        if shared and "column" in text:
            findings.append(f"Exact shared columns: {', '.join(shared[:12])}.")
        findings.extend(relationship["findings"])
        findings.extend(visual_findings)

    # Structured cross-dataset findings from relationship analysis
    rel_type = relationship.get("relationship_type", "")
    join_keys = relationship.get("joinable_shared_columns") or []
    if include_relationship_context and not join_keys and rel_type != "directly_joinable":
        findings.append("These datasets are not reliably joinable — no stable shared identifier was detected.")
    if include_relationship_context and relationship.get("concept_matches") and not join_keys:
        findings.append("The datasets support executive-level comparison through conceptual overlap, but not reliable row-level joins.")
    if include_relationship_context and rel_type == "unrelated_or_incompatible":
        findings.append("These datasets appear analytically distinct — comparison should focus on purpose differences rather than forced mappings.")
    findings.extend(synthesis.get("findings") or [])
    for scope in branch_scopes:
        if include_relationship_context:
            findings.extend(scope.findings[:2])

    metric_result = None if operation in COMPARATIVE_REASONING_OPERATIONS else _cross_metric_rows(text, registry, frames)
    if metric_result:
        artifacts.append(
            {
                "artifact_type": "table",
                "title": "Cross-dataset metric comparison",
                "content": metric_result["rows"],
                "visibility": "user",
                "pinned": True,
            "metadata": {"dataset_scope": DatasetScope.CROSS.value, "dataset_ids": effective_dataset_ids, "metric": metric_result["metric"], "analysis_type": "cross_dataset_metric"},
            }
        )
        findings.append(metric_result["summary"])

    # Deduplicate against prior investigation findings to prevent repetition
    findings = _deduplicate_structural_findings(findings, prior_findings or [])
    if not findings:
        # All findings were duplicates of prior runs — use the intent-specific answer
        # so the response is still meaningful, not a middleware placeholder
        findings = [intent_answer or relationship["summary"]]

    summary = synthesis["answer"] if operation in COMPARATIVE_REASONING_OPERATIONS else (metric_result["summary"] if metric_result and _asks_metric_comparison_or_summary(text) else (intent_answer or synthesis["answer"]))
    critic_warnings = critic_warnings_for_cross_dataset_output(
        question=question,
        operation=operation,
        answer=summary,
        branches=branch_dicts,
        artifact_count=len(artifacts),
    )
    critic_warnings.extend(
        critic_warnings_for_comparative_output(
            question=question,
            operation=operation,
            answer=summary,
            packages=comparative_packages,
            artifact_count=len(artifacts),
        )
    )
    col_counts = [len(e.column_names) for e in selected_entries]
    return {
        "query": question,
        "summary": summary,
        "final_answer": summary,
        "key_findings": findings,
        "evidence": [
            f"Loaded executable runtime for {len(frames)} selected datasets.",
            f"Schema comparison across {len(selected_entries)} datasets ({', '.join(str(c) for c in col_counts)} columns).",
            f"Executed {len(branch_scopes)} isolated dataset branch(es) for operation `{operation}`.",
        ] + ([f"Relationship classified as {relationship['relationship_type']}."] if include_relationship_context else []),
        "limitations": ([] if not include_relationship_context else relationship["limitations"])
        + [item for scope in branch_scopes for item in scope.limitations[:2]]
        + list(synthesis.get("limitations") or [])
        + critic_warnings
        + ([] if len(frames) == len(result.selected_dataset_ids) else ["One or more selected datasets could not be loaded."]),
        "next_steps": synthesis.get("next_steps") or relationship["next_steps"],
        "artifacts": artifacts,
        "trace_metadata": {
            "dataset_resolution": result.to_dict(),
            "dataset_scope": DatasetScope.CROSS.value,
            "dataset_ids": effective_dataset_ids,
            "requested_dataset_ids": result.selected_dataset_ids,
            "dataset_relationships": relationship,
            "analysis_type": "cross_dataset",
            "operation": operation,
            "multi_dataset_synthesis_plan": synthesis_plan.to_dict(),
            "dataset_execution_scopes": branch_dicts,
            "branch_count": len(branch_scopes),
            "cross_dataset_synthesis": synthesis,
            "fallback_cross_dataset_synthesis": fallback_synthesis,
            "comparative_evidence_packages": [package.to_dict() for package in comparative_packages],
            "critic_warnings": critic_warnings,
            "relationship_context_included": include_relationship_context,
        },
        "tool_timeline": [
            {"tool": "cross_dataset_semantic_reasoning", "status": "ok", "metadata": {"dataset_resolution": result.to_dict(), "relationship_type": relationship["relationship_type"]}},
            {"tool": "multi_branch_dataset_execution", "status": "ok", "metadata": {"operation": operation, "branch_count": len(branch_scopes), "dataset_ids": [scope.dataset_id for scope in branch_scopes]}},
        ],
    }


def registry_prompt(registry: list[InvestigationDatasetEntry]) -> str:
    lines = ["Attached dataset registry:"]
    for entry in registry:
        desc = _concise_role_description(entry)
        metric_cols = entry.semantic_profile.get("metric_columns", [])[:6]
        dim_cols = entry.semantic_profile.get("dimension_columns", [])[:6]
        parts = [f"- `{entry.display_name}` ({entry.row_count} rows): {desc}"]
        if metric_cols:
            parts.append(f"  Metrics: {', '.join(f'`{c}`' for c in metric_cols)}")
        if dim_cols:
            parts.append(f"  Dimensions: {', '.join(f'`{c}`' for c in dim_cols)}")
        lines.extend(parts)
    return "\n".join(lines)


def _dataset_entry_mentioned(entry: InvestigationDatasetEntry, text: str) -> bool:
    names = {_norm(entry.display_name), _norm(entry.source_name), _norm(entry.dataset_id)}
    if any(name and name in text for name in names):
        return True
    tokens = set(text.split())
    name_tokens = set(_norm(entry.display_name).split()) | set(_norm(entry.source_name).split())
    distinctive = {token for token in name_tokens if len(token) >= 4}
    return bool(distinctive & tokens)


def analyze_dataset_relationships(entries: list[InvestigationDatasetEntry]) -> dict[str, Any]:
    if len(entries) < 2:
        return {
            "relationship_type": "single_dataset",
            "summary": "Only one dataset is selected, so no cross-dataset relationship is needed.",
            "findings": [],
            "limitations": [],
            "next_steps": [],
            "concept_rows": [],
            "concept_matches": [],
        }
    exact_shared = sorted(set.intersection(*(set(entry.column_names) for entry in entries)))
    concept_matches = _concept_matches(entries)
    all_roles = [set(entry.semantic_profile.get("roles", [])) for entry in entries]
    role_overlap = sorted(set.intersection(*all_roles)) if all_roles else []
    all_concepts = [set(entry.semantic_profile.get("concepts", [])) for entry in entries]
    concept_overlap = sorted(set.intersection(*all_concepts)) if all_concepts else []
    join_keys = _joinable_shared_columns(entries, exact_shared)
    relationship_type = _relationship_type(exact_shared, concept_matches, role_overlap, concept_overlap, join_keys)
    dataset_bits = [
        f"`{entry.display_name}` {_concise_role_description(entry)}"
        for entry in entries
    ]
    if relationship_type == "directly_joinable":
        compatibility = f"They may be directly joinable through shared identifier field(s): {', '.join(join_keys[:6])}."
    elif concept_matches:
        compatibility = "They are not directly joinable from the available schema, but they have a small number of validated conceptual overlaps that can support summary-level comparison."
    elif concept_overlap or role_overlap:
        compatibility = "They are related at the analytical-domain level, but no reliable field bridge was found for direct comparison."
    else:
        compatibility = "They appear analytically distinct, so comparison should focus on purpose and limitations rather than forced column mappings."
    findings = []
    if concept_matches:
        top = concept_matches[:3]
        bridge_desc = []
        for item in top:
            concept_label = item["concept"].replace("_", " ")
            bridge_desc.append(f"{concept_label} (`{item['left_column']}` ↔ `{item['right_column']}`")
        strongest = bridge_desc[0] if bridge_desc else ""
        if len(bridge_desc) == 1:
            findings.append(f"The strongest cross-dataset bridge is {strongest}), supporting summary-level comparison.")
        else:
            others = ", ".join(b + ")" for b in bridge_desc[1:])
            findings.append(f"The strongest cross-dataset bridge is {strongest}). Additional bridges: {others}.")
    elif concept_overlap or role_overlap:
        findings.append("No reliable semantic bridge was found; any comparison should stay at dataset-purpose or aggregate-summary level.")
    limitations = []
    if not join_keys:
        limitations.append("No reliable shared column was detected, so joins or row-level linkage should not be assumed.")
    if concept_matches:
        limitations.append("Concept matches are approximate semantic bridges, not proof that fields are equivalent.")
    next_steps = [
        "Choose a conceptual bridge to validate with distributions or summary statistics.",
        "Provide or identify a shared key if row-level joining is required.",
    ]
    return {
        "relationship_type": relationship_type,
        "summary": " ".join([*dataset_bits, compatibility]),
        "findings": findings,
        "limitations": limitations,
        "next_steps": next_steps,
        "concept_rows": _concept_rows(entries, exact_shared, concept_matches, role_overlap, concept_overlap, relationship_type),
        "concept_matches": concept_matches,
        "exact_shared_columns": exact_shared,
        "joinable_shared_columns": join_keys,
        "shared_roles": role_overlap,
        "shared_concepts": concept_overlap,
    }


def _ambiguous(registry: list[InvestigationDatasetEntry], scores: dict[str, int], reasons: list[str]) -> DatasetResolutionResult:
    return DatasetResolutionResult(
        DatasetScope.AMBIGUOUS,
        [],
        0.0,
        reasons,
        ["Choose a dataset before execution."],
        candidate_scores=scores,
        clarification_options=[{"dataset_id": entry.dataset_id, "label": entry.display_name} for entry in registry],
    )


def _profile_or_none(store: Any, dataset_id: str) -> DataSourceProfile | None:
    try:
        return store.get_data_source_profile(dataset_id)
    except Exception:
        return None


def _compact_profile(profile: DataSourceProfile | None) -> dict[str, Any]:
    if profile is None:
        return {}
    return {
        "row_count": profile.row_count,
        "column_count": profile.column_count,
        "missing_summary": dict(profile.missing_summary),
        "numeric_summary": dict(profile.numeric_summary),
        "categorical_summary": dict(profile.categorical_summary),
        "sampled_rows": list(profile.sampled_rows[:5]),
    }


CONCEPT_MARKERS: dict[str, tuple[str, ...]] = {
    "monetary_value": ("sales", "revenue", "amount", "price", "cost", "profit", "income", "spend", "mnt", "payment", "value", "salary", "pay", "wage"),
    "customer_entity": ("customer", "client", "buyer", "user", "member", "account", "person", "household"),
    "purchase_volume": ("quantity", "qty", "count", "orders", "invoice", "transactions", "purchases", "frequency", "num"),
    "product_or_category": ("product", "item", "sku", "category", "catalog", "line", "brand"),
    "campaign_response": ("campaign", "response", "accepted", "offer", "promotion", "marketing"),
    "age_or_birth": ("age", "birth", "year birth", "yearbirth"),
    "education_level": ("education", "degree", "school", "university"),
    "marital_status": ("marital", "marriage", "spouse"),
    "gender": ("gender", "sex"),
    "household_structure": ("household", "kid", "teen", "children", "family"),
    "geography": ("city", "region", "country", "state", "location", "postal", "zip", "territory"),
    "time": ("date", "time", "year", "month", "day", "dt", "created"),
    "operational_process": ("ship", "delivery", "delay", "status", "process", "ticket", "order"),
    "inventory": ("stock", "inventory", "warehouse", "sku", "supply"),
    "survey_or_rating": ("rating", "score", "survey", "satisfaction", "response"),
}

CONCEPT_COMPATIBILITY_GROUPS: dict[str, str] = {
    "monetary_value": "monetary",
    "customer_entity": "entity",
    "purchase_volume": "volume",
    "product_or_category": "category",
    "campaign_response": "campaign",
    "age_or_birth": "temporal_demographic",
    "education_level": "education",
    "marital_status": "marital",
    "gender": "gender",
    "household_structure": "household",
    "geography": "geography",
    "time": "time",
    "operational_process": "process",
    "inventory": "inventory",
    "survey_or_rating": "rating",
}

MIN_CONCEPT_BRIDGE_CONFIDENCE = 0.72

ROLE_CONCEPTS: dict[str, tuple[str, ...]] = {
    "transactional": ("invoice", "transaction", "order", "quantity", "qty", "price", "amount", "tax", "line", "purchase", "sales", "revenue"),
    "customer_behavioral": ("customer", "client", "spend", "purchase", "response", "campaign", "loyalty", "frequency"),
    "demographic": ("birth", "age", "education", "marital", "income", "gender", "household", "kid", "teen"),
    "financial": ("revenue", "sales", "profit", "income", "salary", "amount", "cost", "price", "tax", "payment"),
    "operational": ("ship", "delivery", "delay", "status", "process", "ticket", "queue", "service"),
    "inventory": ("inventory", "stock", "warehouse", "sku", "supply"),
    "temporal": ("date", "time", "year", "month", "day", "dt"),
    "ecommerce": ("cart", "order", "product", "customer", "invoice", "sku", "online"),
    "campaign": ("campaign", "response", "accepted", "offer", "promotion", "marketing"),
    "event": ("event", "session", "click", "visit", "timestamp"),
    "geographic": ("city", "region", "country", "state", "location", "postal", "zip"),
    "kpi_summary": ("total", "average", "rate", "kpi", "summary", "score"),
    "aggregated": ("total", "average", "mean", "count", "group", "summary"),
    "survey": ("survey", "rating", "score", "satisfaction", "response"),
    "time_series": ("date", "time", "year", "month", "period"),
}


def _metric_subtype(column_name: str) -> str:
    """Classify a metric column into a semantic subtype.

    Categories (from most specific to least):
      DIRECT_REVENUE       — transaction value, revenue, amount, profit
      PRICE_COMPONENT      — unit price, cost, price
      BEHAVIORAL_SPENDING  — spending aggregates and prefixed spend fields
      FINANCIAL_PROFILE    — personal income or wage attributes
      COUNT_VOLUME         — quantity, purchase count, orders, frequency
      ENGAGEMENT           — recency, score, response, campaign acceptance
      GENERIC_METRIC       — fallback for any other numeric
    """
    tokens = set(_norm(column_name).split())
    name_lower = _norm(column_name)
    # Direct transactional revenue: requires explicit transaction-oriented words
    if tokens & {"sales", "revenue", "gross", "profit"}:
        return "DIRECT_REVENUE"
    # Amount is revenue ONLY if not preceded by behavioral markers
    if "amount" in tokens and not tokens & {"mnt", "spend", "campaign", "response", "recency"}:
        return "DIRECT_REVENUE"
    # Price components
    if tokens & {"price", "cost", "fee", "charge", "margin", "unit"}:
        return "PRICE_COMPONENT"
    # Behavioral spending aggregates (Mnt* pattern or explicit spend)
    if "mnt" in name_lower or tokens & {"spend", "spending"}:
        return "BEHAVIORAL_SPENDING"
    # Financial profile (personal attributes, not transactional)
    if tokens & {"income", "salary", "wage", "pay"}:
        return "FINANCIAL_PROFILE"
    # Count/volume
    if tokens & {"quantity", "qty", "count", "orders", "transactions", "purchases", "frequency", "num"}:
        return "COUNT_VOLUME"
    # Engagement / campaign metrics
    if tokens & {"recency", "score", "response", "accepted", "campaign", "complain", "rating", "satisfaction"}:
        return "ENGAGEMENT"
    return "GENERIC_METRIC"


def _dataset_primary_purpose(entry: InvestigationDatasetEntry) -> str:
    """Infer the primary purpose of a dataset using evidence-weighted column analysis.

    Returns one of: TRANSACTIONAL, DEMOGRAPHIC, BEHAVIORAL, CAMPAIGN,
    FINANCIAL, PRODUCT, GEOGRAPHIC, LOGISTICS, OPERATIONAL, EVENT, REFERENCE, MIXED, UNKNOWN.

    The algorithm counts evidence columns for each purpose category and picks
    the one with the highest ratio.  This prevents financial-like metrics from
    dominating a dataset that is primarily demographic or behavioral.
    """
    col_tokens_all = set(_norm(" ".join(entry.column_names)).split())
    total = max(len(entry.column_names), 1)

    # Evidence counters
    evidence: dict[str, int] = {
        "TRANSACTIONAL": 0,
        "DEMOGRAPHIC": 0,
        "BEHAVIORAL": 0,
        "CAMPAIGN": 0,
        "FINANCIAL": 0,
        "PRODUCT": 0,
        "GEOGRAPHIC": 0,
        "LOGISTICS": 0,
        "OPERATIONAL": 0,
        "EVENT": 0,
    }

    # Per-transaction identifiers strongly signal transactional
    _TX_ID_MARKERS = {"invoice", "order", "transaction", "receipt", "ticket"}
    _DEMO_MARKERS = {"birth", "age", "education", "marital", "gender", "household", "kid", "teen", "children", "family"}
    _CAMPAIGN_MARKERS = {"campaign", "response", "accepted", "offer", "promotion", "cmp"}
    _PRODUCT_MARKERS = {"product", "item", "sku", "category", "catalog", "brand"}
    _GEO_MARKERS = {"city", "region", "country", "state", "location", "postal", "zip", "territory", "warehouse"}
    _LOGISTICS_MARKERS = {"ship", "delivery", "delay", "route", "carrier", "freight", "warehouse", "dispatch"}
    _EVENT_MARKERS = {"event", "session", "click", "visit", "login", "signup"}

    for col in entry.column_names:
        col_lower = _norm(col)
        ctokens = set(col_lower.split())
        subtype = _metric_subtype(col)

        # Transactional evidence: per-transaction IDs or direct revenue fields
        if ctokens & _TX_ID_MARKERS:
            evidence["TRANSACTIONAL"] += 3  # Strong signal
        if subtype == "DIRECT_REVENUE":
            evidence["TRANSACTIONAL"] += 2
        if subtype == "PRICE_COMPONENT":
            evidence["TRANSACTIONAL"] += 1
        if subtype == "COUNT_VOLUME":
            evidence["TRANSACTIONAL"] += 1

        # Demographic evidence
        if ctokens & _DEMO_MARKERS:
            evidence["DEMOGRAPHIC"] += 2
        if subtype == "FINANCIAL_PROFILE":
            evidence["DEMOGRAPHIC"] += 2  # Income/Salary are demographic attributes

        # Behavioral evidence
        if subtype == "BEHAVIORAL_SPENDING":
            evidence["BEHAVIORAL"] += 2
        if subtype == "ENGAGEMENT":
            evidence["BEHAVIORAL"] += 1

        # Campaign evidence
        if ctokens & _CAMPAIGN_MARKERS:
            evidence["CAMPAIGN"] += 2

        # Product evidence
        if ctokens & _PRODUCT_MARKERS:
            evidence["PRODUCT"] += 1

        # Geographic evidence
        if ctokens & _GEO_MARKERS:
            evidence["GEOGRAPHIC"] += 1

        # Logistics evidence
        if ctokens & _LOGISTICS_MARKERS:
            evidence["LOGISTICS"] += 2

        # Event evidence
        if ctokens & _EVENT_MARKERS:
            evidence["EVENT"] += 2

        # Operational (catch-all for process-oriented columns)
        if ctokens & {"status", "process", "queue", "service", "ticket"}:
            evidence["OPERATIONAL"] += 1

    # Find the primary purpose
    sorted_evidence = sorted(evidence.items(), key=lambda x: -x[1])
    best_purpose, best_score = sorted_evidence[0]
    second_purpose, second_score = sorted_evidence[1] if len(sorted_evidence) > 1 else ("", 0)

    if best_score == 0:
        return "UNKNOWN"
    # If two categories are nearly tied, call it MIXED
    if second_score > 0 and second_score >= best_score * 0.7:
        # But demographic+behavioral is common and should be "DEMOGRAPHIC"
        if {best_purpose, second_purpose} <= {"DEMOGRAPHIC", "BEHAVIORAL", "CAMPAIGN"}:
            # Pick the stronger one
            return best_purpose
        return "MIXED"
    return best_purpose


def _dataset_semantic_profile(*, column_names: list[str], semantic_roles: dict[str, str], profile: dict[str, Any], display_name: str) -> dict[str, Any]:
    tokens = set(_norm(" ".join([display_name, *column_names])).split())
    role_scores: dict[str, int] = {}
    for role, markers in ROLE_CONCEPTS.items():
        role_scores[role] = len(tokens & set(markers))
    if any(value == "timestamp" for value in semantic_roles.values()):
        role_scores["temporal"] = role_scores.get("temporal", 0) + 2
        role_scores["time_series"] = role_scores.get("time_series", 0) + 1
    if len([value for value in semantic_roles.values() if value == "metric"]) >= 3:
        role_scores["financial"] = role_scores.get("financial", 0) + 1
        role_scores["kpi_summary"] = role_scores.get("kpi_summary", 0) + 1
    roles = [role for role, score in sorted(role_scores.items(), key=lambda item: (-item[1], item[0])) if score > 0]
    concepts = _entry_concepts_from_columns(column_names)
    # Compute metric subtypes for all metric columns
    metric_subtypes: dict[str, str] = {}
    for column, role in semantic_roles.items():
        if role == "metric":
            metric_subtypes[column] = _metric_subtype(column)
    return {
        "primary_role": roles[0] if roles else "structured",
        "roles": roles,
        "concepts": concepts,
        "metric_columns": [column for column, role in semantic_roles.items() if role == "metric"],
        "dimension_columns": [column for column, role in semantic_roles.items() if role == "dimension"],
        "timestamp_columns": [column for column, role in semantic_roles.items() if role == "timestamp"],
        "metric_subtypes": metric_subtypes,
        "profile_shape": {"rows": profile.get("row_count", 0), "columns": profile.get("column_count", len(column_names))},
    }


def _semantic_role(name: str, dtype: str) -> str:
    normalized = _norm(name)
    tokens = set(normalized.split())
    _TIMESTAMP_MARKERS = {"date", "time", "month", "year", "dt"}
    _IDENTIFIER_MARKERS = {"id", "uuid", "key"}
    _METRIC_MARKERS = {"sales", "revenue", "amount", "salary", "price", "cost", "profit", "count", "qty", "quantity"}
    if tokens & {"age", "birth"}:
        return "metric" if any(marker in str(dtype).lower() for marker in ("int", "float", "decimal", "number")) else "dimension"
    if tokens & _TIMESTAMP_MARKERS:
        return "timestamp"
    if tokens & _IDENTIFIER_MARKERS:
        return "identifier"
    if tokens & _METRIC_MARKERS:
        return "metric"
    if any(marker in str(dtype).lower() for marker in ("int", "float", "decimal", "number")):
        return "metric"
    return "dimension"


def _entry_concepts_from_columns(columns: list[str]) -> list[str]:
    concepts: list[str] = []
    for column in columns:
        tokens = set(_norm(column).split())
        for concept, markers in CONCEPT_MARKERS.items():
            if tokens & set(markers):
                concepts.append(concept)
    return list(dict.fromkeys(concepts))


def _norm(value: Any) -> str:
    import re
    text = str(value or "")
    # Split camelCase: 'InvoiceNo' → 'Invoice No', 'TaxAmount' → 'Tax Amount'
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = text.casefold().replace("_", " ")
    cleaned = [char if char.isalnum() or char.isspace() else " " for char in text]
    return " ".join("".join(cleaned).split())


def _column_mentioned(column: str, text: str) -> bool:
    normalized = _norm(column)
    return bool(normalized and (normalized in text or normalized.replace(" ", "") in text.replace(" ", "")))


def _matched_values(entry: InvestigationDatasetEntry, text: str) -> list[str]:
    matched: list[str] = []
    for values in entry.sample_values.values():
        for value in values[:8]:
            normalized = _norm(value)
            if len(normalized) >= 3 and normalized in text:
                matched.append(str(value))
    return list(dict.fromkeys(matched))


def _semantic_role_score(entry: InvestigationDatasetEntry, text: str) -> int:
    score = 0
    roles = set(entry.semantic_roles.values())
    text_tokens = set(text.split())
    if "metric" in roles and text_tokens & {"top", "highest", "lowest", "average", "mean", "sum", "total", "revenue", "sales", "salary", "amount"}:
        score += 3
    if "timestamp" in roles and (text_tokens & {"trend", "time", "month", "year"} or "over time" in text):
        score += 3
    return score


def _concept_matches(entries: list[InvestigationDatasetEntry]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for left_index, left in enumerate(entries):
        for right in entries[left_index + 1:]:
            for left_column in left.column_names:
                left_concepts = _column_concepts(left_column)
                if not left_concepts:
                    continue
                for right_column in right.column_names:
                    right_concepts = _column_concepts(right_column)
                    shared = left_concepts & right_concepts
                    if not shared:
                        continue
                    for concept in sorted(shared):
                        if not _concept_pair_is_valid(left_column, right_column, concept, left, right):
                            continue
                        confidence = _concept_confidence(left_column, right_column, concept, left, right)
                        if confidence < MIN_CONCEPT_BRIDGE_CONFIDENCE:
                            continue
                        matches.append(
                            {
                                "left_dataset_id": left.dataset_id,
                                "left_dataset": left.display_name,
                                "left_column": left_column,
                                "right_dataset_id": right.dataset_id,
                                "right_dataset": right.display_name,
                                "right_column": right_column,
                                "concept": concept,
                                "confidence": confidence,
                                "interpretation": "possible semantic overlap; validate before treating as equivalent",
                            }
                        )
    matches.sort(key=lambda item: (-float(item["confidence"]), str(item["concept"]), str(item["left_column"])))
    return matches


def _column_concepts(column: str) -> set[str]:
    tokens = set(_norm(column).split())
    return {concept for concept, markers in CONCEPT_MARKERS.items() if tokens & set(markers)}


def _concept_confidence(left_column: str, right_column: str, concept: str, left: InvestigationDatasetEntry, right: InvestigationDatasetEntry) -> float:
    score = 0.52
    if _norm(left_column) == _norm(right_column):
        score += 0.28
    if concept in set(left.semantic_profile.get("concepts", [])) and concept in set(right.semantic_profile.get("concepts", [])):
        score += 0.12
    if left.semantic_roles.get(left_column) == right.semantic_roles.get(right_column):
        score += 0.08
    if _concept_family(concept) in {"monetary", "geography", "time", "volume"}:
        score += 0.04
    return min(score, 0.95)


def _concept_pair_is_valid(left_column: str, right_column: str, concept: str, left: InvestigationDatasetEntry, right: InvestigationDatasetEntry) -> bool:
    left_role = left.semantic_roles.get(left_column)
    right_role = right.semantic_roles.get(right_column)
    if left_role and right_role and left_role != right_role:
        if {left_role, right_role} == {"metric", "dimension"}:
            return False
        if "timestamp" in {left_role, right_role} and concept != "time":
            return False
    normalized = {_norm(left_column), _norm(right_column)}
    forbidden_pairs = [
        ("gender", "education"),
        ("gender", "marital"),
        ("dt customer", "customer type"),
        ("date customer", "customer type"),
        ("timestamp", "customer type"),
    ]
    for left_marker, right_marker in forbidden_pairs:
        if any(left_marker in value for value in normalized) and any(right_marker in value for value in normalized):
            return False
    if concept in {"gender", "education_level", "marital_status"} and _norm(left_column) != _norm(right_column):
        return False
    if concept == "monetary_value" and not _monetary_subtypes_compatible(left_column, right_column):
        return False
    return True


def _monetary_subtypes_compatible(left_column: str, right_column: str) -> bool:
    left = _monetary_subtype(left_column)
    right = _monetary_subtype(right_column)
    if not left or not right:
        return True
    if left == right:
        return True
    compatible = {
        frozenset({"revenue", "spend"}),
        frozenset({"revenue", "income"}),
        frozenset({"income", "spend"}),
        frozenset({"revenue", "price"}),
        frozenset({"spend", "price"}),
        frozenset({"salary", "income"}),
    }
    return frozenset({left, right}) in compatible


def _monetary_subtype(column: str) -> str:
    tokens = set(_norm(column).split())
    if tokens & {"tax", "fee", "vat"}:
        return "tax"
    if tokens & {"sales", "revenue", "amount"}:
        return "revenue"
    if tokens & {"spend", "mnt", "purchase"}:
        return "spend"
    if tokens & {"income", "salary", "wage", "pay"}:
        return "salary" if "salary" in tokens else "income"
    if tokens & {"price", "cost"}:
        return "price"
    if "profit" in tokens:
        return "profit"
    return ""


def _concept_family(concept: str) -> str:
    return CONCEPT_COMPATIBILITY_GROUPS.get(concept, concept)


def _joinable_shared_columns(entries: list[InvestigationDatasetEntry], exact_shared: list[str]) -> list[str]:
    joinable = []
    for column in exact_shared:
        roles = {entry.semantic_roles.get(column) for entry in entries}
        normalized = _norm(column)
        if "identifier" in roles or any(marker in normalized for marker in ("id", "uuid", "key", "customer id", "account id")):
            joinable.append(column)
    return joinable


def _relationship_type(exact_shared: list[str], concept_matches: list[dict[str, Any]], role_overlap: list[str], concept_overlap: list[str], join_keys: list[str]) -> str:
    if join_keys:
        return "directly_joinable"
    if concept_matches:
        return "indirectly_comparable"
    if concept_overlap:
        return "domain_related"
    if role_overlap:
        return "analytically_related"
    return "unrelated_or_incompatible"


def _concept_rows(
    entries: list[InvestigationDatasetEntry],
    exact_shared: list[str],
    concept_matches: list[dict[str, Any]],
    role_overlap: list[str],
    concept_overlap: list[str],
    relationship_type: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "assessment": "relationship_type",
            "value": relationship_type,
            "evidence": "Exact shared fields, semantic roles, and concept-level column matches were compared.",
        },
        {
            "assessment": "exact_shared_columns",
            "value": ", ".join(exact_shared) if exact_shared else "none reliable",
            "evidence": "Exact column names only; absence does not rule out conceptual comparison.",
        },
        {
            "assessment": "shared_semantic_roles",
            "value": ", ".join(role_overlap) if role_overlap else "none strong",
            "evidence": "Dataset-level role inference from schema/profile metadata.",
        },
        {
            "assessment": "shared_concepts",
            "value": ", ".join(concept_overlap) if concept_overlap else "none strong",
            "evidence": "Column concept markers mapped to generic analytical concepts.",
        },
    ]
    for match in concept_matches[:10]:
        rows.append(
            {
                "assessment": "possible_concept_bridge",
                "value": f"{match['left_dataset']}.{match['left_column']} ↔ {match['right_dataset']}.{match['right_column']}",
                "evidence": f"{match['concept']} confidence {match['confidence']:.2f}; {match['interpretation']}",
            }
        )
    for entry in entries:
        rows.append(
            {
                "assessment": f"dataset_role:{entry.display_name}",
                "value": entry.semantic_profile.get("primary_role", "structured"),
                "evidence": ", ".join(entry.semantic_profile.get("concepts", [])[:8]) or "No strong concepts inferred.",
            }
        )
    return rows


def _intent_specific_cross_dataset_answer(
    text: str,
    entries: list[InvestigationDatasetEntry],
    relationship: dict[str, Any],
    shared: list[str],
) -> str:
    intent = classify_question_intent(text)
    if intent == QuestionIntentType.MULTI_DATASET_WAREHOUSE_DESIGN:
        return _warehouse_design_answer(entries, relationship)
    if intent == QuestionIntentType.MULTI_DATASET_MISSING_LINKS:
        return _missing_links_answer(entries, relationship)
    if _asks_dataset_role(text):
        return _dataset_role_answer(entries)
    if _asks_important_insights(text) and not any(marker in text for marker in ("kpi", "dashboard", "dashboards")):
        return _strategic_insights_answer(entries, relationship)
    if _asks_temporal_comparison(text):
        return _temporal_compatibility_answer(entries)
    if any(marker in text for marker in ("join", "joined", "merge", "key", "соедин", "джойн", "объедин")):
        join_keys = relationship.get("joinable_shared_columns") or []
        if join_keys:
            return f"These datasets may be joinable, but only after validating key quality. Candidate key(s): {', '.join(join_keys[:6])}. Check uniqueness, missing values, and whether the key means the same entity in each dataset before joining records."
        if shared:
            return f"These datasets are not reliably joinable yet. They share {', '.join(shared[:6])}, but those fields are not stable identifiers; use them for grouped comparison, not row-level merges."
        return "These datasets are not reliably joinable yet. No stable shared customer, product, transaction, location, or date key was detected; add a shared identifier or mapping table before attempting row-level joins."
    if any(marker in text for marker in ("common pattern", "common patterns", "patterns across", "общие паттерн", "общие закономер")):
        if relationship.get("concept_matches"):
            return "The datasets can be compared at a business-pattern level, but only through validated aggregate concepts; they should not be joined row by row without a shared identifier."
        return "The datasets may still be compared at the level of purpose, granularity, and aggregate behavior, but no reliable semantic bridge was found and they should not be joined row by row."
    if any(marker in text for marker in ("column", "columns", "schema", "колон", "схем")):
        if shared:
            return f"The datasets share these exact columns: {', '.join(shared[:12])}. Exact column overlap alone does not prove that records can be joined."
        return _concrete_column_comparison(entries)
    return ""


def _non_join_cross_dataset_answer(
    text: str,
    entries: list[InvestigationDatasetEntry],
    relationship: dict[str, Any],
    shared: list[str],
) -> str:
    if _asks_dataset_role(text):
        return _dataset_role_answer(entries)
    if _asks_important_insights(text) and not any(marker in text for marker in ("kpi", "dashboard", "dashboards")):
        return _strategic_insights_answer(entries, relationship)
    if any(marker in text for marker in ("column", "columns", "schema", "field", "fields", "колон", "схем")) and "categorical fields" not in text:
        if shared and ("shared" in text or "same" in text):
            return f"The datasets share these exact columns: {', '.join(shared[:12])}. Exact column overlap alone does not prove that records can be connected."
        return _concrete_column_comparison(entries)
    return ""


def _concrete_column_comparison(entries: list[InvestigationDatasetEntry]) -> str:
    """Generate a concrete per-dataset field breakdown instead of abstract descriptions."""
    parts: list[str] = []
    for entry in entries:
        categories: dict[str, list[str]] = {
            "metrics": [],
            "dimensions": [],
            "entities": [],
            "temporal": [],
            "behavioral": [],
            "operational": [],
        }
        for col in entry.column_names:
            role = entry.semantic_roles.get(col, "")
            col_lower = _norm(col)
            col_concepts = _column_concepts(col)
            if role == "metric":
                if any(m in col_lower for m in ("mnt", "spend", "campaign", "accepted", "response")):
                    categories["behavioral"].append(col)
                else:
                    categories["metrics"].append(col)
            elif role == "timestamp" or "time" in col_concepts:
                categories["temporal"].append(col)
            elif role == "entity" or "customer_entity" in col_concepts:
                categories["entities"].append(col)
            elif role == "dimension" or "geography" in col_concepts:
                categories["dimensions"].append(col)
            elif "operational_process" in col_concepts:
                categories["operational"].append(col)
            else:
                categories["dimensions"].append(col)

        desc = _concise_role_description(entry)
        field_parts: list[str] = []
        for label, cols in categories.items():
            if cols:
                field_parts.append(f"{label}: `{'`, `'.join(cols[:5])}`")
        summary = "; ".join(field_parts) if field_parts else f"{len(entry.column_names)} columns"
        parts.append(f"`{entry.display_name}` {desc} ({summary})")
    result = ". ".join(parts) + "."
    # Add semantic synthesis
    all_concepts: set[str] = set()
    for entry in entries:
        all_concepts.update(entry.semantic_profile.get("concepts", []))
    shared_concepts = []
    for entry in entries:
        entry_concepts = set(entry.semantic_profile.get("concepts", []))
        if not shared_concepts:
            shared_concepts = list(entry_concepts)
        else:
            shared_concepts = [c for c in shared_concepts if c in entry_concepts]
    if shared_concepts:
        result += f" Shared semantic concepts: {', '.join(shared_concepts[:4])}."
    else:
        result += " The datasets do not share exact column names and have distinct semantic purposes."
    return result


def _asks_temporal_comparison(text: str) -> bool:
    return any(marker in text for marker in ("year", "years", "date", "time", "temporal", "года", "годы", "дат", "времен"))


def _temporal_compatibility_answer(entries: list[InvestigationDatasetEntry]) -> str:
    temporal_by_dataset = {
        entry.display_name: [
            column
            for column in entry.column_names
            if entry.semantic_roles.get(column) == "timestamp" or "time" in _column_concepts(column)
        ]
        for entry in entries
    }
    with_temporal = {name: columns for name, columns in temporal_by_dataset.items() if columns}
    without_temporal = [name for name, columns in temporal_by_dataset.items() if not columns]
    if len(with_temporal) == len(entries):
        parts = [f"`{name}` has {', '.join(columns[:4])}" for name, columns in with_temporal.items()]
        return "A temporal comparison is possible only after confirming that these fields represent comparable time concepts: " + "; ".join(parts) + "."
    if with_temporal and without_temporal:
        present = "; ".join(f"`{name}` contains {', '.join(columns[:4])}" for name, columns in with_temporal.items())
        missing = ", ".join(f"`{name}`" for name in without_temporal)
        return f"{present}, while {missing} does not contain a comparable year/date field. A direct year comparison is therefore not possible."
    return "No comparable year/date field was detected in the selected datasets, so a direct temporal comparison is not possible."


def _asks_dataset_role(text: str) -> bool:
    return any(marker in text for marker in (
        "which dataset contains", "which dataset tracks", "which dataset has",
        "which dataset is", "what does each dataset", "role of each dataset",
        "what type of data", "classify the datasets", "classify these datasets",
        "какой датасет содержит", "роль каждого датасет",
    ))


def _asks_important_insights(text: str) -> bool:
    return any(marker in text for marker in (
        "most important", "key insight", "key insights", "top insight",
        "top insights", "important insight", "important insights",
        "strategic insight", "strategic insights", "главн", "ключев",
        "важнейш", "важных", "важные",
    ))


def _strategic_insights_answer(
    entries: list[InvestigationDatasetEntry],
    relationship: dict[str, Any],
) -> str:
    """Generate exactly 3 strategic insights: capability, blocker, strategic implication.

    The three slots are:
    1. **Capability insight** — what cross-dataset analysis IS feasible.
    2. **Blocker insight** — what IS NOT feasible and why.
    3. **Strategic implication** — what this means for the business analytically.

    Each insight must be semantically distinct (no token-overlap > 60%).
    """
    join_keys = relationship.get("joinable_shared_columns") or []
    concept_matches = relationship.get("concept_matches") or []
    all_concepts: set[str] = set()
    per_dataset: dict[str, set[str]] = {}
    for entry in entries:
        entry_concepts = set(entry.semantic_profile.get("concepts", []))
        all_concepts.update(entry_concepts)
        per_dataset[entry.display_name] = entry_concepts

    # --- Slot 1: Capability ---
    capability = ""
    geo_datasets = [name for name, concepts in per_dataset.items() if "geography" in concepts]
    if join_keys:
        capability = f"Direct dataset integration is feasible through shared key(s) {', '.join(join_keys[:4])}, enabling row-level analysis after key quality validation."
    elif len(geo_datasets) >= 2:
        capability = f"Regional comparison is the strongest validated cross-dataset capability, supported by geography fields in {', '.join(f'`{n}`' for n in geo_datasets)}."
    elif concept_matches:
        strongest = concept_matches[0]
        capability = f"Summary-level comparison is possible through the `{strongest['concept'].replace('_', ' ')}` bridge (`{strongest['left_column']}` ↔ `{strongest['right_column']}`), but only at aggregate level, not row-level joins."
    else:
        purposes = [f"`{e.display_name}` ({_dataset_primary_purpose(e).lower().replace('_', ' ')})" for e in entries]
        capability = f"Each dataset serves a distinct analytical purpose: {', '.join(purposes)}. They can be analyzed independently for complementary business views."

    # --- Slot 2: Blocker ---
    blocker = ""
    customer_datasets = [name for name, concepts in per_dataset.items() if "customer_entity" in concepts]
    missing_customer = [name for name, concepts in per_dataset.items() if "customer_entity" not in concepts]
    if customer_datasets and missing_customer:
        blocker = f"Customer-level attribution is blocked because {', '.join(f'`{n}`' for n in missing_customer)} {'does' if len(missing_customer) == 1 else 'do'} not contain stable customer identifiers."
    elif not join_keys and not concept_matches:
        blocker = "No reliable semantic bridge was found between datasets, so any cross-dataset analysis must remain at the aggregate or purpose-comparison level."
    elif not customer_datasets:
        blocker = "None of the loaded datasets contain explicit customer identifiers, which blocks customer-level journey or attribution analysis."
    else:
        # Find a different blocker — metric type mismatch
        purposes = {_dataset_primary_purpose(e) for e in entries}
        if len(purposes) >= 2 and "TRANSACTIONAL" not in purposes:
            blocker = "None of the loaded datasets contain direct transactional records, limiting revenue-based cross-dataset comparisons to behavioral proxies."
        else:
            blocker = "Concept-level bridges are approximate — cross-dataset joins should not be assumed safe without key validation."

    # --- Slot 3: Strategic implication ---
    strategic = ""
    if join_keys:
        strategic = "The datasets are structurally ready for warehouse integration, but key quality (uniqueness, completeness, consistency) must be validated before production use."
    elif concept_matches:
        strategic = "The datasets are complementary for executive analytics, but operational analytics requires warehouse integration and verified bridge mappings."
    else:
        strategic = "These datasets serve different analytical purposes and should be analyzed independently unless a mapping table or shared identifier is introduced."

    # --- Deduplication: ensure semantic diversity ---
    insights = [capability, blocker, strategic]
    # Check pairwise token overlap and replace if too similar
    for i in range(len(insights)):
        for j in range(i + 1, len(insights)):
            overlap = _finding_overlap(_finding_signature(insights[i]), _finding_signature(insights[j]))
            if overlap > 0.6:
                # Replace the later one with a dataset-purpose summary
                purposes = [f"`{e.display_name}` focuses on {_concise_role_description(e)}" for e in entries]
                insights[j] = f"Dataset purpose differences: {'; '.join(purposes)}."
                break

    return " ".join(insights[:3])


def _dataset_role_answer(entries: list[InvestigationDatasetEntry]) -> str:
    parts = []
    for entry in entries:
        desc = _concise_role_description(entry)
        # Add concrete column examples for richer comparison
        metrics = [col for col, role in entry.semantic_roles.items() if role == "metric"]
        dims = [col for col, role in entry.semantic_roles.items() if role == "dimension"]
        examples = []
        if metrics:
            examples.append(f"metric fields: {', '.join(f'`{c}`' for c in metrics[:4])}")
        if dims:
            examples.append(f"dimension fields: {', '.join(f'`{c}`' for c in dims[:4])}")
        suffix = f" ({'; '.join(examples)})" if examples else ""
        parts.append(f"`{entry.display_name}` {desc}{suffix}")
    if len(parts) >= 2:
        return ", while ".join([", ".join(parts[:-1]), parts[-1]]) + "."
    return "; ".join(parts) + "." if parts else "Dataset roles could not be classified from the available schema."


def _concise_role_description(entry: InvestigationDatasetEntry) -> str:
    """Produce a concise, human-readable role description using evidence-weighted purpose.

    Uses ``_dataset_primary_purpose()`` to determine the dominant dataset category,
    then generates a natural-language description.  This prevents financial-like
    metrics from dominating a dataset that is primarily demographic or behavioral.
    """
    purpose = _dataset_primary_purpose(entry)
    concepts = set(entry.semantic_profile.get("concepts", []))
    subtypes = entry.semantic_profile.get("metric_subtypes", {})

    # Purpose-based descriptions
    _PURPOSE_TEMPLATES: dict[str, str] = {
        "TRANSACTIONAL": "tracks transactional records and sales activity",
        "DEMOGRAPHIC": "focuses on customer demographics and attributes",
        "BEHAVIORAL": "focuses on behavioral patterns and spending",
        "CAMPAIGN": "focuses on campaign response and marketing",
        "FINANCIAL": "contains financial records and monetary data",
        "PRODUCT": "covers product catalog and categories",
        "GEOGRAPHIC": "describes geographic or regional structure",
        "LOGISTICS": "describes logistics and distribution",
        "OPERATIONAL": "describes operational workflows",
        "EVENT": "tracks event-level interactions",
        "UNKNOWN": f"is a structured dataset with {len(entry.column_names)} columns",
    }

    base = _PURPOSE_TEMPLATES.get(purpose, "")

    if purpose == "MIXED":
        # For mixed datasets, describe the two strongest signals
        parts: list[str] = []
        if "customer_entity" in concepts or any(s == "FINANCIAL_PROFILE" for s in subtypes.values()):
            parts.append("customer demographics")
        if any(s == "BEHAVIORAL_SPENDING" for s in subtypes.values()):
            parts.append("behavioral spending aggregates")
        if any(s == "ENGAGEMENT" for s in subtypes.values()) or "campaign_response" in concepts:
            parts.append("campaign behavior")
        if any(s == "DIRECT_REVENUE" for s in subtypes.values()):
            parts.append("transactional revenue")
        if "operational_process" in concepts:
            parts.append("operational metrics")
        if not parts:
            parts.append("mixed analytical fields")
        base = "focuses on " + " and ".join(parts[:3])

    if not base:
        base = f"is a structured dataset with {len(entry.column_names)} columns"

    return base


def _article(value: Any) -> str:
    text = str(value or "structured")
    return f"an {text}" if text[:1].lower() in {"a", "e", "i", "o", "u"} else f"a {text}"


def _lineage_dataset_id(context: dict[str, Any] | None) -> str:
    if not isinstance(context, dict):
        return ""
    for container in (
        context.get("active_message_metadata"),
        context.get("latest_chart_context"),
        context.get("conversation_state"),
        context.get("focus"),
    ):
        if isinstance(container, dict):
            value = str(container.get("dataset_id") or "")
            if value:
                return value
    artifacts = context.get("recent_artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            metadata = artifact.get("metadata") if isinstance(artifact, dict) and isinstance(artifact.get("metadata"), dict) else {}
            value = str(metadata.get("dataset_id") or "")
            if value:
                return value
    return ""


def _looks_like_followup(text: str) -> bool:
    return any(marker in text for marker in (
        # Direct references
        "this", "that", "same", "again", "remain", "continue",
        # Comparison continuation
        "compare against", "compare with", "against",
        # Transformation continuation
        "remove", "outlier", "without", "exclude", "normalize",
        "log transform", "clean",
        # Chart continuation
        "explain this", "explain the", "what does this",
        # Analytical refinement
        "split by", "now by", "by region", "by segment", "instead",
        "monthly", "now", "also",
        # Russian markers
        "теперь", "эт", "убери", "остал", "сравни с", "объясни",
    ))


def _is_cross_dataset_question(text: str) -> bool:
    if is_multi_dataset_intent(classify_question_intent(text)):
        return True
    return (
        any(marker in text for marker in ("compare", "across datasets", "between datasets", "both datasets", "all datasets", "common patterns", "common pattern", "patterns across", "shared patterns", "сравн", "между датасет", "оба датасет", "общие паттерн", "общие закономер"))
        or ("shared columns" in text)
        or ("same columns" in text)
    )


def _missing_links_answer(entries: list[InvestigationDatasetEntry], relationship: dict[str, Any]) -> str:
    dataset_names = ", ".join(f"`{entry.display_name}`" for entry in entries)
    join_keys = relationship.get("joinable_shared_columns") or []
    if join_keys:
        return f"The main missing-link risk is not the presence of a key, but whether {', '.join(join_keys[:4])} is unique, complete, and stable enough for row-level analysis across {dataset_names}."
    return (
        f"Several high-value questions cannot be answered reliably across {dataset_names} yet: customer-level journeys, campaign-to-purchase impact, product demand by customer segment, and regional operational drivers. "
        "The missing pieces are stable shared identifiers such as Customer ID, Account ID, Product/SKU ID, invoice/order keys, or a mapping table that connects records across datasets."
    )


def _warehouse_design_answer(entries: list[InvestigationDatasetEntry], relationship: dict[str, Any]) -> str:
    concepts = set()
    for entry in entries:
        concepts.update(entry.semantic_profile.get("concepts", []) or [])
    entities = ["Customer"] if "customer_entity" in concepts else []
    if "product_or_category" in concepts or "inventory" in concepts:
        entities.append("Product")
    if "geography" in concepts:
        entities.append("Location")
    if "time" in concepts:
        entities.append("Date")
    if not entities:
        entities = ["Dataset Source", "Date", "Business Entity"]
    fact_tables = []
    if "monetary_value" in concepts or "purchase_volume" in concepts:
        fact_tables.append("FactTransactions or FactActivity")
    if "campaign_response" in concepts:
        fact_tables.append("FactCampaignResponse")
    if "inventory" in concepts:
        fact_tables.append("FactInventory")
    if not fact_tables:
        fact_tables.append("FactObservation")
    return (
        "A unified analytics warehouse should not start with a blind merge. "
        f"Introduce shared dimensions for {', '.join(entities)}; keep {', '.join(fact_tables)} as separate fact tables; and add bridge tables only where a verified mapping exists. "
        "Required keys should include stable entity IDs, source-system IDs, and effective dates so cross-dataset analysis stays auditable."
    )


_FINANCIAL_MARKERS = {"revenue", "sales", "amount", "price", "cost", "income", "salary", "profit", "spend", "spending", "payment", "total", "fee", "charge", "budget", "margin"}


def _asks_metric_comparison(text: str) -> bool:
    normalized = _norm(text)
    return any(marker in normalized for marker in (
        "compare revenue", "compare metric", "compare financial",
        "revenue-related", "revenue related", "financial metrics",
        "money", "monetary", "сравни метрик", "финансов",
    ))


def _asks_metric_comparison_or_summary(text: str) -> bool:
    normalized = _norm(text)
    return _asks_metric_comparison(normalized) or (
        any(marker in normalized for marker in ("average", "mean", "sum", "row count", "row counts", "metric"))
        and any(marker in normalized for marker in ("compare", "across datasets", "between datasets", "both datasets"))
    )


_REVENUE_MARKERS = {"revenue", "sales", "amount", "profit"}
_PERSONAL_FINANCIAL_MARKERS = {"income", "salary", "wage"}
_SPEND_MARKERS = {"spend", "spending", "mnt", "payment", "cost", "budget"}
_PRICE_MARKERS = {"price", "fee", "charge", "margin"}
_COUNT_MARKERS = {"quantity", "qty", "count", "orders", "transactions", "purchases", "frequency"}


def _classify_financial_column(col_name: str, role: str) -> str:
    """Classify a column into a granular financial metric type."""
    col_lower = _norm(col_name)
    tokens = set(col_lower.split())

    def _any_marker(markers: set[str]) -> bool:
        # Exact word-level match only: a token must equal the marker.
        # _norm splits on word boundaries, so "count_orders" → {"count", "orders"}.
        # This prevents "country" from matching "count".
        return bool(tokens & markers)

    if role == "metric":
        if _any_marker(_REVENUE_MARKERS):
            return "direct_revenue"
        if _any_marker(_PERSONAL_FINANCIAL_MARKERS):
            return "financial_amount"
        if _any_marker(_SPEND_MARKERS):
            return "spend_proxy"
        if _any_marker(_PRICE_MARKERS):
            return "price_proxy"
        if _any_marker(_COUNT_MARKERS):
            return "count_proxy"
        if _any_marker(_FINANCIAL_MARKERS):
            return "behavioral_spend"
    elif _any_marker(_REVENUE_MARKERS):
        return "direct_revenue"
    elif _any_marker(_SPEND_MARKERS):
        return "spend_proxy"
    elif _any_marker(_FINANCIAL_MARKERS):
        return "behavioral_spend"
    return ""


def _classification_label(classification: str) -> str:
    labels = {
        "direct_revenue": "direct transactional revenue",
        "financial_amount": "personal financial attributes (income, salary)",
        "spend_proxy": "spending aggregates that may act as behavioral revenue proxies",
        "price_proxy": "price-level metrics",
        "count_proxy": "count-based volume proxies",
        "behavioral_spend": "behavioral spending indicators",
        "absent": "no direct financial metrics",
    }
    return labels.get(classification, classification)


def _semantic_metric_alignment(registry: list[InvestigationDatasetEntry], frames: dict[str, pd.DataFrame], text: str) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for entry in registry:
        classified: dict[str, list[str]] = {}
        for col in entry.column_names:
            role = entry.semantic_roles.get(col, "")
            cls = _classify_financial_column(col, role)
            if cls:
                classified.setdefault(cls, []).append(col)
        # Pick the strongest classification
        priority = ["direct_revenue", "spend_proxy", "price_proxy", "count_proxy", "behavioral_spend"]
        best_cls = "absent"
        all_fields: list[str] = []
        for cls in priority:
            if cls in classified:
                if best_cls == "absent":
                    best_cls = cls
                all_fields.extend(classified[cls])
        rows.append({
            "dataset": entry.display_name,
            "classification": best_cls,
            "fields": ", ".join(all_fields[:6]) if all_fields else "\u2014",
            "granularity": f"{entry.row_count} rows, {len(entry.column_names)} columns",
        })
    if not rows:
        return None
    parts: list[str] = []
    for row in rows:
        label = _classification_label(row["classification"])
        parts.append(f"`{row['dataset']}` contains {label}" if row["classification"] != "absent" else f"`{row['dataset']}` does not appear to contain direct financial metrics")
    summary = ". ".join(parts) + "."
    classifications = {r["classification"] for r in rows}
    if len(classifications) > 1 and "absent" not in classifications:
        summary += " Cross-dataset revenue comparison is therefore partial and requires normalization assumptions."
    elif "absent" in classifications and len(classifications) > 1:
        summary += " Cross-dataset revenue comparison is therefore partial and requires normalization assumptions."
    return {"metric": "financial_alignment", "rows": rows, "summary": summary}


def _cross_metric_rows(text: str, registry: list[InvestigationDatasetEntry], frames: dict[str, pd.DataFrame]) -> dict[str, Any] | None:
    # Semantic metric alignment: classify as direct, proxy, or absent
    if _asks_metric_comparison(text):
        return _semantic_metric_alignment(registry, frames, text)
    # Exact column match path (original behavior)
    candidates: list[str] = []
    for entry in registry:
        for column in entry.column_names:
            col_tokens = set(_norm(column).split())
            if _column_mentioned(column, text) or (entry.semantic_roles.get(column) == "metric" and col_tokens & {"sales", "revenue", "amount", "salary"}):
                candidates.append(column)
    for metric in dict.fromkeys(candidates):
        rows = []
        for entry in registry:
            df = frames.get(entry.dataset_id)
            if not isinstance(df, pd.DataFrame) or metric not in df.columns:
                break
            values = pd.to_numeric(df[metric], errors="coerce").dropna()
            if values.empty:
                break
            rows.append({"dataset_id": entry.dataset_id, "dataset": entry.display_name, "metric": metric, "average": float(values.mean()), "sum": float(values.sum()), "n": int(len(values))})
        if len(rows) >= 2:
            leader = max(rows, key=lambda row: row["average"])
            return {"metric": metric, "rows": rows, "summary": f"`{leader['dataset']}` has the higher average `{metric}` ({leader['average']:.2f})."}
    return None


def _cross_dataset_operation(text: str, planned_operation: str | None = None) -> str:
    intent = classify_question_intent(text)
    if any(marker in text for marker in ("join", "joined", "joinable", "merge", "shared key", "shared identifier", "warehouse", "connect these datasets", "relationship between these datasets")):
        return "JOINABILITY_ANALYSIS" if "warehouse" not in text else "WAREHOUSE_DESIGN"
    if planned_operation and planned_operation != "CROSS_DATASET_COMPARISON":
        return planned_operation
    if any(marker in text for marker in ("kpi", "dashboard", "dashboards")):
        return "CROSS_DATASET_KPI_SYNTHESIS"
    if any(marker in text for marker in ("comparative visual", "visual analysis showing", "separate charts", "separate visual", "build comparative visual", "build separate visual")):
        return "PARALLEL_VISUAL_ANALYSIS"
    if "diversity" in text:
        return "CROSS_DATASET_DIVERSITY_ANALYSIS"
    if any(marker in text for marker in ("forecast", "time series", "time-series", "temporal trends", "anomaly detection", "best suited", "most suitable", "prioritize one dataset")):
        return "DATASET_CAPABILITY_REASONING"
    if any(marker in text for marker in ("year", "years", "date", "dates", "temporal")):
        return "DATASET_CAPABILITY_REASONING"
    if any(marker in text for marker in ("behavior", "risk concentration", "imbalance", "information-rich", "information rich", "categorical fields", "analytical strengths", "strengths and limitations")):
        return "CROSS_DATASET_SYNTHESIS"
    mapping = {
        QuestionIntentType.PARALLEL_VISUAL_ANALYSIS: "PARALLEL_VISUAL_ANALYSIS",
        QuestionIntentType.MULTI_DATASET_PARALLEL_ANALYSIS: "PARALLEL_VISUAL_ANALYSIS",
        QuestionIntentType.DATASET_CAPABILITY_REASONING: "DATASET_CAPABILITY_REASONING",
        QuestionIntentType.MULTI_DATASET_CAPABILITY_REASONING: "DATASET_CAPABILITY_REASONING",
        QuestionIntentType.CROSS_DATASET_EXECUTIVE_SYNTHESIS: "CROSS_DATASET_EXECUTIVE_SYNTHESIS",
        QuestionIntentType.MULTI_DATASET_EXECUTIVE_SYNTHESIS: "CROSS_DATASET_EXECUTIVE_SYNTHESIS",
        QuestionIntentType.CROSS_DATASET_DIVERSITY_ANALYSIS: "CROSS_DATASET_DIVERSITY_ANALYSIS",
        QuestionIntentType.MULTI_DATASET_DIVERSITY_COMPARISON: "CROSS_DATASET_DIVERSITY_ANALYSIS",
        QuestionIntentType.CROSS_DATASET_TEMPORAL_ANALYSIS: "CROSS_DATASET_TEMPORAL_ANALYSIS",
        QuestionIntentType.MULTI_DATASET_TEMPORAL_COMPARISON: "CROSS_DATASET_TEMPORAL_ANALYSIS",
        QuestionIntentType.MULTI_DATASET_JOINABILITY: "JOINABILITY_ANALYSIS",
        QuestionIntentType.MULTI_DATASET_MISSING_LINKS: "CROSS_DATASET_LIMITATION_ANALYSIS",
        QuestionIntentType.MULTI_DATASET_WAREHOUSE_DESIGN: "WAREHOUSE_DESIGN",
    }
    if any(marker in text for marker in ("limitation", "limitations", "missing", "cannot", "can't", "нельзя", "огранич")):
        return "CROSS_DATASET_LIMITATION_ANALYSIS"
    return mapping.get(intent, "CROSS_DATASET_COMPARISON")


def _build_dataset_execution_scopes(
    *,
    question: str,
    operation: str,
    entries: list[InvestigationDatasetEntry],
    frames: dict[str, pd.DataFrame],
) -> list[DatasetExecutionScope]:
    scopes: list[DatasetExecutionScope] = []
    synthesis_plan = plan_multi_dataset_synthesis(question, [entry.display_name for entry in entries])
    subquestions = {
        str(item.get("dataset_name")): str(item.get("subquestion"))
        for item in synthesis_plan.subquestions
        if isinstance(item, dict)
    }
    for entry in entries:
        df = frames.get(entry.dataset_id)
        scope = DatasetExecutionScope(
            dataset_id=entry.dataset_id,
            dataset_name=entry.display_name,
            question_id=_question_scope_id(question),
            operation=operation,
            compatibility_score=_dataset_operation_score(entry, df, operation),
            semantic_roles=dict(entry.semantic_roles),
            semantic_profile=dict(entry.semantic_profile),
            columns=list(entry.column_names),
            comparison_type=synthesis_plan.comparison_type,
            subquestion=subquestions.get(entry.display_name, ""),
        )
        if not isinstance(df, pd.DataFrame) or df.empty:
            scope.limitations.append("Executable rows were not available for this dataset branch.")
            scopes.append(scope)
            continue
        _populate_branch_scope(scope, entry, df, operation, question)
        scopes.append(scope)
    return scopes


def _question_scope_id(question: str) -> str:
    import hashlib
    return "q_" + hashlib.sha1(str(question or "").encode("utf-8")).hexdigest()[:12]


def _dataset_operation_score(entry: InvestigationDatasetEntry, df: Any, operation: str) -> float:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return 0.0
    timestamps = _temporal_columns(entry)
    metrics = _metric_columns(entry, df)
    dimensions = _dimension_columns(entry, df)
    if operation in {"CROSS_DATASET_TEMPORAL_ANALYSIS", "PARALLEL_VISUAL_ANALYSIS"}:
        return round(min(1.0, 0.35 + 0.35 * bool(timestamps) + 0.2 * bool(metrics) + 0.1 * bool(dimensions)), 2)
    if operation == "CROSS_DATASET_DIVERSITY_ANALYSIS":
        return round(min(1.0, 0.25 + 0.35 * (len(dimensions) >= 2) + 0.2 * bool(metrics) + 0.2 * bool(timestamps)), 2)
    if operation == "DATASET_CAPABILITY_REASONING":
        return round(min(1.0, 0.25 + 0.25 * bool(timestamps) + 0.25 * bool(metrics) + 0.15 * bool(dimensions) + 0.1 * (len(df) >= 10)), 2)
    return round(min(1.0, 0.35 + 0.25 * bool(metrics) + 0.25 * bool(dimensions) + 0.15 * bool(timestamps)), 2)


def _populate_branch_scope(
    scope: DatasetExecutionScope,
    entry: InvestigationDatasetEntry,
    df: pd.DataFrame,
    operation: str,
    question: str,
) -> None:
    if operation == "CROSS_DATASET_DISTRIBUTION_COMPARISON":
        _populate_distribution_scope(scope, entry, df, question)
    elif operation == "CROSS_DATASET_ANOMALY_COMPARISON":
        _populate_anomaly_scope(scope, entry, df, question)
    elif operation in {"CROSS_DATASET_IMBALANCE_COMPARISON", "CROSS_DATASET_CONCENTRATION_COMPARISON"}:
        _populate_imbalance_scope(scope, entry, df, question)
    elif operation in {"CROSS_DATASET_RISK_PERFORMANCE_COMPARISON", "CROSS_DATASET_PROFITABILITY_COMPARISON", "CROSS_DATASET_KPI_SYNTHESIS"}:
        _populate_risk_performance_scope(scope, entry, df, question)
    elif operation == "CROSS_DATASET_RELATIONSHIP_COMPARISON":
        _populate_relationship_scope(scope, entry, df, question)
    elif operation == "CROSS_DATASET_PREDICTABILITY_COMPARISON":
        _populate_predictability_scope(scope, entry, df)
    elif operation == "CROSS_DATASET_OPERATIONAL_OPTIMIZATION":
        _populate_operational_optimization_scope(scope, entry, df, question)
    elif operation == "CROSS_DATASET_DIVERSITY_ANALYSIS":
        _populate_diversity_scope(scope, entry, df)
    elif operation in {"CROSS_DATASET_TEMPORAL_ANALYSIS", "PARALLEL_VISUAL_ANALYSIS"}:
        _populate_visual_temporal_scope(scope, entry, df, question, require_temporal=operation == "CROSS_DATASET_TEMPORAL_ANALYSIS")
    elif operation == "DATASET_CAPABILITY_REASONING":
        _populate_capability_scope(scope, entry, df)
    elif operation == "CROSS_DATASET_EXECUTIVE_SYNTHESIS":
        _populate_risk_performance_scope(scope, entry, df, question)
        if not scope.computed_results:
            _populate_capability_scope(scope, entry, df)
            _populate_default_scope(scope, entry, df)
    else:
        _populate_default_scope(scope, entry, df)
    if not scope.findings:
        scope.findings.append(f"`{entry.display_name}` was evaluated as a separate branch with {len(df):,} rows and {len(df.columns)} columns.")
    _bind_scope_artifacts(scope)


def _populate_visual_temporal_scope(scope: DatasetExecutionScope, entry: InvestigationDatasetEntry, df: pd.DataFrame, question: str, *, require_temporal: bool = False) -> None:
    time_col = _first_valid(df, _temporal_columns(entry))
    metric_col = _preferred_metric_column(df, entry, question, exclude={time_col})
    dimension_col = _preferred_dimension_column(df, entry, question, exclude={time_col, metric_col})
    aggregation = _aggregation_from_question(question, metric_col)
    if time_col:
        rows = _temporal_rows(df, time_col, metric_col, aggregation=aggregation)
        if rows:
            value_key = "value"
            stats = _trend_stats(rows)
            scope.metrics_used.append(metric_col or "record_count")
            scope.computed_results.append({"analysis_type": "temporal_trend", "time_field": time_col, "metric": metric_col or "record_count", "aggregation": aggregation, "rows": rows[:20], **stats})
            scope.findings.append(f"`{entry.display_name}` supports trend analysis using `{time_col}` with `{aggregation}` `{metric_col or 'record count'}` as the measured signal.")
            scope.artifacts.append(_scoped_chart(entry, "line", f"{entry.display_name}: {aggregation} {metric_col or 'records'} over {time_col}", "period", value_key, rows, {"analysis_type": "temporal_trend", "time_axis": time_col, "metric": metric_col or "record_count", "aggregation": aggregation, **stats}))
            return
    if require_temporal and _requires_temporal_branch(question):
        scope.limitations.append(f"`{entry.display_name}` has no true temporal field for the requested trend comparison.")
        return
    if metric_col and dimension_col:
        rows = _group_metric_rows(df, dimension_col, metric_col, aggregation=aggregation)
        if rows:
            top_share = _top_value_share(rows)
            scope.metrics_used.append(metric_col)
            scope.grouping_fields.append(dimension_col)
            scope.computed_results.append({"analysis_type": "grouped_metric", "dimension": dimension_col, "metric": metric_col, "aggregation": aggregation, "top_share": top_share, "rows": rows[:20]})
            scope.findings.append(f"`{entry.display_name}` can be visualized by comparing {aggregation} `{metric_col}` across `{dimension_col}`.")
            scope.artifacts.append(_scoped_chart(entry, "bar", f"{entry.display_name}: {aggregation} {metric_col} by {dimension_col}", dimension_col, "value", rows, {"analysis_type": "grouped_metric", "dimension": dimension_col, "metric": metric_col, "aggregation": aggregation, "top_share": top_share}))
            return
    dimension_col = dimension_col or _first_valid(df, _dimension_columns(entry, df))
    if dimension_col:
        rows = _count_rows(df, dimension_col)
        scope.computed_results.append({"analysis_type": "category_distribution", "dimension": dimension_col, "rows": rows[:20]})
        scope.findings.append(f"`{entry.display_name}` can be visualized as record distribution across `{dimension_col}`.")
        scope.artifacts.append(_scoped_chart(entry, "bar", f"{entry.display_name}: records by {dimension_col}", dimension_col, "count", rows, {"analysis_type": "category_distribution", "dimension": dimension_col, "metric": "record_count"}))
    else:
        scope.limitations.append(f"`{entry.display_name}` has no clear temporal, metric, or categorical field for a visual branch.")


def _populate_distribution_scope(scope: DatasetExecutionScope, entry: InvestigationDatasetEntry, df: pd.DataFrame, question: str) -> None:
    metric = _preferred_metric_column(df, entry, question)
    if not metric:
        scope.limitations.append(f"`{entry.display_name}` has no numeric metric for distribution comparison.")
        return
    rows, stats = _distribution_evidence_rows(df, metric)
    if not rows:
        scope.limitations.append(f"`{entry.display_name}` had no valid numeric values for `{metric}`.")
        return
    scope.metrics_used.append(metric)
    scope.computed_results.append({"analysis_type": "distribution_comparison", "metric": metric, "rows": rows, **stats})
    scope.findings.append(f"`{entry.display_name}` `{metric}` distribution spans {stats['minimum']:.2f} to {stats['maximum']:.2f} with outlier rate {stats['outlier_rate']:.1f}%.")
    scope.artifacts.append(_scoped_chart(entry, "histogram", f"{entry.display_name}: {metric} distribution", "bin", "count", rows, {"analysis_type": "distribution_comparison", "metric": metric, **stats}))


def _populate_anomaly_scope(scope: DatasetExecutionScope, entry: InvestigationDatasetEntry, df: pd.DataFrame, question: str) -> None:
    metric = _preferred_metric_column(df, entry, question)
    if not metric:
        scope.limitations.append(f"`{entry.display_name}` has no numeric metric for anomaly comparison.")
        return
    rows, stats = _outlier_rows(df, metric)
    scope.metrics_used.append(metric)
    scope.computed_results.append({"analysis_type": "anomaly_comparison", "metric": metric, "rows": rows, **stats})
    scope.findings.append(f"`{entry.display_name}` has {stats['outlier_count']} `{metric}` outlier(s), outlier rate {stats['outlier_rate']:.1f}%.")
    scope.artifacts.append({
        "artifact_type": "table",
        "title": f"{entry.display_name}: {metric} anomaly evidence",
        "content": rows,
        "visibility": "user",
        "pinned": False,
        "metadata": {"analysis_type": "anomaly_comparison", "dataset_id": entry.dataset_id, "dataset_name": entry.display_name, "metric": metric, **stats},
    })
    distribution_rows, distribution_stats = _distribution_evidence_rows(df, metric)
    if distribution_rows:
        scope.artifacts.append(_scoped_chart(entry, "histogram", f"{entry.display_name}: {metric} anomaly distribution", "bin", "count", distribution_rows, {"analysis_type": "anomaly_distribution", "metric": metric, **distribution_stats}))


def _populate_imbalance_scope(scope: DatasetExecutionScope, entry: InvestigationDatasetEntry, df: pd.DataFrame, question: str) -> None:
    metric = _preferred_metric_column(df, entry, question)
    dimension = _preferred_dimension_column(df, entry, question, exclude={metric})
    if metric and not _metric_aligned_with_question(metric, question):
        metric = ""
    if dimension and not _dimension_aligned_with_question(dimension, question):
        dimension = ""
    aggregation = _aggregation_from_question(question, metric)
    if metric:
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if not values.empty:
            imbalance = _gini(values.tolist())
            result = {"analysis_type": "imbalance_comparison", "metric": metric, "gini": round(imbalance, 3), "n": int(len(values)), "minimum": float(values.min()), "maximum": float(values.max())}
            scope.metrics_used.append(metric)
            scope.computed_results.append(result)
            scope.findings.append(f"`{entry.display_name}` `{metric}` imbalance score is {result['gini']:.3f}.")
    if metric and dimension:
        rows = _group_metric_rows(df, dimension, metric, aggregation=aggregation)
        if rows:
            top_share = _top_value_share(rows)
            if metric not in scope.metrics_used:
                scope.metrics_used.append(metric)
            scope.grouping_fields.append(dimension)
            scope.computed_results.append({"analysis_type": "grouped_metric", "dimension": dimension, "metric": metric, "aggregation": aggregation, "top_share": top_share, "rows": rows[:20]})
            scope.artifacts.append(_scoped_chart(entry, "bar", f"{entry.display_name}: {aggregation} {metric} by {dimension}", dimension, "value", rows, {"analysis_type": "grouped_metric", "dimension": dimension, "metric": metric, "aggregation": aggregation, "top_share": top_share}))
    if dimension:
        rows = _count_rows(df, dimension)
        total = max(1, sum(int(row.get("count", 0)) for row in rows))
        top_share = max((int(row.get("count", 0)) / total for row in rows), default=0.0)
        if dimension not in scope.grouping_fields:
            scope.grouping_fields.append(dimension)
        scope.computed_results.append({"analysis_type": "category_concentration", "dimension": dimension, "top_share": round(top_share * 100, 2), "rows": rows[:20]})
        scope.artifacts.append(_scoped_chart(entry, "bar", f"{entry.display_name}: {dimension} concentration", dimension, "count", rows, {"analysis_type": "category_concentration", "dimension": dimension, "top_share": round(top_share * 100, 2)}))
    if not scope.computed_results:
        scope.limitations.append(f"`{entry.display_name}` had no usable metric or category for imbalance comparison.")


def _populate_risk_performance_scope(scope: DatasetExecutionScope, entry: InvestigationDatasetEntry, df: pd.DataFrame, question: str) -> None:
    if _should_use_grouped_metric_branch(question) and _populate_grouped_metric_scope(scope, entry, df, question):
        return
    business = execute_business_plan("Where are the strongest profit margin or operational risk patterns?", df)
    if business:
        _merge_business_evidence_into_scope(scope, entry, business)
        return
    if _business_specific_question(question):
        scope.limitations.append(f"`{entry.display_name}` does not expose the requested business metric and grouping fields for this branch.")
        return
    binary = _binary_indicator_columns(df)
    if binary:
        rows = []
        for column in binary[:8]:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            prevalence = float(values.mean() * 100) if not values.empty else 0.0
            rows.append({"indicator": column, "prevalence": round(prevalence, 2), "count": int(values.sum()), "records": int(len(values))})
        rows.sort(key=lambda row: row["prevalence"], reverse=True)
        scope.metrics_used.extend([row["indicator"] for row in rows[:4]])
        scope.derived_kpis.append("indicator_prevalence")
        scope.computed_results.append({"analysis_type": "risk_prevalence", "rows": rows})
        scope.findings.append(f"`{entry.display_name}` strongest risk indicator is `{rows[0]['indicator']}` at {rows[0]['prevalence']}%.")
        scope.artifacts.append(_scoped_chart(entry, "bar", f"{entry.display_name}: risk indicator prevalence", "indicator", "prevalence", rows, {"analysis_type": "risk_prevalence"}))
        return
    _populate_imbalance_scope(scope, entry, df, question)


def _populate_operational_optimization_scope(scope: DatasetExecutionScope, entry: InvestigationDatasetEntry, df: pd.DataFrame, question: str) -> None:
    business = execute_business_plan("Which groups are operationally inefficient?", df)
    if business:
        _merge_business_evidence_into_scope(scope, entry, business)
        score = 0.9 if {"profit_margin", "operational_inefficiency_score"} & set(business.derived_kpis) else 0.65
        scope.computed_results.append({"analysis_type": "operational_optimization_suitability", "score": score, "evidence": business.findings[:3]})
        scope.findings.append(f"`{entry.display_name}` has operational optimization levers from {', '.join(business.derived_kpis[:4])}.")
        return
    _populate_risk_performance_scope(scope, entry, df, question)
    if scope.computed_results:
        score = 0.55 + 0.1 * bool(scope.grouping_fields)
        scope.computed_results.append({"analysis_type": "operational_optimization_suitability", "score": round(score, 2), "evidence": scope.findings[:3]})


def _populate_diversity_scope(scope: DatasetExecutionScope, entry: InvestigationDatasetEntry, df: pd.DataFrame) -> None:
    dimensions = _dimension_columns(entry, df)
    group_col = _first_valid(df, dimensions)
    entity_col = _first_valid(df, dimensions, exclude={group_col})
    if not group_col or not entity_col:
        scope.limitations.append(f"`{entry.display_name}` needs at least two categorical fields for within-dataset diversity analysis.")
        _populate_default_scope(scope, entry, df)
        return
    grouped = df[[group_col, entity_col]].dropna()
    if grouped.empty:
        scope.limitations.append(f"`{entry.display_name}` has no non-null rows for `{group_col}` and `{entity_col}` diversity.")
        return
    counts = grouped.groupby(group_col, observed=True)[entity_col].nunique().sort_values(ascending=False)
    rows = [{"group": str(k), "unique_count": int(v)} for k, v in counts.head(20).items()]
    if rows:
        leader = rows[0]
        scope.computed_results.append({"analysis_type": "diversity_analysis", "group_column": group_col, "entity_column": entity_col, "rows": rows})
        scope.findings.append(f"`{entry.display_name}` diversity is strongest for `{leader['group']}` with {leader['unique_count']} unique `{entity_col}` values.")
        scope.artifacts.append(_scoped_chart(entry, "bar", f"{entry.display_name}: {entity_col} diversity by {group_col}", "group", "unique_count", rows, {"analysis_type": "diversity_analysis", "dimension": group_col, "entity": entity_col}))


def _populate_relationship_scope(scope: DatasetExecutionScope, entry: InvestigationDatasetEntry, df: pd.DataFrame, question: str) -> None:
    metrics = _relationship_metric_pair(df, entry, question)
    if len(metrics) < 2:
        scope.limitations.append(f"`{entry.display_name}` needs at least two numeric metrics for relationship comparison.")
        _populate_default_scope(scope, entry, df)
        return
    x_metric, y_metric = metrics[:2]
    working = pd.DataFrame({
        "x": pd.to_numeric(df[x_metric], errors="coerce"),
        "y": pd.to_numeric(df[y_metric], errors="coerce"),
    }).dropna()
    if len(working) < 2:
        scope.limitations.append(f"`{entry.display_name}` had too few valid rows for `{x_metric}` versus `{y_metric}`.")
        return
    corr = float(working["x"].corr(working["y"]))
    dimension = _preferred_dimension_column(df, entry, question, exclude={x_metric, y_metric})
    rows: list[dict[str, Any]]
    if dimension:
        temp = df[[dimension, x_metric, y_metric]].copy()
        temp["_x"] = pd.to_numeric(temp[x_metric], errors="coerce")
        temp["_y"] = pd.to_numeric(temp[y_metric], errors="coerce")
        grouped = temp.dropna(subset=[dimension, "_x", "_y"]).groupby(dimension, observed=True)[["_x", "_y"]].mean()
        rows = [{"group": str(idx), x_metric: float(row["_x"]), y_metric: float(row["_y"])} for idx, row in grouped.head(20).iterrows()]
    else:
        rows = [{x_metric: float(row["x"]), y_metric: float(row["y"])} for _, row in working.head(100).iterrows()]
    scope.metrics_used.extend([x_metric, y_metric])
    if dimension:
        scope.grouping_fields.append(dimension)
    scope.computed_results.append({"analysis_type": "relationship_comparison", "x_metric": x_metric, "y_metric": y_metric, "correlation": corr, "dimension": dimension, "rows": rows})
    scope.findings.append(f"`{entry.display_name}` relationship evidence compares `{x_metric}` with `{y_metric}` (correlation {corr:.3f}).")
    scope.artifacts.append(_scoped_chart(entry, "scatter", f"{entry.display_name}: {x_metric} versus {y_metric}", x_metric, y_metric, rows, {"analysis_type": "relationship_comparison", "metric": f"{x_metric} vs {y_metric}", "dimension": dimension, "correlation": corr}))


def _populate_predictability_scope(scope: DatasetExecutionScope, entry: InvestigationDatasetEntry, df: pd.DataFrame) -> None:
    metrics = _metric_columns(entry, df)
    dimensions = _dimension_columns(entry, df)
    temporal = _temporal_columns(entry)
    score = round(min(1.0, 0.25 + 0.2 * bool(metrics) + 0.2 * bool(dimensions) + 0.15 * bool(temporal) + 0.1 * min(len(metrics), 5) / 5 + 0.1 * min(len(dimensions), 5) / 5), 3)
    rows = [
        {"signal": "numeric_metrics", "count": len(metrics), "examples": ", ".join(metrics[:4])},
        {"signal": "categorical_dimensions", "count": len(dimensions), "examples": ", ".join(dimensions[:4])},
        {"signal": "temporal_fields", "count": len(temporal), "examples": ", ".join(temporal[:4])},
    ]
    scope.metrics_used.extend(metrics[:4])
    scope.grouping_fields.extend(dimensions[:4])
    scope.computed_results.append({"analysis_type": "feature_structure_predictability", "score": score, "metric_count": len(metrics), "dimension_count": len(dimensions), "temporal_count": len(temporal), "rows": rows})
    scope.findings.append(f"`{entry.display_name}` feature-structure predictability score is {score:.3f}.")
    scope.artifacts.append({
        "artifact_type": "table",
        "title": f"{entry.display_name}: feature-structure evidence",
        "content": rows,
        "visibility": "user",
        "pinned": False,
        "metadata": {"analysis_type": "feature_structure_predictability", "dataset_id": entry.dataset_id, "dataset_name": entry.display_name, "score": score},
    })


def _populate_capability_scope(scope: DatasetExecutionScope, entry: InvestigationDatasetEntry, df: pd.DataFrame) -> None:
    timestamps = _temporal_columns(entry)
    metrics = _metric_columns(entry, df)
    dimensions = _dimension_columns(entry, df)
    capability_rows = [
        {"capability": "time_series", "score": round(0.25 + 0.45 * bool(timestamps) + 0.2 * bool(metrics) + 0.1 * (len(df) >= 10), 2), "evidence": ", ".join(timestamps[:4]) or "No temporal field detected"},
        {"capability": "anomaly_detection", "score": round(0.25 + 0.35 * bool(metrics) + 0.2 * bool(dimensions) + 0.2 * (len(df) >= 20), 2), "evidence": ", ".join(metrics[:4]) or "No clear numeric metric detected"},
        {"capability": "segmentation", "score": round(0.2 + 0.55 * bool(dimensions) + 0.15 * bool(metrics) + 0.1 * (len(df) >= 10), 2), "evidence": ", ".join(dimensions[:4]) or "No clear categorical segment detected"},
    ]
    best = max(capability_rows, key=lambda row: float(row["score"]))
    scope.computed_results.append({"analysis_type": "dataset_capability", "rows": capability_rows})
    scope.findings.append(f"`{entry.display_name}` is strongest for {best['capability'].replace('_', ' ')} because {best['evidence']}.")
    scope.artifacts.append({
        "artifact_type": "table",
        "title": f"{entry.display_name}: capability profile",
        "content": capability_rows,
        "visibility": "user",
        "pinned": False,
        "metadata": {"analysis_type": "dataset_capability", "dataset_id": entry.dataset_id, "dataset_name": entry.display_name},
    })


def _populate_default_scope(scope: DatasetExecutionScope, entry: InvestigationDatasetEntry, df: pd.DataFrame) -> None:
    metric_col = _first_valid(df, _metric_columns(entry, df))
    dimension_col = _first_valid(df, _dimension_columns(entry, df), exclude={metric_col})
    if metric_col:
        values = pd.to_numeric(df[metric_col], errors="coerce").dropna()
        if not values.empty:
            result = {"analysis_type": "metric_summary", "metric": metric_col, "average": float(values.mean()), "minimum": float(values.min()), "maximum": float(values.max()), "n": int(len(values))}
            scope.computed_results.append(result)
            scope.findings.append(f"`{entry.display_name}` has `{metric_col}` average {result['average']:.2f} across {result['n']} valid rows.")
    if dimension_col:
        rows = _count_rows(df, dimension_col)
        if rows:
            scope.computed_results.append({"analysis_type": "category_distribution", "dimension": dimension_col, "rows": rows[:10]})
            scope.artifacts.append(_scoped_chart(entry, "bar", f"{entry.display_name}: records by {dimension_col}", dimension_col, "count", rows, {"analysis_type": "category_distribution", "dimension": dimension_col, "metric": "record_count"}))


def _bind_scope_artifacts(scope: DatasetExecutionScope) -> None:
    for artifact in scope.artifacts:
        metadata = artifact.setdefault("metadata", {})
        metadata.setdefault("dataset_id", scope.dataset_id)
        metadata.setdefault("dataset_ids", [scope.dataset_id])
        metadata.setdefault("dataset_name", scope.dataset_name)
        metadata.setdefault("dataset_scope", "single_dataset")
        metadata.setdefault("question_id", scope.question_id)
        metadata.setdefault("operation", scope.operation)
        metadata.setdefault("semantic_roles", dict(scope.semantic_roles))
        metadata.setdefault("findings", list(scope.findings))


def _merge_business_evidence_into_scope(scope: DatasetExecutionScope, entry: InvestigationDatasetEntry, business: Any) -> None:
    scope.metrics_used.extend([item for item in business.metrics_used if item not in scope.metrics_used])
    scope.derived_kpis.extend([item for item in business.derived_kpis if item not in scope.derived_kpis])
    scope.grouping_fields.extend([item for item in business.grouping_fields if item not in scope.grouping_fields])
    for result in business.computed_results:
        payload = dict(result)
        payload.setdefault("analysis_type", "business_kpi")
        scope.computed_results.append(payload)
    scope.findings.extend(business.findings[:3])
    for artifact in business.artifacts:
        prepared = dict(artifact)
        metadata = prepared.get("metadata") if isinstance(prepared.get("metadata"), dict) else {}
        metadata.update({"dataset_id": entry.dataset_id, "dataset_name": entry.display_name, "dataset_scope": "single_dataset"})
        prepared["metadata"] = metadata
        prepared["title"] = f"{entry.display_name}: {prepared.get('title', 'business KPI evidence')}"
        scope.artifacts.append(prepared)


def _synthesize_dataset_scopes(scopes: list[DatasetExecutionScope], operation: str) -> str:
    active = [scope for scope in scopes if scope.compatibility_score > 0]
    if not active:
        return ""
    names = ", ".join(f"`{scope.dataset_name}`" for scope in active)
    if operation == "PARALLEL_VISUAL_ANALYSIS":
        chart_count = sum(len([a for a in scope.artifacts if a.get("artifact_type") == "chart"]) for scope in active)
        return f"Parallel visual analysis kept {len(active)} dataset branches separate and produced {chart_count} dataset-scoped chart artifact(s) across {names}."
    if operation == "DATASET_CAPABILITY_REASONING":
        ranked = sorted(active, key=lambda scope: scope.compatibility_score, reverse=True)
        return f"Capability ranking is based on temporal fields, metric richness, segmentation fields, and executable row depth; strongest branch: `{ranked[0].dataset_name}` ({ranked[0].compatibility_score:.2f})."
    if operation == "CROSS_DATASET_DIVERSITY_ANALYSIS":
        return f"Diversity was computed per dataset branch, so categorical variety is compared without merging unrelated entities across {names}."
    if operation == "CROSS_DATASET_TEMPORAL_ANALYSIS":
        temporal_ready = [scope.dataset_name for scope in active if any((r.get("analysis_type") == "temporal_trend") for r in scope.computed_results)]
        if temporal_ready:
            return f"Temporal analysis is branch-scoped; datasets with executable trend branches: {', '.join(f'`{name}`' for name in temporal_ready)}."
    return f"Cross-dataset synthesis used {len(active)} isolated dataset branch(es), preserving per-dataset evidence before comparing strengths and limitations."


def _temporal_columns(entry: InvestigationDatasetEntry) -> list[str]:
    columns = list(entry.semantic_profile.get("timestamp_columns") or [])
    for column in entry.column_names:
        concepts = _column_concepts(column)
        if "time" in concepts and column not in columns:
            columns.append(column)
    return [column for column in columns if not _is_ordinal_demographic_column(column)]


def _is_ordinal_demographic_column(column: str) -> bool:
    tokens = set(_norm(column).split())
    return bool(tokens & {"age", "birth", "education"}) or "age group" in _norm(column) or "year birth" in _norm(column) or "income bracket" in _norm(column)


def _metric_columns(entry: InvestigationDatasetEntry, df: pd.DataFrame) -> list[str]:
    columns = [column for column in entry.semantic_profile.get("metric_columns", []) if column in df.columns]
    for column in df.columns:
        if column not in columns and pd.api.types.is_numeric_dtype(df[column]):
            columns.append(str(column))
    return [column for column in columns if column in df.columns and not _looks_identifier_column(column) and not _is_ordinal_demographic_column(column)]


def _dimension_columns(entry: InvestigationDatasetEntry, df: pd.DataFrame) -> list[str]:
    columns = [column for column in entry.semantic_profile.get("dimension_columns", []) if column in df.columns]
    for column in df.columns:
        if column not in columns and not pd.api.types.is_numeric_dtype(df[column]):
            columns.append(str(column))
    return [
        column
        for column in columns
        if column in df.columns
        and not _looks_identifier_column(column)
        and not _looks_entity_label_column(column)
        and not _looks_temporal_dimension_column(entry, df, column)
    ]


def _looks_identifier_column(column: str) -> bool:
    tokens = set(_norm(column).split())
    return bool(tokens & {"id", "uuid", "key", "postal", "zip", "postcode"})


def _looks_entity_label_column(column: str) -> bool:
    tokens = set(_norm(column).split())
    return bool(tokens & {"title", "name", "description", "comment", "text"})


def _looks_temporal_dimension_column(entry: InvestigationDatasetEntry, df: pd.DataFrame, column: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(df[column]):
        return True
    if column in set(_temporal_columns(entry)):
        return True
    tokens = set(_norm(column).split())
    return bool(tokens & {"date", "datetime", "timestamp"}) or _norm(column) in {"year", "month", "period"}


def _preferred_metric_column(df: pd.DataFrame, entry: InvestigationDatasetEntry, question: str, exclude: set[str | None] | None = None) -> str:
    candidates = _metric_columns(entry, df)
    text = _norm(question)
    for column in candidates:
        if _norm(column) in text:
            return _first_valid(df, [column], exclude=exclude)
    priority_markers = ("risk", "outcome", "condition", "disease", "prevalence", "indicator", "flag", "attack", "smoker")
    if any(marker in text for marker in ("prevalence", "risk", "condition", "health", "patient")):
        for column in candidates:
            col_text = _norm(column)
            if any(marker in col_text for marker in priority_markers):
                return _first_valid(df, [column], exclude=exclude)
    return _first_valid(df, candidates, exclude=exclude)


def _relationship_metric_pair(df: pd.DataFrame, entry: InvestigationDatasetEntry, question: str) -> list[str]:
    candidates = _metric_columns(entry, df)
    text = _norm(question)
    mentioned = [column for column in candidates if _norm(column) in text]
    if len(mentioned) >= 2:
        return mentioned[:2]
    business_pair = []
    for marker_group in (("sales", "revenue", "income", "amount"), ("profit", "margin", "earnings")):
        for column in candidates:
            col_text = _norm(column)
            if any(marker in col_text for marker in marker_group):
                business_pair.append(column)
                break
    if len(_unique(business_pair)) >= 2:
        return _unique(business_pair)[:2]
    preferred = _preferred_metric_column(df, entry, question)
    ordered = [preferred] if preferred else []
    ordered.extend([column for column in candidates if column not in ordered])
    return ordered[:2]


def _preferred_dimension_column(df: pd.DataFrame, entry: InvestigationDatasetEntry, question: str, exclude: set[str | None] | None = None) -> str:
    candidates = _dimension_columns(entry, df)
    text = _norm(question)
    for column in candidates:
        if _norm(column) in text:
            return _first_valid(df, [column], exclude=exclude)
    if "age" in text:
        for column in entry.column_names:
            if "age" in _norm(column).split() and column in df.columns:
                return _first_valid(df, [column], exclude=exclude)
    if any(marker in text for marker in ("category", "genre", "product", "segment")):
        for column in candidates:
            col_text = _norm(column)
            if any(marker in col_text for marker in ("category", "genre", "product", "segment")):
                return _first_valid(df, [column], exclude=exclude)
    if any(marker in text for marker in ("city", "cities", "geographic", "geography", "state", "country", "region", "territory", "market")):
        for column in candidates:
            col_text = _norm(column)
            if any(marker in col_text for marker in ("city", "state", "country", "region", "territory", "market", "province")):
                return _first_valid(df, [column], exclude=exclude)
    return _first_valid(df, candidates, exclude=exclude)


def _aggregation_from_question(question: str, metric_col: str | None = None) -> str:
    text = _norm(question)
    metric_text = _norm(metric_col or "")
    if any(marker in text for marker in ("average", "avg", "mean")) and not any(marker in text for marker in ("do not use average", "don't use average", "not average", "no average")):
        return "mean"
    if "median" in text:
        return "median"
    if any(marker in text for marker in ("count", "frequency", "number of records")) and not any(marker in metric_text for marker in ("sales", "revenue", "profit", "income", "cost", "amount")):
        return "count"
    if any(marker in text for marker in ("total", "sum", "overall", "growth", "strongest", "highest", "top ", "leader")):
        return "sum"
    if any(marker in metric_text for marker in ("sales", "revenue", "profit", "income", "cost", "amount", "value")):
        return "sum"
    return "mean"


def _requires_temporal_branch(question: str) -> bool:
    text = _norm(question)
    return any(marker in text for marker in ("trend", "growth", "over time", "yearly", "annual", "time series", "time-series"))


def _trend_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    first = rows[0]
    last = rows[-1]
    first_value = float(first.get("value") or 0.0)
    last_value = float(last.get("value") or 0.0)
    absolute_change = last_value - first_value
    percent_change = (absolute_change / abs(first_value) * 100) if first_value else 0.0
    return {
        "first_period": first.get("period"),
        "last_period": last.get("period"),
        "first_value": first_value,
        "last_value": last_value,
        "absolute_change": absolute_change,
        "percent_change": percent_change,
    }


def _top_value_share(rows: list[dict[str, Any]]) -> float:
    values = [abs(float(row.get("value") or 0.0)) for row in rows if isinstance(row, dict)]
    total = sum(values)
    if total <= 0:
        return 0.0
    return round(max(values, default=0.0) / total * 100, 2)


def _should_use_grouped_metric_branch(question: str) -> bool:
    text = _norm(question)
    return any(marker in text for marker in ("highest", "strongest", "top ", "rank", "ranking", "by total", "total profit", "total sales", "generate", "generates")) and any(marker in text for marker in ("segment", "category", "city", "cities", "region", "product", "customer", "group", "each dataset"))


def _populate_grouped_metric_scope(scope: DatasetExecutionScope, entry: InvestigationDatasetEntry, df: pd.DataFrame, question: str) -> bool:
    metric = _preferred_metric_column(df, entry, question)
    dimension = _preferred_dimension_column(df, entry, question, exclude={metric})
    if not metric or not dimension:
        return False
    if not _metric_aligned_with_question(metric, question) or not _dimension_aligned_with_question(dimension, question):
        return False
    aggregation = _aggregation_from_question(question, metric)
    rows = _group_metric_rows(df, dimension, metric, aggregation=aggregation)
    if not rows:
        return False
    top_share = _top_value_share(rows)
    scope.metrics_used.append(metric)
    scope.grouping_fields.append(dimension)
    scope.computed_results.append({"analysis_type": "grouped_metric", "dimension": dimension, "metric": metric, "aggregation": aggregation, "top_share": top_share, "rows": rows[:20]})
    leader = rows[0]
    scope.findings.append(f"`{entry.display_name}` leading `{dimension}` by {aggregation} `{metric}` is `{leader.get(dimension)}` ({float(leader.get('value') or 0):.2f}).")
    scope.artifacts.append(_scoped_chart(entry, "bar", f"{entry.display_name}: {aggregation} {metric} by {dimension}", dimension, "value", rows, {"analysis_type": "grouped_metric", "dimension": dimension, "metric": metric, "aggregation": aggregation, "top_share": top_share}))
    return True


def _business_specific_question(question: str) -> bool:
    text = _norm(question)
    if any(marker in text for marker in ("health", "medical", "risk", "patient", "disease")):
        return False
    return any(marker in text for marker in ("profit", "sales", "revenue", "customer segment", "segments", "commercial", "business performance"))


def _metric_aligned_with_question(metric: str, question: str) -> bool:
    text = _norm(question)
    metric_text = _norm(metric)
    if any(marker in text for marker in ("profit", "margin", "earnings")):
        return any(marker in metric_text for marker in ("profit", "margin", "earnings"))
    if any(marker in text for marker in ("sales", "revenue", "income", "amount")):
        return any(marker in metric_text for marker in ("sales", "revenue", "income", "amount", "value"))
    return True


def _dimension_aligned_with_question(dimension: str, question: str) -> bool:
    text = _norm(question)
    dimension_text = _norm(dimension)
    if any(marker in text for marker in ("segment", "segments", "customer group", "customer groups")):
        return any(marker in dimension_text for marker in ("segment", "customer", "client", "group"))
    if any(marker in text for marker in ("city", "cities", "geographic", "geography")):
        return any(marker in dimension_text for marker in ("city", "state", "country", "region", "territory", "market", "province"))
    if "category" in text or "product" in text:
        return any(marker in dimension_text for marker in ("category", "product", "segment"))
    return True


def _first_valid(df: pd.DataFrame, columns: list[str], exclude: set[str | None] | None = None) -> str:
    excluded = {item for item in (exclude or set()) if item}
    for column in columns:
        if column in df.columns and column not in excluded:
            series = df[column].dropna()
            if not series.empty:
                return column
    return ""


def _temporal_rows(df: pd.DataFrame, time_col: str, metric_col: str | None, *, aggregation: str = "mean") -> list[dict[str, Any]]:
    working = df[[time_col] + ([metric_col] if metric_col and metric_col in df.columns else [])].copy()
    period = pd.to_datetime(working[time_col], errors="coerce")
    if period.notna().sum() >= max(2, int(len(working) * 0.25)):
        working["_period"] = period.dt.to_period("Y").astype(str)
    else:
        numeric = pd.to_numeric(working[time_col], errors="coerce")
        if numeric.notna().sum() == 0:
            return []
        working["_period"] = numeric.astype("Int64").astype(str)
    if metric_col and metric_col in working.columns:
        working["_value"] = pd.to_numeric(working[metric_col], errors="coerce")
        grouped_raw = working.dropna(subset=["_period", "_value"]).groupby("_period", observed=True)["_value"]
        grouped = grouped_raw.sum() if aggregation == "sum" else grouped_raw.median() if aggregation == "median" else grouped_raw.count() if aggregation == "count" else grouped_raw.mean()
    else:
        grouped = working.dropna(subset=["_period"]).groupby("_period", observed=True).size()
    return [{"period": str(k), "value": float(v)} for k, v in grouped.sort_index().items()]


def _distribution_evidence_rows(df: pd.DataFrame, metric_col: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
    values = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if values.empty:
        return [], {}
    bins = pd.cut(values, bins=min(8, max(3, int(values.nunique()))), duplicates="drop")
    counts = bins.value_counts().sort_index()
    rows = [{"bin": f"{idx.left:.2f} to {idx.right:.2f}", "count": int(count)} for idx, count in counts.items()]
    outliers, stats = _outlier_rows(df, metric_col)
    stats.update({
        "mean": float(values.mean()),
        "median": float(values.median()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "outlier_count": int(len(outliers)),
    })
    return rows, stats


def _outlier_rows(df: pd.DataFrame, metric_col: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
    values = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if values.empty:
        return [], {"outlier_count": 0, "outlier_rate": 0.0, "minimum": 0.0, "maximum": 0.0}
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (values < lower) | (values > upper)
    rows = [{"row_index": int(idx), "metric": metric_col, "value": float(value), "direction": "low" if value < lower else "high"} for idx, value in values[mask].items()]
    return rows, {
        "q1": q1,
        "q3": q3,
        "lower_fence": float(lower),
        "upper_fence": float(upper),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "outlier_count": int(len(rows)),
        "outlier_rate": float(len(rows) / max(1, len(values)) * 100),
    }


def _gini(values: list[float]) -> float:
    clean = sorted(float(value) for value in values if pd.notna(value))
    if not clean:
        return 0.0
    total = sum(clean)
    if total == 0:
        return 0.0
    n = len(clean)
    weighted = sum((idx + 1) * value for idx, value in enumerate(clean))
    return (2 * weighted) / (n * total) - (n + 1) / n


def _binary_indicator_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        values = set(pd.to_numeric(df[column], errors="coerce").dropna().unique().tolist())
        if values and values.issubset({0, 1}) and len(values) >= 1:
            columns.append(str(column))
    return columns


def _group_metric_rows(df: pd.DataFrame, dimension_col: str, metric_col: str, *, aggregation: str = "mean") -> list[dict[str, Any]]:
    working = df[[dimension_col, metric_col]].copy()
    working["_value"] = pd.to_numeric(working[metric_col], errors="coerce")
    grouped_raw = working.dropna(subset=[dimension_col, "_value"]).groupby(dimension_col, observed=True)["_value"]
    if aggregation == "sum":
        grouped = grouped_raw.sum()
    elif aggregation == "median":
        grouped = grouped_raw.median()
    elif aggregation == "count":
        grouped = grouped_raw.count()
    else:
        grouped = grouped_raw.mean()
    grouped = grouped.sort_values(ascending=False)
    return [{dimension_col: str(k), "value": float(v)} for k, v in grouped.head(20).items()]


def _count_rows(df: pd.DataFrame, dimension_col: str) -> list[dict[str, Any]]:
    counts = df[dimension_col].dropna().astype(str).value_counts().head(20)
    return [{dimension_col: str(k), "count": int(v)} for k, v in counts.items()]


def _scoped_chart(
    entry: InvestigationDatasetEntry,
    chart_type: str,
    title: str,
    x: str,
    y: str,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    content = {"chart_type": chart_type, "title": title, "x": x, "y": y, "rows": rows}
    content.update({k: v for k, v in metadata.items() if k in {"metric", "dimension", "time_axis"}})
    return {
        "artifact_type": "chart",
        "title": title,
        "content": content,
        "visibility": "user",
        "pinned": True,
        "metadata": {"dataset_id": entry.dataset_id, "dataset_ids": [entry.dataset_id], "dataset_name": entry.display_name, **metadata},
    }


def _findings_from_visuals(
    entries: list[InvestigationDatasetEntry],
    relationship: dict[str, Any],
) -> list[str]:
    """Generate findings naturally from heatmap/entity matrix analysis."""
    findings: list[str] = []
    join_keys = relationship.get("joinable_shared_columns") or []
    exact_shared = relationship.get("exact_shared_columns") or []
    total_cols = max(sum(len(e.column_names) for e in entries), 1)
    schema_overlap = len(exact_shared) / max(total_cols / len(entries), 1) if entries else 0.0

    # Entity coverage findings
    all_concepts: set[str] = set()
    per_dataset: dict[str, set[str]] = {}
    for entry in entries:
        entry_concepts = set(entry.semantic_profile.get("concepts", []))
        all_concepts.update(entry_concepts)
        per_dataset[entry.display_name] = entry_concepts
    missing_customer = [name for name, concepts in per_dataset.items() if "customer_entity" not in concepts]
    if missing_customer and "customer_entity" in all_concepts:
        findings.append(f"Customer entity is missing from `{'`, `'.join(missing_customer)}`, which blocks customer-level attribution across datasets.")
    if "monetary_value" in all_concepts:
        missing_financial = [name for name, concepts in per_dataset.items() if "monetary_value" not in concepts]
        if missing_financial:
            findings.append(f"`{'`, `'.join(missing_financial)}` lacks financial metrics, limiting cross-dataset revenue analysis.")

    # Compatibility findings
    if schema_overlap < 0.1 and not join_keys:
        findings.append("Low schema overlap combined with no joinable keys means these datasets require separate analytical treatment or a purpose-built bridge.")
    elif schema_overlap > 0.3 and join_keys:
        findings.append("Moderate schema overlap with joinable keys suggests direct analytical integration may be feasible after key validation.")

    # Granularity mismatch
    row_counts = [(entry.display_name, entry.row_count) for entry in entries if entry.row_count > 0]
    if len(row_counts) >= 2:
        sorted_counts = sorted(row_counts, key=lambda x: x[1])
        if sorted_counts[-1][1] > sorted_counts[0][1] * 10:
            findings.append(f"Granularity mismatch: `{sorted_counts[-1][0]}` has {sorted_counts[-1][1]:,} rows while `{sorted_counts[0][0]}` has {sorted_counts[0][1]:,}, suggesting different observation levels.")

    return findings


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _compatibility_heatmap(entries: list[InvestigationDatasetEntry], relationship: dict[str, Any]) -> dict[str, Any]:
    """Generate a compatibility heatmap between datasets."""
    dataset_names = [entry.display_name for entry in entries]
    dimensions = ["Schema Overlap", "Entity Overlap", "Analytical Compatibility", "Joinability"]
    join_keys = relationship.get("joinable_shared_columns") or []
    concept_matches = relationship.get("concept_matches") or []
    shared_roles = relationship.get("shared_roles") or []
    shared_concepts = relationship.get("shared_concepts") or []
    exact_shared = relationship.get("exact_shared_columns") or []
    total_cols = max(sum(len(e.column_names) for e in entries), 1)
    schema_score = min(len(exact_shared) / max(total_cols / len(entries), 1), 1.0) if entries else 0.0
    entity_score = min(len(concept_matches) / max(len(entries), 1), 1.0)
    compat_score = min((len(shared_roles) + len(shared_concepts)) / 6.0, 1.0)
    join_score = 1.0 if join_keys else (0.5 if concept_matches else 0.1)
    scores = {"Schema Overlap": schema_score, "Entity Overlap": entity_score, "Analytical Compatibility": compat_score, "Joinability": join_score}
    rows: list[dict[str, Any]] = []
    for dim in dimensions:
        for name in dataset_names:
            rows.append({"row": dim, "column": name, "value": round(scores[dim], 2)})
    return {
        "artifact_type": "chart",
        "title": "Dataset Compatibility Assessment",
        "visibility": "user",
        "content": {
            "chart_type": "heatmap",
            "title": "Dataset Compatibility Assessment",
            "x": "column",
            "y": "row",
            "rows": rows,
        },
    }


_ENTITY_CONCEPT_MAP: dict[str, str] = {
    "Customer": "customer_entity",
    "Geography": "geography",
    "Product": "product_or_category",
    "Revenue": "monetary_value",
    "Campaign": "campaign_response",
    "Time": "time",
    "Operations": "operational_process",
    "Inventory": "inventory",
    "Education": "education_level",
    "Household": "household_structure",
}


def _entity_coverage_heatmap(entries: list[InvestigationDatasetEntry]) -> dict[str, Any]:
    """Generate a shared entity matrix showing which concepts exist in which datasets."""
    all_concepts: set[str] = set()
    for entry in entries:
        all_concepts.update(entry.semantic_profile.get("concepts", []))
    entity_labels: list[str] = []
    entity_concepts: list[str] = []
    for label, concept in _ENTITY_CONCEPT_MAP.items():
        if concept in all_concepts:
            entity_labels.append(label)
            entity_concepts.append(concept)
    if not entity_labels:
        entity_labels = ["General"]
        entity_concepts = [""]
    rows: list[dict[str, Any]] = []
    for label, concept in zip(entity_labels, entity_concepts):
        for entry in entries:
            entry_concepts = set(entry.semantic_profile.get("concepts", []))
            value = 1.0 if concept in entry_concepts else 0.0
            rows.append({"row": label, "column": entry.display_name, "value": value})
    return {
        "artifact_type": "chart",
        "title": "Shared Entity Coverage",
        "visibility": "user",
        "content": {
            "chart_type": "heatmap",
            "title": "Shared Entity Coverage",
            "x": "column",
            "y": "row",
            "rows": rows,
        },
    }


def _joinability_bar_chart(entries: list[InvestigationDatasetEntry], relationship: dict[str, Any]) -> dict[str, Any] | None:
    """Generate a joinability confidence bar chart for dataset pairs."""
    if len(entries) < 2:
        return None
    join_keys = relationship.get("joinable_shared_columns") or []
    concept_matches = relationship.get("concept_matches") or []
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(entries):
        for right in entries[i + 1:]:
            pair_label = f"{left.display_name} ↔ {right.display_name}"
            pair_concepts = [m for m in concept_matches if (m["left_dataset_id"] == left.dataset_id and m["right_dataset_id"] == right.dataset_id) or (m["left_dataset_id"] == right.dataset_id and m["right_dataset_id"] == left.dataset_id)]
            if join_keys:
                confidence = 0.85
            elif pair_concepts:
                confidence = round(max(float(m.get("confidence", 0.5)) for m in pair_concepts), 2)
            else:
                confidence = 0.1
            rows.append({"pair": pair_label, "confidence": confidence})
    if not rows:
        return None
    return {
        "artifact_type": "chart",
        "title": "Joinability Confidence",
        "visibility": "user",
        "content": {
            "chart_type": "bar",
            "title": "Joinability Confidence",
            "x": "pair",
            "y": "confidence",
            "rows": rows,
        },
    }
