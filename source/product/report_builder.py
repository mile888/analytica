from __future__ import annotations

from typing import Iterable

from source.product.cross_investigation import CrossInvestigationPattern, report_pattern_hints
from source.product.investigation import (
    Artifact,
    ArtifactType,
    ArtifactVisibility,
    Finding,
    FindingStatus,
    Investigation,
    InvestigationMessage,
    ReportSection,
    ShareableReport,
    ShareableReportStatus,
    ShareableReportTemplate,
)
from source.product.report_artifacts import artifact_report_snapshot, ensure_artifact_report_image
from source.product.organizational_workflows import (
    InvestigationReview,
    ReportStandard,
    report_standard_hints,
)


def build_shareable_report(
    investigation: Investigation,
    template: str | ShareableReportTemplate = ShareableReportTemplate.EXECUTIVE_SUMMARY,
    include_technical: bool = False,
    analytical_patterns: list[CrossInvestigationPattern] | None = None,
    organizational_review: InvestigationReview | None = None,
    report_standard: ReportStandard | None = None,
    messages: list[InvestigationMessage] | None = None,
) -> ShareableReport:
    resolved_template = ShareableReportTemplate(template)
    if resolved_template == ShareableReportTemplate.EXECUTIVE_SUMMARY:
        return build_executive_summary_report(
            investigation,
            include_technical=include_technical,
            analytical_patterns=analytical_patterns,
            organizational_review=organizational_review,
            report_standard=report_standard,
            messages=messages,
        )
    if resolved_template == ShareableReportTemplate.PRODUCT_DECISION_MEMO:
        return build_product_decision_memo_report(
            investigation,
            include_technical=include_technical,
            analytical_patterns=analytical_patterns,
            organizational_review=organizational_review,
            report_standard=report_standard,
            messages=messages,
        )
    if resolved_template == ShareableReportTemplate.TECHNICAL_APPENDIX:
        return build_technical_appendix_report(investigation)
    raise ValueError(f"Unsupported report template: {template}")


def build_executive_summary_report(
    investigation: Investigation,
    include_technical: bool = False,
    analytical_patterns: list[CrossInvestigationPattern] | None = None,
    organizational_review: InvestigationReview | None = None,
    report_standard: ReportStandard | None = None,
    messages: list[InvestigationMessage] | None = None,
) -> ShareableReport:
    findings = _selected_findings(investigation)
    artifacts = _selected_artifacts(investigation, include_technical=include_technical)
    history = _history_items(messages or [])
    limitations = _limitations(investigation, findings)
    artifact_snapshots = _artifact_snapshots(artifacts)
    datasets = _dataset_ids(investigation, artifacts)
    branch_ids = _branch_ids(artifacts)
    summary = _investigation_summary(investigation, history, findings, datasets)
    sections = [
        ReportSection(title="Investigation summary", content=summary, order=10, section_type="summary", source_question_ids=_question_ids(history)),
        ReportSection(title="Key findings", content=_insight_report_block(findings), order=20, section_type="key_findings"),
        ReportSection(title="Analytical workflow", content=_workflow_block(history, investigation), order=30, section_type="workflow", source_question_ids=_question_ids(history)),
        ReportSection(title="Complete Q&A transcript", content=_transcript_block(history, investigation), order=35, section_type="transcript", source_question_ids=_question_ids(history)),
        ReportSection(title="Visual analysis", content=_visual_analysis_block(artifact_snapshots), order=40, section_type="visual_analysis", artifact_ids=[item.artifact_id for item in artifacts], source_artifact_ids=[item.artifact_id for item in artifacts], metadata={"artifact_snapshots": artifact_snapshots}),
        ReportSection(title="Evidence", content=_bullets(_artifact_evidence(artifacts)), order=45, section_type="evidence", artifact_ids=[item.artifact_id for item in artifacts], source_artifact_ids=[item.artifact_id for item in artifacts]),
        ReportSection(title="Limitations", content=_bullets(limitations), order=50, section_type="limitations"),
        ReportSection(title="Conclusions", content=_conclusion_block(investigation, findings), order=60, section_type="conclusions"),
        ReportSection(title="Next steps", content=_insight_field_bullets(findings, "recommended_validation"), order=65, section_type="next_steps"),
        ReportSection(title="Evidence standards", content=_bullets(report_standard_hints(report_standard, organizational_review) if report_standard else report_pattern_hints(analytical_patterns or [])), order=70, section_type="evidence_standards"),
    ]
    if analytical_patterns and not report_standard:
        sections.append(ReportSection(title="Common validation checks", content=_bullets(report_pattern_hints(analytical_patterns)), order=75, section_type="validation_checks"))
    return _report(
        investigation=investigation,
        title=f"{investigation.title} — Executive Summary",
        template=ShareableReportTemplate.EXECUTIVE_SUMMARY,
        include_technical=include_technical,
        sections=sections,
        findings=findings,
        artifacts=artifacts,
        dataset_ids=datasets,
        branch_ids=branch_ids,
        included_question_ids=_question_ids(history),
        summary=summary,
        limitations=limitations,
        artifact_snapshots=artifact_snapshots,
        history=history,
    )


def build_product_decision_memo_report(
    investigation: Investigation,
    include_technical: bool = False,
    analytical_patterns: list[CrossInvestigationPattern] | None = None,
    organizational_review: InvestigationReview | None = None,
    report_standard: ReportStandard | None = None,
    messages: list[InvestigationMessage] | None = None,
) -> ShareableReport:
    report = build_executive_summary_report(
        investigation,
        include_technical=include_technical,
        analytical_patterns=analytical_patterns,
        organizational_review=organizational_review,
        report_standard=report_standard,
        messages=messages,
    )
    report.title = f"{investigation.title} — Product Decision Memo"
    report.template = ShareableReportTemplate.PRODUCT_DECISION_MEMO
    return report


def build_technical_appendix_report(investigation: Investigation) -> ShareableReport:
    artifacts = [
        artifact
        for artifact in investigation.artifacts
        if artifact.visibility == ArtifactVisibility.TECHNICAL and artifact.visibility != ArtifactVisibility.HIDDEN
    ]
    artifacts.sort(key=lambda item: (not item.pinned, item.created_at.isoformat(), item.title))
    sections = [
        ReportSection(title="Technical artifacts", content=_artifact_list(artifacts), order=10, artifact_ids=[item.artifact_id for item in artifacts]),
        ReportSection(title="Run trace", content=_bullets(_trace_lines(investigation.trace)), order=20),
    ]
    return _report(
        investigation=investigation,
        title=f"{investigation.title} — Technical Appendix",
        template=ShareableReportTemplate.TECHNICAL_APPENDIX,
        include_technical=True,
        sections=sections,
        findings=[],
        artifacts=artifacts,
    )


def _report(
    investigation: Investigation,
    title: str,
    template: ShareableReportTemplate,
    include_technical: bool,
    sections: list[ReportSection],
    findings: list[Finding],
    artifacts: list[Artifact],
    dataset_ids: list[str] | None = None,
    branch_ids: list[str] | None = None,
    included_question_ids: list[str] | None = None,
    summary: str = "",
    limitations: list[str] | None = None,
    artifact_snapshots: list[dict] | None = None,
    history: list[dict] | None = None,
) -> ShareableReport:
    metadata = {
        "source": "report_builder",
        "dataset_ids": dataset_ids or [],
        "branch_ids": branch_ids or [],
        "included_question_ids": included_question_ids or [],
        "summary": summary,
        "limitations": limitations or [],
        "included_artifacts": artifact_snapshots or [],
        "investigation_history": history or [],
        "included_artifact_ids": [item.artifact_id for item in artifacts],
        "included_finding_ids": [item.finding_id for item in findings],
    }
    return ShareableReport(
        investigation_id=investigation.investigation_id,
        title=title,
        template=template,
        status=ShareableReportStatus.DRAFT,
        version=1,
        sections=[section for section in sections if section.content or section.artifact_ids],
        dataset_ids=dataset_ids or [],
        branch_ids=branch_ids or [],
        included_question_ids=included_question_ids or [],
        summary=summary,
        limitations=limitations or [],
        source_finding_ids=[item.finding_id for item in findings],
        source_artifact_ids=[item.artifact_id for item in artifacts],
        include_technical=include_technical,
        metadata=metadata,
    )


def _selected_findings(investigation: Investigation) -> list[Finding]:
    candidates = [item for item in investigation.findings if _is_high_confidence_report_finding(item)]
    accepted = [item for item in candidates if item.status == FindingStatus.ACCEPTED]
    if accepted:
        return _dedupe_findings(sorted(accepted, key=lambda item: item.created_at.isoformat()))[-5:]
    proposed = [item for item in candidates if item.status == FindingStatus.PROPOSED]
    return _dedupe_findings(sorted(proposed, key=lambda item: item.created_at.isoformat()))[-5:]


def _history_items(messages: list[InvestigationMessage]) -> list[dict]:
    items = []
    for message in sorted(messages, key=lambda item: item.created_at.isoformat()):
        role = getattr(message.role, "value", message.role)
        mtype = getattr(message.message_type, "value", message.message_type)
        if role not in {"user", "assistant"}:
            continue
        text = " ".join(str(message.content or "").split())
        if not text:
            continue
        items.append(
            {
                "message_id": message.message_id,
                "run_id": message.run_id,
                "role": role,
                "type": mtype,
                "content": text,
                "created_at": message.created_at.isoformat(),
                "metadata": message.metadata if isinstance(message.metadata, dict) else {},
            }
        )
    return items


def _question_ids(history: list[dict]) -> list[str]:
    return [str(item.get("message_id")) for item in history if item.get("role") == "user" and item.get("message_id")]


def _dataset_ids(investigation: Investigation, artifacts: list[Artifact]) -> list[str]:
    ids = list(getattr(investigation, "linked_data_source_ids", []) or [])
    for artifact in artifacts:
        metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
        ids.extend(str(item) for item in metadata.get("dataset_ids", []) if str(item).strip())
        if metadata.get("dataset_id"):
            ids.append(str(metadata.get("dataset_id")))
    return list(dict.fromkeys(ids))


def _branch_ids(artifacts: list[Artifact]) -> list[str]:
    ids = []
    for artifact in artifacts:
        metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
        branch_id = str(metadata.get("branch_id") or "")
        if branch_id:
            ids.append(branch_id)
    return list(dict.fromkeys(ids))


def _artifact_snapshots(artifacts: list[Artifact]) -> list[dict]:
    snapshots = []
    for artifact in artifacts:
        image_path = ensure_artifact_report_image(artifact)
        if image_path:
            artifact.metadata = dict(artifact.metadata or {})
            artifact.metadata.setdefault("image_path", image_path)
            artifact.metadata.setdefault("image_bytes_reference", image_path)
        snapshots.append(artifact_report_snapshot(artifact, image_path=image_path))
    return snapshots


def _investigation_summary(investigation: Investigation, history: list[dict], findings: list[Finding], dataset_ids: list[str]) -> str:
    goals = [item["content"] for item in history if item.get("role") == "user"] or [investigation.user_question]
    outcome = _metadata_text(findings[-1].metadata, "conclusion") if findings else ""
    if not outcome and investigation.report:
        outcome = investigation.report.summary or investigation.report.answer
    return (
        f"This report summarizes investigation `{investigation.title}` across {len(dataset_ids) or len(investigation.linked_data_source_ids)} dataset(s). "
        f"The analysis chronology covered {len(goals)} user question(s), starting with: {goals[0]}. "
        + (f"High-level outcome: {outcome}" if outcome else "The report preserves the investigation history, findings, visuals, and limitations.")
    )


def _workflow_block(history: list[dict], investigation: Investigation) -> str:
    user_items = [item for item in history if item.get("role") == "user"]
    assistant_by_run = {item.get("run_id"): item for item in history if item.get("role") == "assistant" and item.get("run_id")}
    lines = []
    for index, item in enumerate(user_items, start=1):
        answer = assistant_by_run.get(item.get("run_id"))
        answer_text = str((answer or {}).get("content") or "").strip()
        suffix = f" Result: {_short(answer_text, 180)}" if answer_text else ""
        lines.append(f"{index}. {item['content']}{suffix}")
    if not lines:
        lines.append(f"1. {investigation.user_question}")
    return "\n".join(lines)


def _transcript_block(history: list[dict], investigation: Investigation) -> str:
    if not history:
        return f"1. Question: {investigation.user_question}"
    turns: list[dict[str, str]] = []
    pending: dict[str, str] | None = None
    for item in history:
        role = item.get("role")
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            pending = {"question": content, "answer": ""}
            turns.append(pending)
        elif role == "assistant":
            if pending is None:
                pending = {"question": "", "answer": content}
                turns.append(pending)
            else:
                pending["answer"] = content
                pending = None
    lines = []
    for index, turn in enumerate(turns, start=1):
        block = [f"{index}. Question: {turn.get('question', '')}"]
        if turn.get("answer"):
            block.append(f"Answer: {turn['answer']}")
        lines.append("\n".join(block))
    return "\n\n".join(lines)


def _visual_analysis_block(snapshots: list[dict]) -> str:
    if not snapshots:
        return "No chart or table artifacts were selected for this report."
    chart_count = sum(1 for item in snapshots if item.get("artifact_type") == "chart")
    table_count = sum(1 for item in snapshots if item.get("artifact_type") == "table")
    parts = []
    if chart_count:
        parts.append(f"{chart_count} chart{'s' if chart_count != 1 else ''}")
    if table_count:
        parts.append(f"{table_count} table{'s' if table_count != 1 else ''}")
    return f"Selected report artifacts are rendered below: {', '.join(parts) or str(len(snapshots)) + ' artifacts'}."


def _limitations(investigation: Investigation, findings: list[Finding]) -> list[str]:
    values = []
    if investigation.report:
        values.extend(investigation.report.limitations)
    for finding in findings:
        limitation = _metadata_text(finding.metadata or {}, "limitation")
        if limitation:
            values.append(limitation)
    for artifact in investigation.artifacts:
        metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
        for key in ("limitation", "limitations", "warning"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
            elif isinstance(value, list):
                values.extend(str(item) for item in value if str(item).strip())
    return list(dict.fromkeys(values)) or ["Interpretation depends on the available uploaded data, selected artifacts, and computed investigation outputs."]


def _conclusion_block(investigation: Investigation, findings: list[Finding]) -> str:
    if findings:
        conclusions = [_metadata_text(item.metadata or {}, "conclusion") or item.text for item in findings[-3:]]
        return " ".join(item for item in conclusions if item)
    if investigation.report:
        return investigation.report.summary or investigation.report.answer
    return "No final analytical conclusion has been accepted yet; use the investigation chronology and visual evidence as the current investigation record."


def _short(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _is_high_confidence_report_finding(finding: Finding) -> bool:
    metadata = finding.metadata or {}
    confidence = str(metadata.get("confidence_level") or metadata.get("confidence") or "").strip().lower()
    if not confidence and finding.confidence is not None:
        confidence = "high" if finding.confidence >= 0.8 else "medium" if finding.confidence >= 0.55 else "low"
    if confidence != "high":
        return False
    analysis_type = str(metadata.get("analysis_type") or "").strip().lower()
    if analysis_type in {
        "clarification_needed",
        "execution_context_unavailable",
        "fallback",
        "error",
        "non_analytical",
        "validation",
        "limitation",
        "profile",
        "overview",
        "suggestion",
    }:
        return False
    text = " ".join(
        str(part or "")
        for part in (
            finding.title,
            finding.text,
            metadata.get("conclusion"),
        )
    ).lower()
    forbidden_markers = (
        "i cannot explain this chart",
        "exact artifact",
        "not available in the current artifact payload",
        "no active metric/dimension context",
        "execution context is unavailable",
        "raw rows are not attached",
        "this answer uses the latest saved analytical result",
        "needs validation confidence",
        "could not complete",
        "could not compute",
        "no written analytical answer",
    )
    if any(marker in text for marker in forbidden_markers):
        return False
    substantive_markers = (
        " led by ",
        " leads ",
        " ranks ",
        " distribution ",
        " compares ",
        " comparison ",
        " median ",
        " mean ",
        " range ",
        " highest ",
        " lowest ",
        " contributes ",
        " created ",
        " tested against ",
        " correlation ",
        " total ",
        " average ",
        " varies ",
        " outlier",
        " missing ",
        " duplicates",
    )
    return any(marker in text for marker in substantive_markers)


def _selected_artifacts(investigation: Investigation, include_technical: bool) -> list[Artifact]:
    artifacts = [
        artifact
        for artifact in investigation.artifacts
        if artifact.visibility != ArtifactVisibility.HIDDEN
        and artifact.artifact_type in {ArtifactType.CHART, ArtifactType.TABLE}
        and (
            artifact.visibility == ArtifactVisibility.USER
            or (include_technical and artifact.visibility == ArtifactVisibility.TECHNICAL)
        )
    ]
    artifacts.sort(key=lambda item: (not item.pinned, item.created_at.isoformat(), item.title))
    selected = [artifact for artifact in artifacts if _selected_for_report(artifact)]
    if selected:
        return selected[-12:]
    return artifacts[-8:]


def _selected_for_report(artifact: Artifact) -> bool:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    return bool(metadata.get("selected_for_report") or metadata.get("use_in_report"))


def _artifact_list(artifacts: list[Artifact]) -> str:
    if not artifacts:
        return ""
    return _bullets(
        [
            f"{artifact.title or 'Artifact'} ({artifact.artifact_type.value}, {artifact.visibility.value})"
            for artifact in artifacts
        ]
    )


def _artifact_evidence(artifacts: list[Artifact]) -> list[str]:
    return [f"{artifact.title}" for artifact in artifacts if artifact.artifact_type in {ArtifactType.TABLE, ArtifactType.CHART}]


def _insight_report_block(findings: list[Finding]) -> str:
    blocks = [_insight_report_item(item) for item in findings]
    return "\n\n".join(block for block in blocks if block)


def _insight_report_item(finding: Finding) -> str:
    metadata = finding.metadata or {}
    conclusion = _metadata_text(metadata, "conclusion") or finding.text
    if not conclusion:
        return ""
    lines = [f"- {conclusion}"]
    evidence_reason = _metadata_text(metadata, "evidence_reason")
    business_implication = _metadata_text(metadata, "business_implication")
    limitation = _metadata_text(metadata, "limitation")
    recommended_validation = _metadata_text(metadata, "recommended_validation")
    uncertainty_notes = _metadata_list(metadata, "uncertainty_notes")
    if evidence_reason and not _is_generic_report_note(evidence_reason):
        lines.append(f"  Evidence: {evidence_reason}")
    if business_implication and not _is_generic_report_note(business_implication):
        lines.append(f"  Implication: {business_implication}")
    if limitation and not _is_generic_report_note(limitation):
        lines.append(f"  Limitation: {limitation}")
    elif uncertainty_notes and not _is_generic_report_note(uncertainty_notes[0]):
        lines.append(f"  Uncertainty: {uncertainty_notes[0]}")
    if recommended_validation and not _is_generic_report_note(recommended_validation):
        lines.append(f"  Validation: {recommended_validation}")
    return "\n".join(lines)


def _insight_field_bullets(findings: list[Finding], field: str) -> str:
    values = [
        value
        for value in (_metadata_text(item.metadata or {}, field) for item in findings)
        if value and not _is_generic_report_note(value)
    ]
    return _bullets(_dedupe_text(values)[:5])


def _metadata_text(metadata: dict, key: str) -> str:
    value = metadata.get(key)
    return str(value).strip() if isinstance(value, str) and value.strip() else ""


def _metadata_list(metadata: dict, key: str) -> list[str]:
    value = metadata.get(key)
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _trace_lines(trace: Iterable[dict]) -> list[str]:
    lines = []
    for item in trace:
        tool = item.get("tool") or item.get("stage") or "trace"
        status = item.get("status") or item.get("event") or ""
        lines.append(f"{tool}: {status}".strip(": "))
    return lines


def _bullets(items: Iterable[str]) -> str:
    values = _dedupe_text(str(item).strip() for item in items if str(item).strip())
    return "\n".join(f"- {item}" for item in values)


def _dedupe_findings(findings: list[Finding]) -> list[Finding]:
    deduped: list[Finding] = []
    seen: set[str] = set()
    for finding in findings:
        metadata = finding.metadata or {}
        key = " ".join(
            str(part or "")
            for part in (
                metadata.get("conclusion"),
                finding.text,
                metadata.get("analysis_type"),
            )
        )
        normalized = " ".join(key.lower().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(finding)
    return deduped


def _dedupe_text(items: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = " ".join(str(item).strip().lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(str(item).strip())
    return deduped


def _is_generic_report_note(value: str) -> bool:
    text = value.lower()
    generic_markers = (
        "attach supporting evidence",
        "validated with supporting evidence",
        "move into the report",
        "focus the next analytical step",
        "latest saved analytical result",
        "needs validation",
        "before being treated as final",
    )
    return any(marker in text for marker in generic_markers)
