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
    ReportSection,
    ShareableReport,
    ShareableReportStatus,
    ShareableReportTemplate,
)
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
) -> ShareableReport:
    resolved_template = ShareableReportTemplate(template)
    if resolved_template == ShareableReportTemplate.EXECUTIVE_SUMMARY:
        return build_executive_summary_report(
            investigation,
            include_technical=include_technical,
            analytical_patterns=analytical_patterns,
            organizational_review=organizational_review,
            report_standard=report_standard,
        )
    if resolved_template == ShareableReportTemplate.PRODUCT_DECISION_MEMO:
        return build_product_decision_memo_report(
            investigation,
            include_technical=include_technical,
            analytical_patterns=analytical_patterns,
            organizational_review=organizational_review,
            report_standard=report_standard,
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
) -> ShareableReport:
    findings = _selected_findings(investigation)
    artifacts = _selected_artifacts(investigation, include_technical=include_technical)
    report = investigation.report
    sections = [
        ReportSection(title="Question", content=investigation.user_question, order=10),
        ReportSection(title="Answer", content=(report.answer or report.summary) if report else "", order=20),
        ReportSection(title="Key findings", content=_insight_report_block(findings), order=30),
        ReportSection(title="Evidence", content=_bullets(_artifact_evidence(artifacts)), order=40, artifact_ids=[item.artifact_id for item in artifacts]),
        ReportSection(title="Limitations", content=_insight_field_bullets(findings, "limitation"), order=50),
        ReportSection(title="Next steps", content=_insight_field_bullets(findings, "recommended_validation"), order=60),
        ReportSection(title="Common validation checks", content=_bullets(report_pattern_hints(analytical_patterns or [])), order=65),
        ReportSection(title="Evidence standards", content=_bullets(report_standard_hints(report_standard, organizational_review) if report_standard else []), order=66),
    ]
    return _report(
        investigation=investigation,
        title=f"{investigation.title} — Executive Summary",
        template=ShareableReportTemplate.EXECUTIVE_SUMMARY,
        include_technical=include_technical,
        sections=sections,
        findings=findings,
        artifacts=artifacts,
    )


def build_product_decision_memo_report(
    investigation: Investigation,
    include_technical: bool = False,
    analytical_patterns: list[CrossInvestigationPattern] | None = None,
    organizational_review: InvestigationReview | None = None,
    report_standard: ReportStandard | None = None,
) -> ShareableReport:
    findings = _selected_findings(investigation)
    artifacts = _selected_artifacts(investigation, include_technical=include_technical)
    report = investigation.report
    recommendation = (report.answer or report.summary) if report else "No recommendation has been generated yet."
    sections = [
        ReportSection(title="Context", content=f"Investigation status: {investigation.status.value}", order=10),
        ReportSection(title="Decision question", content=investigation.user_question, order=20),
        ReportSection(title="Recommendation", content=recommendation, order=30),
        ReportSection(title="Supporting evidence", content=_insight_report_block(findings), order=40, artifact_ids=[item.artifact_id for item in artifacts]),
        ReportSection(title="Risks / limitations", content=_insight_field_bullets(findings, "limitation"), order=50),
        ReportSection(title="Next steps", content=_insight_field_bullets(findings, "recommended_validation"), order=60),
        ReportSection(title="Reusable validation checks", content=_bullets(report_pattern_hints(analytical_patterns or [])), order=65),
        ReportSection(title="Evidence standards", content=_bullets(report_standard_hints(report_standard, organizational_review) if report_standard else []), order=66),
    ]
    return _report(
        investigation=investigation,
        title=f"{investigation.title} — Product Decision Memo",
        template=ShareableReportTemplate.PRODUCT_DECISION_MEMO,
        include_technical=include_technical,
        sections=sections,
        findings=findings,
        artifacts=artifacts,
    )


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
) -> ShareableReport:
    return ShareableReport(
        investigation_id=investigation.investigation_id,
        title=title,
        template=template,
        status=ShareableReportStatus.DRAFT,
        version=1,
        sections=[section for section in sections if section.content or section.artifact_ids],
        source_finding_ids=[item.finding_id for item in findings],
        source_artifact_ids=[item.artifact_id for item in artifacts],
        include_technical=include_technical,
        metadata={"source": "report_builder"},
    )


def _selected_findings(investigation: Investigation) -> list[Finding]:
    candidates = [item for item in investigation.findings if _is_high_confidence_report_finding(item)]
    accepted = [item for item in candidates if item.status == FindingStatus.ACCEPTED]
    if accepted:
        return _dedupe_findings(sorted(accepted, key=lambda item: item.created_at.isoformat()))[-5:]
    proposed = [item for item in candidates if item.status == FindingStatus.PROPOSED]
    return _dedupe_findings(sorted(proposed, key=lambda item: item.created_at.isoformat()))[-5:]


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
