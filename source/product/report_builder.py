from __future__ import annotations

from typing import Iterable

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


def build_shareable_report(
    investigation: Investigation,
    template: str | ShareableReportTemplate = ShareableReportTemplate.EXECUTIVE_SUMMARY,
    include_technical: bool = False,
) -> ShareableReport:
    resolved_template = ShareableReportTemplate(template)
    if resolved_template == ShareableReportTemplate.EXECUTIVE_SUMMARY:
        return build_executive_summary_report(investigation, include_technical=include_technical)
    if resolved_template == ShareableReportTemplate.PRODUCT_DECISION_MEMO:
        return build_product_decision_memo_report(investigation, include_technical=include_technical)
    if resolved_template == ShareableReportTemplate.TECHNICAL_APPENDIX:
        return build_technical_appendix_report(investigation)
    raise ValueError(f"Unsupported report template: {template}")


def build_executive_summary_report(
    investigation: Investigation,
    include_technical: bool = False,
) -> ShareableReport:
    findings = _selected_findings(investigation)
    artifacts = _selected_artifacts(investigation, include_technical=include_technical)
    report = investigation.report
    sections = [
        ReportSection(title="Question", content=investigation.user_question, order=10),
        ReportSection(title="Answer", content=(report.answer or report.summary) if report else "", order=20),
        ReportSection(title="Key findings", content=_bullets([item.text for item in findings] or (report.key_findings if report else [])), order=30),
        ReportSection(title="Evidence", content=_bullets((report.evidence if report else []) + _artifact_evidence(artifacts)), order=40, artifact_ids=[item.artifact_id for item in artifacts]),
        ReportSection(title="Limitations", content=_bullets(report.limitations if report else []), order=50),
        ReportSection(title="Next steps", content=_bullets(report.next_steps if report else []), order=60),
        ReportSection(title="Selected artifacts", content=_artifact_list(artifacts), order=70, artifact_ids=[item.artifact_id for item in artifacts]),
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
) -> ShareableReport:
    findings = _selected_findings(investigation)
    artifacts = _selected_artifacts(investigation, include_technical=include_technical)
    report = investigation.report
    recommendation = (report.answer or report.summary) if report else "No recommendation has been generated yet."
    sections = [
        ReportSection(title="Context", content=f"Investigation status: {investigation.status.value}", order=10),
        ReportSection(title="Decision question", content=investigation.user_question, order=20),
        ReportSection(title="Recommendation", content=recommendation, order=30),
        ReportSection(title="Supporting evidence", content=_bullets([item.text for item in findings] + (report.evidence if report else [])), order=40, artifact_ids=[item.artifact_id for item in artifacts]),
        ReportSection(title="Risks / limitations", content=_bullets(report.limitations if report else []), order=50),
        ReportSection(title="Next steps", content=_bullets(report.next_steps if report else []), order=60),
        ReportSection(title="Artifacts", content=_artifact_list(artifacts), order=70, artifact_ids=[item.artifact_id for item in artifacts]),
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
    accepted = [item for item in investigation.findings if item.status == FindingStatus.ACCEPTED]
    if accepted:
        return sorted(accepted, key=lambda item: item.created_at.isoformat())
    proposed = [item for item in investigation.findings if item.status == FindingStatus.PROPOSED]
    return sorted(proposed, key=lambda item: item.created_at.isoformat())


def _selected_artifacts(investigation: Investigation, include_technical: bool) -> list[Artifact]:
    artifacts = [
        artifact
        for artifact in investigation.artifacts
        if artifact.visibility != ArtifactVisibility.HIDDEN
        and (
            artifact.visibility == ArtifactVisibility.USER
            or (include_technical and artifact.visibility == ArtifactVisibility.TECHNICAL)
        )
    ]
    artifacts.sort(key=lambda item: (not item.pinned, item.created_at.isoformat(), item.title))
    return artifacts


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
    return [f"Artifact: {artifact.title}" for artifact in artifacts if artifact.artifact_type in {ArtifactType.TABLE, ArtifactType.CHART, ArtifactType.REPORT}]


def _trace_lines(trace: Iterable[dict]) -> list[str]:
    lines = []
    for item in trace:
        tool = item.get("tool") or item.get("stage") or "trace"
        status = item.get("status") or item.get("event") or ""
        lines.append(f"{tool}: {status}".strip(": "))
    return lines


def _bullets(items: Iterable[str]) -> str:
    values = [str(item).strip() for item in items if str(item).strip()]
    return "\n".join(f"- {item}" for item in values)
