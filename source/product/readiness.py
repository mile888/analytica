from __future__ import annotations

from source.product.investigation import (
    ArtifactType,
    ArtifactVisibility,
    FindingStatus,
    Investigation,
    ReadinessCheck,
    ReadinessCheckStatus,
    ReadinessSeverity,
    ReportApprovalStatus,
    ReportComment,
    ReportCommentStatus,
    ReportReadinessResult,
    SectionReviewStatus,
    ShareableReport,
    ShareableReportTemplate,
)


def evaluate_report_readiness(
    report: ShareableReport,
    investigation: Investigation | None = None,
    comments: list[ReportComment] | None = None,
) -> ReportReadinessResult:
    checks: list[ReadinessCheck] = []
    comments = comments or []

    open_comments = [comment for comment in comments if comment.status == ReportCommentStatus.OPEN]
    _add(
        checks,
        name="open_comments",
        passed=not open_comments,
        failed_message=f"{len(open_comments)} open review comment(s) must be resolved.",
        passed_message="No open review comments.",
        metadata={"open_comment_count": len(open_comments)},
    )

    _add(
        checks,
        name="sections_present",
        passed=bool(report.sections),
        failed_message="Report has no sections.",
        passed_message="Report has sections.",
    )

    empty_sections = [section for section in report.sections if not section.content.strip()]
    _add(
        checks,
        name="section_content",
        passed=not empty_sections,
        failed_message=f"{len(empty_sections)} section(s) have empty content.",
        passed_message="All sections have content.",
        metadata={"section_ids": [section.section_id for section in empty_sections]},
    )

    change_sections = [
        section for section in report.sections if section.review_status == SectionReviewStatus.CHANGES_REQUESTED
    ]
    _add(
        checks,
        name="section_review_status",
        passed=not change_sections,
        failed_message=f"{len(change_sections)} section(s) have requested changes.",
        passed_message="No sections have requested changes.",
        metadata={"section_ids": [section.section_id for section in change_sections]},
    )

    _add(
        checks,
        name="report_approval_status",
        passed=report.approval_status != ReportApprovalStatus.CHANGES_REQUESTED,
        failed_message="Report has requested changes.",
        passed_message="Report is not marked as changes requested.",
    )

    if investigation is not None:
        accepted = [finding for finding in investigation.findings if finding.status == FindingStatus.ACCEPTED]
        _warn(
            checks,
            name="accepted_findings",
            passed=bool(accepted),
            warning_message="No accepted findings are linked to the source Investigation.",
            passed_message="Source Investigation has accepted findings.",
        )

    _warn(
        checks,
        name="evidence_artifacts",
        passed=_has_valid_evidence_artifact(report, investigation),
        warning_message="No user-valid evidence artifacts are linked.",
        passed_message="Report has valid evidence artifacts.",
    )

    approved_sections = [section for section in report.sections if section.review_status == SectionReviewStatus.APPROVED]
    _warn(
        checks,
        name="approved_sections",
        passed=bool(approved_sections),
        warning_message="No report sections are approved yet.",
        passed_message="At least one section is approved.",
    )

    _warn(
        checks,
        name="review_state",
        passed=report.approval_status in {ReportApprovalStatus.IN_REVIEW, ReportApprovalStatus.APPROVED},
        warning_message="Report is not in review or approved.",
        passed_message="Report is in review or approved.",
    )

    business_template = report.template != ShareableReportTemplate.TECHNICAL_APPENDIX
    _warn(
        checks,
        name="technical_appendix_in_business_report",
        passed=not (business_template and report.include_technical),
        warning_message="Technical appendix is included in a business-facing report template.",
        passed_message="Technical appendix setting matches report template.",
    )

    blocking_count = sum(1 for check in checks if check.blocking and check.status == ReadinessCheckStatus.FAILED)
    warning_count = sum(1 for check in checks if check.status == ReadinessCheckStatus.WARNING)
    return ReportReadinessResult(
        report_id=report.report_id,
        is_ready=blocking_count == 0,
        blocking_count=blocking_count,
        warning_count=warning_count,
        checks=checks,
    )


def _has_valid_evidence_artifact(report: ShareableReport, investigation: Investigation | None) -> bool:
    if not report.source_artifact_ids:
        return False
    if investigation is None:
        return bool(report.source_artifact_ids)

    by_id = {artifact.artifact_id: artifact for artifact in investigation.artifacts}
    for artifact_id in report.source_artifact_ids:
        artifact = by_id.get(artifact_id)
        if artifact is None or artifact.visibility == ArtifactVisibility.HIDDEN:
            continue
        if report.template == ShareableReportTemplate.TECHNICAL_APPENDIX:
            if artifact.visibility in {ArtifactVisibility.USER, ArtifactVisibility.TECHNICAL}:
                return True
        elif artifact.visibility == ArtifactVisibility.USER and artifact.artifact_type != ArtifactType.PYTHON_CODE:
            return True
    return False


def _add(
    checks: list[ReadinessCheck],
    name: str,
    passed: bool,
    failed_message: str,
    passed_message: str,
    metadata: dict | None = None,
) -> None:
    checks.append(
        ReadinessCheck(
            name=name,
            status=ReadinessCheckStatus.PASSED if passed else ReadinessCheckStatus.FAILED,
            severity=ReadinessSeverity.INFO if passed else ReadinessSeverity.ERROR,
            message=passed_message if passed else failed_message,
            blocking=not passed,
            metadata=metadata or {},
        )
    )


def _warn(
    checks: list[ReadinessCheck],
    name: str,
    passed: bool,
    warning_message: str,
    passed_message: str,
    metadata: dict | None = None,
) -> None:
    checks.append(
        ReadinessCheck(
            name=name,
            status=ReadinessCheckStatus.PASSED if passed else ReadinessCheckStatus.WARNING,
            severity=ReadinessSeverity.INFO if passed else ReadinessSeverity.WARNING,
            message=passed_message if passed else warning_message,
            blocking=False,
            metadata=metadata or {},
        )
    )
