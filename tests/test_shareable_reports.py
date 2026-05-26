from __future__ import annotations

from pathlib import Path

from source.product.exporter import (
    export_final_report_snapshot_txt,
    export_shareable_report_html,
    export_shareable_report_markdown,
    export_shareable_report_txt,
)
from source.product.final_report_registry import get_published_report_summary, list_published_reports
from source.product.investigation import (
    Artifact,
    ArtifactType,
    ArtifactVisibility,
    DecisionReport,
    DecisionMetadata,
    DecisionStatus,
    FinalReportStatus,
    Finding,
    FindingStatus,
    Investigation,
    ReportApprovalStatus,
    ReportComment,
    ReportCommentStatus,
    ReadinessCheckStatus,
    ReportSection,
    SectionReviewStatus,
    ShareableReport,
    ShareableReportStatus,
    ShareableReportTemplate,
)
from source.product.report_builder import build_shareable_report
from source.product.readiness import evaluate_report_readiness
from source.product.report_service import ReportEditingService
from source.product.sqlite_store import SQLiteInvestigationStore
from source.product.store import InvestigationStore


def test_build_report_from_empty_investigation_does_not_crash() -> None:
    investigation = Investigation(title="Empty", user_question="What happened?")

    report = build_shareable_report(investigation)

    assert report.investigation_id == investigation.investigation_id
    assert report.sections
    assert report.template == ShareableReportTemplate.EXECUTIVE_SUMMARY


def test_executive_report_uses_accepted_findings_before_proposed_and_excludes_rejected() -> None:
    investigation = Investigation(title="Findings", user_question="Question")
    investigation.findings = [
        Finding(title="Proposed", text="Metric distribution has a high mean.", status=FindingStatus.PROPOSED, metadata={"confidence": "High"}),
        Finding(title="Rejected", text="Rejected text has a high mean.", status=FindingStatus.REJECTED, metadata={"confidence": "High"}),
        Finding(title="Accepted", text="Accepted metric distribution has a high mean.", status=FindingStatus.ACCEPTED, metadata={"confidence": "High"}),
    ]

    report = build_shareable_report(investigation)
    findings_section = next(section for section in report.sections if section.title == "Key findings")

    assert "Accepted metric distribution has a high mean." in findings_section.content
    assert "Metric distribution has a high mean." not in findings_section.content
    assert "Rejected text" not in findings_section.content
    assert report.source_finding_ids == [investigation.findings[2].finding_id]


def test_executive_report_uses_proposed_when_no_accepted_findings() -> None:
    investigation = Investigation(title="Findings", user_question="Question")
    investigation.findings = [Finding(title="Proposed", text="Proposed metric distribution has a high mean.", status=FindingStatus.PROPOSED, metadata={"confidence": "High"})]

    report = build_shareable_report(investigation)
    findings_section = next(section for section in report.sections if section.title == "Key findings")

    assert "Proposed metric distribution has a high mean." in findings_section.content


def test_report_excludes_medium_and_error_findings() -> None:
    investigation = Investigation(title="Findings", user_question="Question")
    investigation.findings = [
        Finding(title="Medium", text="Metric distribution has a high mean.", status=FindingStatus.PROPOSED, metadata={"confidence": "Medium"}),
        Finding(title="Error", text="I cannot explain this chart because the exact artifact is missing.", status=FindingStatus.PROPOSED, metadata={"confidence": "High"}),
        Finding(title="High", text="Salary distribution has a high mean.", status=FindingStatus.PROPOSED, metadata={"confidence": "High"}),
    ]

    report = build_shareable_report(investigation)
    findings_section = next(section for section in report.sections if section.title == "Key findings")

    assert "Salary distribution has a high mean." in findings_section.content
    assert "Metric distribution has a high mean." not in findings_section.content
    assert "exact artifact" not in findings_section.content


def test_report_prefers_artifacts_selected_for_report() -> None:
    investigation = Investigation(title="Selected charts", user_question="Question")
    first = Artifact(
        artifact_type=ArtifactType.CHART,
        title="Unselected chart",
        content={"chart_type": "bar", "rows": [{"label": "A", "value": 1}]},
    )
    selected = Artifact(
        artifact_type=ArtifactType.CHART,
        title="Selected chart",
        content={"chart_type": "bar", "rows": [{"label": "B", "value": 2}]},
        metadata={"selected_for_report": True},
    )
    table = Artifact(
        artifact_type=ArtifactType.TABLE,
        title="Selected table",
        content=[{"Metric": "B", "Value": 2}],
        metadata={"selected_for_report": True},
    )
    investigation.artifacts = [first, selected, table]

    report = build_shareable_report(investigation)

    assert report.source_artifact_ids == [selected.artifact_id, table.artifact_id]
    evidence = next(section for section in report.sections if section.title == "Evidence")
    assert "Selected chart" in evidence.content
    assert "Selected table" in evidence.content
    assert "Unselected chart" not in evidence.content


def test_report_builder_prefers_structured_insight_fields() -> None:
    investigation = Investigation(title="Insight report", user_question="Question")
    investigation.findings = [
        Finding(
            title="Salary variance",
            text="Raw fallback text",
            status=FindingStatus.PROPOSED,
            metadata={
                "conclusion": "Salary_LPA varies strongly across Job_Title.",
                "confidence": "High",
                "evidence_reason": "Supported by grouped table and bar chart.",
                "limitation": "Small groups may distort ranking.",
                "recommended_validation": "Inspect low-sample high-variance groups.",
                "business_implication": "Compensation bands may be inconsistent across roles.",
            },
        )
    ]

    report = build_shareable_report(investigation)
    findings_section = next(section for section in report.sections if section.title == "Key findings")
    limitations_section = next(section for section in report.sections if section.title == "Limitations")
    next_steps_section = next(section for section in report.sections if section.title == "Next steps")
    markdown = export_shareable_report_markdown(report)

    assert "- Salary_LPA varies strongly across Job_Title." in findings_section.content
    assert "Confidence: High" not in findings_section.content
    assert "Evidence: Supported by grouped table and bar chart." in findings_section.content
    assert "Implication: Compensation bands may be inconsistent across roles." in findings_section.content
    assert "Small groups may distort ranking." in limitations_section.content
    assert "Inspect low-sample high-variance groups." in next_steps_section.content
    assert "metadata" not in markdown.lower()


def test_report_artifacts_are_limited_to_user_charts_and_tables() -> None:
    investigation = Investigation(title="Artifacts", user_question="Question")
    investigation.artifacts = [
        Artifact(artifact_type=ArtifactType.TEXT, title="Unpinned", content="B", pinned=False),
        Artifact(artifact_type=ArtifactType.TEXT, title="Pinned text", content="A", pinned=True),
        Artifact(artifact_type=ArtifactType.TABLE, title="Pinned table", content={"rows": []}, pinned=True),
        Artifact(artifact_type=ArtifactType.CHART, title="Chart", content={"chart_type": "bar"}, pinned=False),
    ]

    report = build_shareable_report(investigation)

    assert report.source_artifact_ids == [
        investigation.artifacts[2].artifact_id,
        investigation.artifacts[3].artifact_id,
    ]


def test_hidden_artifacts_are_never_included() -> None:
    investigation = Investigation(title="Artifacts", user_question="Question")
    investigation.artifacts = [
        Artifact(
            artifact_type=ArtifactType.TEXT,
            title="Hidden",
            content="secret",
            visibility=ArtifactVisibility.HIDDEN,
            pinned=True,
        )
    ]

    report = build_shareable_report(investigation, include_technical=True)

    assert report.source_artifact_ids == []
    assert "Hidden" not in export_shareable_report_markdown(report)


def test_technical_code_artifacts_excluded_from_submission_report() -> None:
    investigation = Investigation(title="Artifacts", user_question="Question")
    technical = Artifact(
        artifact_type=ArtifactType.PYTHON_CODE,
        title="Generated Python",
        content="result = df.head()",
        visibility=ArtifactVisibility.TECHNICAL,
    )
    investigation.artifacts = [technical]

    clean = build_shareable_report(investigation)
    with_technical = build_shareable_report(investigation, include_technical=True)

    assert clean.source_artifact_ids == []
    assert with_technical.source_artifact_ids == []


def test_shareable_report_persists_in_sqlite_and_migration_v5_is_applied(tmp_path: Path) -> None:
    store = SQLiteInvestigationStore(tmp_path / "investigations.sqlite")
    investigation = store.create_investigation("Question")
    report = build_shareable_report(investigation)

    saved = store.create_shareable_report(report)
    reloaded_store = SQLiteInvestigationStore(tmp_path / "investigations.sqlite")
    loaded = reloaded_store.get_shareable_report(saved.report_id)

    assert reloaded_store.get_schema_version() == 14
    assert loaded.title == report.title
    assert loaded.sections[0].title == report.sections[0].title
    assert reloaded_store.list_shareable_reports(investigation.investigation_id)[0].report_id == saved.report_id


def test_editing_section_creates_new_version_and_marks_user_edit(tmp_path: Path) -> None:
    store = SQLiteInvestigationStore(tmp_path / "investigations.sqlite")
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    section = report.sections[0]

    edited = ReportEditingService(store).update_section_content(report.report_id, section.section_id, "Edited content")

    assert edited.report_id != report.report_id
    assert edited.previous_version_id == report.report_id
    assert edited.version == 2
    assert edited.sections[0].content == "Edited content"
    assert edited.sections[0].edited_by_user is True
    assert edited.sections[0].version == section.version + 1
    assert store.get_shareable_report(report.report_id).is_latest is False
    assert store.get_latest_report_version(report.report_id).report_id == edited.report_id


def test_restore_previous_version_works(tmp_path: Path) -> None:
    store = SQLiteInvestigationStore(tmp_path / "investigations.sqlite")
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    edited = ReportEditingService(store).update_section_content(report.report_id, report.sections[0].section_id, "Edited")

    restored = store.restore_report_version(report.report_id)

    assert restored.version == 3
    assert restored.previous_version_id == edited.report_id
    assert restored.sections[0].content == report.sections[0].content
    assert store.get_latest_report_version(report.report_id).report_id == restored.report_id


def test_section_reorder_delete_and_duplicate_persist(tmp_path: Path) -> None:
    store = SQLiteInvestigationStore(tmp_path / "investigations.sqlite")
    investigation = store.create_investigation("Question")
    report = build_shareable_report(investigation)
    report.sections.append(ReportSection(title="Extra", content="Extra", order=999))
    saved = store.create_shareable_report(report)
    service = ReportEditingService(store)

    reordered = service.reorder_sections(saved.report_id, [saved.sections[1].section_id, saved.sections[0].section_id])
    assert reordered.sections[0].title == saved.sections[1].title

    duplicated = service.duplicate_section(reordered.report_id, reordered.sections[0].section_id)
    assert len(duplicated.sections) == len(reordered.sections) + 1

    removed = service.remove_section(duplicated.report_id, duplicated.sections[0].section_id)
    reloaded = store.get_shareable_report(removed.report_id)
    assert len(reloaded.sections) == len(duplicated.sections) - 1


def test_report_version_chain_valid_in_memory_store() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    edited = ReportEditingService(store).update_section_title(report.report_id, report.sections[0].section_id, "Edited title")

    versions = store.list_report_versions(report.report_id)

    assert [item.version for item in versions] == [1, 2]
    assert edited.previous_version_id == report.report_id
    assert store.get_latest_report_version(report.report_id).report_id == edited.report_id


def test_archive_shareable_report_persists_in_memory_store() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = build_shareable_report(investigation)
    store.create_shareable_report(report)

    archived = store.archive_shareable_report(report.report_id)

    assert archived.status == ShareableReportStatus.ARCHIVED
    assert store.get_shareable_report(report.report_id).status == ShareableReportStatus.ARCHIVED


def test_export_shareable_report_to_markdown_and_html_works() -> None:
    report = ShareableReport(
        investigation_id="inv_1",
        title="Final report",
        template=ShareableReportTemplate.PRODUCT_DECISION_MEMO,
        sections=[ReportSection(title="Answer", content="Use Standard Class", order=1)],
    )

    markdown = export_shareable_report_markdown(report)
    html = export_shareable_report_html(report)
    txt = export_shareable_report_txt(report)

    assert "# Final report" in markdown
    assert "| Approval | `draft` |" in markdown
    assert "## Answer" in markdown
    assert "<h1>Final report</h1>" in html
    assert "Use Standard Class" in html
    assert "FINAL REPORT" in txt
    assert "ANSWER" in txt


def test_export_shareable_report_html_escapes_unsafe_content() -> None:
    report = ShareableReport(
        investigation_id="inv_1",
        title="<script>alert(1)</script>",
        sections=[ReportSection(title="<b>Bad</b>", content="<img src=x onerror=alert(1)>", order=1)],
    )

    html = export_shareable_report_html(report)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_add_list_and_resolve_comments_in_memory_store() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    service = ReportEditingService(store)

    comment = service.add_comment(report.report_id, report.sections[0].section_id, "Clarify this")
    comments = store.list_report_comments(report.report_id)
    resolved = service.resolve_comment(comment.comment_id)

    assert comments == [comment]
    assert resolved.status == ReportCommentStatus.RESOLVED
    assert resolved.resolved_at is not None


def test_comments_persist_in_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "investigations.sqlite"
    store = SQLiteInvestigationStore(db_path)
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    comment = ReportEditingService(store).add_comment(
        report.report_id,
        report.sections[0].section_id,
        "Review note",
    )

    reloaded = SQLiteInvestigationStore(db_path)
    comments = reloaded.list_report_comments(report.report_id)

    assert comments[0].comment_id == comment.comment_id
    assert comments[0].text == "Review note"


def test_report_approval_lifecycle() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    service = ReportEditingService(store)

    assert report.approval_status == ReportApprovalStatus.DRAFT
    assert service.send_to_review(report.report_id).approval_status == ReportApprovalStatus.IN_REVIEW
    changes = service.request_changes(report.report_id, "Needs tighter evidence")
    assert changes.approval_status == ReportApprovalStatus.CHANGES_REQUESTED
    assert changes.reviewer_notes == "Needs tighter evidence"
    service.send_to_review(report.report_id)
    approved = service.approve_report(report.report_id, approved_by="lead")

    assert approved.approval_status == ReportApprovalStatus.APPROVED
    assert approved.approved_by == "lead"
    assert approved.approved_at is not None


def test_approve_report_with_open_comments_requires_force() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    service = ReportEditingService(store)
    service.add_comment(report.report_id, report.sections[0].section_id, "Open question")

    try:
        service.approve_report(report.report_id)
    except ValueError as exc:
        assert "open review comment" in str(exc)
    else:
        raise AssertionError("Expected approval to fail with open comments")

    approved = service.approve_report(report.report_id, force=True)
    assert approved.approval_status == ReportApprovalStatus.APPROVED


def test_shareable_report_export_comments_are_optional() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    comment = ReportEditingService(store).add_comment(
        report.report_id,
        report.sections[0].section_id,
        "Private review note",
    )

    clean = export_shareable_report_markdown(report)
    with_comments = export_shareable_report_markdown(report, include_comments=True, comments=[comment])

    assert "Private review note" not in clean
    assert "Private review note" in with_comments
    assert "| Approval | `draft` |" in clean


def test_txt_export_works_for_empty_report_and_contains_no_raw_json() -> None:
    report = ShareableReport(investigation_id="inv_1", title="Empty report")

    txt = export_shareable_report_txt(report)

    assert "EMPTY REPORT" in txt
    assert "No sections yet." in txt
    assert "{" not in txt
    assert "}" not in txt


def test_final_snapshot_txt_export_uses_stored_content() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    ReportEditingService(store).approve_report(report.report_id, force=True)
    snapshot = ReportEditingService(store).create_final_report_snapshot(report.report_id, force=True)

    assert snapshot.txt_content
    assert export_final_report_snapshot_txt(snapshot) == snapshot.txt_content


def test_empty_report_fails_readiness() -> None:
    report = ShareableReport(investigation_id="inv_1", title="Empty", sections=[])

    readiness = evaluate_report_readiness(report)

    assert readiness.is_ready is False
    assert any(check.name == "sections_present" and check.status == ReadinessCheckStatus.FAILED for check in readiness.checks)


def test_report_with_open_comments_fails_readiness() -> None:
    report = ShareableReport(
        investigation_id="inv_1",
        title="Report",
        sections=[ReportSection(title="Answer", content="Answer", order=1)],
    )
    readiness = evaluate_report_readiness(
        report,
        comments=[ReportComment(report_id=report.report_id, section_id=report.sections[0].section_id, text="Open")],
    )

    assert readiness.is_ready is False
    assert any(check.name == "open_comments" for check in readiness.checks if check.status == ReadinessCheckStatus.FAILED)


def test_empty_section_and_changes_requested_section_fail_readiness() -> None:
    report = ShareableReport(
        investigation_id="inv_1",
        title="Report",
        sections=[
            ReportSection(title="Empty", content="", order=1),
            ReportSection(
                title="Changes",
                content="Needs work",
                order=2,
                review_status=SectionReviewStatus.CHANGES_REQUESTED,
            ),
        ],
    )

    readiness = evaluate_report_readiness(report)

    assert readiness.is_ready is False
    failed_names = {check.name for check in readiness.checks if check.status == ReadinessCheckStatus.FAILED}
    assert {"section_content", "section_review_status"}.issubset(failed_names)


def test_no_accepted_findings_gives_warning() -> None:
    investigation = Investigation(title="Investigation", user_question="Question")
    investigation.findings = [Finding(text="Proposed", status=FindingStatus.PROPOSED)]
    report = ShareableReport(
        investigation_id=investigation.investigation_id,
        title="Report",
        sections=[ReportSection(title="Answer", content="Answer", order=1)],
    )

    readiness = evaluate_report_readiness(report, investigation=investigation)

    assert readiness.is_ready is True
    assert any(check.name == "accepted_findings" and check.status == ReadinessCheckStatus.WARNING for check in readiness.checks)


def test_hidden_and_technical_artifacts_do_not_count_as_business_evidence() -> None:
    hidden = Artifact(title="Hidden", visibility=ArtifactVisibility.HIDDEN)
    technical = Artifact(
        artifact_type=ArtifactType.PYTHON_CODE,
        title="Code",
        visibility=ArtifactVisibility.TECHNICAL,
    )
    investigation = Investigation(title="Investigation", user_question="Question")
    investigation.artifacts = [hidden, technical]
    report = ShareableReport(
        investigation_id=investigation.investigation_id,
        title="Report",
        sections=[ReportSection(title="Answer", content="Answer", order=1)],
        source_artifact_ids=[hidden.artifact_id, technical.artifact_id],
    )

    readiness = evaluate_report_readiness(report, investigation=investigation)

    assert any(check.name == "evidence_artifacts" and check.status == ReadinessCheckStatus.WARNING for check in readiness.checks)


def test_technical_artifact_counts_for_technical_appendix() -> None:
    technical = Artifact(
        artifact_type=ArtifactType.PYTHON_CODE,
        title="Code",
        visibility=ArtifactVisibility.TECHNICAL,
    )
    investigation = Investigation(title="Investigation", user_question="Question")
    investigation.artifacts = [technical]
    report = ShareableReport(
        investigation_id=investigation.investigation_id,
        title="Appendix",
        template=ShareableReportTemplate.TECHNICAL_APPENDIX,
        sections=[ReportSection(title="Code", content="Details", order=1)],
        source_artifact_ids=[technical.artifact_id],
    )

    readiness = evaluate_report_readiness(report, investigation=investigation)

    assert any(check.name == "evidence_artifacts" and check.status == ReadinessCheckStatus.PASSED for check in readiness.checks)


def test_approve_report_blocked_by_readiness_and_force_bypasses() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = ShareableReport(investigation_id=investigation.investigation_id, title="Empty", sections=[])
    store.create_shareable_report(report)
    service = ReportEditingService(store)

    try:
        service.approve_report(report.report_id)
    except ValueError as exc:
        assert "Report has no sections" in str(exc)
    else:
        raise AssertionError("Expected readiness to block approval")

    approved = service.approve_report(report.report_id, force=True)
    assert approved.approval_status == ReportApprovalStatus.APPROVED


def test_section_approval_persists(tmp_path: Path) -> None:
    store = SQLiteInvestigationStore(tmp_path / "investigations.sqlite")
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))

    approved = ReportEditingService(store).approve_section(report.report_id, report.sections[0].section_id)
    reloaded = SQLiteInvestigationStore(tmp_path / "investigations.sqlite").get_shareable_report(approved.report_id)

    assert reloaded.sections[0].review_status == SectionReviewStatus.APPROVED


def test_final_snapshot_requires_approved_report_by_default() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))

    try:
        ReportEditingService(store).create_final_report_snapshot(report.report_id)
    except ValueError as exc:
        assert "not approved" in str(exc)
    else:
        raise AssertionError("Expected final snapshot to require approval")


def test_final_snapshot_requires_readiness_by_default() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = ShareableReport(investigation_id=investigation.investigation_id, title="Empty", sections=[])
    store.create_shareable_report(report)
    service = ReportEditingService(store)
    service.approve_report(report.report_id, force=True)

    try:
        service.create_final_report_snapshot(report.report_id)
    except ValueError as exc:
        assert "Report has no sections" in str(exc)
    else:
        raise AssertionError("Expected readiness to block final snapshot")


def test_force_final_snapshot_bypasses_approval_and_readiness() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = ShareableReport(investigation_id=investigation.investigation_id, title="Empty", sections=[])
    store.create_shareable_report(report)

    snapshot = ReportEditingService(store).create_final_report_snapshot(report.report_id, force=True)

    assert snapshot.status == FinalReportStatus.FINAL
    assert snapshot.readiness_snapshot["is_ready"] is False
    assert snapshot.approval_status == ReportApprovalStatus.DRAFT


def test_final_snapshot_stores_content_and_is_immutable_after_report_changes() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    service = ReportEditingService(store)
    service.approve_report(report.report_id)
    snapshot = service.create_final_report_snapshot(report.report_id)

    edited = service.update_section_content(report.report_id, report.sections[0].section_id, "Changed after final")

    assert snapshot.markdown_content
    assert snapshot.html_content
    assert "Changed after final" not in snapshot.markdown_content
    assert edited.report_id != snapshot.report_id


def test_list_and_revoke_final_snapshots() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    service = ReportEditingService(store)
    service.approve_report(report.report_id)
    snapshot = service.create_final_report_snapshot(report.report_id)

    by_report = store.list_final_report_snapshots(report_id=report.report_id)
    by_investigation = store.list_final_report_snapshots(investigation_id=investigation.investigation_id)
    revoked = service.revoke_final_report_snapshot(snapshot.snapshot_id, reason="Outdated")

    assert by_report == [snapshot]
    assert by_investigation == [snapshot]
    assert revoked.status == FinalReportStatus.REVOKED
    assert store.get_final_report_snapshot(snapshot.snapshot_id).metadata["revocation_reason"] == "Outdated"


def test_final_snapshot_persists_in_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "investigations.sqlite"
    store = SQLiteInvestigationStore(db_path)
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    service = ReportEditingService(store)
    service.approve_report(report.report_id)
    snapshot = service.create_final_report_snapshot(report.report_id)

    reloaded = SQLiteInvestigationStore(db_path)
    loaded = reloaded.get_final_report_snapshot(snapshot.snapshot_id)

    assert reloaded.get_schema_version() == 14
    assert loaded.snapshot_id == snapshot.snapshot_id
    assert loaded.markdown_content == snapshot.markdown_content
    assert loaded.txt_content == snapshot.txt_content
    assert loaded.decision_metadata.decision_status == DecisionStatus.UNKNOWN


def test_registry_lists_final_snapshots_newest_first_and_includes_revoked() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    service = ReportEditingService(store)
    service.approve_report(report.report_id)
    first = service.create_final_report_snapshot(report.report_id)
    second = service.create_final_report_snapshot(report.report_id)
    service.revoke_final_report_snapshot(first.snapshot_id, reason="Old")

    summaries = list_published_reports(store)

    assert [item["snapshot_id"] for item in summaries] == [second.snapshot_id, first.snapshot_id]
    assert {item["status"] for item in summaries} == {"final", "revoked"}


def test_registry_filters_by_status_and_investigation_id() -> None:
    store = InvestigationStore()
    first_inv = store.create_investigation("Question", title="First investigation")
    second_inv = store.create_investigation("Other", title="Second investigation")
    first_report = store.create_shareable_report(build_shareable_report(first_inv))
    second_report = store.create_shareable_report(build_shareable_report(second_inv))
    service = ReportEditingService(store)
    service.approve_report(first_report.report_id)
    service.approve_report(second_report.report_id)
    first_snapshot = service.create_final_report_snapshot(first_report.report_id)
    second_snapshot = service.create_final_report_snapshot(second_report.report_id)
    service.revoke_final_report_snapshot(second_snapshot.snapshot_id)

    final_only = list_published_reports(store, status="final")
    first_only = list_published_reports(store, investigation_id=first_inv.investigation_id)

    assert [item["snapshot_id"] for item in final_only] == [first_snapshot.snapshot_id]
    assert [item["snapshot_id"] for item in first_only] == [first_snapshot.snapshot_id]


def test_registry_search_matches_title_and_investigation_title() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question", title="Metric drop analysis")
    report = build_shareable_report(investigation)
    report.title = "Executive memo"
    saved = store.create_shareable_report(report)
    service = ReportEditingService(store)
    service.approve_report(saved.report_id)
    snapshot = service.create_final_report_snapshot(saved.report_id)

    by_title = list_published_reports(store, search="executive")
    by_investigation = list_published_reports(store, search="metric drop")

    assert by_title[0]["snapshot_id"] == snapshot.snapshot_id
    assert by_investigation[0]["snapshot_id"] == snapshot.snapshot_id


def test_registry_summary_includes_readiness_counts() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    service = ReportEditingService(store)
    service.approve_report(report.report_id)
    snapshot = service.create_final_report_snapshot(report.report_id)

    summary = get_published_report_summary(snapshot, investigation=investigation, report=report)

    assert summary["readiness_is_ready"] is True
    assert isinstance(summary["readiness_blocking_count"], int)
    assert isinstance(summary["readiness_warning_count"], int)
    assert summary["download_available"] is True


def test_update_decision_metadata_persists_without_changing_content(tmp_path: Path) -> None:
    db_path = tmp_path / "investigations.sqlite"
    store = SQLiteInvestigationStore(db_path)
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    service = ReportEditingService(store)
    service.approve_report(report.report_id)
    snapshot = service.create_final_report_snapshot(report.report_id)
    original_markdown = snapshot.markdown_content
    original_html = snapshot.html_content
    original_txt = snapshot.txt_content

    service.update_final_report_metadata(
        snapshot.snapshot_id,
        DecisionMetadata(
            tags=["Metric", " Growth "],
            owner="Ira",
            audience="Leadership",
            business_area="Metric Value",
            decision_date="2026-05-11",
            decision_status=DecisionStatus.ACCEPTED,
            short_description="Decision on metric growth.",
        ),
    )
    loaded = SQLiteInvestigationStore(db_path).get_final_report_snapshot(snapshot.snapshot_id)

    assert loaded.decision_metadata.tags == ["metric", "growth"]
    assert loaded.decision_metadata.owner == "Ira"
    assert loaded.decision_metadata.decision_status == DecisionStatus.ACCEPTED
    assert loaded.markdown_content == original_markdown
    assert loaded.html_content == original_html
    assert loaded.txt_content == original_txt


def test_add_and_remove_final_report_tags() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    service = ReportEditingService(store)
    service.approve_report(report.report_id)
    snapshot = service.create_final_report_snapshot(report.report_id)

    tagged = service.add_final_report_tags(snapshot.snapshot_id, [" Metric ", "metric", "Retention"])
    assert tagged.decision_metadata.tags == ["metric", "retention"]

    removed = service.remove_final_report_tag(snapshot.snapshot_id, "METRIC")

    assert removed.decision_metadata.tags == ["retention"]


def test_registry_search_and_filters_use_decision_metadata() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question", title="Entity retention study")
    report = store.create_shareable_report(build_shareable_report(investigation))
    service = ReportEditingService(store)
    service.approve_report(report.report_id)
    snapshot = service.create_final_report_snapshot(report.report_id)
    service.update_final_report_metadata(
        snapshot.snapshot_id,
        {
            "tags": ["retention", "entity success"],
            "owner": "Masha",
            "audience": "Exec team",
            "business_area": "Growth",
            "decision_status": "accepted",
            "short_description": "Reduce churn through onboarding changes.",
        },
    )

    by_tag_search = list_published_reports(store, search="retention")
    by_area_search = list_published_reports(store, search="growth")
    by_status = list_published_reports(store, decision_status="accepted")
    by_tag = list_published_reports(store, tag="Entity Success")

    assert by_tag_search[0]["snapshot_id"] == snapshot.snapshot_id
    assert by_area_search[0]["snapshot_id"] == snapshot.snapshot_id
    assert by_status[0]["snapshot_id"] == snapshot.snapshot_id
    assert by_tag[0]["snapshot_id"] == snapshot.snapshot_id
    assert by_status[0]["business_area"] == "Growth"
