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

def test_shareable_report_persists_in_sqlite_and_migration_v5_is_applied(tmp_path: Path) -> None:
    store = SQLiteInvestigationStore(tmp_path / "investigations.sqlite")
    investigation = store.create_investigation("Question")
    report = build_shareable_report(investigation)

    saved = store.create_shareable_report(report)
    reloaded_store = SQLiteInvestigationStore(tmp_path / "investigations.sqlite")
    loaded = reloaded_store.get_shareable_report(saved.report_id)

    assert reloaded_store.get_schema_version() == 15
    assert loaded.title == report.title
    assert loaded.sections[0].title == report.sections[0].title
    assert reloaded_store.list_shareable_reports(investigation.investigation_id)[0].report_id == saved.report_id

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
