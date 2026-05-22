from __future__ import annotations

from pathlib import Path
import math

from fastapi.testclient import TestClient

from source.api.app import app
from source.api.deps import set_store_for_testing
from source.product.data_sources import DataSource, DataSourceProfile, DataSourceSemanticNotes, DataSourceType
from source.product.investigation import (
    Artifact,
    ArtifactType,
    Finding,
    InvestigationMemoryItem,
    InvestigationMemoryType,
    InvestigationMessage,
    InvestigationRun,
    InvestigationRunEvent,
)
from source.product.report_builder import build_shareable_report
from source.product.report_service import ReportEditingService
from source.product.sqlite_store import SQLiteInvestigationStore
from source.product.store import InvestigationStore


def test_api_investigation_endpoints_return_payloads() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    set_store_for_testing(store)
    client = TestClient(app)

    list_response = client.get("/investigations")
    get_response = client.get(f"/investigations/{investigation.investigation_id}")

    assert list_response.status_code == 200
    assert list_response.json()[0]["investigation_id"] == investigation.investigation_id
    assert get_response.status_code == 200
    assert get_response.json()["user_question"] == "Question"


def test_api_serializes_nan_artifact_values_as_null() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    store.add_artifact(
        investigation.investigation_id,
        Artifact(
            artifact_type=ArtifactType.TABLE,
            title="NaN table",
            content=[{"metric": "x", "value": math.nan, "ratio": math.inf}],
            metadata={"score": math.nan},
        ),
    )
    set_store_for_testing(store)
    client = TestClient(app)

    get_response = client.get(f"/investigations/{investigation.investigation_id}")
    list_response = client.get("/investigations")

    assert get_response.status_code == 200
    artifact = get_response.json()["artifacts"][0]
    assert artifact["content"][0]["value"] is None
    assert artifact["content"][0]["ratio"] is None
    assert artifact["metadata"]["score"] is None
    assert list_response.status_code == 200


def test_api_investigation_memory_endpoints_work() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    set_store_for_testing(store)
    client = TestClient(app)

    created = client.post(
        f"/investigations/{investigation.investigation_id}/memory",
        json={"type": "assumption", "content": "The uploaded data is representative."},
    )
    listed = client.get(f"/investigations/{investigation.investigation_id}/memory")
    updated = client.patch(
        f"/investigations/{investigation.investigation_id}/memory/{created.json()['memory_id']}",
        json={"status": "resolved"},
    )

    assert created.status_code == 200
    assert created.json()["memory_type"] == "assumption"
    assert listed.status_code == 200
    assert listed.json()[0]["content"] == "The uploaded data is representative."
    assert updated.status_code == 200
    assert updated.json()["status"] == "resolved"


def test_api_promotes_findings_and_links_evidence() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    source = store.create_data_source(DataSource(name="Evidence source"))
    finding = Finding(title="Pattern", text="A useful analytical pattern.")
    store.add_finding(investigation.investigation_id, finding)
    set_store_for_testing(store)
    client = TestClient(app)

    promoted = client.post(
        f"/investigations/{investigation.investigation_id}/findings/{finding.finding_id}/promote-memory",
        json={"type": "risk"},
    )
    linked = client.post(
        f"/investigations/{investigation.investigation_id}/findings/{finding.finding_id}/evidence",
        json={"type": "data_source", "id": source.data_source_id, "label": source.name},
    )

    assert promoted.status_code == 200
    assert promoted.json()["memory_type"] == "risk"
    assert promoted.json()["metadata"]["source_type"] == "finding"
    assert linked.status_code == 200
    assert linked.json()["metadata"]["linked_evidence"][0]["id"] == source.data_source_id


def test_api_marks_artifact_for_report_selection() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    artifact = Artifact(
        artifact_type=ArtifactType.CHART,
        title="Metric distribution",
        content={"chart_type": "histogram", "bins": [{"label": "A", "count": 2}], "row_count": 2},
    )
    store.add_artifact(investigation.investigation_id, artifact)
    set_store_for_testing(store)
    client = TestClient(app)

    response = client.post(
        f"/investigations/{investigation.investigation_id}/artifacts/{artifact.artifact_id}/report-selection",
        json={"selected": True},
    )
    report_response = client.post(
        f"/investigations/{investigation.investigation_id}/reports",
        json={"template": "executive_summary", "include_technical": False},
    )

    assert response.status_code == 200
    assert response.json()["metadata"]["selected_for_report"] is True
    assert report_response.status_code == 200
    assert report_response.json()["source_artifact_ids"] == [artifact.artifact_id]


def test_api_create_updated_report_versions_previous_report() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    set_store_for_testing(store)
    client = TestClient(app)

    first = client.post(
        f"/investigations/{investigation.investigation_id}/reports",
        json={"template": "executive_summary", "include_technical": False},
    )
    second = client.post(
        f"/investigations/{investigation.investigation_id}/reports",
        json={"template": "executive_summary", "include_technical": False},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["previous_version_id"] == first.json()["report_id"]
    assert second.json()["version"] == first.json()["version"] + 1
    assert store.get_shareable_report(first.json()["report_id"]).is_latest is False
    assert store.get_shareable_report(second.json()["report_id"]).is_latest is True


def test_api_investigation_suggested_questions_endpoint_work() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    source = store.create_data_source(DataSource(name="Generic source"))
    store.link_data_source_to_investigation(investigation.investigation_id, source.data_source_id)
    set_store_for_testing(store)
    client = TestClient(app)

    response = client.get(f"/investigations/{investigation.investigation_id}/suggested-questions")

    assert response.status_code == 200
    assert response.json()["suggestions"]


def test_api_report_endpoints_return_versions() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    edited = ReportEditingService(store).update_section_content(
        report.report_id,
        report.sections[0].section_id,
        "Edited",
    )
    set_store_for_testing(store)
    client = TestClient(app)

    report_response = client.get(f"/reports/{edited.report_id}")
    versions_response = client.get(f"/reports/{edited.report_id}/versions")

    assert report_response.status_code == 200
    assert report_response.json()["report_id"] == edited.report_id
    assert versions_response.status_code == 200
    assert [item["version"] for item in versions_response.json()] == [1, 2]


def test_api_investigation_report_actions_endpoints_work() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("What should we inspect?")
    set_store_for_testing(store)
    client = TestClient(app)

    empty = client.get(f"/investigations/{investigation.investigation_id}/reports")
    created = client.post(
        f"/investigations/{investigation.investigation_id}/reports",
        json={"template": "product_decision_memo", "include_technical": False},
    )
    listing = client.get(f"/investigations/{investigation.investigation_id}/reports")

    assert empty.status_code == 200
    assert empty.json() == []
    assert created.status_code == 200
    assert created.json()["investigation_id"] == investigation.investigation_id
    assert created.json()["template"] == "product_decision_memo"
    assert listing.status_code == 200
    assert listing.json()[0]["report_id"] == created.json()["report_id"]


def test_api_delete_investigation_route_removes_related_records() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    source = store.create_data_source(DataSource(name="Source"))
    store.link_data_source_to_investigation(investigation.investigation_id, source.data_source_id)
    report = store.create_shareable_report(build_shareable_report(investigation))
    comment = ReportEditingService(store).add_comment(report.report_id, report.sections[0].section_id, "Review")
    run = store.create_investigation_run(InvestigationRun(investigation_id=investigation.investigation_id))
    set_store_for_testing(store)
    client = TestClient(app)

    deleted = client.delete(f"/investigations/{investigation.investigation_id}")
    missing = client.get(f"/investigations/{investigation.investigation_id}")

    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "id": investigation.investigation_id}
    assert missing.status_code == 404
    assert store.get_data_source(source.data_source_id).linked_investigation_ids == []
    assert store.list_shareable_reports(investigation_id=investigation.investigation_id) == []
    assert comment.comment_id not in store._report_comments
    try:
        store.get_investigation_run(run.run_id)
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("run should have been deleted")


def test_api_archive_lifecycle_routes_are_not_exposed() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    source = store.create_data_source(DataSource(name="Generic source"))
    report = store.create_shareable_report(build_shareable_report(investigation))
    set_store_for_testing(store)
    client = TestClient(app)

    assert client.post(f"/investigations/{investigation.investigation_id}/archive").status_code in {404, 405}
    assert client.post(f"/data-sources/{source.data_source_id}/archive").status_code in {404, 405}
    assert client.post(f"/reports/{report.report_id}/archive").status_code in {404, 405}


def test_sqlite_delete_investigation_removes_related_records(tmp_path: Path) -> None:
    store = SQLiteInvestigationStore(tmp_path / "delete-investigation.sqlite")
    investigation = store.create_investigation("Question")
    source = store.create_data_source(DataSource(name="Source"))
    store.link_data_source_to_investigation(investigation.investigation_id, source.data_source_id)
    report = store.create_shareable_report(build_shareable_report(investigation))
    ReportEditingService(store).add_comment(report.report_id, report.sections[0].section_id, "Review")
    run = store.create_investigation_run(InvestigationRun(investigation_id=investigation.investigation_id))
    store.add_investigation_run_event(
        InvestigationRunEvent(
            run_id=run.run_id,
            investigation_id=investigation.investigation_id,
            message="Started",
        )
    )
    store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            run_id=run.run_id,
            content="Follow-up",
        )
    )
    store.add_investigation_memory_item(
        InvestigationMemoryItem(
            investigation_id=investigation.investigation_id,
            memory_type=InvestigationMemoryType.ASSUMPTION,
            content="Assumption",
        )
    )

    store.delete_investigation(investigation.investigation_id)

    try:
        store.get_investigation(investigation.investigation_id)
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("investigation should have been deleted")
    assert store.get_data_source(source.data_source_id).linked_investigation_ids == []
    assert store.list_shareable_reports(investigation_id=investigation.investigation_id) == []
    with store._connect() as conn:
        comment_count = conn.execute(
            "SELECT COUNT(*) FROM report_comments WHERE report_id = ?",
            (report.report_id,),
        ).fetchone()[0]
    assert comment_count == 0
    with store._connect() as conn:
        event_count = conn.execute(
            "SELECT COUNT(*) FROM investigation_run_events WHERE run_id = ?",
            (run.run_id,),
        ).fetchone()[0]
        message_count = conn.execute(
            "SELECT COUNT(*) FROM investigation_messages WHERE investigation_id = ?",
            (investigation.investigation_id,),
        ).fetchone()[0]
        memory_count = conn.execute(
            "SELECT COUNT(*) FROM investigation_memory WHERE investigation_id = ?",
            (investigation.investigation_id,),
        ).fetchone()[0]
    assert event_count == 0
    assert message_count == 0
    assert memory_count == 0


def test_api_delete_data_source_route_removes_source_links() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    source = store.create_data_source(DataSource(name="Source"))
    store.link_data_source_to_investigation(investigation.investigation_id, source.data_source_id)
    set_store_for_testing(store)
    client = TestClient(app)

    deleted = client.delete(f"/data-sources/{source.data_source_id}")
    missing = client.get(f"/data-sources/{source.data_source_id}")

    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert deleted.json()["file_deleted"] is False
    assert missing.status_code == 404
    assert store.get_investigation(investigation.investigation_id).linked_data_source_ids == []


def test_sqlite_delete_data_source_deletes_only_managed_upload_files(tmp_path: Path) -> None:
    store = SQLiteInvestigationStore(tmp_path / "delete-source.sqlite")
    upload_path = Path(".analytica/uploads/test_delete_data_source.csv")
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_text("metric_value\n1\n", encoding="utf-8")
    source = store.create_data_source(
        DataSource(
            name="Uploaded source",
            data_source_type=DataSourceType.CSV,
            location=str(upload_path),
        )
    )
    investigation = store.create_investigation("Question")
    store.link_data_source_to_investigation(investigation.investigation_id, source.data_source_id)
    store.save_data_source_profile(source.data_source_id, DataSourceProfile(row_count=1, column_count=1))
    store.save_data_source_semantic_notes(DataSourceSemanticNotes(data_source_id=source.data_source_id))

    file_deleted = store.delete_data_source(source.data_source_id)

    assert file_deleted is True
    assert not upload_path.exists()
    assert store.get_investigation(investigation.investigation_id).linked_data_source_ids == []
    try:
        store.get_data_source(source.data_source_id)
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("data source should have been deleted")
    try:
        store.get_data_source_profile(source.data_source_id)
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("profile should have been deleted")


def test_sqlite_delete_data_source_does_not_delete_external_file(tmp_path: Path) -> None:
    store = SQLiteInvestigationStore(tmp_path / "delete-external-source.sqlite")
    external_file = tmp_path / "external.csv"
    external_file.write_text("metric_value\n1\n", encoding="utf-8")
    source = store.create_data_source(
        DataSource(
            name="External source",
            data_source_type=DataSourceType.CSV,
            location=str(external_file),
        )
    )

    file_deleted = store.delete_data_source(source.data_source_id)

    assert file_deleted is False
    assert external_file.exists()


def test_api_shareable_report_txt_download_works() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    set_store_for_testing(store)
    client = TestClient(app)

    response = client.get(f"/reports/{report.report_id}/download/txt")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert report.title.upper() in response.text


def test_api_shareable_report_pdf_download_includes_chart_artifacts() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    investigation.artifacts.append(
        Artifact(
            artifact_type=ArtifactType.CHART,
            title="Revenue by Region",
            content={
                "chart_type": "bar",
                "x": "Region",
                "y": "Revenue",
                "rows": [
                    {"Region": "North", "Revenue": 120},
                    {"Region": "South", "Revenue": 90},
                ],
            },
        )
    )
    report = store.create_shareable_report(build_shareable_report(investigation))
    set_store_for_testing(store)
    client = TestClient(app)

    response = client.get(f"/reports/{report.report_id}/download/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF")
    assert b"/Image" in response.content or b"Revenue by Region" in response.content


def test_api_shareable_report_pdf_download_handles_histogram_comparison() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    store.add_artifact(
        investigation.investigation_id,
        Artifact(
            artifact_type=ArtifactType.CHART,
            title="Metric distribution comparison",
            content={
                "chart_type": "histogram",
                "comparison_groups": [
                    {
                        "label": "A",
                        "record_count": 5,
                        "bins": [
                            {"label": "0 to 10", "count": 2},
                            {"label": "10 to 20", "count": 3},
                        ],
                    },
                    {
                        "label": "B",
                        "record_count": 5,
                        "bins": [
                            {"label": "0 to 10", "count": 4},
                            {"label": "10 to 20", "count": 1},
                        ],
                    },
                ],
                "row_count": 10,
            },
            metadata={"selected_for_report": True},
        ),
    )
    report = store.create_shareable_report(build_shareable_report(store.get_investigation(investigation.investigation_id)))
    set_store_for_testing(store)
    client = TestClient(app)

    response = client.get(f"/reports/{report.report_id}/download/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF")
    assert len(response.content) > 8000


def test_api_shareable_report_pdf_download_handles_non_ascii_title() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question", title="тест работа")
    report = build_shareable_report(investigation)
    report.title = "тест работа — Executive Summary"
    report = store.create_shareable_report(report)
    set_store_for_testing(store)
    client = TestClient(app)

    response = client.get(f"/reports/{report.report_id}/download/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert "filename=" in response.headers["content-disposition"]
    assert "filename*=UTF-8''" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF")


def test_api_report_comments_and_review_endpoints_work() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    set_store_for_testing(store)
    client = TestClient(app)

    create_response = client.post(
        f"/reports/{report.report_id}/comments",
        json={"section_id": report.sections[0].section_id, "text": "Please clarify", "author": "reviewer"},
    )
    comments_response = client.get(f"/reports/{report.report_id}/comments")
    comment_id = create_response.json()["comment_id"]
    blocked_approval = client.post(f"/reports/{report.report_id}/approve", json={"approved_by": "lead"})
    promoted_comment = client.post(f"/reports/{report.report_id}/comments/{comment_id}/promote-memory")
    resolve_response = client.post(f"/reports/{report.report_id}/comments/{comment_id}/resolve")
    delete_response = client.delete(f"/reports/{report.report_id}/comments/{comment_id}")
    request_changes = client.post(f"/reports/{report.report_id}/request-changes", json={"notes": "Needs edits"})
    forced_approval = client.post(
        f"/reports/{report.report_id}/approve",
        json={"approved_by": "lead", "force": True},
    )

    assert create_response.status_code == 200
    assert create_response.json()["text"] == "Please clarify"
    assert comments_response.status_code == 200
    assert comments_response.json()[0]["status"] == "open"
    assert blocked_approval.status_code == 409
    assert promoted_comment.status_code == 200
    assert promoted_comment.json()["memory_type"] == "open_question"
    assert promoted_comment.json()["metadata"]["source_type"] == "report_comment"
    assert resolve_response.status_code == 200
    assert resolve_response.json()["status"] == "resolved"
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True
    assert request_changes.status_code == 200
    assert request_changes.json()["approval_status"] == "changes_requested"
    assert forced_approval.status_code == 200
    assert forced_approval.json()["approval_status"] == "approved"
    assert forced_approval.json()["approved_by"] == "lead"


def test_api_report_readiness_and_section_review_endpoints_work() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    set_store_for_testing(store)
    client = TestClient(app)

    readiness_response = client.get(f"/reports/{report.report_id}/readiness")
    approve_section = client.post(f"/reports/{report.report_id}/sections/{report.sections[0].section_id}/approve")
    promoted_section = client.post(f"/reports/{report.report_id}/sections/{report.sections[0].section_id}/promote-memory")
    request_changes = client.post(
        f"/reports/{approve_section.json()['report_id']}/sections/{report.sections[0].section_id}/request-changes"
    )

    assert readiness_response.status_code == 200
    assert "checks" in readiness_response.json()
    assert approve_section.status_code == 200
    assert approve_section.json()["sections"][0]["review_status"] == "approved"
    assert promoted_section.status_code == 200
    assert promoted_section.json()["memory_type"] == "decision"
    assert promoted_section.json()["metadata"]["source_type"] == "report_section"
    assert request_changes.status_code == 200
    assert request_changes.json()["sections"][0]["review_status"] == "changes_requested"


def test_api_report_section_editing_endpoints_work() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    set_store_for_testing(store)
    client = TestClient(app)

    section_id = report.sections[0].section_id
    updated = client.patch(
        f"/reports/{report.report_id}/sections/{section_id}",
        json={"title": "Edited question", "content": "Edited content"},
    )
    updated_report_id = updated.json()["report_id"]
    added = client.post(
        f"/reports/{updated_report_id}/sections",
        json={"title": "New section", "content": "New content"},
    )
    added_report_id = added.json()["report_id"]
    new_section_id = added.json()["sections"][-1]["section_id"]
    duplicated = client.post(f"/reports/{added_report_id}/sections/{new_section_id}/duplicate")
    duplicated_report_id = duplicated.json()["report_id"]
    section_ids = [item["section_id"] for item in duplicated.json()["sections"]]
    reordered = client.post(
        f"/reports/{duplicated_report_id}/sections/reorder",
        json={"section_ids": list(reversed(section_ids))},
    )
    reordered_report_id = reordered.json()["report_id"]
    deleted = client.delete(f"/reports/{reordered_report_id}/sections/{new_section_id}")

    assert updated.status_code == 200
    assert updated.json()["version"] == 2
    assert updated.json()["sections"][0]["title"] == "Edited question"
    assert updated.json()["sections"][0]["content"] == "Edited content"
    assert updated.json()["sections"][0]["edited_by_user"] is True
    assert added.status_code == 200
    assert any(item["title"] == "New section" for item in added.json()["sections"])
    assert duplicated.status_code == 200
    assert any(item["title"] == "New section copy" for item in duplicated.json()["sections"])
    assert reordered.status_code == 200
    assert deleted.status_code == 200
    assert all(item["section_id"] != new_section_id for item in deleted.json()["sections"])


def test_api_finalize_and_final_snapshot_endpoints_work() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    ReportEditingService(store).approve_report(report.report_id)
    set_store_for_testing(store)
    client = TestClient(app)

    finalize = client.post(f"/reports/{report.report_id}/finalize", json={"created_by": "api"})
    snapshots = client.get(f"/reports/{report.report_id}/final-snapshots")
    snapshot_id = finalize.json()["snapshot_id"]
    get_snapshot = client.get(f"/reports/final-snapshots/{snapshot_id}")
    revoke = client.post(f"/reports/final-snapshots/{snapshot_id}/revoke", json={"reason": "Superseded"})

    assert finalize.status_code == 200
    assert finalize.json()["created_by"] == "api"
    assert snapshots.status_code == 200
    assert snapshots.json()[0]["snapshot_id"] == snapshot_id
    assert get_snapshot.status_code == 200
    assert get_snapshot.json()["markdown_content"]
    assert revoke.status_code == 200
    assert revoke.json()["status"] == "revoked"


def test_api_final_reports_library_and_downloads_work() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question", title="Published investigation")
    report = store.create_shareable_report(build_shareable_report(investigation))
    service = ReportEditingService(store)
    service.approve_report(report.report_id)
    snapshot = service.create_final_report_snapshot(report.report_id)
    set_store_for_testing(store)
    client = TestClient(app)

    listing = client.get("/final-reports")
    markdown = client.get(f"/final-reports/{snapshot.snapshot_id}/download/markdown")
    txt = client.get(f"/final-reports/{snapshot.snapshot_id}/download/txt")
    html = client.get(f"/final-reports/{snapshot.snapshot_id}/download/html")
    detail = client.get(f"/final-reports/{snapshot.snapshot_id}")

    assert listing.status_code == 200
    assert listing.json()[0]["snapshot_id"] == snapshot.snapshot_id
    assert listing.json()[0]["investigation_title"] == "Published investigation"
    assert markdown.status_code == 200
    assert markdown.text == snapshot.markdown_content
    assert markdown.headers["content-type"].startswith("text/markdown")
    assert txt.status_code == 200
    assert txt.text == snapshot.txt_content
    assert txt.headers["content-type"].startswith("text/plain")
    assert html.status_code == 200
    assert html.text == snapshot.html_content
    assert html.headers["content-type"].startswith("text/html")
    assert detail.status_code == 200
    assert detail.json()["snapshot_id"] == snapshot.snapshot_id


def test_api_final_report_metadata_routes_work() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    report = store.create_shareable_report(build_shareable_report(investigation))
    service = ReportEditingService(store)
    service.approve_report(report.report_id)
    snapshot = service.create_final_report_snapshot(report.report_id)
    set_store_for_testing(store)
    client = TestClient(app)

    patch = client.patch(
        f"/final-reports/{snapshot.snapshot_id}/metadata",
        json={
            "tags": ["Metric", "metric"],
            "owner": "Ira",
            "audience": "Leadership",
            "business_area": "Metric",
            "decision_status": "accepted",
            "short_description": "Metric decision.",
        },
    )
    add_tags = client.post(f"/final-reports/{snapshot.snapshot_id}/tags", json={"tags": ["Retention"]})
    delete_tag = client.delete(f"/final-reports/{snapshot.snapshot_id}/tags/metric")
    status = client.post(f"/final-reports/{snapshot.snapshot_id}/decision-status", json={"status": "superseded"})
    filtered = client.get("/final-reports", params={"decision_status": "superseded", "tag": "retention"})

    assert patch.status_code == 200
    assert patch.json()["decision_metadata"]["tags"] == ["metric"]
    assert add_tags.status_code == 200
    assert add_tags.json()["decision_metadata"]["tags"] == ["metric", "retention"]
    assert delete_tag.status_code == 200
    assert delete_tag.json()["decision_metadata"]["tags"] == ["retention"]
    assert status.status_code == 200
    assert status.json()["decision_metadata"]["decision_status"] == "superseded"
    assert filtered.status_code == 200
    assert filtered.json()[0]["snapshot_id"] == snapshot.snapshot_id
