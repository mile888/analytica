from __future__ import annotations

from fastapi.testclient import TestClient

from source.api.app import app
from source.api.deps import set_store_for_testing
from source.product.report_builder import build_shareable_report
from source.product.report_service import ReportEditingService
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
    blocked_approval = client.post(f"/reports/{report.report_id}/approve", json={"approved_by": "lead"})
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
    request_changes = client.post(
        f"/reports/{approve_section.json()['report_id']}/sections/{report.sections[0].section_id}/request-changes"
    )

    assert readiness_response.status_code == 200
    assert "checks" in readiness_response.json()
    assert approve_section.status_code == 200
    assert approve_section.json()["sections"][0]["review_status"] == "approved"
    assert request_changes.status_code == 200
    assert request_changes.json()["sections"][0]["review_status"] == "changes_requested"


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
    html = client.get(f"/final-reports/{snapshot.snapshot_id}/download/html")
    detail = client.get(f"/final-reports/{snapshot.snapshot_id}")

    assert listing.status_code == 200
    assert listing.json()[0]["snapshot_id"] == snapshot.snapshot_id
    assert listing.json()[0]["investigation_title"] == "Published investigation"
    assert markdown.status_code == 200
    assert markdown.text == snapshot.markdown_content
    assert markdown.headers["content-type"].startswith("text/markdown")
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
            "tags": ["Sales", "sales"],
            "owner": "Ira",
            "audience": "Leadership",
            "business_area": "Revenue",
            "decision_status": "accepted",
            "short_description": "Revenue decision.",
        },
    )
    add_tags = client.post(f"/final-reports/{snapshot.snapshot_id}/tags", json={"tags": ["Retention"]})
    delete_tag = client.delete(f"/final-reports/{snapshot.snapshot_id}/tags/sales")
    status = client.post(f"/final-reports/{snapshot.snapshot_id}/decision-status", json={"status": "superseded"})
    filtered = client.get("/final-reports", params={"decision_status": "superseded", "tag": "retention"})

    assert patch.status_code == 200
    assert patch.json()["decision_metadata"]["tags"] == ["sales"]
    assert add_tags.status_code == 200
    assert add_tags.json()["decision_metadata"]["tags"] == ["sales", "retention"]
    assert delete_tag.status_code == 200
    assert delete_tag.json()["decision_metadata"]["tags"] == ["retention"]
    assert status.status_code == 200
    assert status.json()["decision_metadata"]["decision_status"] == "superseded"
    assert filtered.status_code == 200
    assert filtered.json()[0]["snapshot_id"] == snapshot.snapshot_id
