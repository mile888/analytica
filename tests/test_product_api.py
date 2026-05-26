from __future__ import annotations

from fastapi.testclient import TestClient

from source.api.app import app
from source.api.deps import set_store_for_testing
from source.product.store import InvestigationStore


def test_api_investigation_endpoints_return_payloads() -> None:
    store = InvestigationStore()
    set_store_for_testing(store)
    client = TestClient(app)

    create_response = client.post("/investigations", json={"question": "Question"})
    investigation_id = create_response.json()["investigation_id"]
    list_response = client.get("/investigations")
    get_response = client.get(f"/investigations/{investigation_id}")

    assert create_response.status_code == 200
    assert list_response.status_code == 200
    assert list_response.json()[0]["investigation_id"] == investigation_id
    assert get_response.status_code == 200
    assert get_response.json()["user_question"] == "Question"


def test_api_finalize_and_final_snapshot_endpoints_work() -> None:
    store = InvestigationStore()
    set_store_for_testing(store)
    client = TestClient(app)

    investigation = client.post("/investigations", json={"question": "Question"}).json()
    report = client.post(f"/investigations/{investigation['investigation_id']}/reports").json()
    approve = client.post(f"/reports/{report['report_id']}/approve", json={"approved_by": "api", "force": True})
    finalize = client.post(f"/reports/{report['report_id']}/finalize", json={"created_by": "api", "force": True})
    snapshots = client.get(f"/reports/{report['report_id']}/final-snapshots")
    snapshot_id = finalize.json()["snapshot_id"]
    get_snapshot = client.get(f"/reports/final-snapshots/{snapshot_id}")
    revoke = client.post(f"/reports/final-snapshots/{snapshot_id}/revoke", json={"reason": "Superseded"})

    assert approve.status_code == 200
    assert finalize.status_code == 200
    assert finalize.json()["created_by"] == "api"
    assert snapshots.status_code == 200
    assert snapshots.json()[0]["snapshot_id"] == snapshot_id
    assert get_snapshot.status_code == 200
    assert get_snapshot.json()["markdown_content"]
    assert revoke.status_code == 200
    assert revoke.json()["status"] == "revoked"
