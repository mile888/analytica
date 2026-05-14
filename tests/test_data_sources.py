from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from source.api.app import app
from source.api.deps import set_store_for_testing
from source.product.data_context import build_data_source_usage_context
from source.product.data_profiling import profile_csv, profile_dataframe
from source.product.data_sources import (
    ColumnInferredRole,
    ColumnSemanticNote,
    ColumnSemanticRole,
    DataSource,
    DataSourceSemanticNotes,
    DataSourceStatus,
    DataSourceType,
)
from source.product.event_stream import (
    build_event_stream_response,
    decode_event_cursor,
    encode_event_cursor,
    filter_events_after,
)
from source.product.investigation import (
    InvestigationRun,
    InvestigationRunEvent,
    InvestigationRunEventSeverity,
    InvestigationRunEventType,
    InvestigationRunStage,
    InvestigationRunStatus,
)
from source.product.run_service import InvestigationRunService
from source.product.service import InvestigationService
from source.product.sqlite_store import SQLiteInvestigationStore
from source.product.store import InvestigationStore


def test_create_list_get_data_source_in_memory_store() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders", data_source_type=DataSourceType.CSV, tags=[" Sales "]))

    assert store.get_data_source(source.data_source_id).name == "Orders"
    assert store.list_data_sources()[0].data_source_id == source.data_source_id
    assert source.tags == ["sales"]


def test_profile_dataframe_and_empty_dataframe_work() -> None:
    df = pd.DataFrame({"sales": [10.0, 20.0, None], "segment": ["a", "b", "a"]})
    profile = profile_dataframe(df)
    empty = profile_dataframe(pd.DataFrame())

    assert profile.row_count == 3
    assert profile.column_count == 2
    assert profile.missing_summary["sales"] == 1
    assert "sales" in profile.numeric_summary
    assert "segment" in profile.categorical_summary
    assert empty.row_count == 0
    assert empty.column_count == 0


def test_profile_csv_works(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    path.write_text("sales,segment\n10,a\n20,b\n", encoding="utf-8")

    profile = profile_csv(path)

    assert profile.row_count == 2
    assert [column.name for column in profile.columns] == ["sales", "segment"]


def test_profile_and_links_persist_in_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "investigations.sqlite"
    store = SQLiteInvestigationStore(db_path)
    source = store.create_data_source(DataSource(name="Orders", data_source_type=DataSourceType.CSV))
    investigation = store.create_investigation("Question")
    profile = profile_dataframe(pd.DataFrame({"sales": [1, 2]}))

    store.save_data_source_profile(source.data_source_id, profile)
    store.link_data_source_to_investigation(investigation.investigation_id, source.data_source_id)

    reloaded = SQLiteInvestigationStore(db_path)
    loaded_investigation = reloaded.get_investigation(investigation.investigation_id)
    loaded_source = reloaded.get_data_source(source.data_source_id)
    loaded_profile = reloaded.get_data_source_profile(source.data_source_id)

    assert reloaded.get_schema_version() == 11
    assert loaded_investigation.linked_data_source_ids == [source.data_source_id]
    assert loaded_source.linked_investigation_ids == [investigation.investigation_id]
    assert loaded_profile.row_count == 2


def test_archive_data_source() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders"))

    archived = store.archive_data_source(source.data_source_id)

    assert archived.status == DataSourceStatus.ARCHIVED
    assert store.list_data_sources(status="archived")[0].data_source_id == source.data_source_id


def test_api_data_source_routes_work() -> None:
    store = InvestigationStore()
    set_store_for_testing(store)
    client = TestClient(app)

    created = client.post(
        "/data-sources",
        json={"name": "Orders", "type": "csv", "location": "/tmp/orders.csv", "tags": ["Sales"]},
    )
    source_id = created.json()["data_source_id"]
    listed = client.get("/data-sources")
    fetched = client.get(f"/data-sources/{source_id}")
    updated = client.patch(f"/data-sources/{source_id}", json={"description": "Orders data", "tags": ["Revenue"]})
    archived = client.post(f"/data-sources/{source_id}/archive")

    assert created.status_code == 200
    assert listed.json()[0]["data_source_id"] == source_id
    assert fetched.json()["name"] == "Orders"
    assert updated.json()["description"] == "Orders data"
    assert updated.json()["tags"] == ["revenue"]
    assert archived.json()["status"] == "archived"


def test_api_upload_csv_creates_source_saves_file_and_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    store = InvestigationStore()
    set_store_for_testing(store)
    client = TestClient(app)

    response = client.post(
        "/data-sources/upload-csv",
        data={"name": "Orders", "description": "Uploaded orders", "tags": ["Sales, Retail"]},
        files={"file": ("orders.csv", b"sales,segment\n10,a\n20,b\n", "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    source = payload["data_source"]
    assert source["name"] == "Orders"
    assert source["data_source_type"] == "csv"
    assert source["tags"] == ["sales", "retail"]
    assert Path(source["location"]).exists()
    assert Path(source["location"]).parent == Path(".analytica/uploads")
    assert payload["profile"]["row_count"] == 2
    assert store.get_data_source_profile(source["data_source_id"]).column_count == 2


def test_api_upload_csv_rejects_non_csv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    set_store_for_testing(InvestigationStore())
    client = TestClient(app)

    response = client.post(
        "/data-sources/upload-csv",
        files={"file": ("orders.txt", b"sales\n10\n", "text/plain")},
    )

    assert response.status_code == 400
    assert "Only .csv files" in response.json()["detail"]


def test_api_upload_empty_csv_profiles_empty_frame(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    set_store_for_testing(InvestigationStore())
    client = TestClient(app)

    response = client.post(
        "/data-sources/upload-csv",
        files={"file": ("empty.csv", b"", "text/csv")},
    )

    assert response.status_code == 200
    assert response.json()["profile"]["row_count"] == 0
    assert response.json()["profile"]["column_count"] == 0


def test_api_upload_invalid_csv_returns_controlled_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    set_store_for_testing(InvestigationStore())
    client = TestClient(app)

    response = client.post(
        "/data-sources/upload-csv",
        files={"file": ("broken.csv", b'a,b\n"unterminated\n', "text/csv")},
    )

    assert response.status_code == 400
    assert "Could not parse CSV" in response.json()["detail"]


def test_api_create_investigation_links_data_source() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders"))
    set_store_for_testing(store)
    client = TestClient(app)

    response = client.post(
        "/investigations",
        json={"question": "What changed?", "data_source_ids": [source.data_source_id]},
    )

    assert response.status_code == 200
    assert response.json()["linked_data_source_ids"] == [source.data_source_id]


def test_investigation_service_links_data_source_before_run() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders"))
    investigation = store.create_investigation("Question")

    def runner(**kwargs):
        return {"summary": "Done"}

    service = InvestigationService(store=store, runner=runner)
    service.run_investigation(
        investigation.investigation_id,
        data_context={"data_source_ids": [source.data_source_id]},
    )

    loaded = store.get_investigation(investigation.investigation_id)
    assert loaded.linked_data_source_ids == [source.data_source_id]


def test_build_usage_context_with_full_profile_and_roles() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders", data_source_type=DataSourceType.CSV))
    profile = profile_dataframe(
        pd.DataFrame(
            {
                "created_at": ["2026-01-01", "2026-01-02"],
                "customer_id": ["c1", "c2"],
                "sales": [10.0, 20.0],
                "segment": ["enterprise", "smb"],
                "notes": [
                    "This customer provided a long qualitative note that should be treated as text.",
                    "Another long qualitative note with enough words to look like free-form text.",
                ],
            }
        )
    )
    store.save_data_source_profile(source.data_source_id, profile)

    context = build_data_source_usage_context(store, source.data_source_id)
    roles = {column.name: column.inferred_role for column in context.column_summaries}

    assert context.schema_summary["row_count"] == 2
    assert roles["created_at"] == ColumnInferredRole.TIMESTAMP
    assert roles["customer_id"] == ColumnInferredRole.IDENTIFIER
    assert roles["sales"] == ColumnInferredRole.METRIC
    assert roles["segment"] == ColumnInferredRole.DIMENSION
    assert roles["notes"] == ColumnInferredRole.TEXT


def test_build_usage_context_without_profile_does_not_crash() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders"))

    context = build_data_source_usage_context(store, source.data_source_id)

    assert context.schema_summary["row_count"] is None
    assert "No profile is available for this data source." in context.caveats


def test_usage_context_caveats_for_archived_source_and_missing_values() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders", status=DataSourceStatus.ARCHIVED))
    profile = profile_dataframe(pd.DataFrame({"sales": [None, None, 3.0]}))
    store.save_data_source_profile(source.data_source_id, profile)

    context = build_data_source_usage_context(store, source.data_source_id)

    assert "Source status is archived." in context.caveats
    assert "Column sales has high missingness." in context.caveats


def test_usage_context_includes_previous_questions_from_linked_investigations() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders"))
    investigation = store.create_investigation("Why did revenue change?")
    store.link_data_source_to_investigation(investigation.investigation_id, source.data_source_id)

    context = build_data_source_usage_context(store, source.data_source_id)

    assert context.previous_questions == ["Why did revenue change?"]


def test_api_usage_context_endpoint_works() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders"))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(pd.DataFrame({"sales": [1, 2]})))
    set_store_for_testing(store)
    client = TestClient(app)

    response = client.get(f"/data-sources/{source.data_source_id}/usage-context")

    assert response.status_code == 200
    assert response.json()["data_source_id"] == source.data_source_id
    assert response.json()["column_summaries"][0]["inferred_role"] == "metric"


def test_investigation_service_passes_usage_context_to_runner_for_linked_source() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders"))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(pd.DataFrame({"sales": [1, 2]})))
    investigation = store.create_investigation("What changed?")
    store.link_data_source_to_investigation(investigation.investigation_id, source.data_source_id)
    captured = {}

    def runner(**kwargs):
        captured.update(kwargs)
        return {"summary": "Done"}

    service = InvestigationService(store=store, runner=runner)
    service.run_investigation(investigation.investigation_id)

    context = captured["data_context"]
    assert context["data_source_ids"] == [source.data_source_id]
    assert context["data_source_usage_contexts"][0]["name"] == "Orders"
    assert "Data source usage context" in context["data_context_prompt"]


def test_empty_semantic_notes_load_by_default() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders"))

    notes = store.get_data_source_semantic_notes(source.data_source_id)

    assert notes.data_source_id == source.data_source_id
    assert notes.column_notes == []


def test_semantic_notes_persist_in_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "investigations.sqlite"
    store = SQLiteInvestigationStore(db_path)
    source = store.create_data_source(DataSource(name="Orders"))
    notes = DataSourceSemanticNotes(
        data_source_id=source.data_source_id,
        source_description="Orders source",
        business_context="US market only",
        global_caveats=["Returns are not included"],
        column_notes=[
            ColumnSemanticNote(
                column_name="Sales",
                business_meaning="Revenue amount before discount",
                semantic_role=ColumnSemanticRole.TARGET,
            )
        ],
    )

    store.save_data_source_semantic_notes(notes)
    reloaded = SQLiteInvestigationStore(db_path)
    loaded = reloaded.get_data_source_semantic_notes(source.data_source_id)

    assert reloaded.get_schema_version() == 11
    assert loaded.business_context == "US market only"
    assert loaded.global_caveats == ["Returns are not included"]
    assert loaded.column_notes[0].semantic_role == ColumnSemanticRole.TARGET


def test_update_and_delete_column_semantic_note() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders"))

    updated = store.update_column_semantic_note(
        source.data_source_id,
        "Sales",
        ColumnSemanticNote(
            column_name="Sales",
            description="Revenue",
            semantic_role=ColumnSemanticRole.METRIC,
        ),
    )
    assert updated.column_notes[0].description == "Revenue"

    deleted = store.delete_column_semantic_note(source.data_source_id, "Sales")

    assert deleted.column_notes == []


def test_semantic_role_overrides_usage_context_role_and_adds_notes() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders"))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(pd.DataFrame({"Sales": [1.0, 2.0]})))
    store.update_column_semantic_note(
        source.data_source_id,
        "Sales",
        ColumnSemanticNote(
            column_name="Sales",
            display_name="Revenue",
            business_meaning="Revenue amount before discount",
            semantic_role=ColumnSemanticRole.TARGET,
            caveats=["Excludes returns"],
            examples=["100.25"],
        ),
    )

    context = build_data_source_usage_context(store, source.data_source_id)
    sales = context.column_summaries[0]

    assert sales.inferred_role == ColumnSemanticRole.TARGET
    assert any("overrides deterministic role metric" in note for note in sales.notes)
    assert any("Business meaning: Revenue amount before discount." in note for note in sales.notes)


def test_semantic_context_and_caveats_appear_in_usage_context() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders", description="Raw order export"))
    store.save_data_source_semantic_notes(
        DataSourceSemanticNotes(
            data_source_id=source.data_source_id,
            source_description="Curated orders table",
            business_context="Used for revenue reporting",
            global_caveats=["US market only"],
        )
    )

    context = build_data_source_usage_context(store, source.data_source_id)

    assert "Curated orders table" in context.description
    assert "Used for revenue reporting" in context.description
    assert "US market only" in context.caveats


def test_api_semantic_notes_endpoints_work() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders"))
    set_store_for_testing(store)
    client = TestClient(app)

    default_response = client.get(f"/data-sources/{source.data_source_id}/semantic-notes")
    saved = client.put(
        f"/data-sources/{source.data_source_id}/semantic-notes",
        json={
            "source_description": "Orders source",
            "business_context": "Retail revenue",
            "global_caveats": ["US only"],
        },
    )
    patched = client.patch(
        f"/data-sources/{source.data_source_id}/semantic-notes/columns/Sales",
        json={"business_meaning": "Revenue before discount", "semantic_role": "metric"},
    )
    deleted = client.delete(f"/data-sources/{source.data_source_id}/semantic-notes/columns/Sales")

    assert default_response.status_code == 200
    assert default_response.json()["column_notes"] == []
    assert saved.json()["business_context"] == "Retail revenue"
    assert patched.json()["column_notes"][0]["semantic_role"] == "metric"
    assert deleted.json()["column_notes"] == []


def test_create_list_investigation_runs_in_memory_store() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    run = store.create_investigation_run(
        InvestigationRun(
            investigation_id=investigation.investigation_id,
            status=InvestigationRunStatus.QUEUED,
            data_source_ids=["ds_1"],
        )
    )

    run.current_stage = InvestigationRunStage.BUILDING_CONTEXT
    run.status = InvestigationRunStatus.RUNNING
    updated = store.update_investigation_run(run)

    assert store.get_investigation_run(run.run_id).run_id == run.run_id
    assert store.list_runs_for_investigation(investigation.investigation_id)[0].run_id == run.run_id
    assert updated.current_stage == InvestigationRunStage.BUILDING_CONTEXT


def test_investigation_run_service_builds_context_and_completes() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Orders"))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(pd.DataFrame({"Sales": [1, 2]})))
    store.update_column_semantic_note(
        source.data_source_id,
        "Sales",
        ColumnSemanticNote(
            column_name="Sales",
            business_meaning="Revenue amount",
            semantic_role=ColumnSemanticRole.TARGET,
        ),
    )
    investigation = store.create_investigation("What changed?")

    def runner(**kwargs):
        return {"summary": "Done", "key_findings": ["Sales changed"]}

    service = InvestigationRunService(
        store=store,
        investigation_service=InvestigationService(store=store, runner=runner),
    )
    run = service.run_investigation(investigation.investigation_id, data_source_ids=[source.data_source_id])

    assert run.status == InvestigationRunStatus.COMPLETED
    assert run.current_stage == InvestigationRunStage.COMPLETED
    assert run.run_context_summary["data_sources"][0]["semantic_column_notes_count"] == 1
    assert run.artifact_ids
    assert run.report_ids


def test_investigation_run_service_loads_uploaded_csv_dataframe(tmp_path: Path) -> None:
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text("sales,segment\n10,a\n20,b\n", encoding="utf-8")
    store = InvestigationStore()
    source = store.create_data_source(
        DataSource(name="Orders", data_source_type=DataSourceType.CSV, location=str(csv_path))
    )
    store.save_data_source_profile(source.data_source_id, profile_csv(csv_path))
    investigation = store.create_investigation("What changed?")
    captured = {}

    def runner(**kwargs):
        captured.update(kwargs)
        return {"summary": "Done", "key_findings": ["Sales changed"]}

    service = InvestigationRunService(
        store=store,
        investigation_service=InvestigationService(store=store, runner=runner),
    )
    run = service.run_investigation(investigation.investigation_id, data_source_ids=[source.data_source_id])

    assert run.status == InvestigationRunStatus.COMPLETED
    assert captured["df"].shape == (2, 2)
    assert list(captured["df"].columns) == ["sales", "segment"]


def test_investigation_run_service_failed_run_stores_error() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")

    def runner(**kwargs):
        raise RuntimeError("runner exploded")

    service = InvestigationRunService(
        store=store,
        investigation_service=InvestigationService(store=store, runner=runner),
    )
    run = service.run_investigation(investigation.investigation_id)

    assert run.status == InvestigationRunStatus.FAILED
    assert "runner exploded" in run.error_message


def test_investigation_runs_persist_in_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "investigations.sqlite"
    store = SQLiteInvestigationStore(db_path)
    investigation = store.create_investigation("Question")
    run = store.create_investigation_run(
        InvestigationRun(
            investigation_id=investigation.investigation_id,
            status=InvestigationRunStatus.RUNNING,
            current_stage=InvestigationRunStage.RUNNING_ANALYSIS,
            run_context_summary={"data_sources_count": 0},
        )
    )

    reloaded = SQLiteInvestigationStore(db_path)
    loaded = reloaded.get_investigation_run(run.run_id)

    assert reloaded.get_schema_version() == 11
    assert loaded.current_stage == InvestigationRunStage.RUNNING_ANALYSIS
    assert reloaded.list_runs_for_investigation(investigation.investigation_id)[0].run_id == run.run_id


def test_api_run_endpoints_work(monkeypatch) -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    set_store_for_testing(store)
    client = TestClient(app)

    class FakeRunService:
        def __init__(self, store):
            self.store = store

        def run_investigation(self, investigation_id, data_source_ids=None, force_refresh_context=False):
            return self.store.create_investigation_run(
                InvestigationRun(
                    investigation_id=investigation_id,
                    status=InvestigationRunStatus.COMPLETED,
                    current_stage=InvestigationRunStage.COMPLETED,
                    data_source_ids=data_source_ids or [],
                    run_context_summary={"data_sources_count": len(data_source_ids or [])},
                )
            )

    monkeypatch.setattr("source.api.routes.investigations.InvestigationRunService", FakeRunService)
    created = client.post(f"/investigations/{investigation.investigation_id}/run", json={"data_source_ids": []})
    run_id = created.json()["run_id"]
    listed = client.get(f"/investigations/{investigation.investigation_id}/runs")
    fetched = client.get(f"/investigation-runs/{run_id}")

    assert created.status_code == 200
    assert created.json()["status"] == "completed"
    assert listed.json()[0]["run_id"] == run_id
    assert fetched.json()["current_stage"] == "completed"


def test_add_and_list_run_events_in_memory_store() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    run = store.create_investigation_run(
        InvestigationRun(
            investigation_id=investigation.investigation_id,
            status=InvestigationRunStatus.RUNNING,
        )
    )
    first = store.add_investigation_run_event(
        InvestigationRunEvent(
            run_id=run.run_id,
            investigation_id=investigation.investigation_id,
            event_type=InvestigationRunEventType.STAGE_STARTED,
            stage=InvestigationRunStage.PREPARING_DATA,
            message="Preparing data started.",
        )
    )
    second = store.add_investigation_run_event(
        InvestigationRunEvent(
            run_id=run.run_id,
            investigation_id=investigation.investigation_id,
            event_type=InvestigationRunEventType.INFO,
            stage=InvestigationRunStage.PREPARING_DATA,
            message="Prepared source.",
        )
    )

    events = store.list_investigation_run_events(run.run_id)

    assert [event.event_id for event in events] == [first.event_id, second.event_id]
    assert store.list_events_for_investigation(investigation.investigation_id)[-1].message == "Prepared source."


def test_run_events_persist_in_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "investigations.sqlite"
    store = SQLiteInvestigationStore(db_path)
    investigation = store.create_investigation("Question")
    run = store.create_investigation_run(InvestigationRun(investigation_id=investigation.investigation_id))
    event = store.add_investigation_run_event(
        InvestigationRunEvent(
            run_id=run.run_id,
            investigation_id=investigation.investigation_id,
            event_type=InvestigationRunEventType.WARNING,
            stage=InvestigationRunStage.BUILDING_CONTEXT,
            message="No profile found.",
            severity=InvestigationRunEventSeverity.WARNING,
            metadata={"data_source_id": "ds_missing"},
        )
    )

    reloaded = SQLiteInvestigationStore(db_path)
    events = reloaded.list_investigation_run_events(run.run_id)

    assert reloaded.get_schema_version() == 11
    assert events[0].event_id == event.event_id
    assert events[0].severity == InvestigationRunEventSeverity.WARNING
    assert events[0].metadata["data_source_id"] == "ds_missing"


def test_run_service_emits_stage_and_validation_events() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")

    def runner(**kwargs):
        return {"summary": "Done", "key_findings": ["A finding"]}

    service = InvestigationRunService(
        store=store,
        investigation_service=InvestigationService(store=store, runner=runner),
    )
    run = service.run_investigation(investigation.investigation_id)
    events = store.list_investigation_run_events(run.run_id)
    event_types = [event.event_type for event in events]

    assert InvestigationRunEventType.STAGE_STARTED in event_types
    assert InvestigationRunEventType.STAGE_COMPLETED in event_types
    assert InvestigationRunEventType.VALIDATION_CHECK in event_types
    assert any(event.stage == InvestigationRunStage.COMPLETED for event in events)


def test_run_service_emits_error_event_on_failure() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")

    def runner(**kwargs):
        raise RuntimeError("boom")

    service = InvestigationRunService(
        store=store,
        investigation_service=InvestigationService(store=store, runner=runner),
    )
    run = service.run_investigation(investigation.investigation_id)
    events = store.list_investigation_run_events(run.run_id)

    assert run.status == InvestigationRunStatus.FAILED
    assert any(event.event_type == InvestigationRunEventType.ERROR for event in events)


def test_api_run_events_endpoints_work(monkeypatch) -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    run = store.create_investigation_run(
        InvestigationRun(
            investigation_id=investigation.investigation_id,
            status=InvestigationRunStatus.COMPLETED,
            current_stage=InvestigationRunStage.COMPLETED,
        )
    )
    event = store.add_investigation_run_event(
        InvestigationRunEvent(
            run_id=run.run_id,
            investigation_id=investigation.investigation_id,
            event_type=InvestigationRunEventType.INFO,
            stage=InvestigationRunStage.COMPLETED,
            message="Run completed.",
        )
    )
    set_store_for_testing(store)
    client = TestClient(app)

    run_events = client.get(f"/investigation-runs/{run.run_id}/events")
    investigation_events = client.get(f"/investigations/{investigation.investigation_id}/events")

    assert run_events.status_code == 200
    assert run_events.json()["events"][0]["event_id"] == event.event_id
    assert run_events.json()["next_cursor"]
    assert investigation_events.status_code == 200
    assert investigation_events.json()["events"][0]["message"] == "Run completed."


def test_event_cursor_roundtrip_and_filtering() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    run = store.create_investigation_run(InvestigationRun(investigation_id=investigation.investigation_id))
    first = store.add_investigation_run_event(
        InvestigationRunEvent(run_id=run.run_id, investigation_id=investigation.investigation_id, message="First")
    )
    second = store.add_investigation_run_event(
        InvestigationRunEvent(run_id=run.run_id, investigation_id=investigation.investigation_id, message="Second")
    )
    cursor = encode_event_cursor(first)

    assert decode_event_cursor(cursor) == (first.created_at.isoformat(), first.event_id)
    assert filter_events_after([first, second], cursor)[0].event_id == second.event_id


def test_invalid_event_cursor_returns_from_beginning() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    run = store.create_investigation_run(InvestigationRun(investigation_id=investigation.investigation_id))
    event = store.add_investigation_run_event(
        InvestigationRunEvent(run_id=run.run_id, investigation_id=investigation.investigation_id, message="First")
    )

    assert filter_events_after([event], "not-a-cursor") == [event]


def test_event_stream_response_limit_next_cursor_and_has_more() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    run = store.create_investigation_run(InvestigationRun(investigation_id=investigation.investigation_id))
    events = [
        store.add_investigation_run_event(
            InvestigationRunEvent(run_id=run.run_id, investigation_id=investigation.investigation_id, message=str(idx))
        )
        for idx in range(3)
    ]

    first_page = build_event_stream_response(events, cursor=None, limit=2)
    second_page = build_event_stream_response(events, cursor=first_page["next_cursor"], limit=2)

    assert [event.message for event in first_page["events"]] == ["0", "1"]
    assert first_page["has_more"] is True
    assert first_page["next_cursor"] == encode_event_cursor(events[1])
    assert [event.message for event in second_page["events"]] == ["2"]
    assert second_page["has_more"] is False


def test_api_events_endpoint_supports_after_and_limit() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Question")
    run = store.create_investigation_run(InvestigationRun(investigation_id=investigation.investigation_id))
    first = store.add_investigation_run_event(
        InvestigationRunEvent(run_id=run.run_id, investigation_id=investigation.investigation_id, message="First")
    )
    second = store.add_investigation_run_event(
        InvestigationRunEvent(run_id=run.run_id, investigation_id=investigation.investigation_id, message="Second")
    )
    set_store_for_testing(store)
    client = TestClient(app)

    response = client.get(
        f"/investigation-runs/{run.run_id}/events",
        params={"after": encode_event_cursor(first), "limit": 1},
    )

    assert response.status_code == 200
    assert response.json()["events"][0]["event_id"] == second.event_id
    assert response.json()["has_more"] is False
