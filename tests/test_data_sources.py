from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from source.api.app import app
from source.api.deps import set_store_for_testing
from source.dataframe import read_csv_dataset
from source.product.data_context import build_data_source_usage_context
from source.product.dataframe_resolver import resolve_dataframe_from_data_source
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
    ArtifactType,
    InvestigationMessage,
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
    source = store.create_data_source(DataSource(name="Dataset", data_source_type=DataSourceType.CSV, tags=[" Metric "]))

    assert store.get_data_source(source.data_source_id).name == "Dataset"
    assert store.list_data_sources()[0].data_source_id == source.data_source_id
    assert source.tags == ["metric"]


def test_profile_dataframe_and_empty_dataframe_work() -> None:
    df = pd.DataFrame({"metric_value": [10.0, 20.0, None], "category_label": ["a", "b", "a"]})
    profile = profile_dataframe(df)
    empty = profile_dataframe(pd.DataFrame())

    assert profile.row_count == 3
    assert profile.column_count == 2
    assert profile.missing_summary["metric_value"] == 1
    assert "metric_value" in profile.numeric_summary
    assert "category_label" in profile.categorical_summary
    assert empty.row_count == 0
    assert empty.column_count == 0


def test_profile_csv_works(tmp_path: Path) -> None:
    path = tmp_path / "dataset.csv"
    path.write_text("metric_value,category_label\n10,a\n20,b\n", encoding="utf-8")

    profile = profile_csv(path)

    assert profile.row_count == 2
    assert [column.name for column in profile.columns] == ["metric_value", "category_label"]


def test_csv_reader_recovers_semicolon_delimiter_when_default_parse_glues_columns(tmp_path: Path) -> None:
    path = tmp_path / "dataset.csv"
    path.write_text("metric_value;category_label\n10;a\n20;b\n", encoding="utf-8")

    df = read_csv_dataset(path)
    profile = profile_csv(path)

    assert list(df.columns) == ["metric_value", "category_label"]
    assert df.to_dict(orient="records") == [
        {"metric_value": 10, "category_label": "a"},
        {"metric_value": 20, "category_label": "b"},
    ]
    assert profile.column_count == 2


def test_profile_and_links_persist_in_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "investigations.sqlite"
    store = SQLiteInvestigationStore(db_path)
    source = store.create_data_source(DataSource(name="Dataset", data_source_type=DataSourceType.CSV))
    investigation = store.create_investigation("Question")
    profile = profile_dataframe(pd.DataFrame({"metric_value": [1, 2]}))

    store.save_data_source_profile(source.data_source_id, profile)
    store.link_data_source_to_investigation(investigation.investigation_id, source.data_source_id)

    reloaded = SQLiteInvestigationStore(db_path)
    loaded_investigation = reloaded.get_investigation(investigation.investigation_id)
    loaded_source = reloaded.get_data_source(source.data_source_id)
    loaded_profile = reloaded.get_data_source_profile(source.data_source_id)

    assert reloaded.get_schema_version() == 14
    assert loaded_investigation.linked_data_source_ids == [source.data_source_id]
    assert loaded_source.linked_investigation_ids == [investigation.investigation_id]
    assert loaded_profile.row_count == 2


def test_archive_data_source() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Dataset"))

    archived = store.archive_data_source(source.data_source_id)

    assert archived.status == DataSourceStatus.ARCHIVED
    assert store.list_data_sources(status="archived")[0].data_source_id == source.data_source_id


def test_api_data_source_routes_work() -> None:
    store = InvestigationStore()
    set_store_for_testing(store)
    client = TestClient(app)

    created = client.post(
        "/data-sources",
        json={"name": "Dataset", "type": "csv", "location": "/tmp/dataset.csv", "tags": ["Metric Value"]},
    )
    source_id = created.json()["data_source_id"]
    listed = client.get("/data-sources")
    fetched = client.get(f"/data-sources/{source_id}")
    updated = client.patch(f"/data-sources/{source_id}", json={"description": "Dataset data", "tags": ["Metric"]})

    assert created.status_code == 200
    assert listed.json()[0]["data_source_id"] == source_id
    assert fetched.json()["name"] == "Dataset"
    assert updated.json()["description"] == "Dataset data"
    assert updated.json()["tags"] == ["metric"]


def test_api_upload_csv_creates_source_saves_file_and_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    store = InvestigationStore()
    set_store_for_testing(store)
    client = TestClient(app)

    response = client.post(
        "/data-sources/upload-csv",
        data={"name": "Dataset", "description": "Uploaded dataset", "tags": ["Metric, Operations"]},
        files={"file": ("dataset.csv", b"metric_value,category_label\n10,a\n20,b\n", "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    source = payload["data_source"]
    assert source["name"] == "Dataset"
    assert source["data_source_type"] == "csv"
    assert source["tags"] == ["metric", "operations"]
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
        files={"file": ("dataset.txt", b"metric_value\n10\n", "text/plain")},
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

    # Zero-byte files are not valid CSV — return 400
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_api_upload_ragged_csv_profiles_with_bad_lines_skipped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    set_store_for_testing(InvestigationStore())
    client = TestClient(app)

    response = client.post(
        "/data-sources/upload-csv",
        files={"file": ("ragged.csv", b"metric_value,category_label\n10,a\n20,b,extra\n30,c\n", "text/csv")},
    )

    assert response.status_code == 200
    assert response.json()["profile"]["row_count"] == 2
    assert response.json()["profile"]["column_count"] == 2


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
    source = store.create_data_source(DataSource(name="Dataset"))
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
    source = store.create_data_source(DataSource(name="Dataset"))
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
    source = store.create_data_source(DataSource(name="Dataset", data_source_type=DataSourceType.CSV))
    profile = profile_dataframe(
        pd.DataFrame(
            {
                "created_at": ["2026-01-01", "2026-01-02"],
                "entity_id": ["c1", "c2"],
                "metric_value": [10.0, 20.0],
                "category_label": ["group_a", "group_b"],
                "notes": [
                    "This entity provided a long qualitative note that should be treated as text.",
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
    assert roles["entity_id"] == ColumnInferredRole.IDENTIFIER
    assert roles["metric_value"] == ColumnInferredRole.METRIC
    assert roles["category_label"] == ColumnInferredRole.DIMENSION
    assert roles["notes"] == ColumnInferredRole.TEXT


def test_build_usage_context_without_profile_does_not_crash() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Dataset"))

    context = build_data_source_usage_context(store, source.data_source_id)

    assert context.schema_summary["row_count"] is None
    assert "No profile is available for this data source." in context.caveats


def test_usage_context_caveats_for_archived_source_and_missing_values() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Dataset", status=DataSourceStatus.ARCHIVED))
    profile = profile_dataframe(pd.DataFrame({"metric_value": [None, None, 3.0]}))
    store.save_data_source_profile(source.data_source_id, profile)

    context = build_data_source_usage_context(store, source.data_source_id)

    assert "Source status is archived." in context.caveats
    assert "Column metric_value has high missingness." in context.caveats


def test_usage_context_includes_previous_questions_from_linked_investigations() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Dataset"))
    investigation = store.create_investigation("Why did metric change?")
    store.link_data_source_to_investigation(investigation.investigation_id, source.data_source_id)

    context = build_data_source_usage_context(store, source.data_source_id)

    assert context.previous_questions == ["Why did metric change?"]


def test_api_usage_context_endpoint_works() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Dataset"))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(pd.DataFrame({"metric_value": [1, 2]})))
    set_store_for_testing(store)
    client = TestClient(app)

    response = client.get(f"/data-sources/{source.data_source_id}/usage-context")

    assert response.status_code == 200
    assert response.json()["data_source_id"] == source.data_source_id
    assert response.json()["column_summaries"][0]["inferred_role"] == "metric"


def test_investigation_service_passes_usage_context_to_runner_for_linked_source() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Dataset"))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(pd.DataFrame({"metric_value": [1, 2]})))
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
    assert context["data_source_usage_contexts"][0]["name"] == "Dataset"
    assert "Data source usage context" in context["data_context_prompt"]


def test_empty_semantic_notes_load_by_default() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Dataset"))

    notes = store.get_data_source_semantic_notes(source.data_source_id)

    assert notes.data_source_id == source.data_source_id
    assert notes.column_notes == []


def test_semantic_notes_persist_in_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "investigations.sqlite"
    store = SQLiteInvestigationStore(db_path)
    source = store.create_data_source(DataSource(name="Dataset"))
    notes = DataSourceSemanticNotes(
        data_source_id=source.data_source_id,
        source_description="Dataset source",
        business_context="US market only",
        global_caveats=["Returns are not included"],
        column_notes=[
            ColumnSemanticNote(
                column_name="Metric Value",
                business_meaning="Metric amount before discount",
                semantic_role=ColumnSemanticRole.TARGET,
            )
        ],
    )

    store.save_data_source_semantic_notes(notes)
    reloaded = SQLiteInvestigationStore(db_path)
    loaded = reloaded.get_data_source_semantic_notes(source.data_source_id)

    assert reloaded.get_schema_version() == 14
    assert loaded.business_context == "US market only"
    assert loaded.global_caveats == ["Returns are not included"]
    assert loaded.column_notes[0].semantic_role == ColumnSemanticRole.TARGET


def test_update_and_delete_column_semantic_note() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Dataset"))

    updated = store.update_column_semantic_note(
        source.data_source_id,
        "Metric Value",
        ColumnSemanticNote(
            column_name="Metric Value",
            description="Metric",
            semantic_role=ColumnSemanticRole.METRIC,
        ),
    )
    assert updated.column_notes[0].description == "Metric"

    deleted = store.delete_column_semantic_note(source.data_source_id, "Metric Value")

    assert deleted.column_notes == []


def test_semantic_role_overrides_usage_context_role_and_adds_notes() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Dataset"))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(pd.DataFrame({"Metric Value": [1.0, 2.0]})))
    store.update_column_semantic_note(
        source.data_source_id,
        "Metric Value",
        ColumnSemanticNote(
            column_name="Metric Value",
            display_name="Metric",
            business_meaning="Metric amount before discount",
            semantic_role=ColumnSemanticRole.TARGET,
            caveats=["Excludes returns"],
            examples=["100.25"],
        ),
    )

    context = build_data_source_usage_context(store, source.data_source_id)
    metric_value = context.column_summaries[0]

    assert metric_value.inferred_role == ColumnSemanticRole.TARGET
    assert any("overrides deterministic role metric" in note for note in metric_value.notes)
    assert any("Business meaning: Metric amount before discount." in note for note in metric_value.notes)


def test_semantic_context_and_caveats_appear_in_usage_context() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Dataset", description="Raw dataset export"))
    store.save_data_source_semantic_notes(
        DataSourceSemanticNotes(
            data_source_id=source.data_source_id,
            source_description="Curated dataset table",
            business_context="Used for metric reporting",
            global_caveats=["US market only"],
        )
    )

    context = build_data_source_usage_context(store, source.data_source_id)

    assert "Curated dataset table" in context.description
    assert "Used for metric reporting" in context.description
    assert "US market only" in context.caveats


def test_api_semantic_notes_endpoints_work() -> None:
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Dataset"))
    set_store_for_testing(store)
    client = TestClient(app)

    default_response = client.get(f"/data-sources/{source.data_source_id}/semantic-notes")
    saved = client.put(
        f"/data-sources/{source.data_source_id}/semantic-notes",
        json={
            "source_description": "Dataset source",
            "business_context": "Operations metric",
            "global_caveats": ["US only"],
        },
    )
    patched = client.patch(
        f"/data-sources/{source.data_source_id}/semantic-notes/columns/Metric Value",
        json={"business_meaning": "Metric before discount", "semantic_role": "metric"},
    )
    deleted = client.delete(f"/data-sources/{source.data_source_id}/semantic-notes/columns/Metric Value")

    assert default_response.status_code == 200
    assert default_response.json()["column_notes"] == []
    assert saved.json()["business_context"] == "Operations metric"
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
    source = store.create_data_source(DataSource(name="Dataset"))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(pd.DataFrame({"Metric Value": [1, 2]})))
    store.update_column_semantic_note(
        source.data_source_id,
        "Metric Value",
        ColumnSemanticNote(
            column_name="Metric Value",
            business_meaning="Metric amount",
            semantic_role=ColumnSemanticRole.TARGET,
        ),
    )
    investigation = store.create_investigation("What changed?")

    def runner(**kwargs):
        return {"summary": "Done", "key_findings": ["Metric Value changed"]}

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
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("metric_value,category_label\n10,a\n20,b\n", encoding="utf-8")
    store = InvestigationStore()
    source = store.create_data_source(
        DataSource(name="Dataset", data_source_type=DataSourceType.CSV, location=str(csv_path))
    )
    store.save_data_source_profile(source.data_source_id, profile_csv(csv_path))
    investigation = store.create_investigation("What changed?")
    captured = {}

    def runner(**kwargs):
        captured.update(kwargs)
        return {"summary": "Done", "key_findings": ["Metric Value changed"]}

    service = InvestigationRunService(
        store=store,
        investigation_service=InvestigationService(store=store, runner=runner),
    )
    run = service.run_investigation(investigation.investigation_id, data_source_ids=[source.data_source_id])

    assert run.status == InvestigationRunStatus.COMPLETED
    assert captured["df"].shape == (2, 2)
    assert list(captured["df"].columns) == ["metric_value", "category_label"]


def test_dataframe_resolver_loads_relative_project_csv_path(tmp_path: Path, monkeypatch) -> None:
    relative_path = Path(".analytica/uploads/dataset.csv")
    csv_path = tmp_path / relative_path
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("metric_value,category_label\n10,a\n20,b\n", encoding="utf-8")
    monkeypatch.setattr("source.product.dataframe_resolver.PROJECT_ROOT", tmp_path)
    store = InvestigationStore()
    source = store.create_data_source(
        DataSource(name="Dataset", data_source_type=DataSourceType.CSV, location=str(relative_path))
    )

    df = resolve_dataframe_from_data_source(store, source.data_source_id)

    assert df is not None
    assert df.shape == (2, 2)
    assert list(df.columns) == ["metric_value", "category_label"]


def test_dataframe_resolver_missing_file_returns_none(tmp_path: Path) -> None:
    store = InvestigationStore()
    source = store.create_data_source(
        DataSource(name="Dataset", data_source_type=DataSourceType.CSV, location=str(tmp_path / "missing.csv"))
    )

    assert resolve_dataframe_from_data_source(store, source.data_source_id) is None


def test_follow_up_run_uses_linked_csv_for_concrete_group_analysis(tmp_path: Path) -> None:
    csv_path = tmp_path / "careers.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Job_Title,Salary_LPA",
                "Research Scientist,80",
                "Research Scientist,120",
                "NLP Engineer,90",
                "NLP Engineer,95",
                "UI Designer,40",
                "UI Designer,45",
                "Support Analyst,50",
            ]
        ),
        encoding="utf-8",
    )
    store = InvestigationStore()
    source = store.create_data_source(
        DataSource(name="Careers", data_source_type=DataSourceType.CSV, location=str(csv_path))
    )
    store.save_data_source_profile(source.data_source_id, profile_csv(csv_path))
    investigation = store.create_investigation("What can you say about this dataset?")
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="How does Salary_LPA vary across Job_Title?",
        )
    )

    def error_runner(**kwargs):
        return {"exec_error": "runner unavailable", "critic_verdict": "ERROR"}

    service = InvestigationRunService(
        store=store,
        investigation_service=InvestigationService(store=store, runner=error_runner),
    )
    run = service.run_investigation(
        investigation.investigation_id,
        data_source_ids=[source.data_source_id],
        message_id=message.message_id,
    )
    updated = store.get_investigation(investigation.investigation_id)

    assert run.status == InvestigationRunStatus.COMPLETED
    assert "Research Scientist" in updated.report.summary
    assert "UI Designer" in updated.report.summary
    assert "100.00" in updated.report.summary
    assert any(artifact.artifact_type == ArtifactType.CHART for artifact in updated.artifacts)


def test_unusual_group_follow_up_uses_concrete_groups_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "careers.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Job_Title,Salary_LPA",
                "Research Scientist,75",
                "Research Scientist,150",
                "Research Scientist,82",
                "NLP Engineer,88",
                "NLP Engineer,90",
                "NLP Engineer,92",
                "UI Designer,40",
                "UI Designer,42",
                "UI Designer,43",
            ]
        ),
        encoding="utf-8",
    )
    store = InvestigationStore()
    source = store.create_data_source(
        DataSource(name="Careers", data_source_type=DataSourceType.CSV, location=str(csv_path))
    )
    investigation = store.create_investigation("What can you say about this dataset?")

    def error_runner(**kwargs):
        return {"exec_error": "runner unavailable", "critic_verdict": "ERROR"}

    service = InvestigationRunService(
        store=store,
        investigation_service=InvestigationService(store=store, runner=error_runner),
    )
    run = service.run_investigation(
        investigation.investigation_id,
        data_source_ids=[source.data_source_id],
        message_id=store.add_investigation_message(
            InvestigationMessage(
                investigation_id=investigation.investigation_id,
                content="Which Job_Title groups have unusual Salary_LPA values?",
            )
        ).message_id,
    )
    updated = store.get_investigation(investigation.investigation_id)

    assert run.status == InvestigationRunStatus.COMPLETED
    assert "Research Scientist" in updated.report.summary
    assert "least stable group" in updated.report.summary.lower() or "group spread" in updated.report.summary.lower()
    assert any("Unusual Salary_LPA groups" in artifact.title for artifact in updated.artifacts)


def test_chart_follow_up_uses_previous_metric_by_category_intent(tmp_path: Path) -> None:
    csv_path = tmp_path / "careers.csv"
    csv_path.write_text(
        "Job_Title,Salary_LPA,Applicants\nAI Engineer,80,20\nAI Engineer,100,30\nDesigner,40,12\nDesigner,45,10\n",
        encoding="utf-8",
    )
    store = InvestigationStore()
    source = store.create_data_source(
        DataSource(name="Careers", data_source_type=DataSourceType.CSV, location=str(csv_path))
    )
    investigation = store.create_investigation("What can you say about this dataset?")
    store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="How does Salary_LPA vary across Job_Title?",
        )
    )
    chart_message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="Build a chart.",
        )
    )

    def error_runner(**kwargs):
        return {"exec_error": "runner unavailable", "critic_verdict": "ERROR"}

    service = InvestigationRunService(
        store=store,
        investigation_service=InvestigationService(store=store, runner=error_runner),
    )
    run = service.run_investigation(
        investigation.investigation_id,
        data_source_ids=[source.data_source_id],
        message_id=chart_message.message_id,
    )
    updated = store.get_investigation(investigation.investigation_id)
    chart_titles = [artifact.title for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.CHART]

    assert run.status == InvestigationRunStatus.COMPLETED
    assert any("Salary_LPA" in title and "Job_Title" in title for title in chart_titles)


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

    assert reloaded.get_schema_version() == 14
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

        def run_investigation(
            self,
            investigation_id,
            data_source_ids=None,
            force_refresh_context=False,
            message_id=None,
            analysis_mode="exploration",
        ):
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

    assert reloaded.get_schema_version() == 14
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
            message="Latest analytical pass is available.",
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
    assert investigation_events.json()["events"][0]["message"] == "Latest analytical pass is available."


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
