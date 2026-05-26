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

    assert reloaded.get_schema_version() == 15
    assert loaded_investigation.linked_data_source_ids == [source.data_source_id]
    assert loaded_source.linked_investigation_ids == [investigation.investigation_id]
    assert loaded_profile.row_count == 2

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
