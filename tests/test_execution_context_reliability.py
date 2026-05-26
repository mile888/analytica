from __future__ import annotations

import pandas as pd
from dataclasses import asdict

from source.product.data_context import build_data_source_usage_context
from source.product.data_profiling import profile_dataframe
from source.product.data_sources import DataSource, DataSourceType
from source.product.execution_context import (
    execution_required_for_question,
    persist_dataset_runtime,
    resolve_dataset_runtime,
)
from source.product.fallback_analysis import deterministic_context_fallback
from source.product.investigation import ArtifactType, InvestigationMessage, InvestigationMessageRole, InvestigationMessageType
from source.product.run_service import InvestigationRunService
from source.product.service import InvestigationService
from source.product.store import InvestigationStore


def _neutral_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Metric": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "Location": ["North", "South", "North", "East", "South", "East"],
            "Entity": ["A", "B", "C", "D", "E", "F"],
            "Timestamp": pd.date_range("2026-01-01", periods=6, freq="D"),
            "Value": [1.0, 2.0, 3.0, 2.0, 4.0, 5.0],
            "Region": ["R1", "R1", "R2", "R2", "R1", "R2"],
        }
    )


def _service(store: InvestigationStore) -> InvestigationService:
    return InvestigationService(
        store=store,
        runner=lambda **_: {"summary": "Generic analytical narration without execution.", "artifacts": []},
    )


def _salary_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Salary": [120.0, 80.0, 90.0, 130.0],
            "City": ["A", "B", "A", "C"],
            "Team": ["X", "X", "Y", "Y"],
        }
    )


def _error_runner(**_: object) -> dict:
    return {"exec_error": "runner unavailable", "critic_verdict": "ERROR"}


def _ask(store: InvestigationStore, service: InvestigationRunService, investigation_id: str, text: str, data_source_id: str) -> str:
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation_id,
            role=InvestigationMessageRole.USER,
            message_type=InvestigationMessageType.FOLLOW_UP,
            content=text,
        )
    )
    service.run_investigation(investigation_id, data_source_ids=[data_source_id], message_id=message.message_id)
    assistant = [
        item
        for item in store.list_investigation_messages(investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT and item.metadata.get("response_to_message_id") == message.message_id
    ]
    return assistant[-1].content


def test_executable_dataset_runtime_resolves_and_queries_materialize_artifacts() -> None:
    df = _neutral_df()
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Neutral", data_source_type=DataSourceType.CSV))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(df))
    persisted = persist_dataset_runtime(store, source.data_source_id, df)

    loaded, resolved = resolve_dataset_runtime(store, source.data_source_id)
    assert persisted.executable_available is True
    assert resolved.executable_available is True
    assert loaded.shape == df.shape

    queries = [
        "Top Location by Metric",
        "Average Metric by Location",
        "Histogram of Metric",
        "Compare Metric by Region",
        "Trend of Metric over Timestamp",
    ]
    for query in queries:
        investigation = store.create_investigation(query)
        updated = _service(store).run_investigation(investigation.investigation_id, df=df)
        assert updated.report is not None
        assert updated.artifacts
        assert not any(
            artifact.metadata.get("execution_context_unavailable")
            for artifact in updated.artifacts
        )


def test_profile_only_dataset_returns_execution_context_error_without_fake_artifacts() -> None:
    df = _neutral_df()
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Profile only", data_source_type=DataSourceType.CSV))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(df))
    context = build_data_source_usage_context(store, source.data_source_id)

    queries = [
        "Top Location by Metric",
        "Average Metric by Location",
        "Histogram of Metric",
        "Compare Metric by Region",
        "Trend of Metric over Timestamp",
    ]
    for query in queries:
        investigation = store.create_investigation(query)
        updated = _service(store).run_investigation(
            investigation.investigation_id,
            df=None,
            data_context={
                "data_source_ids": [source.data_source_id],
                "data_source_usage_contexts": [asdict(context)],
                "active_question": query,
            },
        )
        assert updated.report is not None
        assert "cannot compute" in updated.report.summary.lower()
        assert "raw rows" in updated.report.summary.lower()
        assert updated.artifacts == []
        assert updated.findings == []


def test_context_fallback_does_not_create_profile_based_analytical_narration() -> None:
    df = _neutral_df()
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Profile only", data_source_type=DataSourceType.CSV))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(df))
    context = build_data_source_usage_context(store, source.data_source_id)

    result = deterministic_context_fallback(
        "Build a histogram of Metric",
        {"data_source_usage_contexts": [asdict(context)]},
    )

    assert result is not None
    assert result["artifacts"] == []
    assert result["trace_metadata"]["execution_context_unavailable"] is True
    assert "histogram of" not in result["summary"].lower()
    assert "clearest view" not in result["summary"].lower()


def test_histogram_unavailable_runtime_creates_no_chart_artifact() -> None:
    df = _neutral_df()
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Profile only", data_source_type=DataSourceType.CSV))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(df))
    context = build_data_source_usage_context(store, source.data_source_id)

    investigation = store.create_investigation("Histogram of Metric")
    updated = _service(store).run_investigation(
        investigation.investigation_id,
        df=None,
        data_context={
            "data_source_ids": [source.data_source_id],
            "data_source_usage_contexts": [asdict(context)],
            "active_question": "Histogram of Metric",
        },
    )

    assert not any(artifact.artifact_type == ArtifactType.CHART for artifact in updated.artifacts)
    assert updated.artifacts == []


def test_execution_required_detection_uses_generic_neutral_schema() -> None:
    df = _neutral_df()
    context = {
        "data_source_usage_contexts": [
            {
                "column_summaries": [
                    {"name": column, "inferred_role": "metric" if column in {"Metric", "Value"} else "dimension"}
                    for column in df.columns
                ]
            }
        ]
    }

    assert execution_required_for_question("Top Location by Metric", data_context=context)
    assert execution_required_for_question("Histogram of Value", data_context=context)
    assert execution_required_for_question("Trend of Metric over Timestamp", data_context=context)


def test_repeated_executable_query_after_missing_runtime_retries_error_not_vague_followup() -> None:
    df = _salary_df()
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Profile only", data_source_type=DataSourceType.CSV))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(df))
    investigation = store.create_investigation("top cities by salary")
    store.link_data_source_to_investigation(investigation.investigation_id, source.data_source_id)
    service = InvestigationRunService(store, InvestigationService(store=store, runner=_error_runner))

    first = _ask(store, service, investigation.investigation_id, "top cities by salary", source.data_source_id)
    second = _ask(store, service, investigation.investigation_id, "top cities by salary", source.data_source_id)

    assert "raw rows are not currently attached" in first.lower()
    assert "raw rows are not currently attached" in second.lower()
    assert "more specific analytical anchor" not in second.lower()
    state = store.get_investigation(investigation.investigation_id).metadata["conversation_state"]
    assert state["last_error_type"] == "execution_context_unavailable"
    assert not state.get("active_artifact_id")


def test_repeated_executable_query_executes_after_runtime_reattached() -> None:
    df = _salary_df()
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Profile only", data_source_type=DataSourceType.CSV))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(df))
    investigation = store.create_investigation("top cities by salary")
    store.link_data_source_to_investigation(investigation.investigation_id, source.data_source_id)
    service = InvestigationRunService(store, InvestigationService(store=store, runner=_error_runner))

    first = _ask(store, service, investigation.investigation_id, "top cities by salary", source.data_source_id)
    persist_dataset_runtime(store, source.data_source_id, df)
    second = _ask(store, service, investigation.investigation_id, "top cities by salary", source.data_source_id)
    updated = store.get_investigation(investigation.investigation_id)

    assert "raw rows are not currently attached" in first.lower()
    assert "raw rows are not currently attached" not in second.lower()
    assert any(artifact.artifact_type == ArtifactType.CHART for artifact in updated.artifacts)


def test_execution_error_does_not_create_active_analytical_branch_for_anomaly_followup() -> None:
    df = _salary_df()
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Profile only", data_source_type=DataSourceType.CSV))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(df))
    investigation = store.create_investigation("top cities by salary")
    store.link_data_source_to_investigation(investigation.investigation_id, source.data_source_id)
    service = InvestigationRunService(store, InvestigationService(store=store, runner=_error_runner))

    _ask(store, service, investigation.investigation_id, "top cities by salary", source.data_source_id)
    anomaly = _ask(store, service, investigation.investigation_id, "Are there anomalies?", source.data_source_id)
    state = store.get_investigation(investigation.investigation_id).metadata["conversation_state"]

    assert "raw rows are not currently attached" in anomaly.lower()
    assert state["last_error_type"] == "execution_context_unavailable"
    assert not state.get("active_branch_id")


def test_executable_query_is_not_classified_as_vague_followup_after_error() -> None:
    df = _salary_df()
    store = InvestigationStore()
    source = store.create_data_source(DataSource(name="Profile only", data_source_type=DataSourceType.CSV))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(df))
    investigation = store.create_investigation("average salary by city")
    store.link_data_source_to_investigation(investigation.investigation_id, source.data_source_id)
    service = InvestigationRunService(store, InvestigationService(store=store, runner=_error_runner))

    _ask(store, service, investigation.investigation_id, "top cities by salary", source.data_source_id)
    answer = _ask(store, service, investigation.investigation_id, "average salary by city", source.data_source_id)

    assert execution_required_for_question("average salary by city")
    assert "more specific analytical anchor" not in answer.lower()
    assert "raw rows are not currently attached" in answer.lower()
