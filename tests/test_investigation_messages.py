from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd

from source.api.app import app
from source.api.deps import set_store_for_testing
from source.product.investigation import (
    Artifact,
    ArtifactType,
    Finding,
    InvestigationMessage,
    InvestigationMessageRole,
    InvestigationMessageType,
    InvestigationRun,
    InvestigationRunStatus,
)
from source.product.run_service import InvestigationRunService
from source.product.run_service import is_semantically_redundant_response
from source.product.service import InvestigationService
from source.product.sqlite_store import SQLiteInvestigationStore
from source.product.store import InvestigationStore


def test_add_and_list_investigation_messages_in_memory_store() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Initial question")

    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            role=InvestigationMessageRole.USER,
            message_type=InvestigationMessageType.FOLLOW_UP,
            content="Follow-up question",
        )
    )

    assert store.get_investigation_message(message.message_id).content == "Follow-up question"
    assert store.list_investigation_messages(investigation.investigation_id)[0].message_id == message.message_id


def test_investigation_messages_persist_in_sqlite(tmp_path) -> None:
    db_path = tmp_path / "investigations.sqlite"
    store = SQLiteInvestigationStore(db_path)
    investigation = store.create_investigation("Initial question")
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="Persisted follow-up",
        )
    )

    reloaded = SQLiteInvestigationStore(db_path)

    assert reloaded.get_schema_version() == 14
    assert reloaded.get_investigation_message(message.message_id).content == "Persisted follow-up"
    assert reloaded.list_investigation_messages(investigation.investigation_id)[0].message_id == message.message_id


def test_follow_up_run_uses_message_context_and_adds_summary() -> None:
    captured: dict[str, object] = {}

    def runner(*, question, df=None, data_context=None):
        captured["question"] = question
        captured["data_context"] = data_context
        return {"summary": "Follow-up answer", "key_findings": ["A useful finding"]}

    store = InvestigationStore()
    investigation = store.create_investigation("Initial question")
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="Follow-up question",
        )
    )
    service = InvestigationRunService(store, InvestigationService(store, runner=runner))

    run = service.run_investigation(investigation.investigation_id, message_id=message.message_id)
    messages = store.list_investigation_messages(investigation.investigation_id)

    assert run.status == InvestigationRunStatus.COMPLETED
    assert run.metadata["message_id"] == message.message_id
    assert captured["question"] == "Follow-up question"
    conversation = captured["data_context"]["conversation_context"]
    assert conversation["active_message_id"] == message.message_id
    assert conversation["messages"][0]["content"] == "Follow-up question"
    summary = next(item for item in messages if item.message_type == InvestigationMessageType.RUN_SUMMARY)
    assert summary.content
    forbidden = ("finished the analysis", "analysis completed", "available above", "done")
    assert not any(phrase in summary.content.lower() for phrase in forbidden)


def test_old_run_without_messages_still_uses_initial_question() -> None:
    captured: dict[str, object] = {}

    def runner(*, question, df=None, data_context=None):
        captured["question"] = question
        return {"summary": "Initial answer"}

    store = InvestigationStore()
    investigation = store.create_investigation("Initial question")
    service = InvestigationRunService(store, InvestigationService(store, runner=runner))

    run = service.run_investigation(investigation.investigation_id)

    assert run.status == InvestigationRunStatus.COMPLETED
    assert captured["question"] == "Initial question"


def test_terminal_runner_summary_is_rewritten_as_open_investigation_language() -> None:
    def runner(*, question, df=None, data_context=None):
        return {"summary": "Done"}

    store = InvestigationStore()
    investigation = store.create_investigation("Initial question")
    service = InvestigationRunService(store, InvestigationService(store, runner=runner))

    service.run_investigation(investigation.investigation_id)
    messages = store.list_investigation_messages(investigation.investigation_id)
    summary = next(item for item in messages if item.message_type == InvestigationMessageType.RUN_SUMMARY)

    assert summary.content.lower() != "done"
    assert "keep exploring" not in summary.content.lower()
    assert "investigation has been updated" not in summary.content.lower()
    assert "focused question" in summary.content.lower()


def test_terminal_runner_summary_uses_latest_finding_instead_of_filler() -> None:
    def runner(*, question, df=None, data_context=None):
        return {
            "summary": "Analysis completed",
            "key_findings": ["Revenue is concentrated in a small number of regions."],
            "artifacts": [
                {
                    "artifact_type": "chart",
                    "title": "Revenue by region",
                    "content": {"chart_type": "bar"},
                }
            ],
        }

    store = InvestigationStore()
    investigation = store.create_investigation("Build a chart of revenue by region")
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="Could this be related to order volume?",
        )
    )
    service = InvestigationRunService(store, InvestigationService(store, runner=runner))

    service.run_investigation(investigation.investigation_id, message_id=message.message_id)
    messages = store.list_investigation_messages(investigation.investigation_id)
    summary = next(item for item in messages if item.message_type == InvestigationMessageType.RUN_SUMMARY)

    assert "Revenue is concentrated" in summary.content
    assert "analysis completed" not in summary.content.lower()
    assert "keep exploring" not in summary.content.lower()
    assert "investigation has been updated" not in summary.content.lower()


def test_follow_up_context_contains_thread_material_for_continuity() -> None:
    captured: dict[str, object] = {}

    def runner(*, question, df=None, data_context=None):
        captured["data_context"] = data_context
        return {"summary": "Order volume explains part of the regional gap."}

    store = InvestigationStore()
    investigation = store.create_investigation("Build a chart of revenue by region")
    store.add_finding(
        investigation.investigation_id,
        Finding(
            text="Revenue is concentrated in the strongest regions.",
            metadata={"analysis_type": "grouped_metric", "related_metrics": ["revenue"], "supporting_dimensions": ["region"]},
        ),
    )
    store.add_artifact(
        investigation.investigation_id,
        Artifact(
            artifact_type=ArtifactType.CHART,
            title="Revenue by region",
            content={"chart_type": "bar", "metric": "revenue", "x": "region"},
        ),
    )
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="Could this be related to order volume?",
        )
    )
    service = InvestigationRunService(store, InvestigationService(store, runner=runner))

    service.run_investigation(investigation.investigation_id, message_id=message.message_id)

    conversation = captured["data_context"]["conversation_context"]
    assert conversation["latest_chart_context"]["metric"] == "revenue"
    assert conversation["latest_chart_context"]["dimension"] == "region"
    assert conversation["latest_findings"]
    assert "Could this be related" in conversation["messages"][-1]["content"]


def test_empty_runner_output_triggers_visible_deterministic_answer() -> None:
    def runner(*, question, df=None, data_context=None):
        return {"summary": ""}

    df = pd.DataFrame({"metric_value": [10, 20, 30], "segment_label": ["A", "B", "B"]})
    store = InvestigationStore()
    investigation = store.create_investigation("Compare metric by segment")
    message = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="How does metric_value vary by segment_label?")
    )
    service = InvestigationRunService(store, InvestigationService(store, runner=runner))

    service.run_investigation(investigation.investigation_id, df=df, message_id=message.message_id)
    assistant_messages = [
        item for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert len(assistant_messages) == 1
    assert "metric_value" in assistant_messages[0].content
    assert "segment_label" in assistant_messages[0].content
    assert "keep exploring" not in assistant_messages[0].content.lower()


def test_multiple_follow_ups_each_get_one_assistant_response() -> None:
    def runner(*, question, df=None, data_context=None):
        return {"summary": f"Answer for {question}", "key_findings": [f"Finding for {question}"]}

    store = InvestigationStore()
    investigation = store.create_investigation("Initial")
    service = InvestigationRunService(store, InvestigationService(store, runner=runner))
    first = store.add_investigation_message(InvestigationMessage(investigation_id=investigation.investigation_id, content="First follow-up"))
    second = store.add_investigation_message(InvestigationMessage(investigation_id=investigation.investigation_id, content="Second follow-up"))

    service.run_investigation(investigation.investigation_id, message_id=first.message_id)
    service.run_investigation(investigation.investigation_id, message_id=second.message_id)
    assistant_messages = [
        item for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert len(assistant_messages) == 2
    assert {item.metadata.get("response_to_message_id") for item in assistant_messages} == {first.message_id, second.message_id}


def test_redundant_follow_up_answer_is_rewritten_with_current_focus() -> None:
    repeated = "The dataset has 9,898 rows and 18 columns. Useful quantitative fields include Sales."

    def runner(*, question, df=None, data_context=None):
        return {"summary": repeated}

    store = InvestigationStore()
    investigation = store.create_investigation("Build a chart of Sales by City")
    store.add_artifact(
        investigation.investigation_id,
        Artifact(
            artifact_type=ArtifactType.CHART,
            title="Average Sales by City",
            content={"chart_type": "bar", "metric": "Sales", "x": "City"},
        ),
    )
    first = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Which cities are strongest?")
    )
    second = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Does volume explain this gap?")
    )
    service = InvestigationRunService(store, InvestigationService(store, runner=runner))

    service.run_investigation(investigation.investigation_id, message_id=first.message_id)
    service.run_investigation(investigation.investigation_id, message_id=second.message_id)
    assistant_messages = [
        item for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert len(assistant_messages) == 2
    assert assistant_messages[0].content != repeated
    assert "Sales" in assistant_messages[0].content
    assert "City" in assistant_messages[0].content
    assert assistant_messages[1].content != repeated
    assert "Sales" in assistant_messages[1].content
    assert "City" in assistant_messages[1].content
    assert "dataset has 9,898 rows" not in assistant_messages[1].content.lower()


def test_long_assistant_answer_is_not_compacted_when_persisted() -> None:
    long_answer = "Sales by City shows a sustained difference. " + "Moscow remains the strongest group. " * 80

    def runner(*, question, df=None, data_context=None):
        return {"summary": long_answer}

    store = InvestigationStore()
    investigation = store.create_investigation("Explain the city chart")
    message = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Explain the city chart in detail.")
    )
    service = InvestigationRunService(store, InvestigationService(store, runner=runner))

    service.run_investigation(investigation.investigation_id, message_id=message.message_id)
    assistant = next(
        item for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    )

    assert assistant.content == long_answer.strip()
    assert not assistant.content.endswith("…")
    assert len(assistant.content) > 900


def test_semantic_redundancy_detection_catches_near_duplicate_answers() -> None:
    first = "Sales differs sharply across City. Moscow is highest, Kazan is lower, and the spread needs validation."
    second = "Sales differs sharply across City: Moscow is highest; Kazan is lower. The spread needs validation."

    assert is_semantically_redundant_response(second, first)
    assert not is_semantically_redundant_response("Duplicates affect group counts and should be checked.", first)


def test_compound_overview_dependency_question_answers_both_parts() -> None:
    df = pd.DataFrame(
        {
            "Sales": [100, 120, 80, 250],
            "Profit": [10, 15, 8, 35],
            "City": ["A", "A", "B", "C"],
            "Order Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"]),
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("что ты можешь сказать о данных, какие зависимости видишь")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    summary = store.get_investigation(investigation.investigation_id).report.summary

    assert "4 records" in summary or "4 rows" in summary or "4 строк" in summary
    assert "завис" in summary.lower() or "relationships" in summary.lower()
    assert "Sales" in summary
    assert "City" in summary or "Profit" in summary
    assert "Postal Code" not in summary
    assert "Key findings should" not in summary


def test_repeated_dataset_dependency_question_repeats_concrete_scan_not_generic_meta() -> None:
    df = pd.DataFrame(
        {
            "Postal Code": [10001, 10002, None, 10003, 10004],
            "Sales": [100, 500, 20, 700, 50],
            "City": ["A", "B", "A", "B", "C"],
            "Customer Name": ["Ann", "Bob", "Ann", "Cara", "Dan"],
            "Order Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]),
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("что ты можешь сказать о данных, какие зависимости видишь?")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    repeated = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="что ты можешь сказать о данных, какие зависимости видишь?",
        )
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=repeated.message_id)
    assistant = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert len(assistant) == 2
    assert all("Sales" in item for item in assistant)
    assert all("active conclusion" not in item.lower() for item in assistant)
    assert all("to move beyond" not in item.lower() for item in assistant)
    assert all("key findings should" not in item.lower() for item in assistant)
    assert all("compare `Postal Code`" not in item for item in assistant)


def test_quality_and_duplicate_followups_get_visible_answers() -> None:
    df = pd.DataFrame(
        {
            "Sales": [100, 100, 200, None],
            "City": ["A", "A", "B", "C"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Analyze sales by city")
    service = InvestigationRunService(store)
    quality = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Which quality issue affects the strongest conclusion most?")
    )
    duplicates = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Do duplicates inflate important group counts?")
    )

    service.run_investigation(investigation.investigation_id, df=df, message_id=quality.message_id)
    service.run_investigation(investigation.investigation_id, df=df, message_id=duplicates.message_id)
    assistant = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert len(assistant) == 2
    assert any("quality" in item.lower() or "missing" in item.lower() or "duplicate" in item.lower() for item in assistant)
    assert any("duplicate" in item.lower() for item in assistant)


def test_duplicate_orders_trigger_quality_branch_not_country_grouping() -> None:
    df = pd.DataFrame(
        {
            "Order ID": ["A", "A", "B", "C", "C"],
            "Sales": [100.0, 50.0, 200.0, 300.0, 300.0],
            "Country": ["United States"] * 5,
            "City": ["NYC", "NYC", "Boston", "Seattle", "Seattle"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Show Sales by Country")
    service = InvestigationRunService(store)
    message = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Есть ли duplicate orders?")
    )

    service.run_investigation(investigation.investigation_id, df=df, message_id=message.message_id)
    updated = store.get_investigation(investigation.investigation_id)

    assert "Duplicate-order check" in updated.report.summary
    assert "line items" in updated.report.summary
    assert "Sales by Country" not in updated.report.summary
    assert not any(artifact.artifact_type == ArtifactType.CHART and artifact.content.get("x") == "Country" for artifact in updated.artifacts)


def test_duplicate_impact_followup_estimates_sales_without_exact_duplicates() -> None:
    df = pd.DataFrame(
        {
            "Order ID": ["A", "A", "B", "C", "C"],
            "Sales": [100.0, 50.0, 200.0, 300.0, 300.0],
            "Country": ["United States"] * 5,
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Show Sales trend")
    service = InvestigationRunService(store)
    first = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Есть ли duplicate orders?")
    )
    second = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Как они влияют на sales?")
    )

    service.run_investigation(investigation.investigation_id, df=df, message_id=first.message_id)
    service.run_investigation(investigation.investigation_id, df=df, message_id=second.message_id)
    assistant = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert any("total is 950.00" in item and "after" in item and "650.00" in item for item in assistant)
    assert all("Sales by Country" not in item for item in assistant)


def test_hypothesis_high_sales_cities_grounded_to_city_concentration() -> None:
    df = pd.DataFrame(
        {
            "Sales": [1000.0, 20.0, 30.0, 400.0, 410.0, 390.0],
            "Country": ["United States"] * 6,
            "City": ["A", "A", "A", "B", "B", "B"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Sales by Country")
    service = InvestigationRunService(store)
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="Hypothesis: high sales cities are driven by a few large orders. Проверь.",
        )
    )

    service.run_investigation(investigation.investigation_id, df=df, message_id=message.message_id)
    updated = store.get_investigation(investigation.investigation_id)

    assert "`Sales` by `City`" in updated.report.summary
    assert "top-order concentration" in updated.report.summary
    assert "Country" not in updated.report.summary


def test_hypothesis_standard_class_matches_ship_mode_and_tests_volume() -> None:
    df = pd.DataFrame(
        {
            "Ship Mode": ["Standard Class", "Standard Class", "Standard Class", "Same Day", "Second Class"],
            "Sales": [100.0, 110.0, 90.0, 250.0, 240.0],
            "Country": ["United States"] * 5,
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Sales by Country")
    service = InvestigationRunService(store)
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="Hypothesis: Standard Class dominates because of volume, not order value.",
        )
    )

    service.run_investigation(investigation.investigation_id, df=df, message_id=message.message_id)
    updated = store.get_investigation(investigation.investigation_id)

    assert "`Standard Class` in `Ship Mode`" in updated.report.summary
    assert "volume" in updated.report.summary.lower()
    assert "Country" not in updated.report.summary


def test_hypothesis_technology_category_validates_outlier_concentration_not_segment_fallback() -> None:
    df = pd.DataFrame(
        {
            "Category": ["Technology", "Technology", "Furniture", "Office Supplies", "Furniture", "Technology"],
            "Segment": ["Consumer", "Corporate", "Consumer", "Home Office", "Corporate", "Consumer"],
            "Sales": [20.0, 5000.0, 25.0, 30.0, 22.0, 24.0],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Sales by Segment")
    service = InvestigationRunService(store)
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="Hypothesis: Technology category creates most outliers.",
        )
    )

    service.run_investigation(investigation.investigation_id, df=df, message_id=message.message_id)
    updated = store.get_investigation(investigation.investigation_id)

    assert "`Technology` in `Category`" in updated.report.summary or "`Technology`" in updated.report.summary
    assert "outlier" in updated.report.summary.lower()
    assert "Sales` between groups `Segment`" not in updated.report.summary


def test_hypothesis_sparse_cities_uses_active_city_branch_not_segment_fallback() -> None:
    df = pd.DataFrame(
        {
            "City": ["A", "B", "B", "B", "C", "C", "C"],
            "Segment": ["Consumer", "Consumer", "Corporate", "Corporate", "Home Office", "Consumer", "Corporate"],
            "Sales": [1000.0, 100.0, 110.0, 105.0, 90.0, 95.0, 100.0],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Построй график top cities by sales")
    service = InvestigationRunService(store)
    service.run_investigation(investigation.investigation_id, df=df)
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="Hypothesis: Some cities look strong only because of sparse data.",
        )
    )

    service.run_investigation(investigation.investigation_id, df=df, message_id=message.message_id)
    updated = store.get_investigation(investigation.investigation_id)

    assert "`City`" in updated.report.summary
    assert "sparse" in updated.report.summary.lower()
    assert "groups `Segment`" not in updated.report.summary


def test_russian_city_chart_request_chooses_city_not_country() -> None:
    df = pd.DataFrame(
        {
            "Sales": [100, 300, 50, 200],
            "Country": ["RU", "RU", "US", "US"],
            "City": ["Moscow", "Kazan", "Boston", "Chicago"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("построй график топ городов по sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    updated = store.get_investigation(investigation.investigation_id)
    chart = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.CHART)

    assert chart.content["x"] == "City"
    assert "City" in chart.title
    assert "Country" not in chart.title


def test_english_top_cities_chart_request_chooses_city_not_country() -> None:
    df = pd.DataFrame(
        {
            "Sales": [100, 300, 50, 200],
            "Country": ["United States", "United States", "United States", "United States"],
            "City": ["Boston", "Chicago", "Austin", "Chicago"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Построй график top cities by sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    updated = store.get_investigation(investigation.investigation_id)
    chart = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.CHART)

    assert chart.content["x"] == "City"
    assert chart.content["metric"] == "Sales"
    assert "Sales by City" == chart.title
    assert chart.content["aggregation"] == "sum"
    assert "Country" not in updated.report.summary


def test_top_cities_chart_keeps_high_cardinality_city_dimension() -> None:
    rows = 140
    df = pd.DataFrame(
        {
            "Sales": [float((idx % 25) + 1) for idx in range(rows)],
            "Country": ["United States"] * rows,
            "City": [f"City {idx:03d}" for idx in range(rows)],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Построй график top cities by sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    updated = store.get_investigation(investigation.investigation_id)
    chart = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.CHART)

    assert chart.content["x"] == "City"
    assert chart.content["metric"] == "Sales"
    assert "Sales by City" == chart.title
    assert chart.content["aggregation"] == "sum"
    assert "Country" not in updated.report.summary


def test_city_request_without_city_column_does_not_silently_use_country() -> None:
    df = pd.DataFrame({"Sales": [100, 300, 50], "Country": ["RU", "RU", "US"]})
    store = InvestigationStore()
    investigation = store.create_investigation("построй график топ городов по sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    updated = store.get_investigation(investigation.investigation_id)

    assert "could not find a city-level field" in updated.report.summary
    assert not any(artifact.artifact_type == ArtifactType.CHART and artifact.content.get("x") == "Country" for artifact in updated.artifacts)


def test_api_investigation_message_endpoints_and_run_with_message(monkeypatch) -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Initial question")
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
            run = self.store.create_investigation_run(
                InvestigationRun(
                    investigation_id=investigation_id,
                    status=InvestigationRunStatus.COMPLETED,
                    metadata={"message_id": message_id},
                )
            )
            return run

    monkeypatch.setattr("source.api.routes.investigations.InvestigationRunService", FakeRunService)

    created = client.post(
        f"/investigations/{investigation.investigation_id}/messages",
        json={"content": "API follow-up", "type": "follow_up"},
    )
    message_id = created.json()["message_id"]
    listed = client.get(f"/investigations/{investigation.investigation_id}/messages")
    run = client.post(f"/investigations/{investigation.investigation_id}/run", json={"message_id": message_id})

    assert created.status_code == 200
    assert listed.status_code == 200
    assert listed.json()[0]["content"] == "API follow-up"
    assert run.status_code == 200
    assert run.json()["metadata"]["message_id"] == message_id


def test_full_conversational_thread_stays_on_city_sales_chart() -> None:
    def generic_runner(*, question, df=None, data_context=None):
        return {"summary": "The dataset has rows and columns. A useful next step would be to keep exploring."}

    df = pd.DataFrame(
        {
            "Sales": [2354, 1600, 1260, 50, 70, 22638, 30, 20, 900, 850],
            "City": ["Jamestown", "Cheyenne", "Bellingham", "Norman", "Norman", "Jacksonville", "Abilene", "Elyria", "Buffalo", "Sparks"],
            "Order ID": ["o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9", "o10"],
            "Segment": ["Consumer", "Corporate", "Consumer", "Home", "Home", "Corporate", "Consumer", "Consumer", "Corporate", "Home"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("что ты можешь сказать о данных, какие зависимости видишь?")
    service = InvestigationRunService(store, InvestigationService(store, runner=generic_runner))

    service.run_investigation(investigation.investigation_id, df=df)
    chart_request = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="построй график топ городов по sales")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=chart_request.message_id)
    strongest = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="какие города самые сильные?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=strongest.message_id)
    anomalies = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="есть ли аномалии?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=anomalies.message_id)
    volume = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="связано ли это с объемом заказов?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=volume.message_id)

    assistant = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]
    updated = store.get_investigation(investigation.investigation_id)
    state = updated.metadata["conversation_state"]

    assert len(assistant) == 5
    assert all("useful next step would be" not in item.lower() for item in assistant)
    assert all("i would keep this follow-up" not in item.lower() for item in assistant)
    assert all("dataset has rows and columns" not in item.lower() for item in assistant[1:])
    assert any("`Sales` by `City` is led" in item and "total `Sales`" in item for item in assistant)
    assert any("strongest `City`" in item and "Jamestown" in item for item in assistant)
    assert any("anomal" in item.lower() or "unusual" in item.lower() for item in assistant)
    assert any("record volume" in item.lower() or "volume" in item.lower() or "объем" in item.lower() for item in assistant)
    assert all("average `Postal Code`" not in item for item in assistant[2:])
    assert all("`Postal Code` by `Ship Mode`" not in item for item in assistant[2:])
    assert state["active_metric"] == "Sales"
    assert state["active_dimension"] == "City"
    assert state["phase"] in {"focused_investigation", "validation", "anomaly_analysis"}


def test_persistent_conversation_state_is_used_in_follow_up_context() -> None:
    captured: dict[str, object] = {}

    def runner(*, question, df=None, data_context=None):
        captured["context"] = data_context["conversation_context"]
        return {"summary": "Record volume explains only part of the city gap."}

    store = InvestigationStore()
    investigation = store.create_investigation("построй график топ городов по sales")
    store.add_artifact(
        investigation.investigation_id,
        Artifact(
            artifact_type=ArtifactType.CHART,
            title="Average Sales by City",
            content={"chart_type": "bar", "metric": "Sales", "x": "City", "y": "mean", "rows": [{"City": "A", "mean": 10}]},
            metadata={"metric": "Sales", "dimension": "City", "aggregation": "mean", "ranking_scope": "Top cities by average Sales"},
        ),
    )
    message = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="связано ли это с объемом заказов?")
    )
    service = InvestigationRunService(store, InvestigationService(store, runner=runner))

    service.run_investigation(investigation.investigation_id, message_id=message.message_id)

    context = captured["context"]
    assert context["resolved_intent"]["primary"] == "causal_hypothesis"
    assert context["conversation_state"]["active_metric"] == "Sales"
    assert context["conversation_state"]["active_dimension"] == "City"
    assert context["latest_chart_context"]["aggregation"] == "mean"


def test_authoritative_state_prevents_silent_metric_and_dimension_reassignment() -> None:
    df = pd.DataFrame(
        {
            "Postal Code": [10001, 99301, 1040, 98208, 2149, 60601],
            "Sales": [2354, 1603, 1, 22638, 20, 900],
            "Ship Mode": ["Standard Class", "Second Class", "First Class", "Standard Class", "Same Day", "First Class"],
            "City": ["Jamestown", "Cheyenne", "Abilene", "Jacksonville", "Elyria", "Buffalo"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("построй график топ городов по sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    strongest = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="какие города самые сильные?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=strongest.message_id)
    anomalies = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="есть ли аномалии?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=anomalies.message_id)
    volume = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="связано ли это с объемом заказов?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=volume.message_id)

    assistant = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert any(("среднему `Sales`" in item or "average `Sales`" in item) and "`City`" in item for item in assistant)
    assert any(("Самые сильные `City`" in item or "strongest `City`" in item) and "`Sales`" in item for item in assistant)
    assert any("`Sales`" in item and ("record volume across `City`" in item or "объем" in item.lower() or "volume" in item.lower()) for item in assistant)
    assert all("average `Postal Code`" not in item for item in assistant)
    assert all("`Postal Code` closely tracks" not in item for item in assistant)
    assert all("`Ship Mode` groups" not in item for item in assistant[1:])


def test_explicit_metric_switch_is_allowed_but_not_silent() -> None:
    df = pd.DataFrame(
        {
            "Sales": [10, 20, 30, 40],
            "Profit": [1, 8, 2, 12],
            "City": ["A", "A", "B", "C"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("построй график топ городов по sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    switched = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="теперь сравни Profit по городам")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=switched.message_id)
    state = store.get_investigation(investigation.investigation_id).metadata["conversation_state"]
    assistant = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert state["active_metric"] == "Profit"
    assert state["active_dimension"] == "City"
    assert any("`Profit` by `City`" in item for item in assistant)


def test_explicit_dimension_switch_is_allowed_but_metric_is_preserved() -> None:
    df = pd.DataFrame(
        {
            "Sales": [10, 20, 30, 40],
            "City": ["A", "A", "B", "C"],
            "Segment": ["Consumer", "Corporate", "Consumer", "Corporate"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("построй график топ городов по sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    switched = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="а теперь по Segment")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=switched.message_id)
    state = store.get_investigation(investigation.investigation_id).metadata["conversation_state"]
    assistant = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert state["active_metric"] == "Sales"
    assert state["active_dimension"] == "Segment"
    assert any(("`Sales` заметно различается по `Segment`" in item or "`Sales` differs sharply across `Segment`" in item) for item in assistant)


def test_broad_exploratory_questions_do_not_inherit_stale_branch() -> None:
    df = pd.DataFrame(
        {
            "Row ID": [1, 2, 3, 4, 5, 6],
            "Postal Code": [10001, 99301, 1040, 98208, 2149, 60601],
            "Sales": [100, 500, 20, 700, 50, 300],
            "Ship Mode": ["Standard Class", "Second Class", "First Class", "Standard Class", "Same Day", "First Class"],
            "City": ["A", "B", "A", "C", "D", "E"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Что ты можешь сказать о данных?")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    dependencies = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Какие зависимости здесь выглядят самыми сильными?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=dependencies.message_id)
    business = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Какие бизнес-вопросы можно исследовать на этом датасете?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=business.message_id)
    fields = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Какие поля здесь наиболее важны?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=fields.message_id)

    assistant = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert all("average `Postal Code`" not in item for item in assistant)
    assert all("`Postal Code` by `Ship Mode`" not in item for item in assistant)
    assert "not binding it to the previous branch" in assistant[-2]
    assert "`Sales`" in assistant[-1]
    assert "identifier" in assistant[0].lower() or "identifier" in assistant[-1].lower()


def test_metric_prioritization_rejects_identifier_and_code_columns_across_domains() -> None:
    df = pd.DataFrame(
        {
            "patient_id": [101, 102, 103, 104, 105],
            "diagnosis_code": [10, 11, 10, 12, 13],
            "claim_amount": [1200.0, 250.0, 800.0, 1600.0, 90.0],
            "risk_score": [0.7, 0.2, 0.5, 0.9, 0.1],
            "department": ["A", "B", "A", "C", "B"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Какие поля здесь наиболее важны?")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    summary = store.get_investigation(investigation.investigation_id).report.summary

    assert "`claim_amount`" in summary or "`risk_score`" in summary
    assert summary.find("`patient_id`") == -1 or "not as default average metrics" in summary
    assert "average `patient_id`" not in summary
    assert "average `diagnosis_code`" not in summary


def test_ambiguous_anomaly_followup_continues_current_chart_but_conceptual_question_does_not() -> None:
    df = pd.DataFrame(
        {
            "ticket_id": [1, 2, 3, 4],
            "resolution_hours": [2.0, 3.0, 40.0, 5.0],
            "team": ["Ops", "Ops", "IT", "IT"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("построй график resolution_hours по team")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    anomaly = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="есть ли аномалии?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=anomaly.message_id)
    conceptual = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="какие бизнес-вопросы можно исследовать?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=conceptual.message_id)
    assistant = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert any("active `resolution_hours` by `team`" in item for item in assistant)
    assert "not binding it to the previous branch" in assistant[-1]


def test_chart_request_missing_explicit_metric_does_not_substitute_salary_for_sales() -> None:
    df = pd.DataFrame(
        {
            "Salary_LPA": [10.0, 22.0, 14.0, 18.0],
            "Openings": [3, 5, 2, 8],
            "Applicants": [100, 220, 80, 160],
            "City": ["Bengaluru", "Delhi", "Mumbai", "Bengaluru"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Построй график top cities by sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    updated = store.get_investigation(investigation.investigation_id)

    assert updated.report is not None
    assert "нет такого поля" in updated.report.summary
    assert "`sales`" in updated.report.summary
    assert "`Salary_LPA`" in updated.report.summary
    assert "Salary_LPA across `City` best matches" not in updated.report.summary
    assert not any(artifact.artifact_type == ArtifactType.CHART for artifact in updated.artifacts)


def test_exact_column_mentions_override_identifier_penalties_and_defaults() -> None:
    df = pd.DataFrame(
        {
            "Postal Code": [10001, 10002, 10003, 10004],
            "Sales": [10.0, 20.0, 30.0, 40.0],
            "City": ["A", "B", "A", "C"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Построй график Postal Code by City")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    updated = store.get_investigation(investigation.investigation_id)
    chart = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.CHART)

    assert chart.content["metric"] == "Postal Code"
    assert chart.content["x"] == "City"
    assert "Average Postal Code by City" == chart.title


def test_exact_dimension_column_mention_overrides_lower_cardinality_default() -> None:
    df = pd.DataFrame(
        {
            "Sales": [10.0, 20.0, 30.0, 40.0],
            "Country": ["US", "US", "US", "US"],
            "Company_Type": ["Startup", "Enterprise", "Startup", "SMB"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Построй график Sales by Company_Type")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    updated = store.get_investigation(investigation.investigation_id)
    chart = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.CHART)

    assert chart.content["metric"] == "Sales"
    assert chart.content["x"] == "Company_Type"
    assert "Country" not in updated.report.summary


def test_conditional_remove_extreme_orders_recomputes_active_city_sales_thread() -> None:
    df = pd.DataFrame(
        {
            "Sales": [
                2354.39, 2354.39, 1603.14, 1263.41, 1099.50, 1427.32, 1208.68, 1208.68,
                1082.39, 1279.97, 884.81, 1.17, 22638.48, 12.0, 34.0, 500.0, 520.0, 510.0,
            ],
            "City": [
                "Jamestown", "Jamestown", "Cheyenne", "Bellingham", "Bellingham", "Bellingham", "Independence", "Independence",
                "Burbank", "Burbank", "Burbank", "Jacksonville", "Jacksonville", "Abilene", "Elyria", "Seattle", "Seattle", "Seattle",
            ],
            "Order ID": [f"o{idx}" for idx in range(18)],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Построй график top cities by sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    strongest = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Какие города самые сильные?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=strongest.message_id)
    anomalies = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Есть ли аномалии?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=anomalies.message_id)
    volume = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Это связано с объемом заказов?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=volume.message_id)
    transformed = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="А если убрать extreme orders?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=transformed.message_id)
    remaining = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Какие города остаются лидерами?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=remaining.message_id)
    hypothesis = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Сформируй hypothesis")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=hypothesis.message_id)
    validation = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Как это проверить?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=validation.message_id)

    updated = store.get_investigation(investigation.investigation_id)
    assistant = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert all("основу для аналитического расследования" not in item for item in assistant[-4:])
    assert "After removing extreme `Sales` records" in assistant[-4]
    assert "`City`" in assistant[-4]
    assert "the `City` story changes" in assistant[-4]
    assert "After filtering" in assistant[-3]
    assert "outlier" in assistant[-3]
    assert "Hypothesis" in assistant[-2]
    assert "outlier-driven" in assistant[-2]
    assert "Validate this" in assistant[-1]
    assert "raw average ranking" in assistant[-1]
    assert updated.metadata["conversation_state"]["active_metric"] == "Sales"
    assert updated.metadata["conversation_state"]["active_dimension"] == "City"
    assert any(
        artifact.artifact_type == ArtifactType.CHART
        and (
            artifact.metadata.get("analysis_type") == "remove_outliers"
            or (artifact.metadata.get("metadata") or {}).get("analysis_type") == "remove_outliers"
        )
        and artifact.content.get("x") == "City"
        for artifact in updated.artifacts
    )


def test_repeated_strongest_question_deepens_instead_of_duplicate_answer() -> None:
    df = pd.DataFrame(
        {
            "Sales": [100.0, 110.0, 900.0, 50.0, 55.0, 500.0],
            "City": ["A", "A", "B", "C", "C", "D"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Построй график top cities by sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    first = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Какие города самые сильные?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=first.message_id)
    second = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Какие города самые сильные?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=second.message_id)

    assistant = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert assistant[-1] != assistant[-2]
    assert "same ranking" in assistant[-1].lower()
    assert "reliability" in assistant[-1].lower()


def test_conditional_transformation_without_active_context_asks_clarification() -> None:
    df = pd.DataFrame({"Revenue": [10.0, 20.0, 1000.0], "Segment": ["A", "A", "B"]})
    store = InvestigationStore()
    investigation = store.create_investigation("А если убрать extreme orders?")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    summary = store.get_investigation(investigation.investigation_id).report.summary

    assert "Which metric and grouping" in summary
    assert "основу для аналитического расследования" not in summary


def test_generic_dataset_median_and_sparse_transformations_use_active_context() -> None:
    df = pd.DataFrame(
        {
            "claim_amount": [100.0, 120.0, 5000.0, 90.0, 95.0, 80.0, 200.0],
            "department": ["ER", "ER", "ER", "Cardiology", "Cardiology", "Lab", "Oncology"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Построй график claim_amount by department")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    median = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="а если смотреть медиану?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=median.message_id)
    sparse = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="если убрать маленькие выборки?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=sparse.message_id)

    assistant = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert "median `claim_amount`" in assistant[-2] or "median" in assistant[-2]
    assert "claim_amount" in assistant[-1]
    assert "department" in assistant[-1]
    assert all("Sales" not in item and "City" not in item for item in assistant)


def test_post_transformation_followup_is_generic_for_finance_dataset() -> None:
    df = pd.DataFrame(
        {
            "portfolio_return": [0.05, 0.04, 0.50, 0.03, 0.02, 0.01, 0.08, 0.07],
            "sector": ["Tech", "Tech", "Tech", "Retail", "Retail", "Energy", "Health", "Health"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("plot portfolio_return by sector")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    transform = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="without outliers")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=transform.message_id)
    leaders = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="which sectors remain leaders?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=leaders.message_id)

    assistant = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert "portfolio_return" in assistant[-1]
    assert "sector" in assistant[-1]
    assert "After filtering" in assistant[-1]
    assert "dataset has" not in assistant[-1].lower()


def test_quality_gate_blocks_overview_after_active_healthcare_analysis() -> None:
    df = pd.DataFrame(
        {
            "claim_amount": [100.0, 120.0, 5000.0, 90.0, 95.0, 80.0, 200.0],
            "department": ["ER", "ER", "ER", "Cardiology", "Cardiology", "Lab", "Oncology"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("plot claim_amount by department")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    transform = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="remove outliers")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=transform.message_id)
    validation = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="how to validate this?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=validation.message_id)

    summary = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ][-1]

    assert "claim_amount" in summary
    assert "department" in summary
    assert "raw average ranking" in summary
    assert "dataset has" not in summary.lower()


def test_temporal_branch_preserves_sales_time_objective_across_followups() -> None:
    dates = pd.date_range("2020-01-01", periods=18, freq="MS")
    df = pd.DataFrame(
        {
            "Order Date": list(dates) * 3,
            "Sales": [
                100, 120, 160, 140, 180, 260, 240, 300, 420, 380, 450, 520, 500, 620, 700, 650, 760, 820,
                80, 90, 95, 100, 120, 140, 150, 170, 190, 210, 230, 250, 260, 280, 300, 320, 340, 360,
                30, 35, 40, 42, 45, 50, 60, 58, 70, 75, 80, 85, 90, 92, 95, 100, 110, 115,
            ],
            "Category": ["Technology"] * 18 + ["Furniture"] * 18 + ["Office Supplies"] * 18,
            "Segment": ["Consumer", "Corporate", "Home Office"] * 18,
            "Ship Mode": ["Standard Class", "Second Class", "First Class"] * 18,
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Покажи sales trend over time")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    growth = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Где strongest growth?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=growth.message_id)
    categories = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Какие категории объясняют рост?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=categories.message_id)
    seasonality = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Есть ли seasonality?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=seasonality.message_id)
    anomalous = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Есть ли anomalous periods?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=anomalous.message_id)
    shipping = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Это связано с shipping behavior?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=shipping.message_id)

    updated = store.get_investigation(investigation.investigation_id)
    assistant = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert "over `Order Date`" in assistant[0]
    assert "Strongest growth" in assistant[1]
    assert "decomposed growth by `Category` over time" in assistant[2]
    assert "Seasonality check" in assistant[3]
    assert "Anomalous periods" in assistant[4]
    assert "`Ship Mode`" in assistant[5]
    assert all("основу для аналитического расследования" not in item for item in assistant)
    assert all("I ranked all `Segment` groups" not in item for item in assistant[2:])
    assert updated.metadata["conversation_state"]["active_metric"] == "Sales"
    assert updated.metadata["conversation_state"]["active_time_axis"] == "Order Date"
    assert updated.metadata["conversation_state"]["active_branch_type"] in {"trend_analysis", "temporal_decomposition"}


def test_temporal_branch_is_generic_for_operations_dataset() -> None:
    df = pd.DataFrame(
        {
            "event_date": pd.date_range("2024-01-01", periods=8, freq="MS"),
            "duration_hours": [10, 12, 18, 16, 25, 30, 28, 40],
            "site": ["A", "A", "B", "B", "A", "B", "A", "B"],
            "delivery_method": ["Air", "Ground", "Ground", "Air", "Ground", "Ground", "Air", "Ground"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("show duration_hours trend over time")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    behavior = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="is this related to delivery behavior?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=behavior.message_id)

    assistant = [
        item.content for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    assert "duration_hours" in assistant[-1]
    assert "delivery_method" in assistant[-1]
    assert "over time" in assistant[-1]
    assert "dataset has" not in assistant[-1].lower()
