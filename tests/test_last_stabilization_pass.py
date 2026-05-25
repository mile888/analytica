from __future__ import annotations

import pandas as pd
import pytest

from source.product.analytical_graph import (
    EvidenceExecutionResult,
    HypothesisExecutionResult,
    QualityExecutionResult,
    TransformationExecutionResult,
    validate_execution_result,
)
from source.product.investigation import InvestigationMessage, InvestigationMessageRole
from source.product.run_service import InvestigationRunService
from source.product.store import InvestigationStore


def _assistant_messages(store: InvestigationStore, investigation_id: str) -> list[str]:
    return [
        item.content
        for item in store.list_investigation_messages(investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]


def _run_followup(store: InvestigationStore, service: InvestigationRunService, investigation_id: str, df: pd.DataFrame, content: str) -> None:
    message = store.add_investigation_message(InvestigationMessage(investigation_id=investigation_id, content=content))
    service.run_investigation(investigation_id, df=df, message_id=message.message_id)


def _city_sales_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sales": [
                2354.39, 2354.39, 1603.14, 1263.41, 1099.50, 1427.32, 1208.68, 1208.68,
                1082.39, 1279.97, 884.81, 1.17, 22638.48, 12.0, 34.0, 500.0, 520.0, 510.0,
                410.0, 420.0, 430.0, 440.0,
            ],
            "City": [
                "Jamestown", "Jamestown", "Cheyenne", "Bellingham", "Bellingham", "Bellingham", "Independence", "Independence",
                "Burbank", "Burbank", "Burbank", "Jacksonville", "Jacksonville", "Abilene", "Elyria", "Seattle", "Seattle", "Seattle",
                "El Cajon", "El Cajon", "El Cajon", "El Cajon",
            ],
            "Order ID": [f"o{idx}" for idx in range(22)],
        }
    )


def test_transformation_followups_compare_raw_and_adjusted_rankings() -> None:
    df = _city_sales_df()
    store = InvestigationStore()
    investigation = store.create_investigation("Top cities by sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    _run_followup(store, service, investigation.investigation_id, df, "Remove extreme orders")
    _run_followup(store, service, investigation.investigation_id, df, "What changed after filtering?")
    _run_followup(store, service, investigation.investigation_id, df, "Which findings became unreliable?")

    assistant = _assistant_messages(store, investigation.investigation_id)
    changed = assistant[-2]
    unreliable = assistant[-1]

    assert "Who dropped" in changed
    assert "Who stayed stable" in changed
    assert "rank #" in changed
    assert "raw average ranking" in unreliable or "raw leaders collapsed" in unreliable
    assert "Jamestown" in changed or "Jamestown" in unreliable


@pytest.mark.integration
def test_evidence_and_contradiction_use_real_transformation_state() -> None:
    df = _city_sales_df()
    store = InvestigationStore()
    investigation = store.create_investigation("Top cities by sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    _run_followup(store, service, investigation.investigation_id, df, "Remove extreme orders")
    _run_followup(store, service, investigation.investigation_id, df, "Hypothesis: high sales cities are driven by a few large orders.")
    _run_followup(store, service, investigation.investigation_id, df, "What evidence supports the hypothesis?")
    _run_followup(store, service, investigation.investigation_id, df, "What contradicts the hypothesis?")

    assistant = _assistant_messages(store, investigation.investigation_id)
    evidence = assistant[-2]
    contradiction = assistant[-1]

    assert "rank #" in evidence
    assert "collapsed after extreme `Sales` records were removed" in evidence
    assert "active transformation" not in evidence
    assert "of 0" not in evidence
    assert "based on 0 valid rows" not in evidence
    assert "No strong concrete counterevidence" in contradiction or "stayed near the top" in contradiction
    assert "`City`" in evidence


def test_execution_result_metadata_validation_rejects_impossible_counts() -> None:
    invalid = TransformationExecutionResult(
        metric="Sales",
        dimension="City",
        original_row_count=0,
        filtered_row_count=0,
        removed_row_count=1162,
        threshold={"lower": -272.30, "upper": 499.86},
        adjusted_rankings=[{"City": "Missoula", "adjusted_mean": 487.98, "adjusted_count": 1}],
    )
    valid = TransformationExecutionResult(
        metric="Sales",
        dimension="City",
        original_row_count=9898,
        filtered_row_count=8736,
        removed_row_count=1162,
        threshold={"lower": -272.30, "upper": 499.86},
    )

    assert not validate_execution_result(invalid, dataframe_row_count=9898).valid
    assert validate_execution_result(valid, dataframe_row_count=9898).valid
    assert HypothesisExecutionResult(hypothesis="x").to_payload()["hypothesis"] == "x"
    assert EvidenceExecutionResult(metric="Sales").to_payload()["metric"] == "Sales"
    assert QualityExecutionResult(quality_issue="duplicates").to_payload()["quality_issue"] == "duplicates"


def test_temporal_category_resolution_prefers_category_over_segment() -> None:
    df = pd.DataFrame(
        {
            "Order Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"]),
            "Sales": [100.0, 120.0, 300.0, 80.0],
            "Category": ["Technology", "Furniture", "Technology", "Furniture"],
            "Segment": ["Consumer", "Corporate", "Consumer", "Corporate"],
            "Ship Mode": ["Standard Class", "Second Class", "Standard Class", "Same Day"],
            "Ship Date": pd.to_datetime(["2024-01-03", "2024-01-05", "2024-02-08", "2024-02-02"]),
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Show sales trend over time")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    _run_followup(store, service, investigation.investigation_id, df, "Which categories explain the growth?")

    answer = _assistant_messages(store, investigation.investigation_id)[-1]
    assert "`Category`" in answer
    assert "`Segment`" not in answer


@pytest.mark.integration
def test_temporal_shipping_evidence_is_concrete() -> None:
    df = pd.DataFrame(
        {
            "Order Date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02", "2024-03-01", "2024-03-02"]
            ),
            "Ship Date": pd.to_datetime(
                ["2024-01-03", "2024-01-06", "2024-02-10", "2024-02-12", "2024-03-03", "2024-03-04"]
            ),
            "Sales": [100.0, 140.0, 900.0, 850.0, 120.0, 130.0],
            "Category": ["Office Supplies", "Furniture", "Technology", "Technology", "Furniture", "Office Supplies"],
            "Segment": ["Consumer", "Corporate", "Consumer", "Corporate", "Consumer", "Corporate"],
            "Ship Mode": ["Standard Class", "Second Class", "Standard Class", "Same Day", "First Class", "Standard Class"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Show sales trend over time")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    _run_followup(store, service, investigation.investigation_id, df, "Which categories explain the growth?")
    _run_followup(store, service, investigation.investigation_id, df, "Is this related to shipping behavior?")
    _run_followup(store, service, investigation.investigation_id, df, "Hypothesis: shipping delays drive revenue spikes.")
    _run_followup(store, service, investigation.investigation_id, df, "What evidence supports this?")

    answer = _assistant_messages(store, investigation.investigation_id)[-1]
    assert "`Category`" in _assistant_messages(store, investigation.investigation_id)[1]
    assert "Ship Mode" in " ".join(_assistant_messages(store, investigation.investigation_id))
    assert "delay" in answer.lower()
    assert "spike" in answer.lower()
    assert "The evidence should stay" not in answer
    assert "Limitation:" not in answer
    assert "Next validation:" not in answer
    assert "hypothesis is partially supported" not in answer


def test_duplicate_quality_repeated_question_recomputes_branch_not_filler() -> None:
    df = pd.DataFrame(
        {
            "Order ID": ["A", "A", "B", "C", "C", "C"],
            "Sales": [100.0, 100.0, 200.0, 50.0, 50.0, 60.0],
            "City": ["Austin", "Austin", "Boston", "Chicago", "Chicago", "Chicago"],
        }
    )
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    store = InvestigationStore()
    investigation = store.create_investigation("Are there duplicate orders?")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    _run_followup(store, service, investigation.investigation_id, df, "Are there duplicate orders?")
    _run_followup(store, service, investigation.investigation_id, df, "How do they affect Sales?")
    _run_followup(store, service, investigation.investigation_id, df, "Which findings become unreliable?")

    messages = _assistant_messages(store, investigation.investigation_id)
    repeated = messages[-3]
    unreliable = messages[-1]
    assert "Duplicate-order check" in repeated or "duplicate" in repeated.lower()
    assert "latest active finding" not in repeated.lower()
    assert "new computed result" not in repeated.lower()
    assert "saved finding" not in repeated.lower()
    assert "estimated inflation" in unreliable.lower()
    assert "total `Sales`" in unreliable or "total Sales" in unreliable


def test_repeated_duplicate_question_answers_naturally() -> None:
    df = pd.DataFrame(
        {
            "Order ID": ["A", "A", "B", "C", "C"],
            "Sales": [100.0, 50.0, 200.0, 300.0, 300.0],
            "City": ["New York", "New York", "Boston", "Seattle", "Seattle"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Show Sales by City")
    service = InvestigationRunService(store)

    _run_followup(store, service, investigation.investigation_id, df, "Are there duplicate orders?")
    _run_followup(store, service, investigation.investigation_id, df, "Are there duplicate orders?")

    answer = _assistant_messages(store, investigation.investigation_id)[-1]
    assert "The duplicate picture is unchanged" in answer
    assert "saved finding" not in answer.lower()
    assert "new computed result" not in answer.lower()


@pytest.mark.integration
def test_sparse_hypothesis_evidence_is_fact_first() -> None:
    df = _city_sales_df()
    store = InvestigationStore()
    investigation = store.create_investigation("Top cities by sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    _run_followup(store, service, investigation.investigation_id, df, "Hypothesis: Some cities look strong only because of sparse data.")
    _run_followup(store, service, investigation.investigation_id, df, "What supports the hypothesis?")
    _run_followup(store, service, investigation.investigation_id, df, "What contradicts the hypothesis?")

    evidence = _assistant_messages(store, investigation.investigation_id)[-2]
    contradiction = _assistant_messages(store, investigation.investigation_id)[-1]
    assert evidence.startswith("`")
    assert any(city in evidence for city in ("Jamestown", "Cheyenne", "Independence", "Jacksonville"))
    assert "n=" in evidence
    assert not evidence.startswith("The evidence supports")
    assert "No strong concrete counterevidence" in contradiction or "sufficiently sampled groups" in contradiction
    assert "It would contradict" not in contradiction


def test_outlier_affected_findings_uses_transformation_not_clarification() -> None:
    df = _city_sales_df()
    store = InvestigationStore()
    investigation = store.create_investigation("Top cities by sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    _run_followup(store, service, investigation.investigation_id, df, "Remove extreme orders")
    _run_followup(store, service, investigation.investigation_id, df, "Which findings become unreliable after removing outliers?")

    answer = _assistant_messages(store, investigation.investigation_id)[-1]
    assert "Do you want to continue" not in answer
    assert "raw leaders collapsed" in answer or "ranking becomes unreliable" in answer
    assert "Jamestown" in answer or "Cheyenne" in answer or "Bellingham" in answer


def test_branch_switch_to_cities_is_natural_not_onboarding_dump() -> None:
    df = _city_sales_df()
    store = InvestigationStore()
    investigation = store.create_investigation("Show sales trend over time")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df.assign(**{"Order Date": pd.date_range("2024-01-01", periods=len(df))}))
    _run_followup(store, service, investigation.investigation_id, df, "Now switch back to cities")

    answer = _assistant_messages(store, investigation.investigation_id)[-1]
    assert answer.startswith("Back to the `City`-level `Sales` view.")
    assert "I ranked all" not in answer
    assert "Switching back" not in answer
