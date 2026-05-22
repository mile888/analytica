from __future__ import annotations

import pandas as pd

from source.product.affected_findings import AffectedFindingsAnalyzer
from source.product.conversation_engine import response_quality_gate
from source.product.evidence_resolution import ActiveAnalyticalTarget
from source.product.fallback_analysis import deterministic_investigation_fallback
from source.product.hypothesis_reasoning import HypothesisEngine
from source.product.investigation import InvestigationMessage, InvestigationMessageRole
from source.product.run_service import InvestigationRunService
from source.product.semantic_layer import build_semantic_dataset_profile
from source.product.store import InvestigationStore


def test_affected_findings_after_duplicate_quality_issue_identifies_totals_and_counts() -> None:
    df = pd.DataFrame(
        {
            "Order ID": ["A", "A", "B", "C", "C"],
            "Sales": [100.0, 50.0, 200.0, 300.0, 300.0],
            "City": ["NYC", "NYC", "Boston", "Seattle", "Seattle"],
        }
    )
    target = ActiveAnalyticalTarget(
        branch_type="data_quality",
        metric="Sales",
        active_mechanism="duplicate_inflation",
        response_language="ru",
    )

    result = AffectedFindingsAnalyzer.analyze(
        question="Какие findings become unreliable?",
        dataframe=df,
        active_target=target.to_payload(),
        branch_state={"active_branch_type": "data_quality", "active_metric": "Sales"},
    )

    assert result is not None
    assert "total `Sales`" in result.summary
    assert "record counts" in result.summary or "counts" in result.summary
    assert "Repeated `Order ID`" in " ".join(result.findings)


def test_missingness_issue_identifies_affected_comparisons() -> None:
    df = pd.DataFrame({"Region": ["A", "B", None], "Revenue": [10.0, 20.0, 30.0]})

    result = AffectedFindingsAnalyzer.analyze(
        question="Which findings become unreliable?",
        dataframe=df,
        active_target=ActiveAnalyticalTarget(branch_type="data_quality", active_mechanism="missingness_bias").to_payload(),
    )

    assert result is not None
    assert "Missingness" in result.summary
    assert "comparisons" in result.summary


def test_outlier_issue_identifies_affected_rankings_and_averages() -> None:
    df = pd.DataFrame({"City": ["A", "A", "B", "B", "C"], "Sales": [10, 11, 12, 13, 1000]})

    result = AffectedFindingsAnalyzer.analyze(
        question="Which findings become unreliable?",
        dataframe=df,
        active_target=ActiveAnalyticalTarget(branch_type="data_quality", metric="Sales", active_mechanism="outlier_effect").to_payload(),
    )

    assert result is not None
    assert "average-ranking" in result.summary
    assert "Median" in " ".join(result.findings + result.summary.split())


def test_mixed_russian_english_evidence_question_gets_english_answer() -> None:
    df = pd.DataFrame({"City": ["A", "B"], "Sales": [100.0, 200.0]})
    target = ActiveAnalyticalTarget(
        branch_type="hypothesis_validation",
        metric="Sales",
        dimension="City",
        active_mechanism="sparse_group_instability",
        response_language="ru",
    ).to_payload()

    result = deterministic_investigation_fallback(
        "Какие evidence supports this?",
        df,
        data_context={"conversation_context": {"conversation_state": {"active_analytical_target": target}}},
    )

    assert result is not None
    assert result["final_answer"].startswith("The evidence")
    assert "Поддерживающие" not in result["final_answer"]


def test_hypothesis_answer_includes_calibrated_confidence_limitation_and_next_validation() -> None:
    df = pd.DataFrame(
        {
            "Ship Mode": ["Standard Class", "Standard Class", "Same Day"],
            "Sales": [100.0, 110.0, 250.0],
        }
    )
    result = HypothesisEngine.validate(
        question="Hypothesis: Standard Class dominates because of volume, not order value.",
        dataframe=df,
        branch_state={},
        semantic_profile=build_semantic_dataset_profile(df=df),
    )

    assert result is not None
    answer = result["final_answer"]
    assert "Confidence is" in answer
    assert "Limitation:" in answer
    assert "Next validation:" in answer


def test_hypothesis_artifact_metadata_links_to_current_hypothesis() -> None:
    df = pd.DataFrame(
        {
            "Category": ["Technology", "Technology", "Furniture", "Furniture"],
            "Sales": [10.0, 1000.0, 20.0, 30.0],
        }
    )
    result = HypothesisEngine.validate(
        question="Hypothesis: Technology category creates most outliers.",
        dataframe=df,
        branch_state={},
        semantic_profile=build_semantic_dataset_profile(df=df),
    )

    artifact = result["artifacts"][0]
    assert artifact["metadata"]["evidence_role"] == "supports_current_hypothesis"
    assert artifact["metadata"]["supports_current_target"] is True
    assert artifact["metadata"]["mechanism"] == "outlier_concentration"


def test_quality_gate_rejects_wrong_language_response() -> None:
    valid, reason = response_quality_gate(
        question="Какие evidence supports this?",
        response_text="Поддерживающие данные показывают, что выборка маленькая и рейтинг хрупкий.",
        conversation_context={"conversation_state": {"active_metric": "Sales", "active_dimension": "City"}},
    )

    assert not valid
    assert reason == "wrong_response_language"


def test_quality_gate_rejects_shallow_hypothesis_verdict() -> None:
    valid, reason = response_quality_gate(
        question="Hypothesis: Standard Class dominates because of volume.",
        response_text="The hypothesis is supported.",
        conversation_context={"conversation_state": {"active_metric": "Sales", "active_dimension": "Ship Mode"}},
    )

    assert not valid
    assert reason == "shallow_hypothesis_verdict"


def test_quality_gate_rejects_raw_ship_date_shipping_answer() -> None:
    valid, reason = response_quality_gate(
        question="Is this related to shipping behavior?",
        response_text="I compared `Sales` over time by `Ship Date`. Largest `Ship Date` contributors are `2020-01-03` and `2020-02-03`.",
        conversation_context={"conversation_state": {"active_metric": "Sales", "active_time_axis": "Order Date", "active_branch_type": "trend_analysis"}},
    )

    assert not valid
    assert reason == "shipping_used_raw_ship_date"


def test_quality_gate_rejects_raw_ranking_after_adjusted_transformation() -> None:
    valid, reason = response_quality_gate(
        question="Which cities remain leaders?",
        response_text="The leaders are `Jamestown`, `Cheyenne`, and `Bellingham`.",
        conversation_context={
            "conversation_state": {
                "active_metric": "Sales",
                "active_dimension": "City",
                "active_adjusted_ranking": [
                    {"City": "Missoula", "adjusted_mean": 510.0},
                    {"City": "Murrieta", "adjusted_mean": 480.0},
                ],
            }
        },
    )

    assert not valid
    assert reason == "contradicts_active_transformation"


def test_shipping_behavior_uses_ship_mode_and_delay_not_raw_ship_date() -> None:
    df = pd.DataFrame(
        {
            "Order Date": pd.date_range("2020-01-01", periods=8, freq="MS"),
            "Ship Date": pd.date_range("2020-01-03", periods=8, freq="MS"),
            "Ship Mode": ["Standard Class", "Standard Class", "Second Class", "First Class"] * 2,
            "Sales": [100, 120, 180, 140, 240, 260, 220, 300],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("show Sales trend over time")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    message = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Это связано с shipping behavior?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=message.message_id)
    assistant = [
        item.content
        for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    answer = assistant[-1]
    assert "`Ship Mode`" in answer
    assert "Delivery delay" in answer
    assert "shipping date field is treated as a timestamp" in answer
    assert "contributors are `2020" not in answer


def test_adjusted_transformation_leaders_are_authoritative_for_followup() -> None:
    df = pd.DataFrame(
        {
            "Sales": [
                10000, 9000, 8000,
                500, 520, 510,
                480, 490, 470,
                460, 455, 465,
                450, 440, 445,
                430, 420, 425,
            ],
            "City": [
                "Jamestown", "Cheyenne", "Bellingham",
                "Missoula", "Missoula", "Missoula",
                "Murrieta", "Murrieta", "Murrieta",
                "Whittier", "Whittier", "Whittier",
                "El Cajon", "El Cajon", "El Cajon",
                "Vacaville", "Vacaville", "Vacaville",
            ],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("top cities by Sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    transform = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="remove extreme orders")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=transform.message_id)
    leaders = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="which cities remain leaders?")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=leaders.message_id)

    assistant = [
        item.content
        for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]
    answer = assistant[-1]

    for city in ["Missoula", "Murrieta", "Whittier", "El Cajon", "Vacaville"]:
        assert city in answer
    assert "Jamestown" not in answer
    assert "After filtering" in answer


def test_transformation_artifact_persists_dimension_and_metric_lineage() -> None:
    df = pd.DataFrame(
        {
            "Sales": [10000, 9000, 8000, 500, 520, 510, 480, 490, 470],
            "City": ["Jamestown", "Cheyenne", "Bellingham", "Missoula", "Missoula", "Missoula", "Murrieta", "Murrieta", "Murrieta"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Top cities by Sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    transform = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Remove outliers")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=transform.message_id)

    updated = store.get_investigation(investigation.investigation_id)
    adjusted = next(artifact for artifact in updated.artifacts if artifact.title == "Adjusted average Sales by City")

    assert adjusted.metadata["base_metric"] == "Sales"
    assert adjusted.metadata["base_dimension"] == "City"
    assert adjusted.metadata["base_aggregation"] == "mean"
    assert adjusted.metadata["transformation_type"] == "remove_outliers"
    assert adjusted.metadata["base_artifact_id"]
    assert adjusted.metadata["parent_artifact_id"] == adjusted.metadata["base_artifact_id"]
    assert adjusted.content["base_metric"] == "Sales"
    assert adjusted.content["base_dimension"] == "City"


def test_explain_adjusted_chart_uses_transformed_artifact_not_fresh_grouping() -> None:
    df = pd.DataFrame(
        {
            "Sales": [10000, 9000, 8000, 500, 520, 510, 480, 490, 470],
            "City": ["Jamestown", "Cheyenne", "Bellingham", "Missoula", "Missoula", "Missoula", "Murrieta", "Murrieta", "Murrieta"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Top cities by Sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    for prompt in ["Remove outliers", "Explain the adjusted chart"]:
        msg = store.add_investigation_message(
            InvestigationMessage(investigation_id=investigation.investigation_id, content=prompt)
        )
        service.run_investigation(investigation.investigation_id, df=df, message_id=msg.message_id)

    answer = [
        item.content
        for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ][-1]

    assert "`Adjusted average Sales by City`" in answer
    assert "after remove outliers" in answer or "after removing" in answer
    assert "adjusted leaders" in answer
    assert "Missoula" in answer
    assert "Murrieta" in answer
    assert "Sub-Category" not in answer


def test_histogram_explanation_uses_distribution_language() -> None:
    df = pd.DataFrame(
        {
            "Amount": [10.0, 12.0, 13.0, 15.0, 18.0, 40.0, 45.0, 100.0, 120.0, 130.0],
            "Location": ["East", "East", "East", "East", "East", "West", "West", "West", "West", "West"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Build histogram of amount in east")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    for prompt in ["Compare against west", "Explain this chart"]:
        msg = store.add_investigation_message(
            InvestigationMessage(investigation_id=investigation.investigation_id, content=prompt)
        )
        service.run_investigation(investigation.investigation_id, df=df, message_id=msg.message_id)

    answer = [
        item.content
        for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ][-1]

    assert "distribution" in answer
    assert "median" in answer
    assert "spread" in answer
    assert "tail" in answer
    assert "ranks `Location`" not in answer


def test_quality_risks_prompt_is_english_only() -> None:
    df = pd.DataFrame(
        {
            "Order ID": ["A", "B"],
            "Sales": [100.0, 200.0],
            "City": ["NYC", "Boston"],
        }
    )

    result = deterministic_investigation_fallback("Какие quality risks здесь самые важные?", df)
    answer = result["final_answer"]

    assert "The most important fields" in answer
    assert "Наиболее" not in answer
    assert "`Sales`" in answer


def test_transformation_reuses_active_target_without_clarification() -> None:
    df = pd.DataFrame(
        {
            "Sales": [1000.0, 900.0, 100.0, 120.0, 130.0, 140.0],
            "City": ["A", "A", "B", "B", "C", "C"],
        }
    )

    result = deterministic_investigation_fallback(
        "А если убрать extreme orders?",
        df,
        data_context={
            "conversation_context": {
                "conversation_state": {"active_metric": "Sales", "active_dimension": "City"},
            }
        },
    )
    answer = result["final_answer"]

    assert answer.startswith("After removing")
    assert "Which metric and grouping" not in answer
    assert "`Sales`" in answer
    assert "`City`" in answer


def test_contradiction_followup_stays_on_active_city_hypothesis() -> None:
    df = pd.DataFrame({"City": ["A", "A", "B"], "Sales": [100.0, 110.0, 1000.0]})
    target = ActiveAnalyticalTarget(
        branch_type="hypothesis_validation",
        metric="Sales",
        dimension="City",
        active_mechanism="outlier_concentration",
        hypothesis="High sales cities are driven by a few large orders.",
        active_entities=["Sales", "City"],
    ).to_payload()

    result = deterministic_investigation_fallback(
        "Что противоречит hypothesis?",
        df,
        data_context={"conversation_context": {"conversation_state": {"active_analytical_target": target}}},
    )
    answer = result["final_answer"]

    assert "counterevidence" in answer.lower()
    assert "`City`" in answer
    assert "`Sales`" in answer
    assert "Standard Class" not in answer
    assert "`Ship Mode`" not in answer


def test_repeated_explicit_hypotheses_each_run_full_validation() -> None:
    df = pd.DataFrame(
        {
            "Ship Mode": ["Standard Class"] * 6 + ["Second Class"] * 2,
            "Category": ["Technology", "Technology", "Furniture", "Furniture", "Technology", "Furniture", "Furniture", "Office Supplies"],
            "Sales": [100.0, 105.0, 110.0, 95.0, 120.0, 90.0, 1000.0, 30.0],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Hypothesis: Standard Class dominates because of volume, not order value.")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    second = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="Hypothesis: Technology category creates most outliers.",
        )
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=second.message_id)

    assistant = [
        item.content
        for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]
    answer = assistant[-1]

    assert "The active conclusion is" not in answer
    assert "`Technology`" in answer
    assert "`Category`" in answer
    assert "Confidence is" in answer
    assert "Limitation:" in answer
    assert "Next validation:" in answer


def test_quality_gate_rejects_active_conclusion_filler_for_explicit_hypothesis() -> None:
    valid, reason = response_quality_gate(
        question="Hypothesis: Standard Class dominates because of volume, not order value.",
        response_text="The active conclusion is: Record count is a proxy for volume.",
        conversation_context={"conversation_state": {"active_metric": "Sales", "active_dimension": "Ship Mode"}},
    )

    assert not valid
    assert reason in {"generic_hypothesis_filler", "workflow_filler"}


def test_transformation_persists_across_long_followup_chain() -> None:
    df = pd.DataFrame(
        {
            "Sales": [
                10000, 9000, 8000,
                500, 520, 510,
                480, 490, 470,
                460, 455, 465,
                450, 440, 445,
            ],
            "City": [
                "Jamestown", "Cheyenne", "Bellingham",
                "Missoula", "Missoula", "Missoula",
                "Murrieta", "Murrieta", "Murrieta",
                "Whittier", "Whittier", "Whittier",
                "El Cajon", "El Cajon", "El Cajon",
            ],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Top cities by Sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    for prompt in [
        "Remove extreme orders",
        "Which cities remain leaders?",
        "What changed?",
        "Какие города самые сильные?",
    ]:
        msg = store.add_investigation_message(
            InvestigationMessage(investigation_id=investigation.investigation_id, content=prompt)
        )
        service.run_investigation(investigation.investigation_id, df=df, message_id=msg.message_id)

    answer = [
        item.content
        for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ][-1]

    assert "After filtering" in answer
    assert "Missoula" in answer
    assert "Murrieta" in answer
    assert "Jamestown" not in answer


def test_shipping_delay_hypothesis_and_followups_stay_operational() -> None:
    df = pd.DataFrame(
        {
            "Order Date": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "Ship Date": pd.date_range("2020-01-03", periods=12, freq="MS")
            + pd.to_timedelta([0, 1, 0, 5, 6, 1, 0, 7, 2, 0, 6, 1], unit="D"),
            "Ship Mode": ["Standard Class", "Second Class", "First Class"] * 4,
            "Sales": [100, 120, 130, 600, 650, 110, 105, 700, 140, 120, 680, 150],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Show Sales trend over time")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    for prompt in [
        "Is this related to shipping behavior?",
        "Hypothesis: shipping delays drive revenue spikes.",
        "What evidence supports this?",
        "What contradicts it?",
        "How would you validate it?",
    ]:
        msg = store.add_investigation_message(
            InvestigationMessage(investigation_id=investigation.investigation_id, content=prompt)
        )
        service.run_investigation(investigation.investigation_id, df=df, message_id=msg.message_id)

    assistant = [
        item.content
        for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]
    hypothesis_answer = assistant[-4]
    evidence_answer = assistant[-3]
    contradiction_answer = assistant[-2]
    validation_answer = assistant[-1]

    assert "shipping-delay hypothesis" in hypothesis_answer
    assert "delivery delay" in hypothesis_answer
    assert "Do you want to continue" not in hypothesis_answer
    assert "delivery" in evidence_answer.lower() or "shipping" in evidence_answer.lower()
    assert "delivery" in contradiction_answer.lower() or "shipping" in contradiction_answer.lower()
    assert "delivery delay" in validation_answer
    assert "`Segment`" not in validation_answer


def test_city_hypothesis_validation_followup_does_not_reset_to_grouped_chart() -> None:
    df = pd.DataFrame(
        {
            "City": ["A", "A", "B", "B", "C", "C"],
            "Sales": [1000.0, 10.0, 80.0, 90.0, 70.0, 75.0],
        }
    )
    target = ActiveAnalyticalTarget(
        branch_type="hypothesis_validation",
        metric="Sales",
        dimension="City",
        active_mechanism="concentration",
        hypothesis="High sales cities are driven by a few large orders.",
        active_entities=["Sales", "City"],
    ).to_payload()

    result = deterministic_investigation_fallback(
        "How can we validate it further?",
        df,
        data_context={"conversation_context": {"conversation_state": {"active_analytical_target": target}}},
    )
    answer = result["final_answer"]

    assert "Validate the active large-record hypothesis" in answer
    assert "`Sales`" in answer
    assert "`City`" in answer
    assert "I ranked all" not in answer
