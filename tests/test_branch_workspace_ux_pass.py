from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from source.api.app import app
from source.api.deps import get_store
from source.product.branch_workspace import branch_dtos_for_investigation
from source.product.fallback_analysis import deterministic_investigation_fallback
from source.product.investigation import Artifact, ArtifactType, InvestigationMessage
from source.product.run_service import InvestigationRunService
from source.product.service import InvestigationService
from source.product.store import InvestigationStore


def _workspace_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sales": [100.0, 40.0, 300.0, 80.0, 50.0, 500.0, 120.0, 90.0],
            "City": ["Los Angeles", "Los Angeles", "San Francisco", "San Francisco", "Boston", "Seattle", "Seattle", "Seattle"],
            "Customer Name": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "Order Date": pd.to_datetime(["2024-01-01", "2024-01-12", "2024-02-01", "2024-02-15", "2024-03-01", "2024-03-20", "2024-04-01", "2024-04-12"]),
            "Ship Date": pd.to_datetime(["2024-01-03", "2024-01-15", "2024-02-04", "2024-02-18", "2024-03-03", "2024-03-30", "2024-04-08", "2024-04-18"]),
            "Ship Mode": ["Standard Class", "Second Class", "Standard Class", "Same Day", "First Class", "Standard Class", "Second Class", "Standard Class"],
        }
    )


def _extremum_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sales": [2.88, 100.0, 4.26, 10.0, 90.0, 40.0],
            "City": ["Los Angeles", "Los Angeles", "Los Angeles", "San Francisco", "Los Angeles", "Los Angeles"],
            "Customer Name": ["Gary Hansen", "Gary Hansen", "Stuart Calhoun", "Other", "Ana Lee", "Ana Lee"],
            "Order ID": ["LA-1", "LA-2", "LA-3", "SF-1", "LA-4", "LA-5"],
            "Order Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]),
        }
    )


def _run_followup(store: InvestigationStore, service: InvestigationRunService, investigation_id: str, df: pd.DataFrame, content: str) -> str:
    message = store.add_investigation_message(InvestigationMessage(investigation_id=investigation_id, content=content))
    service.run_investigation(investigation_id, df=df, message_id=message.message_id)
    return [
        item.content
        for item in store.list_investigation_messages(investigation_id)
        if item.role == "assistant"
    ][-1]


def test_explicit_top_customers_overrides_transformed_city_branch() -> None:
    df = _workspace_df()
    store = InvestigationStore()
    investigation = store.create_investigation("Top cities by Sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    _run_followup(store, service, investigation.investigation_id, df, "Remove outliers")
    answer = _run_followup(store, service, investigation.investigation_id, df, "Show top customers")

    assert "Customer Name" in answer
    assert "City" not in answer.split(" is led by", 1)[0]
    assert "adjusted City" not in answer


def test_customer_revenue_with_city_filter_executes_without_confirmation() -> None:
    result = deterministic_investigation_fallback("Какая выручка у клиентов из Seattle?", _workspace_df())

    answer = result["final_answer"]
    assert "`выручка` is treated as `Sales`" in answer or "`выручку` is treated as `Sales`" in answer
    assert "`City` = `Seattle`" in answer
    assert "`Sales` by `Customer Name`" in answer
    assert "Do you want" not in answer


def test_basic_constrained_average_executes_directly() -> None:
    result = deterministic_investigation_fallback("Средний Sales у людей живущих в Los Angeles", _workspace_df())

    answer = result["final_answer"]
    assert "`City` = `Los Angeles`" in answer
    assert "average `Sales`" in answer
    assert "across 2 rows" in answer
    assert "Do you want" not in answer
    assert "Sales` by `City`" not in answer
    assert result["trace_metadata"]["analysis_type"] == "constrained_aggregation"


def test_extremum_customer_in_city_executes_directly() -> None:
    result = deterministic_investigation_fallback("Какой customer в Los Angeles имеет наименьший Sales", _extremum_df())

    answer = result["final_answer"]
    assert "`Customer Name`" in answer
    assert "`City` = `Los Angeles`" in answer
    assert "lowest single `Sales`" in answer
    assert "`Gary Hansen`" in answer
    assert "2.88" in answer
    assert "top" not in answer.casefold()
    assert "is led by" not in answer
    assert result["trace_metadata"]["analysis_type"] == "extremum"
    assert result["trace_metadata"]["extremum_scope"] == "row_level"


def test_english_row_level_extremum_customer_in_city() -> None:
    result = deterministic_investigation_fallback("Which customer in Los Angeles has the lowest Sales?", _extremum_df())

    answer = result["final_answer"]
    assert "`Gary Hansen`" in answer
    assert "lowest single `Sales`" in answer
    assert result["trace_metadata"]["query_plan"]["extremum_scope"] == "row_level"


def test_grouped_total_extremum_requires_total_trigger() -> None:
    result = deterministic_investigation_fallback("Which customer in Los Angeles has the lowest total Sales?", _extremum_df())

    answer = result["final_answer"]
    assert "`Stuart Calhoun`" in answer
    assert "lowest total `Sales`" in answer
    assert result["trace_metadata"]["query_plan"]["extremum_scope"] == "group_aggregate"
    assert result["trace_metadata"]["query_plan"]["aggregate_function"] == "sum"


def test_grouped_average_extremum_uses_average_trigger() -> None:
    result = deterministic_investigation_fallback("Which customer in Los Angeles has the highest average Sales?", _extremum_df())

    answer = result["final_answer"]
    assert "`Ana Lee`" in answer
    assert "highest average `Sales`" in answer
    assert result["trace_metadata"]["query_plan"]["extremum_scope"] == "group_aggregate"
    assert result["trace_metadata"]["query_plan"]["aggregate_function"] == "mean"


def test_ambiguous_extremum_row_level_first_with_total_alternative() -> None:
    result = deterministic_investigation_fallback("Customer with lowest Sales in Los Angeles", _extremum_df())

    answer = result["final_answer"]
    assert answer.index("`Gary Hansen`") < answer.index("`Stuart Calhoun`")
    assert "If you meant lowest total `Sales`" in answer
    assert "I ranked all `Customer Name` groups" not in answer
    assert "direct extremum calculation, not a ranked list" not in answer


def test_russian_temporal_average_uses_order_date_not_customer_ranking() -> None:
    result = deterministic_investigation_fallback("Средний Sales по датам", _workspace_df())

    answer = result["final_answer"]
    assert "`Sales` over `Order Date`" in answer
    assert "volatile" in answer or "steady" in answer or "uneven" in answer
    assert "This is a temporal trend calculation" not in answer
    assert "Customer Name" not in answer
    assert result["trace_metadata"]["analysis_type"] == "temporal_trend"
    assert result["artifacts"][0]["content"]["visualization_type"] == "trend"


def test_russian_temporal_chart_creates_trend_artifact() -> None:
    result = deterministic_investigation_fallback("график Средний Sales по датам", _workspace_df())

    content = result["artifacts"][0]["content"]
    assert result["trace_metadata"]["analysis_type"] == "temporal_trend"
    assert result["trace_metadata"]["query_plan"]["intent"] == "chart_request"
    assert content["chart_type"] == "line"
    assert content["visualization_type"] == "trend"
    assert content["time_axis"] == "Order Date"
    assert "Created a line chart" in result["final_answer"]


def test_temporal_branch_identity_dedupes_equivalent_requests() -> None:
    df = _workspace_df()
    store = InvestigationStore()
    investigation = store.create_investigation("Average Sales over time")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    _run_followup(store, service, investigation.investigation_id, df, "Sales trend over Order Date")
    _run_followup(store, service, investigation.investigation_id, df, "график Средний Sales по датам")

    branches = branch_dtos_for_investigation(store.get_investigation(investigation.investigation_id))
    temporal = [item for item in branches if item["branch_type"] == "temporal"]
    titles = [item["title"] for item in temporal]

    assert len(temporal) == 1
    assert titles == ["Sales Trend over Time"]
    assert not any("(2)" in title or "(3)" in title for title in titles)


def test_top_cities_defaults_to_sum_sales() -> None:
    df = pd.DataFrame({"Sales": [100.0, 50.0, 50.0, 1000.0], "City": ["A", "A", "B", "B"]})

    result = deterministic_investigation_fallback("Top cities by Sales", df)
    plan = result["trace_metadata"]["query_plan"]

    assert plan["aggregation"] == "sum"
    assert plan["aggregation_intent"] == "business_revenue_default"
    assert "`B` (total 1050.00" in result["final_answer"]
    assert "average `Sales`" not in result["final_answer"]
    assert "I ranked all" not in result["final_answer"]


def test_explicit_average_cities_uses_average_sales() -> None:
    df = pd.DataFrame({"Sales": [100.0, 50.0, 50.0, 1000.0], "City": ["A", "A", "B", "B"]})

    result = deterministic_investigation_fallback("Cities with highest average Sales", df)
    plan = result["trace_metadata"]["query_plan"]

    assert plan["aggregation"] == "mean"
    assert plan["aggregation_intent"] == "explicit_average_requested"
    assert plan["extremum_scope"] == "group_aggregate"
    assert "highest average `Sales`" in result["final_answer"]


def test_russian_revenue_by_city_defaults_to_total_sales() -> None:
    df = pd.DataFrame(
        {
            "Sales": [253608.90, 175851.34, 119540.74, 1200.0],
            "City": ["New York City", "Los Angeles", "Seattle", "Jamestown"],
        }
    )

    result = deterministic_investigation_fallback("Посчитай выручку в каждом городе", df)
    plan = result["trace_metadata"]["query_plan"]

    assert plan["metric"] == "Sales"
    assert plan["dimension"] == "City"
    assert plan["aggregation"] == "sum"
    assert plan["aggregation_intent"] == "business_revenue_default"
    assert "`New York City`" in result["final_answer"]
    assert "average `Sales`" not in result["final_answer"]


def test_russian_explicit_average_by_city_uses_mean_sales() -> None:
    df = pd.DataFrame({"Sales": [100.0, 50.0, 50.0, 1000.0], "City": ["A", "A", "B", "B"]})

    result = deterministic_investigation_fallback("Средний Sales по городам", df)
    plan = result["trace_metadata"]["query_plan"]

    assert plan["aggregation"] == "mean"
    assert plan["aggregation_intent"] == "explicit_average_requested"


def test_english_revenue_by_city_defaults_to_total_sales() -> None:
    df = pd.DataFrame({"Sales": [100.0, 50.0, 50.0, 1000.0], "City": ["A", "A", "B", "B"]})

    result = deterministic_investigation_fallback("Revenue by city", df)
    plan = result["trace_metadata"]["query_plan"]

    assert plan["aggregation"] == "sum"
    assert plan["aggregation_intent"] == "business_revenue_default"
    assert "total `Sales`" in result["final_answer"]


def test_explain_chart_uses_existing_sum_artifact_metadata() -> None:
    df = pd.DataFrame({"Sales": [100.0, 50.0, 50.0, 1000.0], "City": ["A", "A", "B", "B"]})
    first = deterministic_investigation_fallback("Top cities by Sales", df)
    chart = next(artifact for artifact in first["artifacts"] if artifact["artifact_type"] == "chart")

    result = deterministic_investigation_fallback(
        "Explain the chart",
        df,
        data_context={"conversation_context": {"recent_artifacts": [chart], "conversation_state": {}}},
    )

    answer = result["final_answer"]
    assert result["trace_metadata"]["analysis_type"] == "chart_explanation"
    assert "total `Sales`" in answer
    assert "average `Sales`" not in answer
    assert "`B`" in answer


def test_explain_artifact_action_resolves_artifact_id_before_title() -> None:
    sum_chart = {
        "artifact_id": "chart_sum",
        "artifact_type": "chart",
        "title": "Sales by City",
        "content": {
            "chart_type": "bar",
            "metric": "Sales",
            "dimension": "City",
            "aggregation": "sum",
            "x": "City",
            "y": "sum",
            "rows": [{"City": "New York City", "sum": 253608.90}, {"City": "Los Angeles", "sum": 175851.34}],
        },
        "metadata": {"metric": "Sales", "dimension": "City", "aggregation": "sum", "branch_id": "grouped::Sales::City::sum::"},
    }
    avg_chart = {
        "artifact_id": "chart_avg",
        "artifact_type": "chart",
        "title": "Sales by City",
        "content": {
            "chart_type": "bar",
            "metric": "Sales",
            "dimension": "City",
            "aggregation": "mean",
            "x": "City",
            "y": "mean",
            "rows": [{"City": "Jamestown", "mean": 2354.00}],
        },
        "metadata": {"metric": "Sales", "dimension": "City", "aggregation": "mean", "branch_id": "grouped::Sales::City::mean::"},
    }
    result = deterministic_investigation_fallback(
        "Explain this chart.",
        pd.DataFrame({"Sales": [1], "City": ["A"]}),
        data_context={
            "conversation_context": {
                "active_message_metadata": {"action": "explain_artifact", "artifact_id": "chart_sum", "branch_id": "grouped::Sales::City::sum::"},
                "recent_artifacts": [avg_chart, sum_chart],
                "conversation_state": {},
            }
        },
    )

    assert result["trace_metadata"]["analysis_type"] == "chart_explanation"
    assert "total `Sales`" in result["final_answer"]
    assert "average `Sales`" not in result["final_answer"]


def test_explain_artifact_action_does_not_fallback_when_exact_id_missing() -> None:
    chart = {
        "artifact_id": "chart_actual",
        "artifact_type": "chart",
        "title": "Sales by City",
        "content": {
            "chart_type": "bar",
            "metric": "Sales",
            "dimension": "City",
            "aggregation": "sum",
            "x": "City",
            "y": "sum",
            "rows": [{"City": "A", "sum": 10.0}],
        },
        "metadata": {"metric": "Sales", "dimension": "City", "aggregation": "sum"},
    }

    result = deterministic_investigation_fallback(
        "Explain this chart.",
        pd.DataFrame({"Sales": [10.0], "City": ["A"]}),
        data_context={
            "conversation_context": {
                "active_message_metadata": {"action": "explain_artifact", "artifact_id": "chart_missing"},
                "recent_artifacts": [chart],
                "conversation_state": {},
            }
        },
    )

    assert result["trace_metadata"]["analysis_type"] == "chart_explanation"
    assert "exact artifact `chart_missing`" in result["final_answer"]
    assert "ranks `City`" not in result["final_answer"]


def test_explain_histogram_comparison_artifact_stays_distribution_grounded() -> None:
    chart = {
        "artifact_id": "hist_compare",
        "artifact_type": "chart",
        "title": "Amount distribution comparison",
        "content": {
            "chart_type": "histogram",
            "visualization_type": "histogram",
            "metric": "Amount",
            "series": "Location",
            "row_count": 6,
            "comparison_groups": [
                {
                    "group": "East",
                    "label": "East",
                    "row_count": 3,
                    "mean": 12.0,
                    "median": 11.0,
                    "bins": [{"label": "10 to 12", "left": 10, "right": 12, "count": 2}, {"label": "12 to 14", "left": 12, "right": 14, "count": 1}],
                },
                {
                    "group": "West",
                    "label": "West",
                    "row_count": 3,
                    "mean": 50.0,
                    "median": 45.0,
                    "bins": [{"label": "40 to 50", "left": 40, "right": 50, "count": 2}, {"label": "50 to 60", "left": 50, "right": 60, "count": 1}],
                },
            ],
        },
        "metadata": {"metric": "Amount", "dimension": "Location", "chart_type": "histogram", "branch_type": "distribution_comparison"},
    }

    result = deterministic_investigation_fallback(
        "Explain this chart.",
        pd.DataFrame({"Amount": [1.0], "Location": ["East"]}),
        data_context={
            "conversation_context": {
                "active_message_metadata": {"action": "explain_artifact", "artifact_id": "hist_compare"},
                "recent_artifacts": [chart],
                "conversation_state": {},
            }
        },
    )

    assert result["trace_metadata"]["analysis_type"] == "chart_explanation"
    assert "histogram comparison" in result["final_answer"]
    assert "distribution overlap" in result["final_answer"]
    assert "ranks `Location`" not in result["final_answer"]


def test_explain_artifact_action_activates_artifact_branch() -> None:
    df = pd.DataFrame({"Sales": [253608.90, 175851.34], "City": ["New York City", "Los Angeles"]})
    store = InvestigationStore()
    investigation = store.create_investigation("Top cities by Sales")
    metric_branch = "grouped::Sales::City::sum::::"
    other_branch = "grouped::Sales::Customer Name::sum::::"
    investigation.metadata = {
        "branch_workspace": {
            "active_branch_id": other_branch,
            "branches": {
                metric_branch: {
                    "branch_id": metric_branch,
                    "identity": {"metric": "Sales", "dimension": "City", "aggregation": "sum", "intent": "rank_groups"},
                    "title": "Sales by City",
                    "last_query_plan": {},
                    "findings": [],
                    "artifacts": [],
                },
                other_branch: {
                    "branch_id": other_branch,
                    "identity": {"metric": "Sales", "dimension": "Customer Name", "aggregation": "sum", "intent": "rank_groups"},
                    "title": "Sales by Customer",
                    "last_query_plan": {},
                    "findings": [],
                    "artifacts": [],
                },
            },
        },
        "conversation_state": {"active_branch_id": other_branch},
    }
    chart = Artifact(
        artifact_type=ArtifactType.CHART,
        title="Sales by City",
        content={"chart_type": "bar", "metric": "Sales", "dimension": "City", "aggregation": "sum", "x": "City", "y": "sum", "rows": [{"City": "New York City", "sum": 253608.90}]},
        metadata={"branch_id": metric_branch, "metric": "Sales", "dimension": "City", "aggregation": "sum"},
    )
    store.add_artifact(investigation.investigation_id, chart)
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="Explain this chart.",
            metadata={"action": "explain_artifact", "artifact_id": chart.artifact_id, "branch_id": metric_branch},
        )
    )

    InvestigationRunService(store).run_investigation(investigation.investigation_id, df=df, message_id=message.message_id)
    updated = store.get_investigation(investigation.investigation_id)

    assert updated.metadata["conversation_state"]["active_branch_id"] == metric_branch
    assert "total `Sales`" in updated.report.summary


def test_explain_artifact_action_can_resolve_displayed_chart_outside_recent_tail() -> None:
    df = _workspace_df()
    store = InvestigationStore()
    investigation = store.create_investigation("Top cities by Sales")
    target = Artifact(
        artifact_type=ArtifactType.CHART,
        title="Sales by City",
        content={
            "chart_type": "bar",
            "metric": "Sales",
            "dimension": "City",
            "aggregation": "sum",
            "x": "City",
            "y": "sum",
            "rows": [{"City": "Seattle", "sum": 710.0}, {"City": "San Francisco", "sum": 380.0}],
        },
        metadata={"metric": "Sales", "dimension": "City", "aggregation": "sum"},
    )
    store.add_artifact(investigation.investigation_id, target)
    for index in range(13):
        store.add_artifact(
            investigation.investigation_id,
            Artifact(
                artifact_type=ArtifactType.CHART,
                title=f"Noise chart {index}",
                content={
                    "chart_type": "bar",
                    "metric": "Sales",
                    "dimension": "Customer Name",
                    "aggregation": "sum",
                    "x": "Customer Name",
                    "y": "sum",
                    "rows": [{"Customer Name": f"Customer {index}", "sum": float(index + 1)}],
                },
                metadata={"metric": "Sales", "dimension": "Customer Name", "aggregation": "sum"},
            ),
        )
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="Explain this chart.",
            metadata={"action": "explain_artifact", "artifact_id": target.artifact_id},
        )
    )

    InvestigationRunService(store).run_investigation(investigation.investigation_id, df=df, message_id=message.message_id)
    updated = store.get_investigation(investigation.investigation_id)

    assert "exact artifact" not in updated.report.summary
    assert "total `Sales`" in updated.report.summary
    assert "`Seattle`" in updated.report.summary


def test_outlier_removal_narration_is_concise() -> None:
    df = pd.DataFrame(
        {
            "Sales": [10000, 9000, 8000, 500, 520, 510, 480, 490, 470, 460, 455, 465],
            "City": ["Jamestown", "Cheyenne", "Bellingham", "Missoula", "Missoula", "Missoula", "Murrieta", "Murrieta", "Murrieta", "Whittier", "Whittier", "Whittier"],
        }
    )

    result = deterministic_investigation_fallback(
        "Remove outliers",
        df,
        data_context={"conversation_context": {"conversation_state": {"active_metric": "Sales", "active_dimension": "City"}}},
    )

    answer = result["final_answer"]
    assert "Average-based confidence decreased" not in answer
    assert answer.startswith("After removing extreme `Sales` records")
    assert "Adjusted leaders" in answer or "adjusted leaders" in answer


def test_temporal_narration_interprets_direction_and_volatility() -> None:
    result = deterministic_investigation_fallback("Average Sales over time", _workspace_df())
    answer = result["final_answer"]

    assert any(marker in answer for marker in ("rises", "declines", "stays flat"))
    assert any(marker in answer for marker in ("volatile", "steady", "uneven"))
    assert "Largest" in answer
    assert "moves from" not in answer
    assert "This is a temporal trend calculation" not in answer


def test_distribution_followup_compares_against_second_city() -> None:
    df = _workspace_df()
    first = deterministic_investigation_fallback("Build a histogram of Sales in Los Angeles", df)
    context = {
        "latest_chart_context": {"chart_type": "histogram", "metric": "Sales"},
        "recent_artifacts": first["artifacts"],
        "conversation_state": {},
    }

    result = deterministic_investigation_fallback("Compare against San Francisco", df, data_context={"conversation_context": context})

    assert "distribution comparison" in result["final_answer"]
    assert "Los Angeles" in result["final_answer"]
    assert "San Francisco" in result["final_answer"]
    assert "tail effects" in result["final_answer"]


def test_histogram_artifact_has_ordered_distribution_shape() -> None:
    result = deterministic_investigation_fallback("Build a histogram of Sales in Los Angeles", _workspace_df())
    content = result["artifacts"][0]["content"]

    assert content["chart_type"] == "histogram"
    assert content["visualization_type"] == "histogram"
    assert content["is_ordered_distribution"] is True
    assert content["bins"]
    assert {"left", "right", "count"} <= set(content["bins"][0])


def test_bins_persist_and_are_used_on_followup() -> None:
    df = _workspace_df()
    first = deterministic_investigation_fallback("Create bins automatically from Sales", df)
    derived = first["trace_metadata"]["derived_field"]
    result = deterministic_investigation_fallback(
        "Does record volume explain differences across bins?",
        df,
        data_context={"conversation_context": {"conversation_state": {"derived_field": derived}}},
    )

    assert "Sales_bin" in result["final_answer"]
    assert "record volume" in result["final_answer"]


def test_non_analytical_utterance_does_not_become_state_analysis() -> None:
    result = deterministic_investigation_fallback("stateful heuristics fighting each other", _workspace_df())

    assert "agent behavior" in result["final_answer"]
    assert result["trace_metadata"]["analysis_type"] == "non_analytical"
    assert "Sales by State" not in result["final_answer"]


def test_shipping_delay_relationship_runs_delay_check_not_generic_trend() -> None:
    result = deterministic_investigation_fallback("Is growth related to shipping delays?", _workspace_df())

    answer = result["final_answer"]
    assert "delivery_delay_days" in answer
    assert "Growth periods" in answer
    assert "relationship" in answer
    assert result["trace_metadata"]["analysis_type"] == "growth_delay_relationship"


def test_delay_intent_overrides_active_city_branch_without_confirmation() -> None:
    df = _workspace_df()
    store = InvestigationStore()
    investigation = store.create_investigation("Top cities by Sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    answer = _run_followup(store, service, investigation.investigation_id, df, "late shipments")

    assert "Do you want to continue" not in answer
    assert "delivery_delay_days" in answer
    assert "Average delay" in answer


def test_shipping_delay_sanity_blocks_implausible_delay_claim() -> None:
    df = pd.DataFrame(
        {
            "Sales": [10.0, 20.0, 30.0],
            "Order Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "Ship Date": pd.to_datetime(["2024-05-01", "2024-05-03", "2024-05-05"]),
        }
    )

    result = deterministic_investigation_fallback("Is growth related to shipping delays?", df)

    assert "Average delay" in result["final_answer"]
    assert "Interpret the delay summary cautiously" in result["final_answer"]
    assert result["trace_metadata"]["analysis_type"] == "shipping_delay"
    assert result["artifacts"]


def test_branch_list_api_and_activation() -> None:
    store = get_store()
    investigation = store.create_investigation("Branch API test")
    df = _workspace_df()
    service = InvestigationRunService(store)
    service.run_investigation(investigation.investigation_id, df=df)
    _run_followup(store, service, investigation.investigation_id, df, "Посчитай выручку в каждом городе")

    client = TestClient(app)
    branches = client.get(f"/investigations/{investigation.investigation_id}/branches")

    assert branches.status_code == 200
    payload = branches.json()
    assert payload
    assert any(item["is_active"] for item in payload)
    branch_id = payload[0]["branch_id"]
    activated = client.post(f"/investigations/{investigation.investigation_id}/branches/{branch_id}/activate")
    assert activated.status_code == 200
    assert activated.json()["active_branch_id"] == branch_id
    reloaded = store.get_investigation(investigation.investigation_id)
    assert reloaded.metadata["branch_workspace"]["active_branch_id"] == branch_id
    assert reloaded.metadata["conversation_state"]["active_branch_id"] == branch_id
    assert reloaded.metadata["conversation_state"].get("active_artifact_id")


def test_invalid_branch_activation_returns_safe_error() -> None:
    store = get_store()
    investigation = store.create_investigation("Invalid branch API test")

    client = TestClient(app)
    activated = client.post(f"/investigations/{investigation.investigation_id}/branches/not-a-branch/activate")

    assert activated.status_code == 404


def test_manual_distribution_branch_context_routes_vague_followup() -> None:
    df = _workspace_df()
    store = InvestigationStore()
    investigation = store.create_investigation("Top cities by Sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    _run_followup(store, service, investigation.investigation_id, df, "Build histogram of Sales in Los Angeles")
    _run_followup(store, service, investigation.investigation_id, df, "Show top customers")
    branches = branch_dtos_for_investigation(store.get_investigation(investigation.investigation_id))
    distribution = next(item for item in branches if item["branch_type"] == "distribution")

    from source.product.branch_workspace import activate_branch

    metadata = activate_branch(store.get_investigation(investigation.investigation_id), distribution["branch_id"])
    store.update_investigation_metadata(investigation.investigation_id, metadata)
    answer = _run_followup(store, service, investigation.investigation_id, df, "Compare against San Francisco")

    assert "distribution comparison" in answer
    assert "Los Angeles" in answer
    assert "San Francisco" in answer
    assert "customer" not in answer.casefold().split("compared", 1)[0]
    assert store.get_investigation(investigation.investigation_id).metadata["branch_workspace"]["active_branch_id"] == distribution["branch_id"]


def test_active_distribution_branch_executes_seattle_comparison() -> None:
    df = _workspace_df()
    store = InvestigationStore()
    investigation = store.create_investigation("Top cities by Sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    _run_followup(store, service, investigation.investigation_id, df, "Build histogram of Sales in Los Angeles")
    _run_followup(store, service, investigation.investigation_id, df, "Show top customers")
    branches = branch_dtos_for_investigation(store.get_investigation(investigation.investigation_id))
    distribution = next(item for item in branches if item["branch_type"] == "distribution")

    from source.product.branch_workspace import activate_branch

    store.update_investigation_metadata(
        investigation.investigation_id,
        activate_branch(store.get_investigation(investigation.investigation_id), distribution["branch_id"]),
    )
    answer = _run_followup(store, service, investigation.investigation_id, df, "Compare against Seattle")

    assert "distribution comparison" in answer
    assert "Los Angeles" in answer
    assert "Seattle" in answer
    assert "tail effects" in answer


def test_explicit_task_after_manual_branch_switches_automatically() -> None:
    df = _workspace_df()
    store = InvestigationStore()
    investigation = store.create_investigation("Top cities by Sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    _run_followup(store, service, investigation.investigation_id, df, "Build histogram of Sales in Los Angeles")
    branches = branch_dtos_for_investigation(store.get_investigation(investigation.investigation_id))
    distribution = next(item for item in branches if item["branch_type"] == "distribution")

    from source.product.branch_workspace import activate_branch

    store.update_investigation_metadata(
        investigation.investigation_id,
        activate_branch(store.get_investigation(investigation.investigation_id), distribution["branch_id"]),
    )
    answer = _run_followup(store, service, investigation.investigation_id, df, "Show top customers")

    assert "Customer Name" in answer
    active = store.get_investigation(investigation.investigation_id).metadata["branch_workspace"]["active_branch_id"]
    assert active != distribution["branch_id"]


def test_repeated_temporal_text_then_chart_is_not_duplicate_text() -> None:
    df = _workspace_df()
    store = InvestigationStore()
    investigation = store.create_investigation("Temporal")
    service = InvestigationRunService(store)

    first = _run_followup(store, service, investigation.investigation_id, df, "Средний Sales по датам")
    second = _run_followup(store, service, investigation.investigation_id, df, "график Средний Sales по датам")
    updated = store.get_investigation(investigation.investigation_id)

    assert "Created a line chart" in second
    assert second != first
    assert any(getattr(artifact, "artifact_type", None).value == "chart" for artifact in updated.artifacts)


def test_branch_title_dedupe_updates_same_identity() -> None:
    df = _workspace_df()
    store = InvestigationStore()
    investigation = store.create_investigation("Dedupe")
    service = InvestigationRunService(store)

    _run_followup(store, service, investigation.investigation_id, df, "Build histogram of Sales in Los Angeles")
    _run_followup(store, service, investigation.investigation_id, df, "Build histogram of Sales in Los Angeles")
    branches = branch_dtos_for_investigation(store.get_investigation(investigation.investigation_id))
    titles = [item["title"] for item in branches if "Distribution in Los Angeles" in item["title"]]

    assert titles == ["Sales Distribution in Los Angeles"]


def test_outlier_adjusted_ranking_avoids_single_row_leaders() -> None:
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

    result = deterministic_investigation_fallback(
        "Remove outliers",
        df,
        data_context={"conversation_context": {"conversation_state": {"active_metric": "Sales", "active_dimension": "City"}}},
    )
    chart = next(artifact for artifact in result["artifacts"] if artifact["artifact_type"] == "chart")
    rows = chart["content"]["rows"]

    assert rows
    assert all(int(row["count"]) > 1 for row in rows[:5])
    assert "Jamestown" not in [row["City"] for row in rows[:5]]
