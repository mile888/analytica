from __future__ import annotations

import pandas as pd

from source.product.business_semantic_planner import business_analysis_response, execute_business_plan
from source.product.dataset_registry import (
    DatasetResolutionResult,
    DatasetScope,
    InvestigationDatasetEntry,
    cross_dataset_output,
)
from source.product.fallback_analysis import deterministic_investigation_fallback


def _business_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Region": ["East", "East", "West", "West", "Central", "Central", "South", "South"],
            "Category": ["Furniture", "Furniture", "Tech", "Tech", "Office", "Office", "Office", "Furniture"],
            "Customer Segment": ["Consumer", "Corporate", "Consumer", "Corporate", "Consumer", "Corporate", "Small Business", "Small Business"],
            "Sales": [1000.0, 900.0, 3000.0, 2000.0, 2200.0, 1800.0, 400.0, 500.0],
            "Profit": [20.0, -80.0, 600.0, 350.0, 40.0, -20.0, 30.0, -10.0],
            "Discount": [0.15, 0.30, 0.05, 0.08, 0.40, 0.35, 0.10, 0.25],
            "Shipping Cost": [200.0, 250.0, 100.0, 80.0, 600.0, 500.0, 50.0, 120.0],
        }
    )


def _health_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age Group": ["18-29", "30-44", "45-59", "60+", "60+", "45-59", "30-44", "18-29"],
            "Gender": ["F", "M", "F", "M", "F", "M", "F", "M"],
            "BMI": [22.0, 27.0, 31.0, 36.0, 42.0, 29.0, 24.0, 21.0],
            "Smoker": [0, 1, 1, 1, 0, 1, 0, 0],
            "Disease Flag": [0, 0, 1, 1, 1, 0, 0, 0],
        }
    )


def _entry(dataset_id: str, name: str, df: pd.DataFrame, *, concepts: list[str]) -> InvestigationDatasetEntry:
    metrics = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    dimensions = [column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]
    return InvestigationDatasetEntry(
        dataset_id=dataset_id,
        display_name=name,
        source_name=name,
        row_count=len(df),
        column_names=list(df.columns),
        semantic_profile={"metric_columns": metrics, "dimension_columns": dimensions, "concepts": concepts},
    )


def _cross_output(question: str) -> dict:
    business = _business_df()
    health = _health_df()
    registry = [
        _entry("business", "Business data", business, concepts=["monetary_value", "purchase_volume", "product_or_category"]),
        _entry("health", "Health data", health, concepts=["health", "risk", "age_or_birth", "gender"]),
    ]
    result = DatasetResolutionResult(scope=DatasetScope.CROSS, selected_dataset_ids=["business", "health"], confidence=1.0)
    return cross_dataset_output(question=question, registry=registry, frames={"business": business, "health": health}, result=result)


def test_profit_margin_is_computed_from_profit_over_sales() -> None:
    evidence = execute_business_plan("Where is profit margin weakest?", _business_df())

    assert evidence is not None
    assert evidence.plan.operation == "PROFIT_MARGIN_ANALYSIS"
    rows = evidence.computed_results[0]["rows"]
    furniture = next(row for row in rows if row["group"] == "Furniture")
    assert furniture["profit_margin"] == round((20.0 - 80.0 - 10.0) / (1000.0 + 900.0 + 500.0) * 100, 2)
    assert "profit_margin" in evidence.derived_kpis


def test_operational_inefficiency_uses_derived_kpis_not_average_sales() -> None:
    output = deterministic_investigation_fallback("Which regions are operationally inefficient?", _business_df())
    metadata = output["trace_metadata"]

    assert metadata["fallback"] == "business_semantic_planner"
    assert metadata["analysis_type"] == "business_efficiency_analysis"
    assert {"profit_margin", "operational_inefficiency_score", "shipping_cost_burden"} <= set(metadata["derived_kpis"])
    assert "average sales" not in output["final_answer"].casefold()


def test_high_sales_low_profit_ranking_uses_sales_and_profit() -> None:
    evidence = execute_business_plan("Which categories have high sales but low profit?", _business_df())

    assert evidence is not None
    rows = evidence.computed_results[0]["rows"]
    assert rows[0]["group"] == "Office"
    assert rows[0]["total_revenue"] > 4000
    assert rows[0]["profit_margin"] < 2
    assert "high_sales_low_profit_score" in rows[0]


def test_discount_sensitivity_uses_discount_and_profitability() -> None:
    evidence = execute_business_plan("Compare discount sensitivity against profitability.", _business_df())

    assert evidence is not None
    result = evidence.computed_results[0]
    assert evidence.plan.operation == "DISCOUNT_SENSITIVITY"
    assert {"Discount", "Profit"} <= set(evidence.metrics_used)
    assert result["correlation"] < 0
    assert any(artifact["artifact_type"] == "chart" for artifact in evidence.artifacts)


def test_sales_vs_profit_chart_uses_both_metrics() -> None:
    evidence = execute_business_plan("Build a chart of Sales versus Profit by Category.", _business_df())

    assert evidence is not None
    assert evidence.plan.operation == "RELATIONSHIP_ANALYSIS"
    assert evidence.plan.x_metric == "Sales"
    assert evidence.plan.y_metric == "Profit"
    chart = next(artifact for artifact in evidence.artifacts if artifact["artifact_type"] == "chart")
    assert chart["content"]["chart_type"] == "scatter"
    assert chart["content"]["x_metric"] == "Sales"
    assert chart["content"]["y_metric"] == "Profit"
    assert chart["content"]["grouping"] == "Category"
    assert chart["content"]["x"] == "x"
    assert chart["content"]["y"] == "y"
    assert chart["content"]["points"]
    assert all({"label", "x", "y"} <= set(point) for point in chart["content"]["points"])
    assert {point["label"] for point in chart["content"]["points"]} == {"Furniture", "Office", "Tech"}


def test_explicit_scatter_request_does_not_become_bar_chart() -> None:
    evidence = execute_business_plan(
        "Build a scatter plot of total Sales versus total Profit by Category. Use total Sales on the x-axis, total Profit on the y-axis, and label each point by Category.",
        _business_df(),
    )

    assert evidence is not None
    chart = next(artifact for artifact in evidence.artifacts if artifact["artifact_type"] == "chart")
    assert chart["content"]["chart_type"] == "scatter"
    assert chart["metadata"]["chart_type"] == "scatter"
    assert chart["content"]["points"][0]["label"]


def test_do_not_use_averages_suppresses_unrequested_average_kpis() -> None:
    output = deterministic_investigation_fallback("Which categories have high sales but low profit? Do not use averages.", _business_df())

    assert output["trace_metadata"]["analysis_type"] == "high_sales_low_profit"
    assert "average" not in output["final_answer"].casefold()
    rows = output["artifacts"][0]["content"]
    assert all("average_discount" not in row for row in rows)


def test_prompt_restrictions_and_artifact_deduplication_are_respected() -> None:
    output = business_analysis_response("Focus only on profitability. Where is profit margin weakest? Do not mention schema quality.", _business_df())

    assert output is not None
    text = output["final_answer"].casefold()
    assert "schema" not in text
    assert "data quality" not in text
    signatures = [artifact["metadata"]["artifact_signature"] for artifact in output["artifacts"]]
    assert len(signatures) == len(set(signatures))
    assert {artifact["artifact_type"] for artifact in output["artifacts"]} == {"table", "chart"}


def test_multi_dataset_distribution_comparison_creates_two_branch_evidence_packages() -> None:
    output = _cross_output("Build a side-by-side visualization comparing BMI distribution and Sales distribution.")

    assert output["trace_metadata"]["operation"] == "CROSS_DATASET_DISTRIBUTION_COMPARISON"
    scopes = output["trace_metadata"]["dataset_execution_scopes"]
    assert len(scopes) == 2
    assert all(scope["computed_results"][0]["analysis_type"] == "distribution_comparison" for scope in scopes)
    assert {"Sales", "BMI"} == {scope["metrics_used"][0] for scope in scopes}
    assert sum(1 for artifact in output["artifacts"] if artifact["artifact_type"] == "chart") >= 2


def test_multi_dataset_anomaly_comparison_uses_actual_branch_metrics() -> None:
    output = _cross_output("Compare anomaly patterns between medical metrics and commercial metrics.")

    assert output["trace_metadata"]["operation"] == "CROSS_DATASET_ANOMALY_COMPARISON"
    assert "outlier" in output["summary"].casefold()
    scopes = output["trace_metadata"]["dataset_execution_scopes"]
    assert all(scope["computed_results"][0]["analysis_type"] == "anomaly_comparison" for scope in scopes)


def test_cross_dataset_executive_summary_uses_evidence_from_both_datasets() -> None:
    output = _cross_output("Build a cross-dataset executive summary using concrete evidence from both datasets.")

    assert output["trace_metadata"]["operation"] == "CROSS_DATASET_EXECUTIVE_SYNTHESIS"
    assert "`Business data`" in output["summary"]
    assert "`Health data`" in output["summary"]
    assert "capability" not in output["summary"].casefold()
    scopes = output["trace_metadata"]["dataset_execution_scopes"]
    assert all(scope["computed_results"] for scope in scopes)


def test_operational_optimization_comparison_does_not_collapse_to_capability_or_joinability() -> None:
    output = _cross_output("Which dataset is better suited for operational optimization and why?")

    assert output["trace_metadata"]["operation"] == "CROSS_DATASET_OPERATIONAL_OPTIMIZATION"
    text = output["summary"].casefold()
    assert "which dataset should i use" not in text
    assert "join" not in text
    assert "operational_inefficiency_score" in text or "risk indicator" in text
