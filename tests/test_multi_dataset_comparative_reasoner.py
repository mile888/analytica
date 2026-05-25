from __future__ import annotations

import pandas as pd

from source.product.business_semantic_planner import execute_business_plan
from source.product.dataset_registry import (
    DatasetResolutionResult,
    DatasetScope,
    InvestigationDatasetEntry,
    cross_dataset_output,
)
from source.product.multi_dataset_comparative_reasoner import (
    critic_warnings_for_comparative_output,
    normalize_comparative_evidence,
)


def _business_df(scale: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Order Date": ["2020-01-01", "2020-05-01", "2021-01-01", "2021-08-01", "2022-01-01", "2022-04-01", "2023-01-01", "2023-09-01"],
            "City": ["A", "B", "A", "C", "A", "B", "D", "A"],
            "Customer Segment": ["Consumer", "Corporate", "Consumer", "Corporate", "Home Office", "Consumer", "Corporate", "Home Office"],
            "Category": ["Furniture", "Tech", "Office", "Tech", "Furniture", "Office", "Office", "Tech"],
            "Sales": [1000 * scale, 900 * scale, 3000 * scale, 2000 * scale, 2200 * scale, 1800 * scale, 400 * scale, 5000 * scale],
            "Profit": [20 * scale, -80 * scale, 600 * scale, 350 * scale, 40 * scale, -20 * scale, 30 * scale, 900 * scale],
            "Discount": [0.15, 0.30, 0.05, 0.08, 0.40, 0.35, 0.10, 0.25],
            "Shipping Cost": [200 * scale, 250 * scale, 100 * scale, 80 * scale, 600 * scale, 500 * scale, 50 * scale, 120 * scale],
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


def _entry(dataset_id: str, name: str, df: pd.DataFrame, concepts: list[str]) -> InvestigationDatasetEntry:
    metrics = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    dimensions = [column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]
    timestamps = [column for column in df.columns if "date" in column.casefold() or "year" in column.casefold()]
    return InvestigationDatasetEntry(
        dataset_id=dataset_id,
        display_name=name,
        source_name=name,
        row_count=len(df),
        column_names=list(df.columns),
        semantic_profile={"metric_columns": metrics, "dimension_columns": dimensions, "timestamp_columns": timestamps, "concepts": concepts},
    )


def _registry_and_frames() -> tuple[list[InvestigationDatasetEntry], dict[str, pd.DataFrame]]:
    local = _business_df()
    global_df = _business_df(scale=1.4)
    global_df["City"] = ["X", "Y", "Y", "Z", "X", "X", "Z", "Y"]
    health = _health_df()
    registry = [
        _entry("medical", "Medical metrics", health, ["health", "risk", "gender"]),
        _entry("local", "Local Superstore", local, ["monetary_value", "purchase_volume", "customer_entity", "product_or_category", "geography"]),
        _entry("global", "Global Superstore", global_df, ["monetary_value", "purchase_volume", "customer_entity", "product_or_category", "geography"]),
    ]
    return registry, {"medical": health, "local": local, "global": global_df}


def _cross_output(question: str, selected: list[str] | None = None) -> dict:
    registry, frames = _registry_and_frames()
    result = DatasetResolutionResult(scope=DatasetScope.CROSS, selected_dataset_ids=selected or ["medical", "local", "global"], confidence=1.0)
    return cross_dataset_output(question=question, registry=registry, frames=frames, result=result)


def test_single_dataset_profit_margin_still_uses_profit_over_sales() -> None:
    evidence = execute_business_plan("Where is profit margin weakest?", _business_df())

    assert evidence is not None
    assert evidence.plan.operation == "PROFIT_MARGIN_ANALYSIS"
    assert "profit_margin" in evidence.derived_kpis
    furniture = next(row for row in evidence.computed_results[0]["rows"] if row["group"] == "Furniture")
    assert furniture["profit_margin"] == round((20.0 + 40.0) / (1000.0 + 2200.0) * 100, 2)


def test_single_dataset_do_not_use_averages_still_respected() -> None:
    evidence = execute_business_plan("Which categories have high sales but low profit? Do not use averages.", _business_df())

    assert evidence is not None
    assert evidence.plan.operation == "HIGH_SALES_LOW_PROFIT"
    assert all("average_discount" not in row for row in evidence.computed_results[0]["rows"])


def test_explicit_local_global_trend_filters_to_named_branches() -> None:
    output = _cross_output("Compare sales performance trends between local and global superstore datasets.")

    assert output["trace_metadata"]["operation"] == "CROSS_DATASET_TEMPORAL_ANALYSIS"
    assert output["trace_metadata"]["dataset_ids"] == ["local", "global"]
    assert len(output["trace_metadata"]["dataset_execution_scopes"]) == 2
    assert "`Local Superstore`" in output["summary"]
    assert "`Global Superstore`" in output["summary"]


def test_distribution_comparison_uses_normalized_evidence_and_two_artifacts() -> None:
    output = _cross_output("Build a side-by-side visualization comparing BMI and Sales distributions.")

    packages = output["trace_metadata"]["comparative_evidence_packages"]
    assert output["trace_metadata"]["operation"] == "CROSS_DATASET_DISTRIBUTION_COMPARISON"
    assert sum(1 for package in packages if package["distribution_stats"]) >= 2
    assert sum(1 for artifact in output["artifacts"] if artifact["artifact_type"] == "chart") >= 2


def test_profit_by_customer_segment_does_not_pull_health_fallback_metric() -> None:
    output = _cross_output("Which customer segments generate highest total profit in each dataset?")

    assert output["trace_metadata"]["operation"] == "CROSS_DATASET_PROFITABILITY_COMPARISON"
    assert "BMI" not in output["summary"]
    assert "Age Group" not in output["summary"]
    business_scopes = [scope for scope in output["trace_metadata"]["dataset_execution_scopes"] if scope["dataset_id"] in {"local", "global"}]
    assert all(scope["computed_results"][0]["analysis_type"] == "grouped_metric" for scope in business_scopes)
    assert all(scope["metrics_used"] == ["Profit"] for scope in business_scopes)
    assert all(scope["grouping_fields"] == ["Customer Segment"] for scope in business_scopes)


def test_geographic_concentration_uses_city_not_order_date() -> None:
    output = _cross_output("Find the strongest sales cities in both datasets and compare geographic concentration.")

    business_scopes = [scope for scope in output["trace_metadata"]["dataset_execution_scopes"] if scope["dataset_id"] in {"local", "global"}]
    assert all(scope["grouping_fields"] == ["City"] for scope in business_scopes)
    assert "Order Date" not in output["summary"]
    assert "`Local Superstore`" in output["summary"]
    assert "`Global Superstore`" in output["summary"]


def test_executive_summary_uses_comparative_evidence_not_capability_summary() -> None:
    output = _cross_output("Build a cross-dataset executive summary using concrete evidence from both datasets.")
    text = output["summary"].casefold()

    assert output["trace_metadata"]["operation"] == "CROSS_DATASET_EXECUTIVE_SYNTHESIS"
    assert "executive contrast" in text
    assert "capability summary" not in text
    assert "which dataset should i use" not in text
    assert all(scope["computed_results"] for scope in output["trace_metadata"]["dataset_execution_scopes"])


def test_diversity_comparison_ranks_greater_diversity() -> None:
    output = _cross_output("Which dataset shows greater diversity and what are the strategic implications?")

    assert output["trace_metadata"]["operation"] == "CROSS_DATASET_DIVERSITY_ANALYSIS"
    assert "greater observed diversity" in output["summary"]
    assert "need numeric metric" not in output["summary"].casefold()


def test_relationship_comparison_uses_two_metrics_from_both_branches() -> None:
    output = _cross_output("Compare relationship patterns between Sales and Profit across local and global datasets.")

    assert output["trace_metadata"]["operation"] == "CROSS_DATASET_RELATIONSHIP_COMPARISON"
    packages = output["trace_metadata"]["comparative_evidence_packages"]
    relationship_packages = [package for package in packages if package["relationship_stats"]]
    assert len(relationship_packages) == 2
    assert all(package["relationship_stats"]["x_metric"] == "Sales" for package in relationship_packages)
    assert all(package["relationship_stats"]["y_metric"] == "Profit" for package in relationship_packages)


def test_comparative_visualization_has_branch_scoped_chart_artifacts() -> None:
    output = _cross_output("Build a comparative visualization of yearly sales growth for both datasets.", selected=["local", "global"])

    assert output["trace_metadata"]["operation"] == "CROSS_DATASET_TEMPORAL_ANALYSIS"
    assert sum(1 for artifact in output["artifacts"] if artifact["artifact_type"] == "chart") >= 2
    assert all(package["trend_stats"] for package in output["trace_metadata"]["comparative_evidence_packages"])


def test_predictability_comparison_uses_feature_structure() -> None:
    output = _cross_output("Which dataset appears more predictable based on feature structure?")

    assert output["trace_metadata"]["operation"] == "CROSS_DATASET_PREDICTABILITY_COMPARISON"
    assert "predictability score" in output["summary"]
    assert "feature structure" in output["summary"]


def test_branch_evidence_survives_into_normalized_packages() -> None:
    output = _cross_output("Compare anomaly patterns between medical and commercial metrics.")

    packages = output["trace_metadata"]["comparative_evidence_packages"]
    assert len(packages) == 3
    assert all(package["computed_results"] for package in packages)
    assert any(package["anomaly_stats"] for package in packages)


def test_comparative_critic_flags_single_branch_synthesis() -> None:
    output = _cross_output("Build a side-by-side visualization comparing BMI and Sales distributions.", selected=["medical", "local"])
    packages = normalize_comparative_evidence(
        question=output["query"],
        operation=output["trace_metadata"]["operation"],
        branches=output["trace_metadata"]["dataset_execution_scopes"],
    )

    warnings = critic_warnings_for_comparative_output(
        question=output["query"],
        operation=output["trace_metadata"]["operation"],
        answer="`Medical metrics` has a BMI distribution.",
        packages=packages,
        artifact_count=0,
    )

    assert any("omitted" in warning for warning in warnings)
    assert any("artifacts" in warning for warning in warnings)


def test_prior_single_dataset_memory_does_not_hijack_new_comparison() -> None:
    registry, frames = _registry_and_frames()
    result = DatasetResolutionResult(scope=DatasetScope.CROSS, selected_dataset_ids=["local", "global"], confidence=1.0)
    output = cross_dataset_output(
        question="Compare sales performance trends between local and global superstore datasets.",
        registry=registry,
        frames=frames,
        result=result,
        prior_findings=["Age average was the leading previous finding."],
    )

    assert "Age average" not in output["summary"]
    assert output["trace_metadata"]["dataset_ids"] == ["local", "global"]
