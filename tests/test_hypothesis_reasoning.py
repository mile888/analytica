from __future__ import annotations

import pandas as pd

from source.product.hypothesis_reasoning import HypothesisEngine, MechanismType, resolve_entity_or_value
from source.product.semantic_layer import build_semantic_dataset_profile


def test_decomposes_standard_class_volume_hypothesis() -> None:
    df = pd.DataFrame(
        {
            "Ship Mode": ["Standard Class", "Standard Class", "Same Day", "Second Class"],
            "Sales": [100.0, 110.0, 250.0, 240.0],
        }
    )
    semantic = build_semantic_dataset_profile(df=df)

    frame = HypothesisEngine.decompose(
        "Hypothesis: Standard Class dominates because of volume, not order value.",
        df,
        {},
        semantic,
    )

    assert frame.target.value == "Standard Class"
    assert frame.target.column == "Ship Mode"
    assert frame.target_metric == "Sales"
    assert frame.mechanism == MechanismType.VOLUME_EFFECT
    assert frame.comparison_type == "volume_vs_average_value"


def test_category_value_grounding_is_dataset_agnostic() -> None:
    df = pd.DataFrame(
        {
            "Queue Type": ["Priority", "Backlog", "Backlog"],
            "Resolution Hours": [2.0, 8.0, 9.0],
        }
    )
    semantic = build_semantic_dataset_profile(df=df)

    match = resolve_entity_or_value("Hypothesis: Backlog dominates because of volume", semantic, df)

    assert match is not None
    assert match.matched_column == "Queue Type"
    assert match.matched_value == "Backlog"


def test_validates_outlier_concentration_for_category_value() -> None:
    df = pd.DataFrame(
        {
            "Category": ["Technology", "Technology", "Furniture", "Office Supplies", "Furniture"],
            "Sales": [10.0, 1000.0, 12.0, 13.0, 11.0],
        }
    )
    semantic = build_semantic_dataset_profile(df=df)

    result = HypothesisEngine.validate(
        question="Hypothesis: Technology category creates most outliers.",
        dataframe=df,
        branch_state={},
        semantic_profile=semantic,
    )

    assert result is not None
    assert result["trace_metadata"]["dimension"] == "Category"
    assert result["trace_metadata"]["matched_category_value"] == "Technology"
    assert result["trace_metadata"]["hypothesis_mechanism"] == "outlier_concentration"
    assert "outlier" in result["final_answer"].lower()


def test_validates_sparse_group_instability_from_current_city_branch() -> None:
    df = pd.DataFrame(
        {
            "City": ["A", "B", "B", "B", "C", "C", "C"],
            "Sales": [1000.0, 100.0, 110.0, 105.0, 90.0, 95.0, 100.0],
        }
    )
    semantic = build_semantic_dataset_profile(df=df)

    result = HypothesisEngine.validate(
        question="Hypothesis: Some cities look strong only because of sparse data.",
        dataframe=df,
        branch_state={"active_metric": "Sales", "active_dimension": "City"},
        semantic_profile=semantic,
    )

    assert result is not None
    assert result["trace_metadata"]["dimension"] == "City"
    assert result["trace_metadata"]["hypothesis_mechanism"] == "sparse_group_instability"
    assert "sparse" in result["final_answer"].lower()


def test_validates_temporal_seasonality_from_branch_state() -> None:
    df = pd.DataFrame(
        {
            "Order Date": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "Revenue": [10, 12, 11, 9, 10, 80, 85, 78, 11, 10, 12, 9],
        }
    )
    semantic = build_semantic_dataset_profile(df=df)

    result = HypothesisEngine.validate(
        question="Hypothesis: growth is driven by seasonal spikes.",
        dataframe=df,
        branch_state={"active_metric": "Revenue", "active_time_axis": "Order Date", "active_branch_type": "trend_analysis"},
        semantic_profile=semantic,
    )

    assert result is not None
    assert result["trace_metadata"]["metric"] == "Revenue"
    assert result["trace_metadata"]["hypothesis_mechanism"] == "seasonality"
    assert "seasonality hypothesis" in result["final_answer"].lower()


def test_validates_duplicate_inflation_for_generic_revenue_metric() -> None:
    df = pd.DataFrame(
        {
            "Invoice": ["a", "b", "b"],
            "Revenue": [100.0, 200.0, 200.0],
        }
    )
    semantic = build_semantic_dataset_profile(df=df)

    result = HypothesisEngine.validate(
        question="Hypothesis: duplicates inflate revenue.",
        dataframe=df,
        branch_state={"active_branch_type": "data_quality"},
        semantic_profile=semantic,
    )

    assert result is not None
    assert result["trace_metadata"]["metric"] == "Revenue"
    assert result["trace_metadata"]["hypothesis_mechanism"] == "duplicate_inflation"
    assert "estimated inflation is 200.00" in result["final_answer"]
