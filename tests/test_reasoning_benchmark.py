"""Reasoning benchmark: semantic correctness tests.

Tests that the reasoning pipeline makes semantically correct decisions
about metrics, dimensions, domains, and groupings. Uses synthetic DataFrames,
NOT real datasets.
"""
from __future__ import annotations

import pandas as pd
import pytest


# =========================================================================
# Fixtures: synthetic datasets
# =========================================================================

@pytest.fixture
def retail_df() -> pd.DataFrame:
    """Synthetic retail-like dataset."""
    categories = ["Technology", "Furniture", "Office Supplies"]
    sub_categories = ["Phones", "Chairs", "Binders", "Tables", "Paper"]
    states = ["California", "Texas", "New York", "Florida", "Illinois"]
    regions = ["West", "South", "East", "Central"]
    segments = ["Consumer", "Corporate", "Home Office"]
    ship_modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
    return pd.DataFrame({
        "Row ID": range(1, 101),
        "Order ID": [f"ORD-{i:04d}" for i in range(1, 101)],
        "Customer ID": [f"CUST-{i % 30:03d}" for i in range(1, 101)],
        "Product ID": [f"PROD-{i % 20:03d}" for i in range(1, 101)],
        "Sales": [round(50 + i * 3.7, 2) for i in range(100)],
        "Profit": [round(10 + i * 1.2 - (i % 7) * 5, 2) for i in range(100)],
        "Quantity": [1 + i % 8 for i in range(100)],
        "Discount": [round((i % 5) * 0.05, 2) for i in range(100)],
        "Category": [categories[i % 3] for i in range(100)],
        "Sub-Category": [sub_categories[i % 5] for i in range(100)],
        "State": [states[i % 5] for i in range(100)],
        "City": [f"City_{i % 15}" for i in range(100)],
        "Region": [regions[i % 4] for i in range(100)],
        "Segment": [segments[i % 3] for i in range(100)],
        "Ship Mode": [ship_modes[i % 4] for i in range(100)],
        "Order Date": pd.date_range("2023-01-01", periods=100, freq="D"),
    })


@pytest.fixture
def entertainment_df() -> pd.DataFrame:
    """Synthetic entertainment-like dataset."""
    types = ["Movie", "TV Show"]
    countries = ["United States", "India", "United Kingdom", "Japan", "South Korea"]
    ratings = ["PG-13", "TV-MA", "PG", "R", "TV-14"]
    genres = ["Dramas", "Comedies", "Action", "Documentaries", "Thrillers"]
    topics = ["love", "adventure", "mystery", "science", "history"]
    return pd.DataFrame({
        "show_id": [f"s{i}" for i in range(1, 81)],
        "type": [types[i % 2] for i in range(80)],
        "title": [f"Title {i}" for i in range(1, 81)],
        "director": [f"Director {i % 15}" for i in range(80)],
        "cast": [f"Actor {i % 10}, Actor {(i + 3) % 10}" for i in range(80)],
        "country": [countries[i % 5] for i in range(80)],
        "release_year": [2015 + i % 9 for i in range(80)],
        "rating": [ratings[i % 5] for i in range(80)],
        "duration": [f"{60 + i % 90} min" if i % 2 == 0 else f"{1 + i % 5} Season" for i in range(80)],
        "listed_in": [genres[i % 5] for i in range(80)],
        "description": [f"A story about {topics[i % 5]}" for i in range(80)],
    })


@pytest.fixture
def healthcare_df() -> pd.DataFrame:
    """Synthetic healthcare-like dataset."""
    diagnoses = ["Diabetes", "Hypertension", "Asthma", "Heart Disease"]
    treatments = ["Medication A", "Medication B", "Surgery", "Therapy"]
    return pd.DataFrame({
        "Patient ID": [f"P-{i:04d}" for i in range(1, 61)],
        "Age": [25 + i % 50 for i in range(60)],
        "BMI": [round(18 + i * 0.3, 1) for i in range(60)],
        "Blood Pressure": [round(110 + i % 40, 0) for i in range(60)],
        "Diagnosis": [diagnoses[i % 4] for i in range(60)],
        "Treatment": [treatments[i % 4] for i in range(60)],
        "Hospital": [f"Hospital {i % 5}" for i in range(60)],
        "Admission Date": pd.date_range("2024-01-01", periods=60, freq="W"),
    })


# =========================================================================
# Component 1: Identifier exclusion
# =========================================================================

class TestIdentifierExclusion:
    """Identifiers must never become metrics."""

    def test_row_id_excluded_from_metric_candidates(self, retail_df: pd.DataFrame) -> None:
        from source.product.execution_planner import _metric_candidate_columns
        columns = [str(c) for c in retail_df.columns]
        candidates = _metric_candidate_columns(columns, retail_df)
        assert "Row ID" not in candidates

    def test_order_id_excluded(self, retail_df: pd.DataFrame) -> None:
        from source.product.execution_planner import _metric_candidate_columns
        columns = [str(c) for c in retail_df.columns]
        candidates = _metric_candidate_columns(columns, retail_df)
        assert "Order ID" not in candidates
        assert "Customer ID" not in candidates
        assert "Product ID" not in candidates

    def test_sales_profit_included(self, retail_df: pd.DataFrame) -> None:
        from source.product.execution_planner import _metric_candidate_columns
        columns = [str(c) for c in retail_df.columns]
        candidates = _metric_candidate_columns(columns, retail_df)
        assert "Sales" in candidates
        assert "Profit" in candidates
        assert "Quantity" in candidates
        assert "Discount" in candidates

    def test_plan_uses_sales_not_row_id(self, retail_df: pd.DataFrame) -> None:
        from source.product.execution_planner import AuthoritativeExecutionPlanner
        plan = AuthoritativeExecutionPlanner.plan("Top states by sales", retail_df)
        assert plan.metric != "Row ID"
        assert plan.metric == "Sales"

    def test_show_id_excluded(self, entertainment_df: pd.DataFrame) -> None:
        from source.product.execution_planner import _metric_candidate_columns
        columns = [str(c) for c in entertainment_df.columns]
        candidates = _metric_candidate_columns(columns, entertainment_df)
        assert "show_id" not in candidates

    def test_patient_id_excluded(self, healthcare_df: pd.DataFrame) -> None:
        from source.product.execution_planner import _metric_candidate_columns
        columns = [str(c) for c in healthcare_df.columns]
        candidates = _metric_candidate_columns(columns, healthcare_df)
        assert "Patient ID" not in candidates


# =========================================================================
# Component 2: Metric validation
# =========================================================================

class TestMetricValidation:
    """Validate metric and grouping semantics."""

    def test_row_id_invalid_as_metric(self, retail_df: pd.DataFrame) -> None:
        from source.product.metric_validation import validate_metric_for_analysis
        result = validate_metric_for_analysis("Row ID", retail_df)
        assert not result.valid
        assert result.suggested_alternative is not None

    def test_sales_valid_as_metric(self, retail_df: pd.DataFrame) -> None:
        from source.product.metric_validation import validate_metric_for_analysis
        result = validate_metric_for_analysis("Sales", retail_df)
        assert result.valid

    def test_grouping_row_id_by_state_invalid(self, retail_df: pd.DataFrame) -> None:
        from source.product.metric_validation import validate_grouping
        result = validate_grouping("Row ID", "State", retail_df)
        assert not result.valid

    def test_grouping_sales_by_state_valid(self, retail_df: pd.DataFrame) -> None:
        from source.product.metric_validation import validate_grouping
        result = validate_grouping("Sales", "State", retail_df)
        assert result.valid

    def test_grouping_by_single_value_invalid(self) -> None:
        from source.product.metric_validation import validate_grouping
        df = pd.DataFrame({"Sales": [10, 20, 30], "Status": ["Active"] * 3})
        result = validate_grouping("Sales", "Status", df)
        assert not result.valid


# =========================================================================
# Component 3: Domain profiling
# =========================================================================

class TestDomainProfiling:
    """Domain inference from column structure."""

    def test_retail_domain_detected(self, retail_df: pd.DataFrame) -> None:
        from source.product.semantic_layer import build_semantic_dataset_profile
        profile = build_semantic_dataset_profile(df=retail_df)
        assert profile.domain == "retail"
        assert profile.domain_confidence > 0.3

    def test_entertainment_domain_detected(self, entertainment_df: pd.DataFrame) -> None:
        from source.product.semantic_layer import build_semantic_dataset_profile
        profile = build_semantic_dataset_profile(df=entertainment_df)
        assert profile.domain == "entertainment"
        assert profile.domain_confidence > 0.3

    def test_healthcare_domain_detected(self, healthcare_df: pd.DataFrame) -> None:
        from source.product.semantic_layer import build_semantic_dataset_profile
        profile = build_semantic_dataset_profile(df=healthcare_df)
        assert profile.domain == "healthcare"
        assert profile.domain_confidence > 0.3

    def test_generic_domain_for_ambiguous(self) -> None:
        from source.product.semantic_layer import build_semantic_dataset_profile
        df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
        profile = build_semantic_dataset_profile(df=df)
        assert profile.domain == "general"

    def test_domain_vocabulary_populated(self, retail_df: pd.DataFrame) -> None:
        from source.product.semantic_layer import build_semantic_dataset_profile
        profile = build_semantic_dataset_profile(df=retail_df)
        assert len(profile.domain_vocabulary) > 0


# =========================================================================
# Component 5: Categorical field ranking
# =========================================================================

class TestCategoricalFieldRanking:
    """Genre, listed_in, type should rank as dimensions."""

    def test_genre_recognized_as_dimension(self, entertainment_df: pd.DataFrame) -> None:
        from source.product.semantic_layer import build_semantic_dataset_profile
        profile = build_semantic_dataset_profile(df=entertainment_df)
        dimension_names = [col.name for col in profile.dimensions]
        assert "listed_in" in dimension_names or "type" in dimension_names

    def test_category_recognized_as_dimension(self, retail_df: pd.DataFrame) -> None:
        from source.product.semantic_layer import build_semantic_dataset_profile
        profile = build_semantic_dataset_profile(df=retail_df)
        dimension_names = [col.name for col in profile.dimensions]
        assert "Category" in dimension_names
        assert "State" in dimension_names
        assert "Region" in dimension_names


# =========================================================================
# Component 6: Critic layer
# =========================================================================

class TestCriticSemanticChecks:
    """Quality gate catches semantic nonsense."""

    def test_identifier_as_metric_detected(self) -> None:
        from source.product.conversation_engine import response_quality_gate
        valid, reason = response_quality_gate(
            question="Show top states",
            response_text="The average `Row ID` by `State` shows California leading at 52.3, followed by Texas at 48.1.",
            conversation_context={},
        )
        assert not valid
        assert reason == "identifier_used_as_metric"

    def test_excessive_filler_detected(self) -> None:
        from source.product.conversation_engine import response_quality_gate
        valid, reason = response_quality_gate(
            question="What are the trends?",
            response_text="The operational optimization can improve forecasting stability. Operational risk should be monitored.",
            conversation_context={},
        )
        assert not valid
        assert reason == "excessive_generic_business_filler"

    def test_normal_response_passes(self) -> None:
        from source.product.conversation_engine import response_quality_gate
        valid, reason = response_quality_gate(
            question="Top states by sales",
            response_text="California leads with $45,000 in total `Sales`, followed by New York at $38,000.",
            conversation_context={},
        )
        assert valid


# =========================================================================
# Cross-component: plan integrity
# =========================================================================

class TestPlanIntegrity:
    """Plans should produce semantically valid metric+dimension combinations."""

    def test_rank_groups_uses_real_metric(self, retail_df: pd.DataFrame) -> None:
        from source.product.execution_planner import AuthoritativeExecutionPlanner
        plan = AuthoritativeExecutionPlanner.plan("Top categories by profit", retail_df)
        assert plan.metric == "Profit"
        assert plan.dimension is not None

    def test_temporal_intent_resolved(self, retail_df: pd.DataFrame) -> None:
        from source.product.execution_planner import AuthoritativeExecutionPlanner
        plan = AuthoritativeExecutionPlanner.plan("Sales trend over time", retail_df)
        assert plan.intent in {"temporal_trend", "growth", "chart_request"}
        assert plan.time_axis is not None

    def test_entertainment_count_analysis(self, entertainment_df: pd.DataFrame) -> None:
        from source.product.execution_planner import AuthoritativeExecutionPlanner
        plan = AuthoritativeExecutionPlanner.plan("How many titles by country?", entertainment_df)
        assert plan.metric != "show_id"
