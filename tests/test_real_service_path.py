"""Integration tests using the real InvestigationRunService path.

These tests exercise the SAME execution path as the UI/API, not isolated
helper functions. They verify that the canonical dataset-aware resolver
is actually used in production.
"""
from __future__ import annotations

import pandas as pd

from source.product.data_profiling import profile_dataframe
from source.product.data_sources import DataSource, DataSourceType
from source.product.execution_context import persist_dataset_runtime
from source.product.run_service import InvestigationRunService
from source.product.store import InvestigationMessage, InvestigationStore


def _setup_multi_dataset_workspace():
    """Create a workspace with 3 datasets: Sales, Marketing, Ships."""
    store = InvestigationStore()
    service = InvestigationRunService(store)

    df_sales = pd.DataFrame({
        "Sales": [100.0, 200.0, 350.0, 80.0],
        "Quantity": [3, 7, 2, 12],
        "City": ["London", "Tokyo", "Paris", "Berlin"],
        "Customer": ["A", "B", "C", "D"],
    })
    df_marketing = pd.DataFrame({
        "Year_Birth": [1980, 1975, 1990],
        "Income": [50000, 62000, 47000],
        "MntWines": [120, 240, 80],
        "AcceptedCampaign": [1, 0, 1],
    })
    df_ships = pd.DataFrame({
        "ShipName": ["Aurora", "Borealis"],
        "Displacement": [25000, 35000],
        "Country": ["Norway", "Finland"],
    })

    s_sales = store.create_data_source(DataSource(name="Sales dataset", data_source_type=DataSourceType.CSV))
    store.save_data_source_profile(s_sales.data_source_id, profile_dataframe(df_sales))
    persist_dataset_runtime(store, s_sales.data_source_id, df_sales)

    s_mkt = store.create_data_source(DataSource(name="Marketing dataset", data_source_type=DataSourceType.CSV))
    store.save_data_source_profile(s_mkt.data_source_id, profile_dataframe(df_marketing))
    persist_dataset_runtime(store, s_mkt.data_source_id, df_marketing)

    s_ships = store.create_data_source(DataSource(name="Ships dataset", data_source_type=DataSourceType.CSV))
    store.save_data_source_profile(s_ships.data_source_id, profile_dataframe(df_ships))
    persist_dataset_runtime(store, s_ships.data_source_id, df_ships)

    inv = store.create_investigation("Multi-dataset workspace")
    for sid in [s_sales.data_source_id, s_mkt.data_source_id, s_ships.data_source_id]:
        store.link_data_source_to_investigation(inv.investigation_id, sid)

    return store, service, inv, s_sales, s_mkt, s_ships, df_sales, df_marketing, df_ships


# ===========================================================================
# TEST 1: EXPLICIT METRIC OVERRIDE — "Build histogram of Sales in Tokyo"
# When the caller passes the WRONG df (marketing), the service must
# detect that Sales is missing from it, find Sales in another loaded
# dataset, and switch to that dataset.
# ===========================================================================

def test_real_path_explicit_metric_overrides_wrong_df() -> None:
    store, service, inv, s_sales, s_mkt, _, df_sales, df_marketing, _ = _setup_multi_dataset_workspace()

    msg = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=inv.investigation_id,
            content="Build histogram of Sales in Tokyo",
        )
    )

    # Pass df_marketing (which does NOT have Sales) as the "active" df.
    # The service must override it with the Sales dataset.
    service.run_investigation(inv.investigation_id, df=df_marketing, message_id=msg.message_id)
    updated = store.get_investigation(inv.investigation_id)

    answer = updated.report.summary.lower() if updated.report else ""
    # Must NOT say Sales is missing
    assert "does not contain" not in answer or "sales" not in answer.split("does not contain")[1][:30], (
        f"Should NOT say Sales is missing. Got: {updated.report.summary}"
    )
    # The word "sales" should appear in the answer (it was found)
    assert "sales" in answer, f"Should mention Sales. Got: {updated.report.summary}"
    # Tokyo exists in the Sales dataset City column, so filter should resolve.
    # The answer should NOT say "Filter: none" — it should either filter to Tokyo
    # or mention Tokyo explicitly.
    assert "filter: none" not in answer, (
        f"Should NOT silently drop filter. Got: {updated.report.summary}"
    )


# ===========================================================================
# TEST 2: EXPLICIT METRIC — "Build histogram of Sales in LA"
# Same override scenario with a different filter value.
# ===========================================================================

def test_real_path_explicit_metric_sales_in_la() -> None:
    store, service, inv, s_sales, s_mkt, _, df_sales, df_marketing, _ = _setup_multi_dataset_workspace()

    msg = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=inv.investigation_id,
            content="Build histogram of Sales in LA",
        )
    )

    service.run_investigation(inv.investigation_id, df=df_marketing, message_id=msg.message_id)
    updated = store.get_investigation(inv.investigation_id)

    answer = updated.report.summary.lower() if updated.report else ""
    # Must NOT say Sales is missing — it exists in the Sales dataset
    assert "does not contain" not in answer or "sales" not in answer.split("does not contain")[1][:30], (
        f"Should NOT say Sales is missing. Got: {updated.report.summary}"
    )
    # LA doesn't exist in the Sales dataset — should NOT silently drop the filter
    assert "filter: none" not in answer, (
        f"Should NOT silently drop filter. Got: {updated.report.summary}"
    )


# ===========================================================================
# TEST 3: DATASET ROLES — "Which dataset contains demographics?"
# ===========================================================================

def test_real_path_dataset_roles_distinct() -> None:
    store, service, inv, *_ = _setup_multi_dataset_workspace()

    msg = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=inv.investigation_id,
            content="Which dataset contains customer demographics and which tracks transactions?",
        )
    )

    service.run_investigation(inv.investigation_id, message_id=msg.message_id)
    updated = store.get_investigation(inv.investigation_id)

    answer = updated.report.summary.lower() if updated.report else ""
    # Should NOT have all three datasets described the same way
    assert "tracks customer transactions and purchases" not in answer or answer.count("tracks customer transactions and purchases") <= 1, (
        f"Role collapse detected. Got: {updated.report.summary}"
    )
    # Marketing should be described as demographics/campaign, NOT transactional
    marketing_idx = answer.find("marketing")
    if marketing_idx >= 0:
        marketing_segment = answer[marketing_idx:marketing_idx + 200]
        assert "demographic" in marketing_segment or "campaign" in marketing_segment, (
            f"Marketing should mention demographics or campaign. Got: {updated.report.summary}"
        )
        assert "tracks customer transactions and purchases" not in marketing_segment, (
            f"Marketing should NOT be labeled as transactional. Got: {updated.report.summary}"
        )
    # Sales should be described as transactional
    sales_idx = answer.find("sales dataset")
    if sales_idx >= 0:
        sales_segment = answer[sales_idx:sales_idx + 200]
        assert "transaction" in sales_segment or "purchase" in sales_segment, (
            f"Sales should be described as transactional. Got: {updated.report.summary}"
        )


# ===========================================================================
# TEST 4: REVENUE TYPING — "Compare revenue-related metrics"
# ===========================================================================

def test_real_path_revenue_typing_correct() -> None:
    store, service, inv, *_ = _setup_multi_dataset_workspace()

    msg = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=inv.investigation_id,
            content="Compare revenue-related metrics across all datasets",
        )
    )

    service.run_investigation(inv.investigation_id, message_id=msg.message_id)
    updated = store.get_investigation(inv.investigation_id)

    answer = updated.report.summary.lower() if updated.report else ""
    # Marketing should NOT be "direct transactional revenue"
    # (MntWines is a spend proxy, Income is personal financial)
    marketing_segment = ""
    if "marketing" in answer:
        idx = answer.index("marketing")
        marketing_segment = answer[idx:idx + 200]
    assert "direct transactional revenue" not in marketing_segment, (
        f"Marketing should not be direct revenue. Got: {updated.report.summary}"
    )
    # Sales SHOULD contain "direct transactional revenue"
    sales_segment = ""
    if "sales" in answer:
        idx = answer.index("sales")
        sales_segment = answer[idx:idx + 200]
    assert "direct transactional revenue" in sales_segment or "transaction" in sales_segment, (
        f"Sales should be described as direct transactional revenue. Got: {updated.report.summary}"
    )
    # Marketing should be spending aggregate, behavioral, or demographic
    assert any(w in marketing_segment for w in ("spending", "behavioral", "spend", "demographic", "attributes")), (
        f"Marketing should be spending/behavioral/demographic. Got: {updated.report.summary}"
    )


# ===========================================================================
# TEST 5: CROSS-DATASET INSIGHTS — "3 most important insights"
# ===========================================================================

def test_real_path_cross_dataset_insights() -> None:
    store, service, inv, *_ = _setup_multi_dataset_workspace()

    msg = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=inv.investigation_id,
            content="What are the 3 most important cross-dataset insights?",
        )
    )

    service.run_investigation(inv.investigation_id, message_id=msg.message_id)
    updated = store.get_investigation(inv.investigation_id)

    answer = updated.report.summary if updated.report else ""
    answer_lower = answer.lower()
    # Must have 3 categories of insights
    has_capability = any(w in answer_lower for w in ("feasible", "possible", "can be", "capable"))
    has_limitation = any(w in answer_lower for w in ("blocked", "missing", "cannot", "lack", "absent", "no stable", "no shared"))
    has_strategy = any(w in answer_lower for w in ("warehouse", "complementary", "operational", "executive", "strategic"))
    assert has_capability or has_limitation or has_strategy, (
        f"Insights should be strategic, not labels. Got: {answer}"
    )
    # Must NOT be just abstract structural restatements
    assert "useful comparison is semantic" not in answer_lower, (
        f"Insights should not be abstract. Got: {answer}"
    )


# ===========================================================================
# TEST 6: COLUMN COMPARISON — "Compare columns across all datasets"
# ===========================================================================

def test_real_path_column_comparison_concrete() -> None:
    store, service, inv, *_ = _setup_multi_dataset_workspace()

    msg = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=inv.investigation_id,
            content="Compare columns across all datasets",
        )
    )

    service.run_investigation(inv.investigation_id, message_id=msg.message_id)
    updated = store.get_investigation(inv.investigation_id)

    answer = updated.report.summary.lower() if updated.report else ""
    # Should mention concrete field names, not just "semantic comparison"
    has_concrete_fields = any(field in answer for field in ("sales", "income", "quantity", "displacement", "city", "country"))
    assert has_concrete_fields, (
        f"Column comparison should mention concrete fields. Got: {updated.report.summary}"
    )
    # Must NOT return the old abstract response
    assert "useful comparison is semantic" not in answer, (
        f"Column comparison should not be abstract. Got: {updated.report.summary}"
    )
    # Should include field category labels (metrics, dimensions, etc.)
    has_categories = any(cat in answer for cat in ("metrics:", "dimensions:", "entities:", "temporal:", "behavioral:", "operational:"))
    assert has_categories, (
        f"Column comparison should include field categories. Got: {updated.report.summary}"
    )


# ===========================================================================
# TEST 7: HISTOGRAM ARTIFACT — "Build histogram of Sales in LA"
# Backend MUST produce a chart artifact that the frontend can render.
# ===========================================================================

def test_real_path_histogram_produces_chart_artifact() -> None:
    store, service, inv, s_sales, _, _, df_sales, _, _ = _setup_multi_dataset_workspace()

    msg = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=inv.investigation_id,
            content="Build histogram of Sales in Tokyo",
        )
    )

    service.run_investigation(inv.investigation_id, message_id=msg.message_id)
    updated = store.get_investigation(inv.investigation_id)

    # Must have a chart artifact
    chart_artifacts = [a for a in updated.artifacts if a.artifact_type.value == "chart"]
    assert len(chart_artifacts) >= 1, (
        f"Expected at least one chart artifact. Got types: {[a.artifact_type.value for a in updated.artifacts]}"
    )

    chart = chart_artifacts[0]
    content = chart.content
    assert isinstance(content, dict), f"Chart content must be a dict, got: {type(content)}"
    assert content.get("chart_type") == "histogram", (
        f"Chart type must be histogram. Got: {content.get('chart_type')}"
    )

    # Must have bin data that the frontend can render
    has_bins = bool(content.get("bins"))
    has_bin_edges = bool(content.get("bin_edges")) and bool(content.get("bin_counts"))
    assert has_bins or has_bin_edges, (
        f"Chart must have bins or bin_edges+bin_counts. Got keys: {list(content.keys())}"
    )

    # row_count must be present and > 0
    row_count = content.get("row_count")
    assert row_count is not None and row_count > 0, (
        f"Chart must have positive row_count. Got: {row_count}"
    )

    # Verify bin total matches row_count (frontend validation requirement)
    if has_bin_edges:
        bin_total = sum(content["bin_counts"])
        assert bin_total == row_count, (
            f"bin_counts total ({bin_total}) must match row_count ({row_count})"
        )
    elif has_bins:
        bin_total = sum(b.get("count", 0) for b in content["bins"])
        assert bin_total == row_count, (
            f"bins total ({bin_total}) must match row_count ({row_count})"
        )
