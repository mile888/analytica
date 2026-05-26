"""Tests for multi-dataset intelligence stabilization pass.

Covers:
  1. Metric subtype classification (_metric_subtype)
  2. Dataset primary purpose inference (_dataset_primary_purpose)
  3. Follow-up continuation detection (_looks_like_followup)
  4. Role description accuracy (_concise_role_description)
  5. Strategic insight structure and diversity (_strategic_insights_answer)
  6. Lineage score in resolve_dataset_scope
"""
from __future__ import annotations

import pytest

from source.product.dataset_registry import (
    InvestigationDatasetEntry,
    DatasetScope,
    analyze_dataset_relationships,
    resolve_dataset_scope,
    _concise_role_description,
    _dataset_primary_purpose,
    _dataset_semantic_profile,
    _looks_like_followup,
    _metric_subtype,
    _strategic_insights_answer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    *,
    dataset_id: str = "ds1",
    display_name: str = "test",
    columns: list[str],
    dtypes: list[str] | None = None,
) -> InvestigationDatasetEntry:
    """Create a minimal InvestigationDatasetEntry for testing."""
    from source.product.dataset_registry import _semantic_role
    if dtypes is None:
        dtypes = ["object"] * len(columns)
    semantic_roles = {col: _semantic_role(col, dtype) for col, dtype in zip(columns, dtypes)}
    semantic_profile = _dataset_semantic_profile(
        column_names=columns,
        semantic_roles=semantic_roles,
        profile={"row_count": 100, "column_count": len(columns)},
        display_name=display_name,
    )
    return InvestigationDatasetEntry(
        dataset_id=dataset_id,
        display_name=display_name,
        source_name=display_name,
        column_names=columns,
        semantic_roles=semantic_roles,
        semantic_profile=semantic_profile,
        row_count=100,
    )


# ===========================================================================
# 1. Metric Subtype Classification
# ===========================================================================

class TestMetricSubtype:
    """Verify that metric columns are classified into the correct subtype."""

    def test_direct_revenue(self):
        assert _metric_subtype("Sales") == "DIRECT_REVENUE"
        assert _metric_subtype("Revenue") == "DIRECT_REVENUE"
        assert _metric_subtype("TotalAmount") == "DIRECT_REVENUE"
        assert _metric_subtype("GrossProfit") == "DIRECT_REVENUE"

    def test_price_component(self):
        assert _metric_subtype("UnitPrice") == "PRICE_COMPONENT"
        assert _metric_subtype("Cost") == "PRICE_COMPONENT"
        assert _metric_subtype("ServiceFee") == "PRICE_COMPONENT"

    def test_behavioral_spending(self):
        assert _metric_subtype("MntWines") == "BEHAVIORAL_SPENDING"
        assert _metric_subtype("MntMeatProducts") == "BEHAVIORAL_SPENDING"
        assert _metric_subtype("TotalSpending") == "BEHAVIORAL_SPENDING"

    def test_financial_profile(self):
        assert _metric_subtype("Income") == "FINANCIAL_PROFILE"
        assert _metric_subtype("Salary") == "FINANCIAL_PROFILE"
        assert _metric_subtype("HouseholdIncome") == "FINANCIAL_PROFILE"

    def test_count_volume(self):
        assert _metric_subtype("Quantity") == "COUNT_VOLUME"
        assert _metric_subtype("NumPurchases") == "COUNT_VOLUME"
        assert _metric_subtype("OrderCount") == "COUNT_VOLUME"

    def test_engagement(self):
        assert _metric_subtype("Recency") == "ENGAGEMENT"
        assert _metric_subtype("AcceptedCmp1") == "ENGAGEMENT"
        assert _metric_subtype("SatisfactionScore") == "ENGAGEMENT"

    def test_generic_metric(self):
        assert _metric_subtype("Z_CostContact") == "PRICE_COMPONENT"  # 'cost' is a price marker
        assert _metric_subtype("RandomNumber") == "GENERIC_METRIC"
        assert _metric_subtype("AlphaValue") == "GENERIC_METRIC"


# ===========================================================================
# 2. Dataset Primary Purpose Inference
# ===========================================================================

class TestDatasetPrimaryPurpose:
    """Verify that datasets are classified into the correct purpose category."""

    def test_transactional_dataset(self):
        """A dataset with InvoiceNo, Sales, Quantity should be TRANSACTIONAL."""
        entry = _make_entry(
            columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"],
            dtypes=["object", "object", "object", "int64", "datetime64", "float64", "object", "object"],
            display_name="Online Retail",
        )
        assert _dataset_primary_purpose(entry) == "TRANSACTIONAL"

    def test_demographic_behavioral_dataset(self):
        """A dataset with Year_Birth, Education, Income, MntWines should be DEMOGRAPHIC."""
        entry = _make_entry(
            columns=["Year_Birth", "Education", "Marital_Status", "Income", "MntWines", "MntMeatProducts", "AcceptedCmp1", "Recency"],
            dtypes=["int64", "object", "object", "float64", "float64", "float64", "int64", "int64"],
            display_name="Customer Personality",
        )
        purpose = _dataset_primary_purpose(entry)
        # Should NOT be TRANSACTIONAL
        assert purpose != "TRANSACTIONAL", f"Expected DEMOGRAPHIC but got {purpose}"
        assert purpose in {"DEMOGRAPHIC", "BEHAVIORAL", "MIXED"}, f"Expected DEMOGRAPHIC/BEHAVIORAL/MIXED, got {purpose}"

    def test_operational_logistics_dataset(self):
        """A dataset about shipping/delivery should be LOGISTICS or OPERATIONAL."""
        entry = _make_entry(
            columns=["ShipmentID", "DeliveryDate", "ShipDate", "Delay", "Route", "Carrier", "Status"],
            dtypes=["object", "datetime64", "datetime64", "float64", "object", "object", "object"],
            display_name="Shipping Data",
        )
        purpose = _dataset_primary_purpose(entry)
        assert purpose in {"LOGISTICS", "OPERATIONAL"}, f"Expected LOGISTICS/OPERATIONAL, got {purpose}"

    def test_product_catalog_dataset(self):
        """A dataset about products should be PRODUCT."""
        entry = _make_entry(
            columns=["ProductID", "ProductName", "Category", "Brand", "SKU"],
            dtypes=["object"] * 5,
            display_name="Product Catalog",
        )
        assert _dataset_primary_purpose(entry) == "PRODUCT"

    def test_geographic_dataset(self):
        """A dataset about locations should be GEOGRAPHIC."""
        entry = _make_entry(
            columns=["RegionCode", "City", "State", "Country", "PostalCode", "Population"],
            dtypes=["object", "object", "object", "object", "object", "int64"],
            display_name="Regions",
        )
        assert _dataset_primary_purpose(entry) == "GEOGRAPHIC"

    def test_unseen_generic_dataset(self):
        """A dataset with unknown columns should return UNKNOWN."""
        entry = _make_entry(
            columns=["alpha", "beta", "gamma", "delta"],
            dtypes=["float64"] * 4,
            display_name="Mystery",
        )
        assert _dataset_primary_purpose(entry) == "UNKNOWN"


# ===========================================================================
# 3. Follow-up Continuation Detection
# ===========================================================================

class TestFollowupDetection:
    """Verify that continuation patterns are correctly detected."""

    def test_basic_followup_markers(self):
        assert _looks_like_followup("show this by region")
        assert _looks_like_followup("now compare against France")
        assert _looks_like_followup("remove outliers")
        assert _looks_like_followup("split by segment")

    def test_explanation_followup(self):
        assert _looks_like_followup("explain this chart")
        assert _looks_like_followup("what does this mean")

    def test_transformation_followup(self):
        assert _looks_like_followup("exclude the top 1%")
        assert _looks_like_followup("normalize by volume")

    def test_new_question_is_not_followup(self):
        """Brand new questions should NOT be detected as follow-ups."""
        assert not _looks_like_followup("show me total revenue by country")
        assert not _looks_like_followup("how many customers do we have")


# ===========================================================================
# 4. Role Description Accuracy
# ===========================================================================

class TestRoleDescription:
    """Verify that _concise_role_description produces accurate descriptions."""

    def test_transactional_role_description(self):
        entry = _make_entry(
            columns=["InvoiceNo", "Quantity", "UnitPrice", "Sales", "CustomerID"],
            dtypes=["object", "int64", "float64", "float64", "object"],
            display_name="Retail",
        )
        desc = _concise_role_description(entry)
        assert "transactional" in desc.lower() or "sales" in desc.lower()
        # Must NOT say "demographics"
        assert "demograph" not in desc.lower()

    def test_demographic_role_description(self):
        entry = _make_entry(
            columns=["Year_Birth", "Education", "Marital_Status", "Income", "MntWines", "AcceptedCmp1"],
            dtypes=["int64", "object", "object", "float64", "float64", "int64"],
            display_name="Customer Profile",
        )
        desc = _concise_role_description(entry)
        # Must NOT say "transactional" or "sales"
        assert "transactional" not in desc.lower(), f"Got: {desc}"
        assert "sales" not in desc.lower(), f"Got: {desc}"

    def test_role_description_not_too_long(self):
        """Role descriptions should be concise."""
        entry = _make_entry(
            columns=["A", "B", "C", "D", "E"],
            dtypes=["float64"] * 5,
            display_name="Short",
        )
        desc = _concise_role_description(entry)
        assert len(desc) <= 120, f"Description too long ({len(desc)} chars): {desc}"


# ===========================================================================
# 5. Strategic Insight Structure and Diversity
# ===========================================================================

class TestStrategicInsights:
    """Verify that _strategic_insights_answer produces 3 diverse insights."""

    def test_three_insights_produced(self):
        """Must always produce content that covers capability, blocker, and strategy."""
        entry1 = _make_entry(
            dataset_id="ds1",
            display_name="Sales",
            columns=["InvoiceNo", "Sales", "Country"],
            dtypes=["object", "float64", "object"],
        )
        entry2 = _make_entry(
            dataset_id="ds2",
            display_name="Demographics",
            columns=["Year_Birth", "Education", "Income"],
            dtypes=["int64", "object", "float64"],
        )
        relationship = analyze_dataset_relationships([entry1, entry2])
        result = _strategic_insights_answer([entry1, entry2], relationship)
        # Should have substantial content (3 sentences minimum)
        sentences = [s.strip() for s in result.split(".") if s.strip()]
        assert len(sentences) >= 3, f"Expected ≥3 sentences, got {len(sentences)}: {result}"

    def test_no_duplicate_insights(self):
        """The 3 insights must not have >60% token overlap with each other."""
        from source.product.dataset_registry import _finding_signature, _finding_overlap
        entry1 = _make_entry(
            dataset_id="ds1",
            display_name="Orders",
            columns=["OrderID", "Revenue", "City"],
            dtypes=["object", "float64", "object"],
        )
        entry2 = _make_entry(
            dataset_id="ds2",
            display_name="Products",
            columns=["ProductID", "Category", "Brand"],
            dtypes=["object", "object", "object"],
        )
        relationship = analyze_dataset_relationships([entry1, entry2])
        result = _strategic_insights_answer([entry1, entry2], relationship)
        # Split into the 3 insight sentences (they are space-joined)
        # Each insight is a complete thought ending with a period
        parts = [s.strip() + "." for s in result.split(". ") if s.strip()]
        if len(parts) >= 2:
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    overlap = _finding_overlap(_finding_signature(parts[i]), _finding_signature(parts[j]))
                    assert overlap <= 0.65, (
                        f"Insights {i} and {j} have {overlap:.2f} overlap (max 0.65):\n"
                        f"  [{i}] {parts[i]}\n  [{j}] {parts[j]}"
                    )


# ===========================================================================
# 6. Lineage Score in Dataset Resolution
# ===========================================================================

class TestLineageScore:
    """Verify that follow-up questions correctly use lineage context."""

    def test_followup_preserves_active_dataset(self):
        """A follow-up question should stay on the active dataset even if
        another dataset has a minor column match."""
        entry1 = _make_entry(
            dataset_id="ds_active",
            display_name="Active Dataset",
            columns=["Sales", "Region", "Date"],
            dtypes=["float64", "object", "datetime64"],
        )
        entry2 = _make_entry(
            dataset_id="ds_other",
            display_name="Other Dataset",
            columns=["Region", "Population", "Area"],
            dtypes=["object", "int64", "float64"],
        )
        result = resolve_dataset_scope(
            question="now show this by region",  # follow-up
            registry=[entry1, entry2],
            conversation_context={
                "latest_chart_context": {"dataset_id": "ds_active", "metric": "Sales"},
            },
        )
        assert result.scope == DatasetScope.SINGLE
        assert "ds_active" in result.selected_dataset_ids

    def test_explicit_mention_overrides_lineage(self):
        """An explicit dataset name mention should override lineage."""
        entry1 = _make_entry(
            dataset_id="ds1",
            display_name="Sales Data",
            columns=["Revenue", "City"],
            dtypes=["float64", "object"],
        )
        entry2 = _make_entry(
            dataset_id="ds2",
            display_name="Marketing Data",
            columns=["Campaign", "Clicks"],
            dtypes=["object", "int64"],
        )
        result = resolve_dataset_scope(
            question="show me marketing data campaign analysis",
            registry=[entry1, entry2],
            conversation_context={
                "latest_chart_context": {"dataset_id": "ds1"},
            },
        )
        assert result.scope == DatasetScope.SINGLE
        assert "ds2" in result.selected_dataset_ids


# ===========================================================================
# 7. Metric Subtypes in Semantic Profile
# ===========================================================================

class TestSemanticProfileMetricSubtypes:
    """Verify that semantic profiles include metric_subtypes."""

    def test_metric_subtypes_populated(self):
        from source.product.dataset_registry import _semantic_role
        columns = ["Sales", "Income", "Quantity", "Country"]
        dtypes = ["float64", "float64", "int64", "object"]
        roles = {col: _semantic_role(col, dtype) for col, dtype in zip(columns, dtypes)}
        profile = _dataset_semantic_profile(
            column_names=columns,
            semantic_roles=roles,
            profile={"row_count": 100, "column_count": 4},
            display_name="Test",
        )
        subtypes = profile.get("metric_subtypes", {})
        assert subtypes.get("Sales") == "DIRECT_REVENUE"
        assert subtypes.get("Income") == "FINANCIAL_PROFILE"
        assert subtypes.get("Quantity") == "COUNT_VOLUME"
        # Country is a dimension, not in subtypes
        assert "Country" not in subtypes
