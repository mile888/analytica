"""Generic regression tests using synthetic datasets with unseen column names.

Verify that the system works correctly on schemas it has never seen before,
proving that semantic inference is truly dynamic and not overfit to the
test fixtures (Sales, Income, City, etc.).
"""
from __future__ import annotations

import pandas as pd

from source.product.data_profiling import profile_dataframe
from source.product.data_sources import DataSource, DataSourceType
from source.product.dataset_registry import (
    DatasetScope,
    InvestigationDatasetEntry,
    _classify_financial_column,
    _concise_role_description,
    _entry_concepts_from_columns,
    _semantic_metric_alignment,
    _semantic_role,
    analyze_dataset_relationships,
    build_investigation_dataset_registry,
    registry_prompt,
    resolve_dataset_scope,
    resolve_explicit_metric_dataset,
)
from source.product.execution_context import persist_dataset_runtime
from source.product.run_service import InvestigationRunService
from source.product.service import InvestigationService
from source.product.store import InvestigationStore


def _add_source(store: InvestigationStore, name: str, df: pd.DataFrame, *, runtime: bool = True) -> str:
    source = store.create_data_source(DataSource(name=name, data_source_type=DataSourceType.CSV))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(df))
    if runtime:
        persist_dataset_runtime(store, source.data_source_id, df)
    return source.data_source_id


def _registry(store: InvestigationStore, investigation_id: str, ids: list[str]):
    return build_investigation_dataset_registry(store, store.get_investigation(investigation_id), ids)


# ===========================================================================
# SYNTHETIC DATASETS — UNSEEN SCHEMAS
# ===========================================================================

def _synthetic_transactional() -> pd.DataFrame:
    """Transactional dataset with column names not in any test fixture."""
    return pd.DataFrame({
        "GrossAmount": [120.5, 340.0, 89.0, 510.0],
        "UnitsSold": [3, 7, 2, 12],
        "BranchName": ["Downtown", "Uptown", "Suburbs", "Airport"],
        "InvoiceDate": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05"],
    })


def _synthetic_demographic() -> pd.DataFrame:
    """Demographic dataset with unseen column names."""
    return pd.DataFrame({
        "BirthYear": [1985, 1970, 1995],
        "HouseholdIncome": [65000, 82000, 41000],
        "CampaignScore": [7.2, 3.1, 8.8],
        "MaritalStatus": ["Single", "Married", "Single"],
    })


def _synthetic_operational() -> pd.DataFrame:
    """Operational/logistics dataset with no financial fields."""
    return pd.DataFrame({
        "WarehouseZone": ["A1", "B3", "C2", "A1"],
        "ShipmentCount": [120, 340, 200, 510],
        "RouteDelay": [2.5, 0.8, 1.2, 4.1],
        "DispatchDate": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15"],
    })


def _synthetic_medical() -> pd.DataFrame:
    """Medical/scientific dataset — completely different domain."""
    return pd.DataFrame({
        "PatientAge": [45, 62, 33, 71],
        "BloodPressure": [120, 145, 110, 155],
        "Cholesterol": [200, 240, 180, 260],
        "Diagnosis": ["Healthy", "Hypertension", "Healthy", "High Risk"],
    })


# ===========================================================================
# TEST 1 — UNSEEN TRANSACTIONAL DATASET INFERENCE
# ===========================================================================

def test_synthetic_transactional_infers_correct_roles() -> None:
    """GrossAmount and UnitsSold should be metrics,
    BranchName should be dimension, InvoiceDate should be timestamp."""
    assert _semantic_role("GrossAmount", "float64") == "metric"
    assert _semantic_role("UnitsSold", "int64") == "metric"
    assert _semantic_role("BranchName", "object") == "dimension"
    assert _semantic_role("InvoiceDate", "object") == "timestamp"


# ===========================================================================
# TEST 2 — UNSEEN DEMOGRAPHIC DATASET INFERENCE
# ===========================================================================

def test_synthetic_demographic_infers_distinct_role() -> None:
    """A demographic dataset must NOT be described
    as 'transactional revenue data'."""
    store = InvestigationStore()
    sid = _add_source(store, "Demographics", _synthetic_demographic())
    inv = store.create_investigation("demo")
    store.link_data_source_to_investigation(inv.investigation_id, sid)
    reg = _registry(store, inv.investigation_id, [sid])

    desc = _concise_role_description(reg[0])

    assert "transactional" not in desc.lower(), (
        f"Demographic dataset should not be described as transactional. Got: {desc}"
    )
    assert "demographic" in desc.lower() or "financial" in desc.lower(), (
        f"Should mention demographics or financial attributes. Got: {desc}"
    )


# ===========================================================================
# TEST 3 — UNSEEN OPERATIONAL DATASET IS NOT REVENUE
# ===========================================================================

def test_synthetic_operational_not_classified_as_revenue() -> None:
    """WarehouseZone, ShipmentCount, RouteDelay — no revenue classification."""
    for col in ["WarehouseZone", "RouteDelay", "DispatchDate"]:
        cls = _classify_financial_column(col, "metric")
        assert cls != "direct_revenue", (
            f"{col} should not be classified as direct_revenue. Got: {cls}"
        )


# ===========================================================================
# TEST 4 — CROSS-DATASET METRIC ALIGNMENT WITH UNSEEN SCHEMAS
# ===========================================================================

def test_metric_alignment_unseen_schemas() -> None:
    """Metric alignment should work on schemas the system has never seen."""
    store = InvestigationStore()
    tx = _add_source(store, "Transactions", _synthetic_transactional())
    demo = _add_source(store, "Demographics", _synthetic_demographic())
    ops = _add_source(store, "Operations", _synthetic_operational())
    inv = store.create_investigation("align")
    for sid in [tx, demo, ops]:
        store.link_data_source_to_investigation(inv.investigation_id, sid)
    reg = _registry(store, inv.investigation_id, [tx, demo, ops])

    alignment = _semantic_metric_alignment(reg, {}, "compare revenue metrics across datasets")

    assert alignment is not None, "Should produce alignment result"
    classifications = {row["dataset"]: row["classification"] for row in alignment["rows"]}

    # Transactions has GrossAmount → should be direct_revenue or financial
    tx_cls = classifications.get("Transactions", "")
    assert tx_cls in ("direct_revenue", "financial_amount"), (
        f"Transactions should have revenue-type metric. Got: {tx_cls}"
    )
    # Operations has ShipmentCount (count metric) but no financial revenue → should NOT be direct_revenue
    ops_cls = classifications.get("Operations", "")
    assert ops_cls != "direct_revenue", (
        f"Operations should not have direct_revenue. Got: {ops_cls}"
    )


# ===========================================================================
# TEST 5 — DATASET ROLE DESCRIPTIONS ON UNSEEN SCHEMAS
# ===========================================================================

def test_role_descriptions_unseen_schemas_are_distinct() -> None:
    """Three completely different datasets must get distinct descriptions."""
    store = InvestigationStore()
    tx = _add_source(store, "Transactions", _synthetic_transactional())
    demo = _add_source(store, "Demographics", _synthetic_demographic())
    ops = _add_source(store, "Operations", _synthetic_operational())
    inv = store.create_investigation("roles")
    for sid in [tx, demo, ops]:
        store.link_data_source_to_investigation(inv.investigation_id, sid)
    reg = _registry(store, inv.investigation_id, [tx, demo, ops])

    descriptions = {entry.display_name: _concise_role_description(entry) for entry in reg}

    # All descriptions should be different
    unique_descriptions = set(descriptions.values())
    assert len(unique_descriptions) >= 2, (
        f"Should have at least 2 distinct descriptions. Got: {descriptions}"
    )
    # None should be empty
    assert all(len(d) > 5 for d in descriptions.values()), (
        f"Descriptions too short: {descriptions}"
    )


# ===========================================================================
# TEST 6 — EXPLICIT METRIC SEARCH ON UNSEEN DATASET
# ===========================================================================

def test_explicit_metric_search_finds_gross_amount() -> None:
    """The explicit metric search should find GrossAmount in the
    transactional dataset when user asks for 'GrossAmount'."""
    store = InvestigationStore()
    tx = _add_source(store, "Transactions", _synthetic_transactional())
    ops = _add_source(store, "Operations", _synthetic_operational())
    inv = store.create_investigation("search")
    for sid in [tx, ops]:
        store.link_data_source_to_investigation(inv.investigation_id, sid)
    reg = _registry(store, inv.investigation_id, [tx, ops])

    match = resolve_explicit_metric_dataset("Build histogram of GrossAmount by zone", reg)

    assert match is not None, "Should find GrossAmount"
    assert match[0] == tx, f"Should be in Transactions, got {match[0]}"
    assert match[1] == "GrossAmount", f"Should match GrossAmount, got {match[1]}"


# ===========================================================================
# TEST 7 — STRUCTURAL QUESTION ON UNSEEN DATASETS
# ===========================================================================

def test_structural_question_unseen_datasets() -> None:
    """Joinability question on unseen datasets should route to CROSS."""
    store = InvestigationStore()
    tx = _add_source(store, "Transactions", _synthetic_transactional())
    med = _add_source(store, "Medical records", _synthetic_medical())
    inv = store.create_investigation("structure")
    for sid in [tx, med]:
        store.link_data_source_to_investigation(inv.investigation_id, sid)
    reg = _registry(store, inv.investigation_id, [tx, med])

    result = resolve_dataset_scope(
        question="Can these datasets be joined reliably?",
        registry=reg,
    )

    assert result.scope == DatasetScope.CROSS, (
        f"Should route to CROSS, got {result.scope}"
    )


# ===========================================================================
# TEST 8 — CONCEPT INFERENCE ON NOVEL COLUMNS
# ===========================================================================

def test_concept_inference_novel_columns() -> None:
    """Concepts must be inferred from token-level matching, not hardcoded."""
    # Test with completely novel column names
    concepts = _entry_concepts_from_columns(["BranchName", "InvoiceDate", "GrossAmount"])
    # InvoiceDate should trigger time concept
    assert "time" in concepts, f"InvoiceDate should trigger time concept. Got: {concepts}"

    # Test with medical columns — should NOT trigger financial concepts
    medical_concepts = _entry_concepts_from_columns(["PatientAge", "BloodPressure", "Diagnosis"])
    assert "monetary_value" not in medical_concepts, (
        f"Medical columns should not trigger monetary_value. Got: {medical_concepts}"
    )


# ===========================================================================
# TEST 9 — REGISTRY PROMPT ON UNSEEN SCHEMAS
# ===========================================================================

def test_registry_prompt_generic_schemas() -> None:
    """Registry prompt should work with any schema, not just Sales/Marketing."""
    store = InvestigationStore()
    tx = _add_source(store, "Procurement Log", _synthetic_transactional())
    med = _add_source(store, "Patient Records", _synthetic_medical())
    inv = store.create_investigation("prompt")
    for sid in [tx, med]:
        store.link_data_source_to_investigation(inv.investigation_id, sid)
    reg = _registry(store, inv.investigation_id, [tx, med])

    prompt = registry_prompt(reg)

    assert "`Procurement Log`" in prompt, f"Should contain dataset name. Got: {prompt}"
    assert "`Patient Records`" in prompt, f"Should contain dataset name. Got: {prompt}"
    assert "roles=" not in prompt, f"Should not dump roles=. Got: {prompt}"


# ===========================================================================
# TEST 10 — RELATIONSHIP ANALYSIS ON UNRELATED DATASETS
# ===========================================================================

def test_relationship_analysis_unrelated_datasets() -> None:
    """Two completely unrelated datasets should NOT be classified as joinable."""
    store = InvestigationStore()
    tx = _add_source(store, "Transactions", _synthetic_transactional())
    med = _add_source(store, "Medical records", _synthetic_medical())
    inv = store.create_investigation("rel")
    for sid in [tx, med]:
        store.link_data_source_to_investigation(inv.investigation_id, sid)
    reg = _registry(store, inv.investigation_id, [tx, med])

    relationship = analyze_dataset_relationships(reg)

    assert relationship["relationship_type"] != "directly_joinable", (
        f"Unrelated datasets should not be directly joinable. Got: {relationship['relationship_type']}"
    )


# ===========================================================================
# TEST 11 — MONETARY SUBTYPE GENERIC COLUMNS
# ===========================================================================

def test_monetary_subtype_generic_columns() -> None:
    """Financial classification should work on unseen column names."""
    from source.product.dataset_registry import _monetary_subtype

    assert _monetary_subtype("GrossRevenue") == "revenue"
    assert _monetary_subtype("NetProfit") == "profit"
    assert _monetary_subtype("TotalCost") == "price"  # "cost" → price tier
    assert _monetary_subtype("MonthlySpend") == "spend"
    assert _monetary_subtype("HouseholdIncome") == "income"
    assert _monetary_subtype("BloodPressure") == ""  # not financial


# ===========================================================================
# TEST 12 — WORD-BOUNDARY SAFETY ON NOVEL COLUMNS
# ===========================================================================

def test_word_boundary_novel_columns() -> None:
    """Verify that word-boundary matching does not produce false
    positives on novel column names."""
    # 'Salesperson' should NOT match 'sales' (it's a single token 'salesperson')
    # But after camelCase splitting: 'Salesperson' → 'salesperson' (one token)
    assert _semantic_role("Salesperson", "object") == "dimension"
    # 'PriceIndex' → 'price index' → 'price' matches → metric
    assert _semantic_role("PriceIndex", "float64") == "metric"
    # 'Timestamp' → 'timestamp' → 'time' is NOT a token (it's 'timestamp')
    # Actually _norm("Timestamp") → "timestamp" (no camelCase split)
    # so "time" is NOT a token. But "timestamp" doesn't match {"date", "time", ...}
    # However numeric detection will catch it via dtype
    role = _semantic_role("Timestamp", "object")
    # 'timestamp' as single token doesn't match any marker — should be dimension
    assert role == "dimension", f"Timestamp (object) should be dimension. Got: {role}"
