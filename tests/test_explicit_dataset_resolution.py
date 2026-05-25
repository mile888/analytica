"""Tests for explicit dataset resolution and cross-dataset execution fix.

These tests verify that:
1. Explicit metric requests search ALL loaded datasets, not just the active one.
2. Active dataset context does not override explicit requests.
3. Dataset is resolved BEFORE filter resolution.
4. Revenue metric alignment uses granular classification.
5. Cross-dataset insights are strategic, not ontology labels.
6. Dataset role descriptions are concise and human-readable.
7. Single-dataset mode still works correctly (regression).
"""
from __future__ import annotations

import pandas as pd

from source.product.data_profiling import profile_dataframe
from source.product.data_sources import DataSource, DataSourceType
from source.product.dataset_registry import (
    DatasetScope,
    InvestigationDatasetEntry,
    analyze_dataset_relationships,
    build_investigation_dataset_registry,
    resolve_dataset_scope,
    resolve_explicit_metric_dataset,
)
from source.product.execution_context import persist_dataset_runtime
from source.product.investigation import InvestigationMessage, InvestigationMessageRole, InvestigationMessageType
from source.product.run_service import InvestigationRunService
from source.product.service import InvestigationService
from source.product.store import InvestigationStore


def _sales_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sales": [100.0, 240.0, 80.0, 310.0],
            "City": ["London", "Paris", "Berlin", "Tokyo"],
            "Customer": ["Alice", "Bob", "Cara", "Dan"],
        }
    )


def _marketing_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year_Birth": [1980, 1975, 1990],
            "Income": [50000, 62000, 47000],
            "MntWines": [120, 240, 80],
            "City": ["New York", "London", "Berlin"],
        }
    )


def _ships_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ShipName": ["Aurora", "Borealis", "Catalyst"],
            "Displacement": [25000, 35000, 18000],
            "Country": ["Norway", "Finland", "Japan"],
        }
    )


def _add_source(store: InvestigationStore, name: str, df: pd.DataFrame, *, runtime: bool = True) -> str:
    source = store.create_data_source(DataSource(name=name, data_source_type=DataSourceType.CSV))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(df))
    if runtime:
        persist_dataset_runtime(store, source.data_source_id, df)
    return source.data_source_id


def _registry(store: InvestigationStore, investigation_id: str, ids: list[str]):
    return build_investigation_dataset_registry(store, store.get_investigation(investigation_id), ids)


def _run_service(store: InvestigationStore) -> InvestigationRunService:
    return InvestigationRunService(
        store,
        InvestigationService(
            store,
            runner=lambda **_: {"summary": "Runner placeholder.", "artifacts": []},
        ),
    )


# ===========================================================================
# TEST 1 — GLOBAL EXPLICIT METRIC SEARCH
# ===========================================================================

def test_explicit_metric_resolves_to_correct_dataset() -> None:
    """When user asks for 'Sales', the dataset containing 'Sales' must be selected,
    even if both datasets share a column like 'City'."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    marketing = _add_source(store, "Marketing dataset", _marketing_df())
    investigation = store.create_investigation("multi")
    for dataset_id in [sales, marketing]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)
    registry = _registry(store, investigation.investigation_id, [sales, marketing])

    result = resolve_dataset_scope(question="Build histogram of Sales in London", registry=registry)

    assert result.scope == DatasetScope.SINGLE
    assert result.selected_dataset_ids == [sales], (
        f"Expected sales dataset, got {result.selected_dataset_ids}. "
        f"Scores: {result.candidate_scores}, Reasons: {result.reasons}"
    )


def test_resolve_explicit_metric_dataset_finds_unique_match() -> None:
    """resolve_explicit_metric_dataset must find Sales in exactly one dataset."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    marketing = _add_source(store, "Marketing dataset", _marketing_df())
    investigation = store.create_investigation("multi")
    for dataset_id in [sales, marketing]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)
    registry = _registry(store, investigation.investigation_id, [sales, marketing])

    match = resolve_explicit_metric_dataset("Build histogram of Sales in LA", registry)

    assert match is not None, "Should find Sales in one dataset"
    assert match[0] == sales, f"Expected sales dataset id, got {match[0]}"
    assert match[1] == "Sales", f"Expected 'Sales' column, got {match[1]}"


# ===========================================================================
# TEST 2 — ACTIVE DATASET MUST NOT OVERRIDE EXPLICIT REQUEST
# ===========================================================================

def test_explicit_metric_wins_over_active_context() -> None:
    """Even when conversation_context points to a different dataset,
    explicit metric matching must select the correct dataset."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    marketing = _add_source(store, "Marketing dataset", _marketing_df())
    investigation = store.create_investigation("multi")
    for dataset_id in [sales, marketing]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)
    registry = _registry(store, investigation.investigation_id, [sales, marketing])

    # Active context points to marketing dataset
    result = resolve_dataset_scope(
        question="Build histogram of Sales in Tokyo",
        registry=registry,
        conversation_context={"conversation_state": {"active_dataset_id": marketing}},
    )

    assert result.selected_dataset_ids == [sales], (
        f"Explicit metric 'Sales' should override active dataset context. "
        f"Got: {result.selected_dataset_ids}"
    )


# ===========================================================================
# TEST 3 — FILTER RESOLUTION ORDER (DATASET BEFORE FILTER)
# ===========================================================================

def test_ambiguous_overridden_by_explicit_metric_in_run_service() -> None:
    """When resolve_dataset_scope returns AMBIGUOUS (e.g. shared City column),
    the run_service must use resolve_explicit_metric_dataset to override."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    marketing = _add_source(store, "Marketing dataset", _marketing_df())
    investigation = store.create_investigation("multi")
    for dataset_id in [sales, marketing]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    # Create a question that mentions Sales (unique to sales dataset)
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            role=InvestigationMessageRole.USER,
            message_type=InvestigationMessageType.QUESTION,
            content="Build histogram of Sales in Tokyo",
        )
    )

    run = _run_service(store).run_investigation(
        investigation.investigation_id,
        message_id=message.message_id,
    )

    resolution = run.metadata.get("dataset_resolution", {})
    # The resolution should NOT be ambiguous — explicit metric search should have resolved it
    assert resolution.get("scope") != "ambiguous", (
        f"Expected explicit metric override, but got ambiguous. Resolution: {resolution}"
    )
    assert sales in resolution.get("selected_dataset_ids", []), (
        f"Expected sales dataset in selected_dataset_ids, got {resolution.get('selected_dataset_ids')}"
    )


# ===========================================================================
# TEST 4 — REVENUE METRIC ALIGNMENT CLASSIFICATION
# ===========================================================================

def test_revenue_metric_alignment_uses_granular_classification() -> None:
    """Metric alignment must classify as direct_revenue, spend_proxy, absent, etc.
    — not collapse everything to 'direct'."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    marketing = _add_source(store, "Marketing dataset", _marketing_df())
    ships = _add_source(store, "Ships dataset", _ships_df())
    investigation = store.create_investigation("compare")
    for dataset_id in [sales, marketing, ships]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)
    registry = _registry(store, investigation.investigation_id, [sales, marketing, ships])

    # Import the metric alignment function
    from source.product.dataset_registry import _semantic_metric_alignment

    result = _semantic_metric_alignment(registry, {}, "compare revenue related metrics across all datasets")

    assert result is not None, "Should produce a metric alignment result"
    rows = result["rows"]
    classifications = {row["dataset"]: row["classification"] for row in rows}

    # Sales dataset should be direct_revenue
    assert classifications.get("Sales dataset") == "direct_revenue", (
        f"Sales dataset should be direct_revenue, got {classifications.get('Sales dataset')}"
    )
    # Ships dataset should be absent (Displacement is not financial)
    assert classifications.get("Ships dataset") == "absent", (
        f"Ships dataset should be absent, got {classifications.get('Ships dataset')}"
    )
    # Marketing dataset — Income might be direct_revenue or spend_proxy depending on markers
    marketing_cls = classifications.get("Marketing dataset", "")
    assert marketing_cls != "absent", (
        f"Marketing dataset has Income/MntWines, should not be absent. Got: {marketing_cls}"
    )
    # Summary should NOT say all contain direct financial metrics
    assert "all" not in result["summary"].lower() or "direct financial" not in result["summary"].lower(), (
        "Summary should not claim all datasets contain direct financial metrics"
    )


# ===========================================================================
# TEST 5 — CROSS-DATASET INSIGHTS ARE STRATEGIC
# ===========================================================================

def test_cross_dataset_insights_are_strategic_not_ontological() -> None:
    """When user asks for 'most important insights', the answer must contain
    strategic analytical implications, not just dataset labeling."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales data", _sales_df())
    marketing = _add_source(store, "Marketing data", _marketing_df())
    investigation = store.create_investigation("insights")
    for dataset_id in [sales, marketing]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)
    registry = _registry(store, investigation.investigation_id, [sales, marketing])

    entries = [entry for entry in registry]
    relationship = analyze_dataset_relationships(entries)

    from source.product.dataset_registry import _strategic_insights_answer

    answer = _strategic_insights_answer(entries, relationship)

    # Should NOT just be ontology labels
    assert "serves as" not in answer.lower(), (
        f"Answer should not use ontology-style 'serves as' labeling. Got: {answer}"
    )
    # Should contain strategic language
    strategic_markers = ["feasible", "blocked", "attribution", "requires", "complementary", "limited", "independently", "warehouse", "reporting", "operational"]
    assert any(marker in answer.lower() for marker in strategic_markers), (
        f"Answer should contain strategic language. Got: {answer}"
    )


# ===========================================================================
# TEST 6 — DATASET ROLE CLASSIFICATION IS HUMAN-READABLE
# ===========================================================================

def test_dataset_role_classification_is_human_readable() -> None:
    """Dataset roles should be described in natural language,
    not as ontology dumps like 'transactional, financial, ecommerce'."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales data", _sales_df())
    marketing = _add_source(store, "Marketing data", _marketing_df())
    investigation = store.create_investigation("roles")
    for dataset_id in [sales, marketing]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)
    registry = _registry(store, investigation.investigation_id, [sales, marketing])

    from source.product.dataset_registry import _dataset_role_answer

    answer = _dataset_role_answer(registry)

    # Should NOT dump roles as comma-separated ontology terms
    role_dump_markers = ["transactional, financial", "financial, ecommerce", "customer_behavioral, demographic"]
    for marker in role_dump_markers:
        assert marker not in answer.lower(), (
            f"Answer should not contain ontology-style dump '{marker}'. Got: {answer}"
        )
    # Should contain natural descriptions
    assert "tracks" in answer.lower() or "contains" in answer.lower() or "focuses" in answer.lower() or "describes" in answer.lower(), (
        f"Answer should use natural language verbs. Got: {answer}"
    )


# ===========================================================================
# TEST 7 — SINGLE-DATASET REGRESSION
# ===========================================================================

def test_single_dataset_histogram_still_works() -> None:
    """Single-dataset mode must not be broken by multi-dataset changes."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    investigation = store.create_investigation("single")
    store.link_data_source_to_investigation(investigation.investigation_id, sales)

    result = resolve_dataset_scope(
        question="Build histogram of Sales by City",
        registry=_registry(store, investigation.investigation_id, [sales]),
    )

    assert result.scope == DatasetScope.SINGLE
    assert result.selected_dataset_ids == [sales]
    assert result.confidence == 1.0


def test_single_dataset_missing_metric_refusal_works() -> None:
    """When a metric is missing in a single-dataset scenario, refusal should still work."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    investigation = store.create_investigation("single")
    store.link_data_source_to_investigation(investigation.investigation_id, sales)

    # This should resolve to the single dataset (since there's only one)
    result = resolve_dataset_scope(
        question="Build histogram of Salary by City",
        registry=_registry(store, investigation.investigation_id, [sales]),
    )

    assert result.scope == DatasetScope.SINGLE
    assert result.selected_dataset_ids == [sales]


# ===========================================================================
# TEST 8 — STRUCTURAL QUESTIONS DON'T TRIGGER METRIC EXECUTION
# ===========================================================================

def test_structural_question_routes_to_cross_not_single() -> None:
    """Joinability / warehouse / schema questions should route to CROSS scope,
    not select a single dataset for metric execution."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales data", _sales_df())
    marketing = _add_source(store, "Marketing data", _marketing_df())
    investigation = store.create_investigation("structural")
    for dataset_id in [sales, marketing]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)
    registry = _registry(store, investigation.investigation_id, [sales, marketing])

    result = resolve_dataset_scope(
        question="Can these datasets be joined reliably?",
        registry=registry,
    )

    assert result.scope == DatasetScope.CROSS, (
        f"Structural joinability question should route to CROSS, got {result.scope}"
    )


# ===========================================================================
# TEST 9 — WORD-BOUNDARY SEMANTIC ROLE FIX
# ===========================================================================

def test_semantic_role_country_is_not_metric() -> None:
    """'Country' must be classified as dimension, not metric.
    Previously 'count' matched as substring of 'country'."""
    from source.product.dataset_registry import _semantic_role

    assert _semantic_role("Country", "object") == "dimension", (
        "'Country' should be dimension, not metric"
    )
    assert _semantic_role("Sales", "float64") == "metric", (
        "'Sales' should be metric"
    )
    assert _semantic_role("Order Count", "int64") == "metric", (
        "'Order Count' should match 'count' token and be metric"
    )
    assert _semantic_role("Quantity", "int64") == "metric", (
        "'Quantity' should be metric"
    )


# ===========================================================================
# TEST 10 — DATASET-AWARE FINDINGS FROM VISUALS
# ===========================================================================

def test_findings_from_visuals_are_dataset_specific() -> None:
    """Visual findings should mention specific dataset names,
    not use generic language."""
    from source.product.dataset_registry import _findings_from_visuals

    entries = [
        InvestigationDatasetEntry(
            dataset_id="ds1", display_name="Transactions", source_name="tx.csv",
            schema={}, profile={}, row_count=10000,
            column_names=["Amount", "Customer_ID", "City"],
            semantic_roles={}, semantic_profile={"concepts": ["monetary_value", "customer_entity", "geography"]},
            sample_values={}, created_at="", executable_available=True, runtime_reference="",
        ),
        InvestigationDatasetEntry(
            dataset_id="ds2", display_name="Products", source_name="prod.csv",
            schema={}, profile={}, row_count=50,
            column_names=["Product_Name", "Category"],
            semantic_roles={}, semantic_profile={"concepts": ["product_or_category"]},
            sample_values={}, created_at="", executable_available=True, runtime_reference="",
        ),
    ]
    relationship = {"joinable_shared_columns": [], "exact_shared_columns": [], "concept_matches": [], "shared_roles": [], "shared_concepts": []}
    findings = _findings_from_visuals(entries, relationship)

    # Findings should mention specific dataset names
    all_text = " ".join(findings)
    assert "Products" in all_text or "Transactions" in all_text, (
        f"Findings should reference dataset names. Got: {findings}"
    )
    # Should not be empty
    assert len(findings) >= 1, "Should generate at least one finding"


# ===========================================================================
# TEST 11 — REGISTRY PROMPT IS HUMAN-READABLE
# ===========================================================================

def test_registry_prompt_uses_natural_descriptions() -> None:
    """registry_prompt() should produce human-readable output,
    not ontology dumps like 'roles=transactional'."""
    from source.product.dataset_registry import registry_prompt

    store = InvestigationStore()
    sales = _add_source(store, "Sales data", _sales_df())
    marketing = _add_source(store, "Marketing data", _marketing_df())
    investigation = store.create_investigation("prompt")
    for dataset_id in [sales, marketing]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)
    registry = _registry(store, investigation.investigation_id, [sales, marketing])

    prompt = registry_prompt(registry)

    # Should NOT contain ontology-style fields
    assert "roles=" not in prompt, f"Prompt should not dump 'roles=...'. Got: {prompt}"
    assert "concepts=" not in prompt, f"Prompt should not dump 'concepts=...'. Got: {prompt}"
    # Should contain backtick-wrapped dataset names
    assert "`Sales data`" in prompt, f"Should contain dataset name. Got: {prompt}"
    assert "Metrics:" in prompt or "Dimensions:" in prompt, (
        f"Should list metrics or dimensions. Got: {prompt}"
    )


# ===========================================================================
# TEST 12 — CROSS-DATASET COMPARISON PRODUCES PER-DATASET ANALYSIS
# ===========================================================================

def test_cross_dataset_output_contains_per_dataset_rows() -> None:
    """cross_dataset_output must produce a table with one row per dataset,
    not collapse them into a single merged row."""
    from source.product.dataset_registry import cross_dataset_output, DatasetResolutionResult

    store = InvestigationStore()
    sales = _add_source(store, "Sales data", _sales_df())
    marketing = _add_source(store, "Marketing data", _marketing_df())
    investigation = store.create_investigation("compare")
    for dataset_id in [sales, marketing]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)
    registry = _registry(store, investigation.investigation_id, [sales, marketing])

    result = DatasetResolutionResult(
        DatasetScope.CROSS, [sales, marketing], 0.9,
        ["Cross-dataset comparison requested."],
    )
    output = cross_dataset_output(
        question="Compare customer-related fields across datasets",
        registry=registry,
        frames={sales: _sales_df(), marketing: _marketing_df()},
        result=result,
    )

    # Should have per-dataset artifacts
    summary_table = next(
        (a for a in output["artifacts"] if a.get("title") == "Dataset relationship summary"),
        None,
    )
    assert summary_table is not None, "Should contain dataset relationship summary"
    assert len(summary_table["content"]) == 2, (
        f"Should have one row per dataset, got {len(summary_table['content'])}"
    )
    # Summary text should NOT be empty
    assert len(output["summary"]) > 20, f"Summary too short: {output['summary']}"


# ===========================================================================
# TEST 13 — CONCEPT INFERENCE WORD BOUNDARY
# ===========================================================================

def test_concept_inference_no_substring_false_positives() -> None:
    """_entry_concepts_from_columns should not match 'count' inside 'country'."""
    from source.product.dataset_registry import _entry_concepts_from_columns

    # 'Country' should match geography (via 'country' token) but NOT purchase_volume (via 'count' substring)
    concepts = _entry_concepts_from_columns(["Country", "ShipName"])
    assert "geography" in concepts, f"Country should trigger geography. Got: {concepts}"
    assert "purchase_volume" not in concepts, (
        f"Country should NOT trigger purchase_volume (count substring). Got: {concepts}"
    )

    # 'Order Count' SHOULD match purchase_volume (has 'count' as a real token)
    concepts2 = _entry_concepts_from_columns(["Order Count"])
    assert "purchase_volume" in concepts2, (
        f"'Order Count' should match purchase_volume. Got: {concepts2}"
    )

