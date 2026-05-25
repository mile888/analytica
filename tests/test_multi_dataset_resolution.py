from __future__ import annotations

import pandas as pd

from source.product.branch_workspace import branch_dtos_for_investigation
from source.product.data_profiling import profile_dataframe
from source.product.data_sources import DataSource, DataSourceType
from source.product.dataset_registry import (
    DatasetScope,
    analyze_dataset_relationships,
    build_investigation_dataset_registry,
    resolve_dataset_scope,
)
from source.product.execution_context import persist_dataset_runtime
from source.product.investigation import Artifact, ArtifactType, InvestigationMessage, InvestigationMessageRole, InvestigationMessageType
from source.product.run_service import InvestigationRunService
from source.product.service import InvestigationService
from source.product.store import InvestigationStore


def _sales_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sales": [100.0, 240.0, 80.0],
            "City": ["London", "Paris", "Berlin"],
            "Customer": ["Alice", "Bob", "Cara"],
        }
    )


def _salary_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Salary_LPA": [12.0, 22.0, 18.0],
            "Company": ["Acme", "Globex", "Initech"],
            "City": ["London", "Rome", "Paris"],
        }
    )


def _amount_a() -> pd.DataFrame:
    return pd.DataFrame({"Amount": [10.0, 20.0], "Region": ["A", "B"]})


def _amount_b() -> pd.DataFrame:
    return pd.DataFrame({"Amount": [30.0, 50.0], "Region": ["A", "C"]})


def _transaction_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "InvoiceNo": ["1", "2", "3"],
            "ProductLine": ["A", "B", "A"],
            "Quantity": [2, 1, 4],
            "TaxAmount": [1.2, 0.8, 2.1],
        }
    )


def _marketing_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year_Birth": [1980, 1975, 1990],
            "Income": [50000, 62000, 47000],
            "MntWines": [120, 240, 80],
            "AcceptedCampaign": [1, 0, 1],
        }
    )


def _entertainment_time_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Title": [f"Title {i}" for i in range(12)],
            "Genre": ["Drama", "Comedy", "Action", "Drama", "Documentary", "Comedy", "Action", "Drama", "Comedy", "Action", "Drama", "Documentary"],
            "ReleaseYear": [2018, 2018, 2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023],
            "Format": ["Movie", "Movie", "TV Show", "Movie", "Movie", "TV Show", "Movie", "TV Show", "Movie", "Movie", "TV Show", "Movie"],
        }
    )


def _health_prevalence_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [35, 42, 51, 60, 68, 72, 45, 39, 58, 63, 49, 55],
            "RiskIndicator": [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
            "SmokerFlag": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
            "ActivityLevel": ["High", "Medium", "Low", "Low", "Low", "Medium", "High", "High", "Low", "Medium", "High", "Low"],
        }
    )


def _retail_segment_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "OrderDate": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01", "2022-01-01", "2022-02-01", "2022-03-01"]),
            "Segment": ["Consumer", "Corporate", "Home Office", "Consumer", "Corporate", "Home Office"],
            "ProductCategory": ["Furniture", "Technology", "Office Supplies", "Technology", "Furniture", "Office Supplies"],
            "Sales": [120.0, 220.0, 90.0, 180.0, 260.0, 110.0],
        }
    )


def _add_source(store: InvestigationStore, name: str, df: pd.DataFrame, *, runtime: bool = True) -> str:
    source = store.create_data_source(DataSource(name=name, data_source_type=DataSourceType.CSV))
    store.save_data_source_profile(source.data_source_id, profile_dataframe(df))
    if runtime:
        persist_dataset_runtime(store, source.data_source_id, df)
    return source.data_source_id


def _run_service(store: InvestigationStore) -> InvestigationRunService:
    return InvestigationRunService(
        store,
        InvestigationService(
            store,
            runner=lambda **_: {"summary": "Runner placeholder.", "artifacts": []},
        ),
    )


def _setup_three_domain_workspace(question: str):
    store = InvestigationStore()
    entertainment = _add_source(store, "Entertainment catalog", _entertainment_time_df())
    health = _add_source(store, "Health indicators", _health_prevalence_df())
    retail = _add_source(store, "Retail orders", _retail_segment_df())
    investigation = store.create_investigation(question)
    for dataset_id in [entertainment, health, retail]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)
    return store, investigation, [entertainment, health, retail]


def _registry(store: InvestigationStore, investigation_id: str, ids: list[str]):
    return build_investigation_dataset_registry(store, store.get_investigation(investigation_id), ids)


def test_single_dataset_auto_selects_without_clarification() -> None:
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    investigation = store.create_investigation("top cities by sales")
    store.link_data_source_to_investigation(investigation.investigation_id, sales)

    result = resolve_dataset_scope(question="top cities by sales", registry=_registry(store, investigation.investigation_id, [sales]))

    assert result.scope == DatasetScope.SINGLE
    assert result.selected_dataset_ids == [sales]


def test_column_matches_route_to_distinct_datasets() -> None:
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("multi")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)
    registry = _registry(store, investigation.investigation_id, [sales, salary])

    sales_result = resolve_dataset_scope(question="which customer has highest sales", registry=registry)
    salary_result = resolve_dataset_scope(question="which company has highest salary", registry=registry)

    assert sales_result.selected_dataset_ids == [sales]
    assert salary_result.selected_dataset_ids == [salary]


def test_value_match_selects_correct_dataset() -> None:
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("multi")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    result = resolve_dataset_scope(question="show records for Globex", registry=_registry(store, investigation.investigation_id, [sales, salary]))

    assert result.selected_dataset_ids == [salary]
    assert any("sample value" in reason for reason in result.reasons)


def test_ambiguous_shared_column_does_not_execute() -> None:
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("top cities")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    assert run.metadata["dataset_resolution"]["scope"] == "ambiguous"
    assert "Which dataset should I use?" in updated.report.summary
    assert not [artifact for artifact in updated.artifacts if artifact.run_id == run.run_id and artifact.visibility.value == "user"]


def test_followup_to_artifact_uses_artifact_dataset_lineage() -> None:
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("multi")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)
    artifact = Artifact(
        artifact_type=ArtifactType.CHART,
        title="Salary by City",
        content={"chart_type": "bar", "metric": "Salary_LPA", "dimension": "City", "rows": []},
        metadata={"dataset_id": salary, "dataset_ids": [salary], "dataset_scope": "single_dataset", "branch_id": "grouped::Salary_LPA::City::sum::"},
    )
    store.add_artifact(investigation.investigation_id, artifact)
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            role=InvestigationMessageRole.USER,
            message_type=InvestigationMessageType.FOLLOW_UP,
            content="which cities remain leaders?",
            metadata={"artifact_id": artifact.artifact_id, "dataset_id": salary},
        )
    )

    run = _run_service(store).run_investigation(investigation.investigation_id, message_id=message.message_id)

    assert run.metadata["dataset_resolution"]["selected_dataset_ids"] == [salary]


def test_explicit_dataset_switch_overrides_active_branch() -> None:
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("multi")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)
    registry = _registry(store, investigation.investigation_id, [sales, salary])

    result = resolve_dataset_scope(
        question="now use the Sales dataset and top cities by sales",
        registry=registry,
        conversation_context={"conversation_state": {"active_dataset_id": salary}},
    )

    assert result.selected_dataset_ids == [sales]


def test_cross_dataset_schema_and_metric_comparison_work() -> None:
    store = InvestigationStore()
    left = _add_source(store, "Left amount", _amount_a())
    right = _add_source(store, "Right amount", _amount_b())
    investigation = store.create_investigation("compare row counts and average amount across datasets")
    for dataset_id in [left, right]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    artifacts = [artifact for artifact in updated.artifacts if artifact.run_id == run.run_id and artifact.artifact_type in {ArtifactType.TABLE}]

    assert run.metadata["dataset_resolution"]["scope"] == "cross_dataset"
    assert any(artifact.metadata.get("dataset_scope") == "cross_dataset" for artifact in artifacts)
    assert "higher average `Amount`" in updated.report.summary


def test_cross_dataset_common_patterns_reason_semantically_without_shared_columns() -> None:
    store = InvestigationStore()
    sales = _add_source(store, "Transactional dataset", _transaction_df())
    marketing = _add_source(store, "Marketing dataset", _marketing_df())
    investigation = store.create_investigation("Are there any common patterns across both datasets?")
    for dataset_id in [sales, marketing]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    summary = updated.report.summary.lower()

    assert run.metadata["dataset_resolution"]["scope"] == "cross_dataset"
    assert "shared columns: none detected" not in summary
    assert "semantic bridge" in summary or "aggregate behavior" in summary
    assert "join" not in summary
    assert run.metadata["dataset_relationships"]["relationship_type"] in {"indirectly_comparable", "domain_related", "analytically_related"}
    assert any(artifact.metadata.get("dataset_id") for artifact in updated.artifacts)


def test_dataset_relationship_classifier_keeps_no_fake_join_guarantee() -> None:
    store = InvestigationStore()
    sales = _add_source(store, "Transactional dataset", _transaction_df())
    marketing = _add_source(store, "Marketing dataset", _marketing_df())
    investigation = store.create_investigation("multi")
    registry = _registry(store, investigation.investigation_id, [sales, marketing])

    relationship = analyze_dataset_relationships(registry)

    assert relationship["relationship_type"] != "directly_joinable"
    assert any("No reliable shared column" in item for item in relationship["limitations"])
    assert relationship["concept_rows"]


def test_incompatible_temporal_comparison_explains_constraints() -> None:
    store = InvestigationStore()
    sales = _add_source(store, "Transactional dataset", _transaction_df())
    marketing = _add_source(store, "Marketing dataset", _marketing_df())
    investigation = store.create_investigation("compare years in both datasets")
    for dataset_id in [sales, marketing]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    text = updated.report.summary.lower()

    assert "year" in text or "temporal" in text or "time" in text
    assert "not possible" in text
    assert run.metadata["dataset_relationships"]["concept_rows"]


def test_weak_demographic_bridges_are_rejected() -> None:
    store = InvestigationStore()
    left = _add_source(store, "Profile A", pd.DataFrame({"Education": ["Grad"], "Marital_Status": ["Single"]}))
    right = _add_source(store, "Profile B", pd.DataFrame({"Gender": ["F"], "CustomerType": ["New"]}))
    investigation = store.create_investigation("multi")
    registry = _registry(store, investigation.investigation_id, [left, right])

    relationship = analyze_dataset_relationships(registry)
    bridge_values = " ".join(str(row.get("value", "")) for row in relationship["concept_rows"])

    assert "Education ↔" not in bridge_values
    assert "Marital_Status ↔" not in bridge_values
    assert relationship["concept_matches"] == []


def test_timestamp_customer_type_bridge_is_rejected() -> None:
    store = InvestigationStore()
    left = _add_source(store, "Customer dates", pd.DataFrame({"Dt_Customer": ["2020-01-01"]}))
    right = _add_source(store, "Customer labels", pd.DataFrame({"CustomerType": ["VIP"]}))
    investigation = store.create_investigation("multi")
    registry = _registry(store, investigation.investigation_id, [left, right])

    relationship = analyze_dataset_relationships(registry)

    assert relationship["concept_matches"] == []
    assert all("Dt_Customer" not in str(row.get("value", "")) or "CustomerType" not in str(row.get("value", "")) for row in relationship["concept_rows"])


def test_no_generic_fallback_phrase_in_cross_dataset_answer() -> None:
    store = InvestigationStore()
    sales = _add_source(store, "Transactional dataset", _transaction_df())
    marketing = _add_source(store, "Marketing dataset", _marketing_df())
    investigation = store.create_investigation("are there common patterns?")
    for dataset_id in [sales, marketing]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    _run_service(store).run_investigation(investigation.investigation_id)
    summary = store.get_investigation(investigation.investigation_id).report.summary.lower()

    assert "analytical picture is unchanged" not in summary


def test_runtime_unavailable_for_selected_dataset_returns_scoped_error() -> None:
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df(), runtime=False)
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("top customer by sales")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    assert run.metadata["dataset_resolution"]["selected_dataset_ids"] == [sales]
    assert updated.report.metadata["trace_metadata"]["data_source_id"] == sales
    assert updated.report.metadata["trace_metadata"]["execution_context_unavailable"] is True


def test_artifact_and_branch_store_dataset_lineage() -> None:
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    investigation = store.create_investigation("top cities by sales")
    store.link_data_source_to_investigation(investigation.investigation_id, sales)

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    user_artifacts = [artifact for artifact in updated.artifacts if artifact.run_id == run.run_id and artifact.visibility.value == "user"]
    branches = branch_dtos_for_investigation(updated)

    assert any(artifact.metadata.get("dataset_id") == sales for artifact in user_artifacts)
    assert any(branch.get("dataset_id") == sales or sales in branch.get("dataset_ids", []) for branch in branches)


def test_cross_dataset_output_produces_chart_artifacts() -> None:
    """Cross-dataset analysis must produce at least one chart artifact (not just tables)."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("compare both datasets")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    chart_artifacts = [a for a in updated.artifacts if a.run_id == run.run_id and a.artifact_type == ArtifactType.CHART]

    assert len(chart_artifacts) >= 1, f"Expected chart artifacts, got {len(chart_artifacts)}"


def test_cross_dataset_output_produces_findings() -> None:
    """Cross-dataset analysis must produce meaningful findings."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("can these datasets be joined?")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    assert len(updated.findings) >= 1, f"Expected findings, got {len(updated.findings)}"
    assert run.metadata["dataset_resolution"]["scope"] == "cross_dataset"


def test_bridge_spam_replaced_with_business_summary() -> None:
    """'Strongest validated bridges' must NOT appear in cross-dataset findings."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("compare both datasets")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    summary = updated.report.summary

    assert "Strongest validated bridges" not in summary


def test_compatibility_heatmap_generated() -> None:
    """Cross-dataset analysis must produce a compatibility heatmap."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("compare both datasets")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    heatmaps = [
        a for a in updated.artifacts
        if a.run_id == run.run_id
        and a.artifact_type == ArtifactType.CHART
        and isinstance(a.content, dict)
        and a.content.get("chart_type") == "heatmap"
    ]

    assert len(heatmaps) >= 1, f"Expected at least 1 heatmap, got {len(heatmaps)}"
    titles = {a.title for a in heatmaps}
    assert "Dataset Compatibility Assessment" in titles or "Shared Entity Coverage" in titles


def test_entity_coverage_matrix_generated() -> None:
    """Cross-dataset analysis must produce a shared entity coverage matrix."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    marketing = _add_source(store, "Marketing dataset", _marketing_df())
    investigation = store.create_investigation("what shared entities exist across datasets?")
    for dataset_id in [sales, marketing]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    entity_charts = [
        a for a in updated.artifacts
        if a.run_id == run.run_id
        and a.artifact_type == ArtifactType.CHART
        and a.title == "Shared Entity Coverage"
    ]

    assert len(entity_charts) == 1
    content = entity_charts[0].content
    assert isinstance(content, dict)
    assert content.get("chart_type") == "heatmap"
    assert len(content.get("rows", [])) >= 1
    # Rows must be record objects, not string labels
    first_row = content["rows"][0]
    assert isinstance(first_row, dict), f"Expected dict row, got {type(first_row)}"
    assert "row" in first_row and "column" in first_row and "value" in first_row


def test_heatmap_rows_are_record_objects_not_strings() -> None:
    """Heatmap rows must be record objects (dicts) for the frontend normalizer, not string labels."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("compare both datasets")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    heatmaps = [
        a for a in updated.artifacts
        if a.run_id == run.run_id
        and a.artifact_type == ArtifactType.CHART
        and isinstance(a.content, dict)
        and a.content.get("chart_type") == "heatmap"
    ]

    for hm in heatmaps:
        rows = hm.content.get("rows", [])
        assert rows, f"Heatmap '{hm.title}' has no rows"
        for row in rows:
            assert isinstance(row, dict), f"Heatmap '{hm.title}' has string row, expected dict: {row}"
            assert "row" in row and "column" in row, f"Heatmap row missing row/column keys: {row}"
        # Must have x/y keys for frontend normalizer
        assert hm.content.get("x") == "column", f"Heatmap '{hm.title}' missing x='column'"
        assert hm.content.get("y") == "row", f"Heatmap '{hm.title}' missing y='row'"


def test_dataset_role_no_duplication() -> None:
    """Dataset role summary must not produce 'transactional transactional dataset'."""
    store = InvestigationStore()
    sales = _add_source(store, "Transactional dataset", _transaction_df())
    marketing = _add_source(store, "Marketing dataset", _marketing_df())
    investigation = store.create_investigation("compare both datasets")
    for dataset_id in [sales, marketing]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    registry = _registry(store, investigation.investigation_id, [sales, marketing])
    relationship = analyze_dataset_relationships(registry)
    summary = relationship["summary"]

    # No word should appear twice consecutively (case-insensitive)
    words = summary.lower().split()
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and words[i] not in {"a", "the", "and", "or", "is"}:
            assert False, f"Duplicate consecutive word '{words[i]}' in summary: {summary}"


def test_cross_dataset_findings_appear_in_investigation() -> None:
    """Cross-dataset analysis must produce actual Finding objects, not an empty findings panel."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("can these datasets be joined reliably?")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    assert run.metadata["dataset_resolution"]["scope"] == "cross_dataset"
    assert len(updated.findings) >= 1, f"Expected findings, got {len(updated.findings)}: findings panel is empty"


def test_cross_dataset_findings_have_high_confidence() -> None:
    """Cross-dataset structural findings must have high confidence (>= 0.75)."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("compare both datasets")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    high_conf = [f for f in updated.findings if (f.confidence or 0) >= 0.75]
    assert len(high_conf) >= 1, f"Expected at least 1 high-confidence finding, got {len(high_conf)} out of {len(updated.findings)}"


def test_cross_dataset_analysis_type_is_set() -> None:
    """Cross-dataset trace_metadata must have analysis_type='cross_dataset', not fallback to 'overview'."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("compare both datasets")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    trace = updated.report.metadata.get("trace_metadata") or {} if updated.report else {}

    assert trace.get("analysis_type") == "cross_dataset", f"Expected analysis_type='cross_dataset', got '{trace.get('analysis_type')}'"


def test_findings_contain_structural_insight() -> None:
    """At least one finding must mention structural content (joinability, identifier, bridge, etc.)."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("can these datasets be joined reliably?")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    structural_markers = ("joinab", "identifier", "bridge", "shared", "entity", "overlap", "reliably", "warehouse")
    all_text = " ".join(f.text.lower() for f in updated.findings)
    assert any(m in all_text for m in structural_markers), f"No structural finding found in: {all_text[:200]}"


def test_no_duplicate_bridge_findings() -> None:
    """Repeated cross-dataset queries must not create duplicate findings."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("compare both datasets")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    _run_service(store).run_investigation(investigation.investigation_id)
    _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    # Check for exact text duplicates
    seen: set[str] = set()
    duplicates = 0
    for f in updated.findings:
        key = f.text.lower().strip()
        if key in seen:
            duplicates += 1
        seen.add(key)
    # Some duplication is expected from running twice, but bridge spam should not multiply
    assert duplicates <= len(updated.findings) // 2, f"Too many duplicate findings: {duplicates} out of {len(updated.findings)}"


def test_repeated_cross_dataset_queries_reduce_duplication() -> None:
    """Second cross-dataset run should produce fewer duplicate findings due to dedup against prior findings."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("can these datasets be joined reliably?")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    # First run
    _run_service(store).run_investigation(investigation.investigation_id)
    first_count = len(store.get_investigation(investigation.investigation_id).findings)

    # Second run — should deduplicate against first run's findings
    _run_service(store).run_investigation(investigation.investigation_id)
    total_count = len(store.get_investigation(investigation.investigation_id).findings)
    second_added = total_count - first_count

    # Second run should add fewer findings than first run (dedup working)
    assert second_added < first_count, f"Second run added {second_added} findings (same as first run {first_count}); dedup should reduce this"


def test_structural_findings_evolve_over_queries() -> None:
    """Different cross-dataset questions should produce different findings, not identical ones."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("can these datasets be joined reliably?")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    _run_service(store).run_investigation(investigation.investigation_id)
    first_findings = {f.text for f in store.get_investigation(investigation.investigation_id).findings}

    # Ask a warehouse question
    msg = store.add_investigation_message(InvestigationMessage(
        investigation_id=investigation.investigation_id,
        content="If you had to design a warehouse from these datasets, what shared entities would you use?",
    ))
    _run_service(store).run_investigation(investigation.investigation_id, message_id=msg.message_id)
    all_findings = {f.text for f in store.get_investigation(investigation.investigation_id).findings}
    new_findings = all_findings - first_findings

    # Warehouse question should produce at least 1 new finding not in first run
    assert len(new_findings) >= 1, f"Warehouse question produced 0 new findings; evolution is not working"


def test_findings_no_grounding_leakage() -> None:
    """No finding should contain 'Grounding:' text."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("compare both datasets")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    for f in updated.findings:
        assert "grounding:" not in f.text.lower(), f"Finding contains 'Grounding:' leakage: {f.text[:100]}"


def test_findings_no_main_metric_hallucination() -> None:
    """Cross-dataset structural findings must not mention 'the main metric' or 'primary metric'."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("compare both datasets")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    for f in updated.findings:
        assert "the main metric" not in f.text.lower(), f"Finding hallucinates 'the main metric': {f.text[:100]}"
        assert "primary metric" not in f.text.lower(), f"Finding hallucinates 'primary metric': {f.text[:100]}"


def test_findings_are_concise() -> None:
    """Each cross-dataset finding should be concise (≤ 300 characters)."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("compare both datasets")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    for f in updated.findings:
        assert len(f.text) <= 300, f"Finding too long ({len(f.text)} chars): {f.text[:100]}..."


def test_semantic_deduplication_adapter() -> None:
    """Adapter dedup should remove near-identical findings within a single run output."""
    from source.product.adapter import _deduplicate_finding_texts

    findings = [
        "These datasets are not reliably joinable — no stable shared identifier was detected.",
        "These datasets are not reliably joinable — no stable shared identifier exists.",
        "The strongest cross-dataset bridge is geography.",
        "The datasets support executive-level comparison through conceptual overlap.",
    ]
    deduped = _deduplicate_finding_texts(findings)
    # First two are near-identical, should be collapsed to 1
    assert len(deduped) < len(findings), f"Dedup did not reduce findings: {len(deduped)} vs {len(findings)}"
    assert len(deduped) >= 3, f"Over-deduplication: {len(deduped)} findings remaining"


def test_cross_dataset_executive_summary_synthesizes() -> None:
    """Executive summary across datasets should not repeat bridge lists verbatim."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("summarize the relationship between these datasets for executive review")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    # The report should exist and have findings
    assert updated.report is not None, "No report generated"
    assert len(updated.findings) >= 1, "No findings generated"
    # No finding should just be a list of bridges with no analytical content
    for f in updated.findings:
        assert "bridge:" not in f.text.lower() or len(f.text) > 50, f"Finding is just a bridge list: {f.text[:100]}"


def test_final_multi_dataset_time_series_capability_keeps_all_branches() -> None:
    store, investigation, dataset_ids = _setup_three_domain_workspace(
        "Which datasets support meaningful time-series analysis, and what trends could each analyze?"
    )

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    trace = updated.report.metadata["trace_metadata"]

    assert run.metadata["dataset_resolution"]["scope"] == "cross_dataset"
    assert len(trace["dataset_execution_scopes"]) == 3
    assert {scope["dataset_id"] for scope in trace["dataset_execution_scopes"]} == set(dataset_ids)
    assert "Which dataset should I use?" not in updated.report.summary
    assert any("branch-scoped" in item.lower() or "capability ranking" in item.lower() for item in updated.report.key_findings)


def test_final_multi_dataset_parallel_visual_analysis_creates_dataset_scoped_charts() -> None:
    store, investigation, dataset_ids = _setup_three_domain_workspace(
        "Build separate visual analyses for: releases over time, sales by segment, prevalence by age group."
    )

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    charts = [artifact for artifact in updated.artifacts if artifact.run_id == run.run_id and artifact.artifact_type == ArtifactType.CHART]
    scoped_chart_dataset_ids = {artifact.metadata.get("dataset_id") for artifact in charts if artifact.metadata.get("dataset_id")}

    assert run.metadata["dataset_resolution"]["scope"] == "cross_dataset"
    assert len(run.metadata["dataset_execution_scopes"]) == 3
    assert len(charts) >= 3
    assert set(dataset_ids).issubset(scoped_chart_dataset_ids)
    assert all(artifact.metadata.get("dataset_scope") == "single_dataset" for artifact in charts if artifact.metadata.get("dataset_id"))


def test_final_multi_dataset_diversity_and_executive_synthesis_are_branch_scoped() -> None:
    store, investigation, dataset_ids = _setup_three_domain_workspace(
        "Compare diversity patterns across all datasets: movie genres, customer/product structure, and patient health indicators."
    )

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    trace = updated.report.metadata["trace_metadata"]

    assert trace["operation"] == "CROSS_DATASET_DIVERSITY_ANALYSIS"
    assert len(trace["dataset_execution_scopes"]) == 3
    assert {scope["dataset_id"] for scope in trace["dataset_execution_scopes"]} == set(dataset_ids)
    assert "without merging unrelated entities" in updated.report.summary


def test_final_multi_dataset_anomaly_capability_and_limitations_use_all_datasets() -> None:
    store, investigation, dataset_ids = _setup_three_domain_workspace(
        "Which datasets are best suited for anomaly detection, and why?"
    )

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    trace = updated.report.metadata["trace_metadata"]

    assert trace["operation"] == "DATASET_CAPABILITY_REASONING"
    assert len(trace["dataset_execution_scopes"]) == 3
    assert set(dataset_ids) == {scope["dataset_id"] for scope in trace["dataset_execution_scopes"]}
    assert "Capability ranking" in updated.report.summary or "strongest for" in updated.report.summary
    assert run.metadata["multi_dataset_lifecycle_trace"][-1]["branch_count"] == 3


def test_final_time_series_synthesis_has_no_joinability_and_age_is_ordinal() -> None:
    store, investigation, _ = _setup_three_domain_workspace(
        "Which datasets support meaningful time-series analysis, and what trends could each analyze?"
    )

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    answer = updated.report.summary.lower()

    assert run.metadata["dataset_resolution"]["scope"] == "cross_dataset"
    assert "join" not in answer
    assert "shared identifier" not in answer
    assert "age" in answer
    assert "not time-series" in answer or "not longitudinal forecasting" in answer


def test_final_dashboard_kpi_synthesis_returns_three_kpis_per_dataset() -> None:
    store, investigation, dataset_ids = _setup_three_domain_workspace(
        "If you were building executive dashboards for each dataset, what would be the 3 most important KPIs?"
    )

    run = _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)
    answer = updated.report.summary

    assert run.metadata["dataset_resolution"]["scope"] == "cross_dataset"
    assert len(run.metadata["dataset_execution_scopes"]) == len(dataset_ids)
    assert answer.count("1.") == len(dataset_ids)
    assert answer.count("2.") == len(dataset_ids)
    assert answer.count("3.") == len(dataset_ids)
    assert "join" not in answer.lower()


def test_final_behavior_comparison_is_not_random_metric_average() -> None:
    store, investigation, _ = _setup_three_domain_workspace(
        "Compare how customer behavior, viewer behavior, and patient behavior differ across the three datasets."
    )

    _run_service(store).run_investigation(investigation.investigation_id)
    answer = store.get_investigation(investigation.investigation_id).report.summary.lower()

    assert "different analytical lenses" in answer
    assert "higher average" not in answer
    assert "join" not in answer


def test_dataset_role_classification_no_pm_leakage() -> None:
    """Dataset role classification must classify roles, not trigger PM prioritization."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("Which dataset contains customer demographics and which tracks transactions?")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    # Should have a report with role-classification content
    assert updated.report is not None
    summary_lower = (updated.report.summary or "").lower()
    # Must NOT contain PM prioritization / concentration language
    assert "concentration" not in summary_lower, f"PM leakage: {updated.report.summary[:150]}"
    assert "highest decision leverage" not in summary_lower, f"PM leakage: {updated.report.summary[:150]}"
    assert "the main metric" not in summary_lower, f"Metric hallucination: {updated.report.summary[:150]}"
    # Should mention at least one dataset name
    assert "sales" in summary_lower or "salary" in summary_lower, f"Role classification missing dataset names: {updated.report.summary[:150]}"


def test_revenue_metric_alignment_direct_proxy_absent() -> None:
    """Revenue metric comparison must classify fields as direct/proxy/absent."""
    from source.product.dataset_registry import _semantic_metric_alignment, InvestigationDatasetEntry

    entries = [
        InvestigationDatasetEntry(
            dataset_id="ds1", display_name="Sales", source_name="sales.csv",
            schema={}, profile={}, row_count=100,
            column_names=["Sales", "Revenue", "City", "Category"],
            semantic_roles={"Sales": "metric", "Revenue": "metric", "City": "dimension", "Category": "dimension"},
            semantic_profile={"roles": ["transactional"], "concepts": ["monetary_value"]},
            sample_values={}, created_at="", executable_available=True, runtime_reference="",
        ),
        InvestigationDatasetEntry(
            dataset_id="ds2", display_name="Demographics", source_name="demo.csv",
            schema={}, profile={}, row_count=50,
            column_names=["Age", "Education", "Satisfaction_Score"],
            semantic_roles={"Age": "dimension", "Education": "dimension", "Satisfaction_Score": "metric"},
            semantic_profile={"roles": ["demographic"], "concepts": ["education_level"]},
            sample_values={}, created_at="", executable_available=True, runtime_reference="",
        ),
    ]
    result = _semantic_metric_alignment(entries, {}, "compare revenue-related metrics")
    assert result is not None
    assert len(result["rows"]) == 2
    # Sales has direct financial metrics
    sales_row = next(r for r in result["rows"] if r["dataset"] == "Sales")
    assert sales_row["classification"] == "direct_revenue"
    # Demographics has no direct financial metrics
    demo_row = next(r for r in result["rows"] if r["dataset"] == "Demographics")
    assert demo_row["classification"] in ("proxy", "absent")
    # Summary should mention partial comparison
    assert "partial" in result["summary"].lower() or "normalization" in result["summary"].lower()


def test_visual_findings_generated_from_entity_coverage() -> None:
    """Heatmap/entity matrix analysis must generate analytical findings."""
    from source.product.dataset_registry import _findings_from_visuals, InvestigationDatasetEntry

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

    # Should produce findings about missing entities and compatibility
    assert len(findings) >= 1, f"No visual findings generated"
    all_text = " ".join(findings).lower()
    # Customer entity missing from Products should be flagged
    assert "customer" in all_text or "financial" in all_text or "schema overlap" in all_text or "granularity" in all_text


def test_finding_dedup_silently_removes_duplicates() -> None:
    """When findings are duplicates of prior, dedup removes them silently — no middleware phrases."""
    from source.product.dataset_registry import _deduplicate_structural_findings

    findings = [
        "These datasets are not reliably joinable.",
        "The strongest bridge is geography.",
        "No stable shared identifier exists.",
    ]
    # Prior findings that are near-exact matches (high Jaccard overlap)
    prior = [
        "These datasets are not reliably joinable.",
        "The strongest bridge is geography between datasets.",
        "No stable shared identifier exists.",
    ]
    result = _deduplicate_structural_findings(findings, prior)
    # At least 2 of the 3 findings should be suppressed as duplicates
    assert len(result) <= 2
    # No visible middleware phrases
    combined = " ".join(result).lower()
    assert "previously established" not in combined
    assert "already been established" not in combined
    assert "building on" not in combined


def test_structural_findings_no_hallucination() -> None:
    """Structural investigation findings must not contain metric hallucination phrases."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("What shared business entities exist across these datasets?")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    for f in updated.findings:
        text = f.text.lower()
        assert "highest decision leverage" not in text, f"Hallucination: {f.text[:100]}"
        assert "secondary observation" not in text, f"Hallucination: {f.text[:100]}"
        assert "leading segment" not in text or "leading" not in text, f"PM leakage: {f.text[:100]}"


def test_findings_concise_with_visual_findings() -> None:
    """All findings including visual-derived ones should be concise."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("compare both datasets")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    for f in updated.findings:
        assert len(f.text) <= 300, f"Finding too long ({len(f.text)} chars): {f.text[:100]}..."


def test_explicit_field_search_across_datasets_column_match_wins() -> None:
    """When user explicitly mentions a field, the dataset containing it must be selected,
    even if another dataset has active branch lineage."""
    from source.product.dataset_registry import InvestigationDatasetEntry, resolve_dataset_scope

    # Dataset A: has Sales field
    entry_a = InvestigationDatasetEntry(
        dataset_id="ds_sales", display_name="Sales Report", source_name="sales.csv",
        schema={}, profile={}, row_count=100,
        column_names=["Sales", "City", "Customer"],
        semantic_roles={"Sales": "metric", "City": "dimension"},
        semantic_profile={"concepts": ["monetary_value", "geography"]},
        sample_values={"City": ["London", "Paris", "Berlin"]},
        created_at="", executable_available=True, runtime_reference="",
    )
    # Dataset B: does NOT have Sales
    entry_b = InvestigationDatasetEntry(
        dataset_id="ds_salary", display_name="Salary Dataset", source_name="salary.csv",
        schema={}, profile={}, row_count=100,
        column_names=["Salary_LPA", "Company", "City"],
        semantic_roles={"Salary_LPA": "metric", "Company": "dimension"},
        semantic_profile={"concepts": ["education_level"]},
        sample_values={"City": ["London", "Rome"]},
        created_at="", executable_available=True, runtime_reference="",
    )

    # Active branch lineage points to salary dataset — but query mentions "Sales" explicitly
    result = resolve_dataset_scope(
        question="Build histogram of Sales in Tokyo",
        registry=[entry_a, entry_b],
        conversation_context={"conversation_state": {"dataset_id": "ds_salary"}},
    )
    # Must select the dataset containing Sales, not the active lineage dataset
    assert "ds_sales" in result.selected_dataset_ids, (
        f"Expected ds_sales, got {result.selected_dataset_ids}. "
        f"Explicit column match must outweigh active branch lineage."
    )


def test_no_visible_memory_middleware_in_repeated_queries() -> None:
    """Repeated cross-dataset queries must not produce visible memory middleware phrases."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("Can these datasets be joined reliably?")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    # Run twice to trigger dedup
    _run_service(store).run_investigation(investigation.investigation_id)
    _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    summary = (updated.report.summary or "").lower() if updated.report else ""
    all_findings = " ".join(f.text.lower() for f in updated.findings)

    forbidden = [
        "previously established conclusions",
        "already been established",
        "building on",
        "the key conclusions remain unchanged",
    ]
    for phrase in forbidden:
        assert phrase not in summary, f"Middleware leakage in summary: {phrase}"
        assert phrase not in all_findings, f"Middleware leakage in findings: {phrase}"


def test_memory_evolves_instead_of_suppressing() -> None:
    """When all prior findings are duplicated, the system must still answer meaningfully."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("Can these datasets be joined reliably?")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    # Run first to establish findings
    _run_service(store).run_investigation(investigation.investigation_id)
    # Run second time — must still produce a meaningful response
    _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    summary = (updated.report.summary or "").lower() if updated.report else ""
    # Must contain meaningful content, not just a placeholder
    assert len(summary) > 30, f"Summary too short (suppressed?): {summary}"
    assert "dataset" in summary or "join" in summary or "reliab" in summary, (
        f"Summary does not address the question: {summary[:150]}"
    )


def test_cross_dataset_insights_synthesis_not_middleware() -> None:
    """'3 most important cross-dataset insights' must return real insights, not middleware."""
    store = InvestigationStore()
    sales = _add_source(store, "Sales dataset", _sales_df())
    salary = _add_source(store, "Salary dataset", _salary_df())
    investigation = store.create_investigation("What are the 3 most important cross-dataset insights?")
    for dataset_id in [sales, salary]:
        store.link_data_source_to_investigation(investigation.investigation_id, dataset_id)

    _run_service(store).run_investigation(investigation.investigation_id)
    updated = store.get_investigation(investigation.investigation_id)

    assert updated.report is not None
    summary = (updated.report.summary or "").lower()
    # Must contain substantive analytical content
    assert len(summary) > 50, f"Summary too short: {summary}"
    assert "the structural relationship has already been established" not in summary
    assert "building on" not in summary
    # Should mention datasets or analytical concepts
    assert "dataset" in summary or "sales" in summary or "salary" in summary or "overlap" in summary or "join" in summary
