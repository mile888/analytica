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

