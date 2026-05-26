from __future__ import annotations

from source.product.cross_dataset_synthesis import (
    critic_warnings_for_cross_dataset_output,
    synthesize_cross_dataset_branches,
)

def _branch(
    name: str,
    *,
    concepts: list[str],
    metrics: list[str],
    dimensions: list[str],
    timestamps: list[str] | None = None,
) -> dict:
    roles = {field: "metric" for field in metrics}
    roles.update({field: "dimension" for field in dimensions})
    roles.update({field: "timestamp" for field in timestamps or []})
    return {
        "dataset_id": name.lower().replace(" ", "_"),
        "dataset_name": name,
        "operation": "CROSS_DATASET_SYNTHESIS",
        "compatibility_score": 0.8,
        "semantic_roles": roles,
        "semantic_profile": {
            "concepts": concepts,
            "metric_columns": metrics,
            "dimension_columns": dimensions,
            "timestamp_columns": timestamps or [],
        },
        "columns": [*metrics, *dimensions, *(timestamps or [])],
        "computed_results": [{"analysis_type": "metric_summary", "n": 100}],
        "findings": [],
        "limitations": [],
        "artifacts": [],
    }

def test_kpi_synthesis_returns_three_kpis_per_dataset() -> None:
    output = synthesize_cross_dataset_branches(
        question="What would be the 3 most important KPIs for each dataset?",
        operation="CROSS_DATASET_KPI_SYNTHESIS",
        branches=[
            _branch("Commercial", concepts=["monetary_value", "product_or_category"], metrics=["Revenue"], dimensions=["Segment"], timestamps=["OrderDate"]),
            _branch("Risk", concepts=["age_or_birth"], metrics=["OutcomeFlag"], dimensions=["AgeBand"], timestamps=[]),
        ],
    )

    answer = output["answer"]
    assert "`Commercial`: 1." in answer
    assert "`Risk`: 1." in answer
    assert answer.count("1.") == 2
    assert answer.count("2.") == 2
    assert answer.count("3.") == 2

def test_critic_rejects_joinability_pollution_for_capability_question() -> None:
    warnings = critic_warnings_for_cross_dataset_output(
        question="Which dataset is best suited for forecasting?",
        operation="DATASET_CAPABILITY_REASONING",
        answer="These datasets are not reliably joinable.",
        branches=[_branch("Events", concepts=["time"], metrics=["Value"], dimensions=[], timestamps=["Date"])],
    )

    assert warnings
