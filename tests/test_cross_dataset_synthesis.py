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


def test_time_series_capability_does_not_treat_age_as_time() -> None:
    output = synthesize_cross_dataset_branches(
        question="Which datasets support meaningful time-series analysis?",
        operation="DATASET_CAPABILITY_REASONING",
        branches=[
            _branch("Events", concepts=["time"], metrics=["Value"], dimensions=["Segment"], timestamps=["EventDate"]),
            _branch("Profiles", concepts=["age_or_birth"], metrics=["RiskFlag", "Age"], dimensions=["Group"], timestamps=[]),
        ],
    )

    answer = output["answer"].lower()
    assert "eventdate" in answer
    assert "age" in answer
    assert "not longitudinal forecasting" in answer
    assert "join" not in answer


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


def test_behavior_comparison_is_conceptual_not_random_average() -> None:
    output = synthesize_cross_dataset_branches(
        question="Compare customer behavior, viewer behavior, and patient behavior across datasets.",
        operation="CROSS_DATASET_SYNTHESIS",
        branches=[
            _branch("Purchases", concepts=["customer_entity", "purchase_volume", "monetary_value"], metrics=["Amount"], dimensions=["Segment"]),
            _branch("Catalog", concepts=["survey_or_rating", "geography", "time"], metrics=[], dimensions=["Genre", "Country"], timestamps=["ReleaseYear"]),
            _branch("Health", concepts=["age_or_birth"], metrics=["RiskIndicator"], dimensions=["ActivityLevel"]),
        ],
    )

    answer = output["answer"].lower()
    assert "different analytical lenses" in answer
    assert "average" not in answer
    assert "join" not in answer


def test_diversity_synthesis_does_not_require_numeric_metric() -> None:
    output = synthesize_cross_dataset_branches(
        question="Compare diversity patterns across datasets.",
        operation="CROSS_DATASET_DIVERSITY_ANALYSIS",
        branches=[
            _branch("Catalog", concepts=["geography"], metrics=[], dimensions=["Genre", "Country"]),
            _branch("Operations", concepts=["product_or_category"], metrics=["Amount"], dimensions=["Segment", "Category"]),
        ],
    )

    answer = output["answer"].lower()
    assert "does not require a shared numeric metric" in answer
    assert not critic_warnings_for_cross_dataset_output(
        question="Compare diversity patterns across datasets.",
        operation="CROSS_DATASET_DIVERSITY_ANALYSIS",
        answer=output["answer"],
        branches=[],
    )


def test_critic_rejects_joinability_pollution_for_capability_question() -> None:
    warnings = critic_warnings_for_cross_dataset_output(
        question="Which dataset is best suited for forecasting?",
        operation="DATASET_CAPABILITY_REASONING",
        answer="These datasets are not reliably joinable.",
        branches=[_branch("Events", concepts=["time"], metrics=["Value"], dimensions=[], timestamps=["Date"])],
    )

    assert warnings
