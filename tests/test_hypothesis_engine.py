from __future__ import annotations

from source.product.hypothesis_engine import build_hypothesis_branches, hypothesis_metadata
from source.product.insight_quality import build_insight_metadata


def test_grouped_comparison_generates_driver_hypothesis() -> None:
    metadata = build_insight_metadata(
        "`Revenue` differs sharply across `Region`.",
        evidence=["Grouped `Revenue` by `Region` over 100 rows."],
        limitations=["Small groups and extreme records can distort rankings."],
        next_steps=["Compare average and total `Revenue` side by side."],
        analysis_context={"analysis_type": "grouped_metric", "metric": "Revenue", "dimension": "Region"},
    )

    assert metadata["hypotheses"]
    assert "record volume" in metadata["possible_drivers"]
    assert "sample-size" in metadata["uncertainty_notes"][0] or "Small groups" in metadata["uncertainty_notes"][0]
    assert any("Does record volume explain" in item for item in metadata["validation_questions"])


def test_outlier_branch_includes_uncertainty_reasoning() -> None:
    metadata = build_insight_metadata(
        "`Segment A` contains the strongest outlier candidates.",
        evidence=["Flagged potential outliers with the IQR rule."],
        limitations=["Groups with very small sample sizes can look unusual because of one or two observations."],
        analysis_context={"fallback": "group_unusual_values_check", "metric": "Amount", "dimension": "Segment"},
    )

    assert metadata["analysis_type"] == "outlier"
    assert "small sample" in " ".join(metadata["uncertainty_notes"]).lower()
    assert any("raw extreme records" in item.lower() or "mean vs median" in item.lower() for item in metadata["validation_questions"])


def test_correlation_branch_warns_about_shared_drivers_not_causality() -> None:
    metadata = build_insight_metadata(
        "`Rating` has the strongest positive relationship with `Revenue`.",
        evidence=["Computed pairwise Pearson correlations."],
        limitations=["Correlation describes association, not causality; missing values and outliers can affect the result."],
        analysis_context={"fallback": "correlation_check", "metric": "Revenue"},
    )

    text = " ".join(metadata["hypotheses"] + metadata["uncertainty_notes"])
    assert "shared driver" in text
    assert "does not imply causality" in text
    assert "causes" not in text.lower()


def test_trend_branch_generates_seasonality_or_sparsity_hypothesis() -> None:
    branches = build_hypothesis_branches(
        conclusion="`Revenue` increased over time.",
        analysis_type="trend",
        evidence=["Aggregated `Revenue` by monthly periods."],
        limitations=["Sparse periods can exaggerate movement."],
        metric="Revenue",
        time_axis="Order Date",
    )
    metadata = hypothesis_metadata(branches)

    joined = " ".join(metadata["hypotheses"] + metadata["possible_drivers"] + metadata["uncertainty_notes"])
    assert "seasonality" in joined
    assert "sparse" in joined
