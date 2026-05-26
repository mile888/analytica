from __future__ import annotations

from source.product.cross_investigation import (
    build_analytical_playbooks,
    build_reusable_branches,
    extract_cross_investigation_patterns,
    extract_reusable_hypotheses,
    find_similar_investigations,
    pattern_aware_suggestions,
    report_pattern_hints,
)
from source.product.investigation import Artifact, ArtifactType, Finding, Investigation
from source.product.question_suggestions import build_cross_investigation_question_suggestions
from source.product.report_builder import build_shareable_report


def _grouped_investigation(title: str, metric: str, dimension: str) -> Investigation:
    investigation = Investigation(title=title, user_question=f"Compare {metric} by {dimension}")
    investigation.findings = [
        Finding(
            title=f"{metric} varies by {dimension}",
            text=f"`{metric}` differs sharply across `{dimension}`.",
            metadata={
                "analysis_type": "grouped_metric",
                "related_metrics": [metric],
                "supporting_dimensions": [dimension],
                "investigation_branches": [
                    {
                        "hypothesis": f"The `{metric}` gap across `{dimension}` may be driven by volume effects.",
                        "branch_type": "variance_explanation",
                        "next_questions": [f"Does record volume explain the `{metric}` gap across `{dimension}`?"],
                    }
                ],
            },
        )
    ]
    investigation.artifacts = [
        Artifact(
            artifact_type=ArtifactType.CHART,
            title=f"{metric} by {dimension}",
            content={"chart_type": "bar", "metric": metric, "x": dimension, "rows": []},
        )
    ]
    return investigation


def test_cross_investigation_pattern_extraction_is_semantic_role_based() -> None:
    first = _grouped_investigation("First", "revenue_amount", "region_label")
    second = _grouped_investigation("Second", "score_value", "category_label")

    patterns = extract_cross_investigation_patterns([first, second])
    grouped = [pattern for pattern in patterns if pattern.pattern_type == "grouped_metric"]

    assert grouped
    assert all("grouped_metric" in pattern.semantic_signature for pattern in grouped)
    assert any("Check sample size by group." in pattern.common_validation_steps for pattern in grouped)


def test_reusable_hypotheses_and_branches_are_extracted_from_findings() -> None:
    investigation = _grouped_investigation("Grouped", "metric_value", "segment_name")

    hypotheses = extract_reusable_hypotheses([investigation])
    patterns = extract_cross_investigation_patterns([investigation])
    branches = build_reusable_branches(patterns)

    assert hypotheses
    assert "volume effects" in hypotheses[0].hypothesis
    assert hypotheses[0].validation_questions
    assert branches
    assert branches[0].suggested_question


def test_investigation_similarity_uses_shared_analytical_patterns() -> None:
    target = _grouped_investigation("Target", "revenue_amount", "region_label")
    similar = _grouped_investigation("Similar", "sales_total", "city_name")
    different = Investigation(title="Quality", user_question="Check data quality")
    different.findings = [
        Finding(
            text="Missing values affect reliability.",
            metadata={"analysis_type": "data_quality"},
        )
    ]

    result = find_similar_investigations(target, [similar, different])

    assert result
    assert result[0].investigation_id == similar.investigation_id
    assert result[0].score > 0


def test_playbooks_and_suggestions_use_historical_patterns() -> None:
    investigation = _grouped_investigation("Grouped", "metric_value", "segment_name")
    patterns = extract_cross_investigation_patterns([investigation])
    playbooks = build_analytical_playbooks(patterns)
    suggestions = pattern_aware_suggestions(patterns, extract_reusable_hypotheses([investigation]))
    routed_suggestions = build_cross_investigation_question_suggestions(patterns)

    assert any(playbook.playbook_id == "metric_segmentation" for playbook in playbooks)
    assert any("volume" in suggestion.lower() for suggestion in suggestions)
    assert routed_suggestions


def test_report_builder_can_use_cross_investigation_validation_hints() -> None:
    investigation = _grouped_investigation("Report", "metric_value", "segment_name")
    patterns = extract_cross_investigation_patterns([investigation])

    report = build_shareable_report(investigation, analytical_patterns=patterns)
    content = "\n".join(section.content for section in report.sections)

    assert "Common validation checks" in [section.title for section in report.sections]
    assert "sample size" in content.lower() or "volume" in content.lower()
    assert report_pattern_hints(patterns)
