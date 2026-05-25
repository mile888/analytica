from __future__ import annotations

import re

import pandas as pd

from source.product.cross_dataset_synthesis import synthesize_cross_dataset_branches
from source.product.fallback_analysis import deterministic_investigation_fallback
from source.product.run_service import InvestigationRunService
from source.product.store import InvestigationMessage, InvestigationStore


def _health_df() -> pd.DataFrame:
    return pd.DataFrame({
        "PersonID": [f"P-{i:03d}" for i in range(16)],
        "Age": [25, 31, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 34, 45, 58, 69],
        "Group": ["A", "A", "B", "B", "A", "A", "B", "B", "A", "A", "B", "B", "A", "B", "A", "B"],
        "ConditionAlpha": [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        "IndicatorBeta": [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        "BehaviorGamma": [0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        "AccessCostFlag": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "OutcomeFlag": [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        "Score": [20, 22, 23, 24, 26, 27, 28, 29, 31, 32, 34, 36, 21, 25, 30, 35],
    })


def test_condition_frequency_ranks_binary_indicators() -> None:
    result = deterministic_investigation_fallback("Which health conditions appear most common across the population?", _health_df())
    assert result["trace_metadata"]["analysis_type"] == "binary_indicator_prevalence"
    answer = result["final_answer"]
    assert "ConditionAlpha" in answer
    assert "AccessCostFlag" not in answer
    assert "NoDocbcCost" not in answer


def test_public_health_risk_summary_uses_multiple_evidence_sources() -> None:
    result = deterministic_investigation_fallback("Summarize the main public health risks revealed by this dataset.", _health_df())
    assert result["trace_metadata"]["analysis_type"] == "strategic_synthesis"
    findings = result["structured_report"]["key_findings"]
    assert len(findings) >= 3
    combined = " ".join(findings).lower()
    assert "prevalence" in combined
    assert "association" in combined or "gradient" in combined


def test_outcome_prevalence_by_behavior_factor() -> None:
    result = deterministic_investigation_fallback("Compare outcome prevalence between BehaviorGamma groups.", _health_df())
    assert result["trace_metadata"]["analysis_type"] == "semantic_role_prevalence"
    assert "OutcomeFlag" in result["final_answer"]
    assert "BehaviorGamma" in result["generated_code"] or "BehaviorGamma" in str(result["artifacts"])


def test_age_group_risk_computes_ordered_prevalence() -> None:
    result = deterministic_investigation_fallback("Which age groups show the strongest increase in outcome risk?", _health_df())
    assert result["trace_metadata"]["analysis_type"] == "ordered_group_prevalence"
    assert result["trace_metadata"]["grouping_variable"] == "Age"
    assert "strongest adjacent increase" in result["final_answer"].lower()


def test_indicators_increasing_with_age_rank_gradients() -> None:
    result = deterministic_investigation_fallback("Which indicators become more common among older patients?", _health_df())
    assert result["trace_metadata"]["analysis_type"] == "ordered_indicator_gradient"
    assert result["trace_metadata"]["grouping_variable"] == "Age"
    assert "IndicatorBeta" in result["final_answer"] or "ConditionAlpha" in result["final_answer"]


def test_lifestyle_factor_diversity_no_entertainment_leak() -> None:
    result = deterministic_investigation_fallback("Compare lifestyle-factor diversity across different patient groups.", _health_df())
    assert result["trace_metadata"]["analysis_type"] == "factor_diversity_analysis"
    assert result["trace_metadata"]["dimension"] == "Group"
    assert re.search(r"\bactor\b", result["final_answer"].lower()) is None
    assert re.search(r"\bentertainment\b", result["final_answer"].lower()) is None


def test_prevalence_visualization_creates_chart_artifact() -> None:
    result = deterministic_investigation_fallback("Build a visualization of outcome prevalence by age group and Group.", _health_df())
    assert any(a.get("artifact_type") == "chart" for a in result["artifacts"])
    assert result["trace_metadata"]["analysis_type"] == "semantic_role_prevalence"


def test_policymaker_findings_return_three_evidence_backed_findings() -> None:
    result = deterministic_investigation_fallback("If you were presenting this dataset to policymakers, what would be the 3 most important findings?", _health_df())
    assert result["trace_metadata"]["analysis_type"] == "strategic_synthesis"
    assert len(result["structured_report"]["key_findings"]) >= 3
    assert "mean =" not in result["final_answer"].lower()


def test_churn_risk_by_tenure_band() -> None:
    df = pd.DataFrame({"TenureBand": [1, 1, 2, 2, 3, 3, 4, 4], "Segment": ["S"] * 8, "ChurnFlag": [0, 0, 0, 1, 1, 1, 1, 1]})
    result = deterministic_investigation_fallback("Which tenure bands show the strongest increase in churn risk?", df)
    assert result["trace_metadata"]["analysis_type"] == "ordered_group_prevalence"
    assert result["trace_metadata"]["target_variable"] == "ChurnFlag"


def test_fraud_indicator_prevalence() -> None:
    df = pd.DataFrame({"ClaimID": ["a", "b", "c", "d"], "FraudFlag": [1, 0, 1, 1], "ManualReviewFlag": [0, 1, 1, 0]})
    result = deterministic_investigation_fallback("Which fraud flags are most prevalent?", df)
    assert result["trace_metadata"]["analysis_type"] == "binary_indicator_prevalence"
    assert "FraudFlag" in result["final_answer"]


def test_student_failure_risk_by_grade_level() -> None:
    df = pd.DataFrame({"GradeLevel": [1, 1, 2, 2, 3, 3, 4, 4], "FailureFlag": [0, 0, 0, 1, 1, 1, 1, 1]})
    result = deterministic_investigation_fallback("Which grade levels show the strongest increase in failure risk?", df)
    assert result["trace_metadata"]["analysis_type"] == "ordered_group_prevalence"
    assert result["trace_metadata"]["grouping_variable"] == "GradeLevel"


def test_species_prevalence_by_elevation_band() -> None:
    df = pd.DataFrame({"ElevationBand": [100, 200, 300, 400, 500, 600], "SpeciesPresentFlag": [0, 0, 1, 1, 1, 1]})
    result = deterministic_investigation_fallback("Which elevation bands show the strongest increase in species prevalence?", df)
    assert result["trace_metadata"]["analysis_type"] == "ordered_group_prevalence"
    assert result["trace_metadata"]["grouping_variable"] == "ElevationBand"


def test_cross_dataset_synthesis_still_passes() -> None:
    output = synthesize_cross_dataset_branches(
        question="Compare risk patterns across datasets.",
        operation="CROSS_DATASET_SYNTHESIS",
        branches=[
            {"dataset_id": "a", "dataset_name": "A", "semantic_profile": {"concepts": ["risk"], "metric_columns": ["RiskFlag"], "dimension_columns": ["Segment"]}, "computed_results": [], "findings": ["A finding"], "artifacts": []},
            {"dataset_id": "b", "dataset_name": "B", "semantic_profile": {"concepts": ["risk"], "metric_columns": ["FailureFlag"], "dimension_columns": ["Band"]}, "computed_results": [], "findings": ["B finding"], "artifacts": []},
        ],
    )
    assert output["answer"]


def test_branch_execution_and_dataset_scoped_artifacts_smoke() -> None:
    store = InvestigationStore()
    service = InvestigationRunService(store)
    investigation = store.create_investigation("Smoke")
    message = store.add_investigation_message(InvestigationMessage(investigation_id=investigation.investigation_id, content="Build a visualization of outcome prevalence by Group."))
    service.run_investigation(investigation.investigation_id, df=_health_df(), message_id=message.message_id)
    updated = store.get_investigation(investigation.investigation_id)
    assert updated.report is not None
    assert updated.report.artifact_ids
