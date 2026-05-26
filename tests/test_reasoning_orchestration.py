"""Tests for high-level reasoning orchestration.

Covers: multi-factor association, strategic synthesis, category growth,
evidence isolation, chart enforcement, diversity analysis, and
domain vocabulary isolation.
"""

from __future__ import annotations

import pandas as pd
import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

def _healthcare_df() -> pd.DataFrame:
    """Minimal healthcare-like dataset with binary indicators."""
    return pd.DataFrame({
        "Age": [55, 42, 67, 30, 51, 60, 45, 38, 72, 48],
        "HighBP": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
        "HighChol": [1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
        "Smoker": [1, 0, 0, 0, 1, 1, 0, 0, 1, 0],
        "PhysActivity": [0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
        "BMI": [32.5, 24.1, 29.8, 22.0, 31.2, 28.5, 25.0, 23.4, 30.1, 26.7],
        "Outcome": [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        "Sex": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    })


def _entertainment_df() -> pd.DataFrame:
    """Minimal entertainment-like dataset with category + year."""
    return pd.DataFrame({
        "title": [f"Title {i}" for i in range(20)],
        "listed_in": [
            "Drama", "Comedy", "Action", "Drama", "Horror",
            "Comedy", "Action", "Drama", "Thriller", "Documentary",
            "Drama", "Comedy", "Action", "Sci-Fi", "Horror",
            "Comedy", "Action", "Drama", "Sci-Fi", "Documentary",
        ],
        "release_year": [
            2010, 2011, 2012, 2013, 2014,
            2015, 2016, 2017, 2018, 2019,
            2020, 2020, 2021, 2021, 2021,
            2022, 2022, 2022, 2022, 2023,
        ],
        "country": [
            "US", "UK", "India", "US", "UK",
            "India", "US", "UK", "India", "US",
            "UK", "India", "US", "UK", "India",
            "US", "UK", "India", "US", "UK",
        ],
        "type": ["Movie"] * 15 + ["TV Show"] * 5,
    })


# ── Multi-factor association ─────────────────────────────────────────────────

class TestMultiFactorAssociation:
    def test_detects_factor_question(self):
        from source.product.fallback_analysis import _is_multi_factor_question
        assert _is_multi_factor_question("Which factors are most associated with heart disease?")
        assert _is_multi_factor_question("What are the risk factors for churn?")
        assert _is_multi_factor_question("What drives the outcome variable?")
        assert not _is_multi_factor_question("What is the average BMI?")
        assert not _is_multi_factor_question("Show me the distribution of age")

    def test_multi_factor_produces_ranked_output(self):
        from source.product.fallback_analysis import _multi_factor_association_response
        df = _healthcare_df()
        result = _multi_factor_association_response("Which factors are most associated with Outcome?", df)
        assert result is not None
        assert result.get("final_answer")
        trace = result.get("trace_metadata", {})
        assert trace.get("analysis_type") == "multi_factor_association"
        findings = result.get("structured_report", {}).get("key_findings", [])
        assert len(findings) > 0
        combined = " ".join(findings).lower()
        assert "ranked" in combined or "factor" in combined

    def test_multi_factor_has_chart_artifact(self):
        from source.product.fallback_analysis import _multi_factor_association_response
        df = _healthcare_df()
        result = _multi_factor_association_response("Which factors are most associated with Outcome?", df)
        artifacts = result.get("artifacts", [])
        chart_artifacts = [a for a in artifacts if a.get("artifact_type") == "chart"]
        assert len(chart_artifacts) > 0, "Multi-factor analysis should produce chart artifact"

    def test_multi_factor_no_hardcoded_terms(self):
        from source.product.fallback_analysis import _multi_factor_association_response
        df = _healthcare_df()
        result = _multi_factor_association_response("Which factors are most associated with Outcome?", df)
        # Should not hardcode specific dataset terms
        import inspect
        source = inspect.getsource(_multi_factor_association_response)
        for term in ("HeartDiseaseorAttack", "Netflix", "Superstore", "BMI"):
            assert term not in source, f"Hardcoded term '{term}' found in source"


# ── Strategic synthesis ─────────────────────────────────────────────────────

class TestStrategicSynthesis:
    def test_produces_binary_prevalence(self):
        from source.product.fallback_analysis import _strategic_synthesis_response
        df = _healthcare_df()
        result = _strategic_synthesis_response("What are the 3 most important findings for policymakers?", df)
        findings = result.get("structured_report", {}).get("key_findings", [])
        # Should contain prevalence rates for binary indicators
        combined = " ".join(findings).lower()
        assert "prevalence" in combined or "indicator" in combined, \
            f"Strategic synthesis should compute binary indicator prevalence, got: {combined[:200]}"

    def test_detection_trigger(self):
        from source.product.fallback_analysis import _is_strategic_synthesis_question
        assert _is_strategic_synthesis_question("What are the most important insights for executives?")
        assert _is_strategic_synthesis_question("Summarize the platform strategy")
        assert not _is_strategic_synthesis_question("What is the average BMI?")


# ── Category growth ──────────────────────────────────────────────────────────

class TestCategoryGrowth:
    def test_detects_growth_question(self):
        from source.product.fallback_analysis import _is_growth_question
        assert _is_growth_question("What genres are growing fastest?")
        assert _is_growth_question("Which categories are fastest growing?")
        assert not _is_growth_question("What is the most common genre?")
        assert not _is_growth_question("Compare average durations by genre")

    def test_growth_analysis_output(self):
        from source.product.fallback_analysis import _category_growth_response
        df = _entertainment_df()
        result = _category_growth_response("What genres are growing fastest?", df, "listed_in", "release_year")
        assert result is not None
        trace = result.get("trace_metadata", {})
        assert trace.get("analysis_type") == "category_growth"
        findings = result.get("structured_report", {}).get("key_findings", [])
        assert len(findings) > 0
        assert "growing" in " ".join(findings).lower()


# ── Evidence isolation ───────────────────────────────────────────────────────

class TestEvidenceIsolation:
    def test_filters_irrelevant_findings(self):
        from source.product.run_service import _findings_relevant_to_question
        findings = [
            "The most common genre is Drama with 45% of all titles.",
            "Director Spielberg has the highest average rating of 8.2.",
            "Actor Tom Hanks appears in 12 titles.",
        ]
        # Healthcare question should filter out entertainment findings
        result = _findings_relevant_to_question(findings, "What is the prevalence of heart disease among smokers?")
        # None of these should match because they share no domain vocabulary
        assert len(result) <= 2  # Falls back to keeping at most 2

    def test_preserves_relevant_findings(self):
        from source.product.run_service import _findings_relevant_to_question
        findings = [
            "Heart disease prevalence is 42% among smokers.",
            "BMI above 30 is associated with higher risk.",
        ]
        result = _findings_relevant_to_question(findings, "What is the heart disease rate?")
        assert len(result) >= 1  # Should keep at least the first one

    def test_empty_question_returns_all(self):
        from source.product.run_service import _findings_relevant_to_question
        findings = ["Finding one.", "Finding two."]
        assert _findings_relevant_to_question(findings, "") == findings

    def test_empty_findings_returns_empty(self):
        from source.product.run_service import _findings_relevant_to_question
        assert _findings_relevant_to_question([], "Some question") == []


# ── Chart enforcement ────────────────────────────────────────────────────────

class TestChartEnforcement:
    def test_detects_visual_request(self):
        from source.product.fallback_analysis import _is_visual_request
        assert _is_visual_request("Show me a chart of sales by region")
        assert _is_visual_request("Visualize the distribution of age")
        assert _is_visual_request("Plot revenue over time")
        assert not _is_visual_request("What is the average revenue?")

    def test_injects_chart_when_missing(self):
        from source.product.fallback_analysis import _ensure_chart_artifact
        result = {
            "summary": "Test summary",
            "artifacts": [],
            "trace_metadata": {"metric": "revenue", "dimension": "region", "analysis_type": "grouped_metric"},
        }
        enforced = _ensure_chart_artifact(result, "Show me a chart of revenue by region")
        artifacts = enforced.get("artifacts", [])
        assert len(artifacts) == 1
        assert artifacts[0]["artifact_type"] == "chart"

    def test_preserves_existing_chart(self):
        from source.product.fallback_analysis import _ensure_chart_artifact
        existing_chart = {"artifact_type": "chart", "title": "Existing chart"}
        result = {
            "summary": "Test summary",
            "artifacts": [existing_chart],
            "trace_metadata": {"metric": "revenue", "dimension": "region"},
        }
        enforced = _ensure_chart_artifact(result, "Show me a chart")
        assert len(enforced.get("artifacts", [])) == 1  # Should not duplicate

    def test_adds_limitation_when_no_dimensions(self):
        from source.product.fallback_analysis import _ensure_chart_artifact
        result = {
            "summary": "Test summary",
            "artifacts": [],
            "trace_metadata": {},
            "limitations": [],
        }
        enforced = _ensure_chart_artifact(result, "Show me a chart")
        assert any("visualization" in l.lower() for l in enforced.get("limitations", []))


# ── Diversity analysis ───────────────────────────────────────────────────────

class TestDiversityAnalysis:
    def test_produces_chart_artifact(self):
        from source.product.fallback_analysis import _diversity_analysis_response
        df = _entertainment_df()
        result = _diversity_analysis_response("Compare regional content diversity", df, "country", "listed_in")
        artifacts = result.get("artifacts", [])
        chart_artifacts = [a for a in artifacts if isinstance(a, dict) and a.get("artifact_type") == "chart"]
        assert len(chart_artifacts) > 0, "Diversity analysis should produce chart artifact"

    def test_diversity_trace_metadata(self):
        from source.product.fallback_analysis import _diversity_analysis_response
        df = _entertainment_df()
        result = _diversity_analysis_response("Compare regional content diversity", df, "country", "listed_in")
        trace = result.get("trace_metadata", {})
        assert trace.get("analysis_type") == "diversity_analysis"


# ── Insight quality mappings ─────────────────────────────────────────────────

class TestInsightQualityMappings:
    def test_new_analysis_types_mapped(self):
        from source.product.insight_quality import _normalized_analysis_type
        assert _normalized_analysis_type("diversity_analysis") == "diversity_analysis"
        assert _normalized_analysis_type("category_mix_shift") == "category_mix_shift"
        assert _normalized_analysis_type("strategic_synthesis") == "strategic_synthesis"
        assert _normalized_analysis_type("multi_factor_association") == "multi_factor_association"
        assert _normalized_analysis_type("category_growth") == "category_growth"
        assert _normalized_analysis_type("semantic_role_prevalence") == "prevalence"
        assert _normalized_analysis_type("semantic_role_association") == "association"
        assert _normalized_analysis_type("duration_by_category") == "duration_analysis"


# ── Critic strengthening ────────────────────────────────────────────────────

class TestCriticStrengthening:
    def test_stale_domain_leakage_detected(self):
        from source.product.grounding_critic import stale_domain_leakage_check
        df = _healthcare_df()
        leaks = stale_domain_leakage_check(
            "The most popular movie genre is Drama, with director Spielberg leading.",
            df=df,
        )
        assert len(leaks) > 0, "Should detect entertainment terms in healthcare context"

    def test_no_false_leakage_for_matching_domain(self):
        from source.product.grounding_critic import stale_domain_leakage_check
        df = _healthcare_df()
        leaks = stale_domain_leakage_check(
            "HighBP prevalence is 50% across the dataset.",
            df=df,
        )
        # 'disease' and 'medical' are NOT in column names so they would flag,
        # but the answer doesn't mention them, so should be empty
        assert len(leaks) == 0


# ── No hardcoding audit ─────────────────────────────────────────────────────

class TestNoHardcoding:
    """Verify that production code does not hardcode dataset-specific terms."""

    @pytest.mark.parametrize("term", [
        "HeartDiseaseorAttack", "Netflix", "Superstore", "heart_disease_2015",
    ])
    def test_fallback_analysis_no_hardcoding(self, term):
        import inspect
        from source.product import fallback_analysis
        source = inspect.getsource(fallback_analysis)
        # Allow in comments/docstrings for documentation, but not in string literals
        # used as values (e.g., not in if/elif/return statements)
        executable_lines = [
            line for line in source.split("\n")
            if line.strip() and not line.strip().startswith("#") and not line.strip().startswith('"""') and not line.strip().startswith("'")
        ]
        for line in executable_lines:
            if term in line and "test" not in line.lower() and "example" not in line.lower():
                # Allow in domain evidence dicts (which need canonical terms for domain detection)
                if "_DOMAIN_EVIDENCE" in line or "domain_signal" in line or "_DOMAIN_INDICATORS" in line:
                    continue
                # Allow in comments
                if line.strip().startswith("#"):
                    continue
                assert False, f"Hardcoded term '{term}' found in production code: {line.strip()[:100]}"
