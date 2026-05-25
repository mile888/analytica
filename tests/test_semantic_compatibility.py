"""Tests for the semantic compatibility engine.

Tests question semantic extraction, dataset profiling, compatibility scoring,
incompatibility response generation, fallback protection, and critic validation.

All tests are deterministic — no LLM calls.
"""

from __future__ import annotations

import pandas as pd
import pytest

from source.product.question_semantics import (
    QuestionSemantics,
    extract_question_semantics,
    _is_generic_question,
    _deterministic_extract,
)
from source.product.dataset_semantics import (
    DatasetSemanticProfile,
    build_dataset_semantic_profile,
)
from source.product.compatibility_engine import (
    CompatibilityResult,
    check_compatibility,
    check_question_dataset_compatibility,
    build_incompatibility_output,
)
from source.product.grounding_critic import (
    validate_synthesis,
    _check_domain_relevance,
)
from source.product.grounded_synthesis import SynthesisResult
from source.product.plan_executor import EvidencePackage


# ── Test DataFrames ─────────────────────────────────────────────────────────

def _entertainment_df() -> pd.DataFrame:
    return pd.DataFrame({
        "title": ["Movie A", "Movie B", "Show C", "Show D", "Movie E"],
        "type": ["Movie", "Movie", "TV Show", "TV Show", "Movie"],
        "genre": ["Drama", "Comedy", "Action", "Drama", "Thriller"],
        "director": ["Dir1", "Dir2", "Dir3", "Dir1", "Dir4"],
        "release_year": [2020, 2021, 2019, 2022, 2023],
        "duration": ["90 min", "120 min", "2 Seasons", "1 Season", "105 min"],
        "rating": ["PG-13", "R", "TV-MA", "TV-14", "PG-13"],
    })


def _healthcare_df() -> pd.DataFrame:
    return pd.DataFrame({
        "patient_id": ["P001", "P002", "P003", "P004", "P005"],
        "diagnosis": ["Diabetes", "Heart Disease", "Asthma", "Diabetes", "Heart Disease"],
        "treatment": ["Medication", "Surgery", "Inhaler", "Medication", "Stent"],
        "smoking_status": ["smoker", "non-smoker", "smoker", "non-smoker", "smoker"],
        "blood_pressure": [130, 145, 120, 135, 150],
        "age": [45, 62, 33, 55, 70],
    })


def _retail_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Order ID": ["O1", "O2", "O3", "O4"],
        "Sales": [100.0, 200.0, 150.0, 300.0],
        "Profit": [20.0, 50.0, 30.0, 80.0],
        "Product": ["Widget A", "Widget B", "Widget C", "Widget A"],
        "Customer": ["Cust1", "Cust2", "Cust1", "Cust3"],
        "City": ["New York", "Chicago", "New York", "Boston"],
    })


def _scientific_df() -> pd.DataFrame:
    return pd.DataFrame({
        "species": ["Oak", "Maple", "Pine", "Birch", "Cedar"],
        "habitat": ["Forest", "Forest", "Mountain", "Wetland", "Mountain"],
        "population": [1200, 800, 500, 350, 600],
        "elevation": [200, 150, 1200, 50, 900],
    })


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — Question Semantic Extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestGenericQuestionDetection:
    """Generic questions should always be compatible."""

    def test_outlier_is_generic(self):
        assert _is_generic_question("Find outliers in the data")

    def test_missing_values_is_generic(self):
        assert _is_generic_question("How many missing values are there?")

    def test_distribution_is_generic(self):
        assert _is_generic_question("Show the distribution of values")

    def test_data_quality_is_generic(self):
        assert _is_generic_question("Check data quality")

    def test_overview_is_generic(self):
        assert _is_generic_question("Give me a dataset overview")

    def test_correlation_is_generic(self):
        assert _is_generic_question("What are the correlations?")

    def test_duplicates_is_generic(self):
        assert _is_generic_question("Are there duplicate rows?")

    def test_healthcare_question_is_not_generic(self):
        assert not _is_generic_question("Compare heart disease prevalence between smokers and non-smokers")

    def test_genre_question_is_not_generic(self):
        assert not _is_generic_question("Which genres dominate the platform?")

    def test_profit_question_is_not_generic(self):
        assert not _is_generic_question("Which products generate the most profit?")


class TestDeterministicExtraction:
    """Deterministic fallback extraction from question text."""

    def test_healthcare_domain(self):
        q = _deterministic_extract("Compare heart disease prevalence between smokers and non-smokers")
        assert q.domain == "healthcare"
        assert not q.is_generic
        assert any("disease" in e for e in q.entities)

    def test_entertainment_domain(self):
        q = _deterministic_extract("Which genres dominate the platform?")
        assert q.domain == "entertainment"
        assert not q.is_generic

    def test_retail_domain(self):
        q = _deterministic_extract("Which products generate the most revenue and profit?")
        assert q.domain == "retail"
        assert not q.is_generic

    def test_financial_domain(self):
        q = _deterministic_extract("What is the average salary and income by department?")
        assert q.domain == "financial" or q.domain == "hr"
        assert not q.is_generic

    def test_generic_falls_through(self):
        q = _deterministic_extract("Hello, how are you?")
        assert q.is_generic

    def test_scientific_domain(self):
        q = _deterministic_extract("What species dominate this habitat and ecosystem?")
        assert q.domain in ("scientific", "ecology")
        assert not q.is_generic


class TestQuestionSemanticsExtraction:
    """Full extraction pipeline (deterministic mode, no LLM)."""

    def test_generic_question_fast_path(self):
        result = extract_question_semantics("Find outliers")
        assert result.is_generic
        assert result.confidence > 0

    def test_empty_question(self):
        result = extract_question_semantics("")
        assert result.is_generic


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Dataset Semantic Profiling
# ═══════════════════════════════════════════════════════════════════════════

class TestDatasetSemanticProfile:
    """Dataset semantic profile building."""

    def test_entertainment_profile(self):
        profile = build_dataset_semantic_profile(_entertainment_df())
        assert profile.domain == "entertainment"
        assert profile.domain_confidence > 0
        assert len(profile.semantic_concepts) > 0
        assert len(profile.column_names) == 7

    def test_healthcare_profile(self):
        profile = build_dataset_semantic_profile(_healthcare_df())
        assert profile.domain == "healthcare"
        assert profile.domain_confidence > 0

    def test_retail_profile(self):
        profile = build_dataset_semantic_profile(_retail_df())
        assert profile.domain == "retail"
        assert profile.domain_confidence > 0

    def test_column_concepts_capture_sample_values(self):
        profile = build_dataset_semantic_profile(_entertainment_df())
        # type column should have "Movie" and "TV Show" as concepts
        assert "type" in profile.column_concepts
        values = profile.column_concepts["type"]
        assert "Movie" in values or "TV Show" in values

    def test_field_roles_detected(self):
        profile = build_dataset_semantic_profile(_retail_df())
        assert "Sales" in profile.field_roles
        assert profile.field_roles["Sales"] == "metric"

    def test_empty_df_returns_default(self):
        profile = build_dataset_semantic_profile(pd.DataFrame())
        assert profile.domain == "general"
        assert profile.domain_confidence == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — Compatibility Scoring
# ═══════════════════════════════════════════════════════════════════════════

class TestCompatibilityScoring:
    """Compatibility engine scoring logic."""

    def test_generic_question_always_compatible(self):
        q = QuestionSemantics(is_generic=True, confidence=0.9)
        d = build_dataset_semantic_profile(_entertainment_df())
        result = check_compatibility(q, d)
        assert result.compatible
        assert result.compatibility_score == 1.0

    def test_same_domain_compatible(self):
        q = QuestionSemantics(
            domain="entertainment", entities=["genre"], is_generic=False, confidence=0.8,
        )
        d = build_dataset_semantic_profile(_entertainment_df())
        result = check_compatibility(q, d)
        assert result.compatible
        assert result.compatibility_score > 0.3

    def test_different_domain_incompatible(self):
        q = QuestionSemantics(
            domain="healthcare",
            entities=["heart disease", "smoker", "non-smoker"],
            required_concepts=["disease", "smoking_status"],
            is_generic=False,
            confidence=0.9,
        )
        d = build_dataset_semantic_profile(_entertainment_df())
        result = check_compatibility(q, d)
        assert not result.compatible
        assert result.compatibility_score < 0.3
        assert len(result.missing_entities) > 0

    def test_healthcare_on_retail_incompatible(self):
        q = QuestionSemantics(
            domain="healthcare",
            entities=["diagnosis", "patient", "treatment"],
            required_concepts=["disease", "clinical"],
            is_generic=False,
            confidence=0.9,
        )
        d = build_dataset_semantic_profile(_retail_df())
        result = check_compatibility(q, d)
        assert not result.compatible

    def test_entertainment_on_healthcare_incompatible(self):
        q = QuestionSemantics(
            domain="entertainment",
            entities=["genre", "movie", "director"],
            required_concepts=["genre", "rating"],
            is_generic=False,
            confidence=0.8,
        )
        d = build_dataset_semantic_profile(_healthcare_df())
        result = check_compatibility(q, d)
        assert not result.compatible

    def test_retail_on_scientific_incompatible(self):
        q = QuestionSemantics(
            domain="retail",
            entities=["revenue", "profit", "customer"],
            required_concepts=["sales", "profit"],
            is_generic=False,
            confidence=0.9,
        )
        d = build_dataset_semantic_profile(_scientific_df())
        result = check_compatibility(q, d)
        assert not result.compatible

    def test_matching_entities_increase_score(self):
        q = QuestionSemantics(
            domain="entertainment",
            entities=["genre", "movie"],
            is_generic=False,
            confidence=0.8,
        )
        d = build_dataset_semantic_profile(_entertainment_df())
        result = check_compatibility(q, d)
        assert result.compatibility_score > 0.4
        assert len(result.matched_entities) > 0

    def test_general_domain_question_compatible(self):
        q = QuestionSemantics(domain="general", is_generic=False, confidence=0.3)
        d = build_dataset_semantic_profile(_entertainment_df())
        result = check_compatibility(q, d)
        assert result.compatible


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — End-to-End Compatibility Check
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndCompatibility:
    """Full pipeline: question string → compatibility result."""

    def test_healthcare_question_on_entertainment_dataset(self):
        result = check_question_dataset_compatibility(
            "Compare heart disease prevalence between smokers and non-smokers",
            _entertainment_df(),
        )
        assert result is not None
        assert not result.compatible
        assert "heart disease" in result.reason.lower() or "disease" in result.reason.lower() or "does not contain" in result.reason.lower()

    def test_genre_question_on_healthcare_dataset(self):
        result = check_question_dataset_compatibility(
            "Which genres dominate the platform?",
            _healthcare_df(),
        )
        assert result is not None
        assert not result.compatible

    def test_profit_question_on_scientific_dataset(self):
        result = check_question_dataset_compatibility(
            "Which products generate the most profit and revenue?",
            _scientific_df(),
        )
        assert result is not None
        assert not result.compatible

    def test_generic_outlier_on_entertainment_allowed(self):
        result = check_question_dataset_compatibility(
            "Find outliers in the data",
            _entertainment_df(),
        )
        assert result is not None
        assert result.compatible

    def test_generic_missing_values_on_healthcare_allowed(self):
        result = check_question_dataset_compatibility(
            "How many missing values are there?",
            _healthcare_df(),
        )
        assert result is not None
        assert result.compatible

    def test_generic_distribution_on_retail_allowed(self):
        result = check_question_dataset_compatibility(
            "Show the distribution of values",
            _retail_df(),
        )
        assert result is not None
        assert result.compatible

    def test_matching_question_on_matching_dataset(self):
        result = check_question_dataset_compatibility(
            "Which genres appear most often?",
            _entertainment_df(),
        )
        assert result is not None
        assert result.compatible

    def test_healthcare_question_on_healthcare_dataset_compatible(self):
        result = check_question_dataset_compatibility(
            "Compare heart disease prevalence between smokers and non-smokers",
            _healthcare_df(),
        )
        assert result is not None
        assert result.compatible

    def test_none_for_no_dataframe(self):
        result = check_question_dataset_compatibility("Some question", None)
        assert result is None

    def test_none_for_empty_question(self):
        result = check_question_dataset_compatibility("", _entertainment_df())
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — Incompatibility Response Format
# ═══════════════════════════════════════════════════════════════════════════

class TestIncompatibilityResponse:
    """Incompatibility response should be well-formed."""

    def test_response_has_required_fields(self):
        result = CompatibilityResult(
            compatible=False,
            compatibility_score=0.1,
            missing_entities=["heart disease", "smoker"],
            missing_concepts=["disease"],
            reason="Dataset does not contain healthcare data.",
            dataset_domain="entertainment",
            question_domain="healthcare",
        )
        output = build_incompatibility_output("Healthcare question", result)
        assert "summary" in output
        assert "final_answer" in output
        assert "structured_report" in output
        assert "trace_metadata" in output
        assert output["trace_metadata"]["analysis_type"] == "semantic_incompatibility"
        assert output["artifacts"] == []

    def test_response_mentions_reason(self):
        result = CompatibilityResult(
            compatible=False,
            compatibility_score=0.1,
            missing_entities=["heart disease"],
            reason="This dataset contains entertainment data, not healthcare.",
            dataset_domain="entertainment",
            question_domain="healthcare",
        )
        output = build_incompatibility_output("Healthcare question", result)
        assert "entertainment" in output["summary"].lower() or "does not contain" in output["summary"].lower()

    def test_no_chart_artifacts(self):
        result = CompatibilityResult(
            compatible=False,
            compatibility_score=0.0,
            reason="Incompatible.",
            dataset_domain="entertainment",
            question_domain="healthcare",
        )
        output = build_incompatibility_output("Q", result)
        assert output["artifacts"] == []
        assert output["key_findings"] == []


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — Fallback Protection
# ═══════════════════════════════════════════════════════════════════════════

class TestFallbackProtection:
    """Compatibility check should prevent unrelated fallback analysis."""

    def test_fallback_returns_incompatibility_for_mismatched_question(self):
        from source.product.fallback_analysis import deterministic_investigation_fallback

        result = deterministic_investigation_fallback(
            "Compare heart disease prevalence between smokers and non-smokers",
            _entertainment_df(),
        )
        assert result is not None
        # Should be an incompatibility response, not Movie vs TV Show analysis
        trace = result.get("trace_metadata", {})
        summary = result.get("summary", "") or result.get("final_answer", "")
        is_incompatible = (
            trace.get("analysis_type") == "semantic_incompatibility"
            or "does not contain" in summary.lower()
            or "cannot be" in summary.lower()
        )
        assert is_incompatible, f"Expected incompatibility response, got: {summary[:200]}"


# ═══════════════════════════════════════════════════════════════════════════
# PART 7 — Critic Domain Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestCriticDomainValidation:
    """Critic should detect domain mismatches in answers."""

    def test_detects_entertainment_answer_for_healthcare_question(self):
        mismatch = _check_domain_relevance(
            "The movie genre distribution shows Drama at 40% and Comedy at 30%.",
            question_domain="healthcare",
            dataset_domain="entertainment",
        )
        assert mismatch  # Should detect wrong-domain terms

    def test_passes_matching_domain(self):
        mismatch = _check_domain_relevance(
            "Heart disease diagnosis is present in 40% of patients.",
            question_domain="healthcare",
            dataset_domain="healthcare",
        )
        assert not mismatch

    def test_critic_with_domain_params(self):
        """validate_synthesis still works with new optional params."""
        synthesis = SynthesisResult(
            answer="The top genre is Drama with 42.0% share.",
            findings=[{"finding": "Drama leads", "evidence": "42.0% of 5 records"}],
        )
        evidence = EvidencePackage(
            question="Which genres dominate?",
            operation="COUNT_DISTRIBUTION",
            columns_used=["genre"],
            record_count=5,
            computed_tables=[{"rows": [{"genre": "Drama", "count": 42.0}]}],
        )
        # Without domain params — should pass as before
        result = validate_synthesis(synthesis, evidence)
        assert result.passed

        # With matching domains — should still pass
        result2 = validate_synthesis(
            synthesis, evidence,
            question_domain="entertainment",
            dataset_domain="entertainment",
        )
        assert result2.passed


# ═══════════════════════════════════════════════════════════════════════════
# PART 8 — No Hardcoded Dataset Logic
# ═══════════════════════════════════════════════════════════════════════════

class TestNoHardcoding:
    """Verify the engine doesn't contain dataset-specific hardcoding."""

    def test_question_semantics_no_dataset_names(self):
        import inspect
        import source.product.question_semantics as qs
        code = inspect.getsource(qs)
        for name in ("Netflix", "Superstore", "listed_in"):
            assert name not in code, f"Found hardcoded '{name}' in question_semantics.py"

    def test_dataset_semantics_no_dataset_names(self):
        import inspect
        import source.product.dataset_semantics as ds
        code = inspect.getsource(ds)
        for name in ("Netflix", "Superstore", "listed_in"):
            assert name not in code, f"Found hardcoded '{name}' in dataset_semantics.py"

    def test_compatibility_engine_no_dataset_names(self):
        import inspect
        import source.product.compatibility_engine as ce
        code = inspect.getsource(ce)
        for name in ("Netflix", "Superstore", "listed_in"):
            assert name not in code, f"Found hardcoded '{name}' in compatibility_engine.py"

    def test_works_with_novel_domain(self):
        """Engine should handle unseen domain datasets gracefully."""
        astronomy_df = pd.DataFrame({
            "star_name": ["Sirius", "Betelgeuse", "Vega", "Polaris"],
            "magnitude": [1.46, 0.5, 0.03, 1.98],
            "constellation": ["Canis Major", "Orion", "Lyra", "Ursa Minor"],
            "distance_ly": [8.6, 700, 25, 430],
        })
        result = check_question_dataset_compatibility(
            "Which products generate the most profit?",
            astronomy_df,
        )
        assert result is not None
        # Should detect retail question + no retail data → incompatible
        # or at minimum not produce random analysis
        # (astronomy is an unseen domain, so it may be "general")
