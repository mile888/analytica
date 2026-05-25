"""Tests for global compatibility enforcement.

Validates that the compatibility gate blocks ALL analysis paths for
semantically incompatible questions, while allowing genuinely generic requests.
"""

import pandas as pd
import pytest

from source.product.compatibility_engine import (
    CompatibilityDecision,
    enforce_question_dataset_compatibility,
    check_question_dataset_compatibility,
    build_incompatibility_output,
)
from source.product.question_semantics import extract_question_semantics
from source.product.fallback_analysis import deterministic_investigation_fallback
from source.product.grounding_critic import critic_backstop_check


# ── Test fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def movie_df():
    return pd.DataFrame({
        "show_id": ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
        "type": ["Movie", "TV Show", "Movie", "TV Show", "Movie", "Movie", "TV Show", "Movie"],
        "title": ["Film A", "Show B", "Film C", "Show D", "Film E", "Film F", "Show G", "Film H"],
        "director": ["Dir1", "Dir2", "Dir3", "Dir1", "Dir4", "Dir5", "Dir2", "Dir6"],
        "country": ["US", "India", "UK", "US", "Japan", "India", "UK", "US"],
        "release_year": [2020, 2019, 2021, 2020, 2022, 2021, 2018, 2023],
        "rating": ["PG-13", "TV-MA", "R", "TV-14", "PG", "R", "TV-MA", "PG-13"],
        "duration": ["90 min", "2 Seasons", "120 min", "1 Season", "105 min", "88 min", "3 Seasons", "95 min"],
        "listed_in": ["Dramas", "Comedies", "Action", "Dramas", "Thrillers", "Comedies", "Action", "Dramas"],
        "description": ["A drama", "A comedy", "Action film", "Drama show", "Thriller", "Comedy", "Action show", "Drama film"],
    })


@pytest.fixture
def healthcare_df():
    return pd.DataFrame({
        "patient_id": [1, 2, 3, 4, 5],
        "age": [35, 55, 42, 68, 29],
        "diagnosis": ["Hypertension", "Diabetes", "Heart Disease", "COPD", "Asthma"],
        "treatment": ["Medication", "Insulin", "Surgery", "Inhaler", "Medication"],
        "bmi": [25.3, 31.1, 28.7, 22.5, 24.1],
        "smoker": ["No", "Yes", "Yes", "Yes", "No"],
        "risk_score": [0.3, 0.7, 0.8, 0.6, 0.2],
    })


@pytest.fixture
def retail_df():
    return pd.DataFrame({
        "order_id": ["o1", "o2", "o3", "o4", "o5"],
        "product": ["Widget A", "Widget B", "Widget C", "Widget A", "Widget D"],
        "sales": [100.0, 250.0, 50.0, 300.0, 120.0],
        "profit": [20.0, 80.0, 5.0, 100.0, 30.0],
        "category": ["Electronics", "Furniture", "Office", "Electronics", "Furniture"],
        "city": ["NYC", "LA", "Chicago", "NYC", "SF"],
    })


# ── PART 10: Healthcare questions MUST REFUSE on movie dataset ─────────────

_HEALTHCARE_QUESTIONS_ON_MOVIE = [
    "Which health conditions appear most common across the population?",
    "Summarize the main public health risks revealed by this dataset.",
    "Compare heart disease prevalence between smokers and non-smokers.",
    "Which age groups show the strongest increase in heart disease risk?",
    "Compare lifestyle-factor diversity across different patient groups.",
    "Find unusual outliers in BMI and explain why they may matter clinically.",
    "Which health indicators become more common among older patients?",
]


@pytest.mark.parametrize("question", _HEALTHCARE_QUESTIONS_ON_MOVIE)
def test_healthcare_question_on_movie_dataset_is_refused(movie_df, question):
    """Every healthcare question must be refused on a movie dataset."""
    decision = enforce_question_dataset_compatibility(question, movie_df)
    assert not decision.allowed, f"'{question}' should be refused but was allowed"
    assert decision.question_domain == "healthcare"
    assert decision.dataset_domain == "entertainment"


@pytest.mark.parametrize("question", _HEALTHCARE_QUESTIONS_ON_MOVIE)
def test_healthcare_question_fallback_returns_incompatibility(movie_df, question):
    """The deterministic fallback must return an incompatibility response, not overview/count."""
    result = deterministic_investigation_fallback(question, movie_df)
    assert result is not None
    analysis_type = (result.get("trace_metadata") or {}).get("analysis_type", "")
    assert analysis_type == "semantic_incompatibility", (
        f"'{question}' returned analysis_type='{analysis_type}' instead of 'semantic_incompatibility'"
    )


@pytest.mark.parametrize("question", _HEALTHCARE_QUESTIONS_ON_MOVIE)
def test_healthcare_question_no_overview_response(movie_df, question):
    """Healthcare questions on movie data must NOT produce overview/profiling responses."""
    result = deterministic_investigation_fallback(question, movie_df)
    summary = str(result.get("summary", "")).lower() if result else ""
    # Must NOT contain typical overview/profiling responses
    assert "structured tabular dataset" not in summary
    assert "useful first analysis" not in summary
    assert "release_year" not in summary
    assert "movie vs tv show" not in summary.replace(" ", "").lower()


@pytest.mark.parametrize("question", _HEALTHCARE_QUESTIONS_ON_MOVIE)
def test_healthcare_question_no_chart_artifacts(movie_df, question):
    """No chart artifacts should be generated for incompatible questions."""
    result = deterministic_investigation_fallback(question, movie_df)
    artifacts = result.get("artifacts", []) if result else []
    assert not artifacts, f"'{question}' generated chart artifacts on movie dataset"


# ── Generic questions MUST still work on movie dataset ─────────────────────

_GENERIC_QUESTIONS_ON_MOVIE = [
    "Summarize this dataset.",
    "Which genres dominate the platform?",
    "Show missing values.",
    "Build chart of releases by year.",
    "Find outliers in duration.",
    "What columns are available?",
    "What can be analyzed from this dataset?",
    "Show data quality issues.",
]


@pytest.mark.parametrize("question", _GENERIC_QUESTIONS_ON_MOVIE)
def test_generic_question_on_movie_dataset_is_allowed(movie_df, question):
    """Generic and entertainment questions must still work on movie dataset."""
    decision = enforce_question_dataset_compatibility(question, movie_df)
    assert decision.allowed, f"'{question}' was wrongly blocked"


# ── Question semantics classification ──────────────────────────────────────

def test_public_health_risks_is_not_classified_as_generic():
    sem = extract_question_semantics("Summarize the main public health risks revealed by this dataset.")
    assert sem.domain == "healthcare" or not sem.is_generic
    assert sem.is_generic is False


def test_health_conditions_is_not_classified_as_generic():
    sem = extract_question_semantics("Which health conditions appear most common across the population?")
    assert sem.is_generic is False


def test_bmi_outliers_clinically_is_not_classified_as_generic():
    sem = extract_question_semantics("Find unusual outliers in BMI and explain why they may matter clinically.")
    assert sem.is_generic is False


def test_lifestyle_diversity_patient_groups_is_not_classified_as_generic():
    sem = extract_question_semantics("Compare lifestyle-factor diversity across different patient groups.")
    assert sem.is_generic is False


def test_pure_summarize_dataset_is_generic():
    sem = extract_question_semantics("Summarize this dataset.")
    assert sem.is_generic is True


def test_pure_find_outliers_is_generic():
    sem = extract_question_semantics("Find outliers.")
    assert sem.is_generic is True


def test_pure_show_missing_is_generic():
    sem = extract_question_semantics("Show missing values.")
    assert sem.is_generic is True


# ── Overview vs domain-specific summary distinction ────────────────────────

def test_summarize_dataset_vs_summarize_health_risks(movie_df):
    """'Summarize this dataset' is allowed; 'Summarize health risks' is refused."""
    overview = enforce_question_dataset_compatibility("Summarize this dataset.", movie_df)
    domain = enforce_question_dataset_compatibility("Summarize the main public health risks revealed by this dataset.", movie_df)

    assert overview.allowed is True
    assert domain.allowed is False


# ── enforce_question_dataset_compatibility returns correct structure ────────

def test_enforcement_returns_compatibility_decision(movie_df):
    decision = enforce_question_dataset_compatibility(
        "Compare heart disease prevalence between smokers and non-smokers.",
        movie_df,
    )
    assert isinstance(decision, CompatibilityDecision)
    assert decision.allowed is False
    assert decision.incompatibility_response
    assert decision.dataset_domain == "entertainment"
    assert decision.question_domain == "healthcare"


# ── Cross-domain tests ─────────────────────────────────────────────────────

def test_entertainment_question_on_healthcare_dataset_is_refused(healthcare_df):
    """Movie/genre questions on healthcare data should be refused."""
    decision = enforce_question_dataset_compatibility(
        "Which genres dominate the platform and what director produces most content?",
        healthcare_df,
    )
    assert not decision.allowed


def test_retail_question_on_movie_dataset_is_refused(movie_df):
    """Profit/sales questions on movie data should be refused."""
    decision = enforce_question_dataset_compatibility(
        "Which products generate the highest profit margin?",
        movie_df,
    )
    assert not decision.allowed


def test_healthcare_question_on_healthcare_dataset_is_allowed(healthcare_df):
    """Healthcare questions on healthcare data should be allowed."""
    decision = enforce_question_dataset_compatibility(
        "Compare heart disease prevalence between smokers and non-smokers.",
        healthcare_df,
    )
    assert decision.allowed


def test_retail_question_on_retail_dataset_is_allowed(retail_df):
    """Retail questions on retail data should be allowed."""
    decision = enforce_question_dataset_compatibility(
        "Which products generate the highest profit?",
        retail_df,
    )
    assert decision.allowed


# ── Critic backstop ────────────────────────────────────────────────────────

def test_critic_backstop_catches_wrong_domain_answer(movie_df):
    """Critic should catch a wrong-domain answer that somehow bypassed the gate."""
    wrong_answer = (
        "Movie vs TV Show count analysis: Movies make up 62.5% of the catalog. "
        "The release_year distribution shows a peak in 2021."
    )
    refusal = critic_backstop_check(
        "Compare heart disease prevalence between smokers and non-smokers.",
        wrong_answer,
        df=movie_df,
    )
    assert refusal is not None
    assert "does not contain" in refusal


def test_critic_backstop_allows_compatible_answer(healthcare_df):
    """Critic should not block a correctly-answered compatible question."""
    correct_answer = "Heart disease is more prevalent among smokers (2 out of 3 smoker patients)."
    refusal = critic_backstop_check(
        "Compare heart disease prevalence between smokers and non-smokers.",
        correct_answer,
        df=healthcare_df,
    )
    assert refusal is None


# ── Edge cases ─────────────────────────────────────────────────────────────

def test_empty_question_is_allowed(movie_df):
    decision = enforce_question_dataset_compatibility("", movie_df)
    assert decision.allowed


def test_empty_dataframe_is_allowed():
    decision = enforce_question_dataset_compatibility(
        "Compare heart disease prevalence.",
        pd.DataFrame(),
    )
    assert decision.allowed  # Can't check compatibility without data


def test_none_dataframe_is_allowed():
    decision = enforce_question_dataset_compatibility("Compare heart disease.", None)
    assert decision.allowed


# ── Hardcode audit ─────────────────────────────────────────────────────────

def test_no_hardcoded_dataset_names_in_compatibility_engine():
    """No dataset-specific names should appear in the compatibility engine."""
    import inspect
    from source.product import compatibility_engine, question_semantics

    for module in (compatibility_engine, question_semantics):
        source = inspect.getsource(module)
        for forbidden in ("Netflix", "Superstore", "listed_in", "release_year"):
            assert forbidden not in source, f"'{forbidden}' found in {module.__name__}"


# ── PART 9: Terminal Decision Propagation ──────────────────────────────────

class TestTerminalDecisionPropagation:
    """Verify that terminal decisions are propagated through the orchestration
    layers and never replaced by generic 'no evidence' messages."""

    def test_summarize_run_result_extracts_terminal_decision(self, movie_df):
        """_summarize_run_result should find the incompatibility message from run output."""
        from source.product.run_service import _summarize_run_result, _extract_terminal_decision_from_runs
        from unittest.mock import MagicMock

        output = build_incompatibility_output(
            "Compare heart disease prevalence between smokers and non-smokers.",
            check_question_dataset_compatibility("Compare heart disease prevalence between smokers and non-smokers.", movie_df),
        )

        # Mock a run with the incompatibility output
        mock_run = MagicMock()
        mock_run.output = output

        # Mock an investigation with no report/findings/artifacts but with the run
        mock_investigation = MagicMock()
        mock_investigation.report = None
        mock_investigation.findings = []
        mock_investigation.artifacts = []
        mock_investigation.runs = [mock_run]

        # _extract_terminal_decision_from_runs should find the message
        terminal = _extract_terminal_decision_from_runs(mock_investigation)
        assert terminal is not None
        assert "does not contain" in terminal

        # _summarize_run_result should return the terminal message, not "I do not have..."
        result = _summarize_run_result(mock_investigation)
        assert "does not contain" in result
        assert "do not have a fresh row-level result" not in result

    def test_no_evidence_message_only_for_genuine_no_evidence(self):
        """'I do not have a fresh row-level result' should only appear for genuine no-evidence."""
        from source.product.run_service import _summarize_run_result
        from unittest.mock import MagicMock

        mock_investigation = MagicMock()
        mock_investigation.report = None
        mock_investigation.findings = []
        mock_investigation.artifacts = []
        mock_investigation.runs = []

        result = _summarize_run_result(mock_investigation)
        assert "do not have a fresh row-level result" in result

    def test_terminal_decision_not_overridden_by_quality_gate(self, movie_df):
        """Terminal decisions must not be overridden by quality gate or redundancy checks."""
        from source.product.run_service import _extract_terminal_decision_from_runs
        from unittest.mock import MagicMock

        output = build_incompatibility_output(
            "Summarize the main public health risks revealed by this dataset.",
            check_question_dataset_compatibility("Summarize the main public health risks.", movie_df),
        )

        mock_run = MagicMock()
        mock_run.output = output
        mock_investigation = MagicMock()
        mock_investigation.runs = [mock_run]

        # Terminal check should return the incompatibility message
        terminal = _extract_terminal_decision_from_runs(mock_investigation)
        assert terminal is not None
        assert "does not contain" in terminal

    @pytest.mark.parametrize("question", _HEALTHCARE_QUESTIONS_ON_MOVIE)
    def test_incompatibility_output_has_required_fields(self, movie_df, question):
        """Incompatibility output must have all fields needed for terminal propagation."""
        compat = check_question_dataset_compatibility(question, movie_df)
        output = build_incompatibility_output(question, compat)

        # Required fields for service.py
        assert "summary" in output
        assert "final_answer" in output
        assert "trace_metadata" in output
        assert output["trace_metadata"]["analysis_type"] == "semantic_incompatibility"
        assert output["trace_metadata"]["suppress_key_findings"] is True

        # Required fields for adapter.py to create a report
        assert "structured_report" in output
        assert output["structured_report"]["summary"]

        # Must NOT have chart/finding artifacts
        assert output["artifacts"] == []
        assert output["key_findings"] == []

    @pytest.mark.parametrize("question", _HEALTHCARE_QUESTIONS_ON_MOVIE)
    def test_adapter_creates_report_from_incompatibility_output(self, movie_df, question):
        """agent_output_to_investigation_update must create a report from incompatibility output."""
        from source.product.adapter import agent_output_to_investigation_update

        compat = check_question_dataset_compatibility(question, movie_df)
        output = build_incompatibility_output(question, compat)
        update = agent_output_to_investigation_update(output)

        assert update.report is not None
        assert update.report.summary
        assert "does not contain" in update.report.summary
        assert len(update.findings) == 0  # suppress_key_findings = True


# ── PART 13: Multi-Dataset Tests ───────────────────────────────────────────

class TestMultiDatasetEnforcement:
    """Verify that incompatible datasets are excluded in multi-dataset scenarios."""

    def test_healthcare_question_entertainment_excluded(self, movie_df, healthcare_df):
        """Healthcare question: movie dataset excluded, healthcare allowed."""
        q = "Compare heart disease prevalence between smokers and non-smokers."

        movie_decision = enforce_question_dataset_compatibility(q, movie_df)
        health_decision = enforce_question_dataset_compatibility(q, healthcare_df)

        assert not movie_decision.allowed
        assert health_decision.allowed

    def test_entertainment_question_healthcare_excluded(self, movie_df, healthcare_df):
        """Entertainment question: healthcare excluded, movie allowed."""
        q = "Which genres dominate the platform?"

        movie_decision = enforce_question_dataset_compatibility(q, movie_df)
        health_decision = enforce_question_dataset_compatibility(q, healthcare_df)

        assert movie_decision.allowed
        assert not health_decision.allowed

    def test_retail_question_only_retail_participates(self, movie_df, healthcare_df, retail_df):
        """Retail question: only retail dataset participates."""
        q = "Which products generate the highest profit?"

        movie_decision = enforce_question_dataset_compatibility(q, movie_df)
        health_decision = enforce_question_dataset_compatibility(q, healthcare_df)
        retail_decision = enforce_question_dataset_compatibility(q, retail_df)

        assert not movie_decision.allowed
        assert not health_decision.allowed
        assert retail_decision.allowed

    def test_generic_question_all_datasets_participate(self, movie_df, healthcare_df, retail_df):
        """Generic question: all datasets should participate."""
        q = "Summarize this dataset."

        for df in (movie_df, healthcare_df, retail_df):
            decision = enforce_question_dataset_compatibility(q, df)
            assert decision.allowed, f"Generic question wrongly blocked on {list(df.columns)[:3]}"

    def test_excluded_dataset_produces_no_fallback(self, movie_df):
        """An incompatible dataset must not produce a fallback analysis."""
        q = "Compare heart disease prevalence between smokers and non-smokers."
        result = deterministic_investigation_fallback(q, movie_df)
        assert result is not None
        analysis_type = (result.get("trace_metadata") or {}).get("analysis_type", "")
        assert analysis_type == "semantic_incompatibility"
        # Must not contain any movie-dataset analysis
        summary = str(result.get("summary", "")).lower()
        assert "movie" not in summary or "does not contain" in summary

