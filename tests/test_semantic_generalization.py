"""Semantic generalization regression tests.

These tests verify that the system handles diverse domains correctly:
- Multi-label category splitting
- Count-based analysis (frequency questions)
- Temporal grouping (year as dimension, not metric)
- Duration parsing
- Domain-neutral language
- Artifact generation
- Year metric guard
- Cross-investigation isolation
- Comparison symmetry
"""

import pandas as pd
import pytest

from source.product.fallback_analysis import deterministic_investigation_fallback, _count_based_dimension, _stem_match
from source.product.fallbacks.semantic_resolution import (
    _is_count_based_question,
    _looks_year_like_column,
    _select_metric_column,
    _explicit_metric_column,
    detect_multi_label_column,
    explode_multi_label,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def entertainment_df():
    return pd.DataFrame({
        "title": [f"Title_{i}" for i in range(1, 21)],
        "type": ["Movie"] * 12 + ["TV Show"] * 8,
        "listed_in": [
            "Dramas, International Movies", "Comedies", "Action, Thrillers", "Dramas",
            "Documentaries", "Comedies, International Movies", "Action", "Dramas, International Movies",
            "Thrillers", "Comedies", "Documentaries", "Action, Thrillers",
            "Dramas", "Comedies", "Action", "Thrillers", "Dramas", "Documentaries", "Comedies", "Action",
        ],
        "release_year": list(range(2016, 2036))[:20],
        "rating": ["TV-MA"] * 5 + ["TV-14"] * 5 + ["PG-13"] * 5 + ["R"] * 5,
        "duration": [
            "90 min", "120 min", "95 min", "110 min", "88 min", "105 min", "130 min",
            "92 min", "98 min", "115 min", "140 min", "78 min",
            "1 Season", "2 Seasons", "3 Seasons", "1 Season", "2 Seasons", "1 Season", "4 Seasons", "1 Season",
        ],
        "country": ["US"] * 6 + ["IN"] * 4 + ["UK"] * 3 + ["JP"] * 3 + ["KR"] * 2 + ["BR"] * 2,
    })


@pytest.fixture
def healthcare_df():
    return pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(1, 51)],
        "diagnosis": ["Hypertension"] * 12 + ["Diabetes"] * 10 + ["Asthma"] * 8 + ["COPD"] * 7 + ["Heart Failure"] * 5 + ["Pneumonia"] * 4 + ["Arthritis"] * 4,
        "patient_age": [45 + i % 40 for i in range(50)],
        "treatment": ["Medication A"] * 15 + ["Medication B"] * 12 + ["Surgery"] * 8 + ["Physical Therapy"] * 10 + ["Medication C"] * 5,
        "outcome": ["Improved"] * 22 + ["Stable"] * 15 + ["Deteriorated"] * 8 + ["Resolved"] * 5,
        "risk_score": [2.1 + i * 0.3 for i in range(50)],
        "symptoms": ["Fatigue, Shortness of breath"] * 10 + ["Chest pain, Dizziness"] * 10 + ["Joint pain"] * 10 + ["Cough, Fever"] * 10 + ["Headache, Nausea"] * 10,
        "hospital": ["General Hospital"] * 20 + ["University Medical"] * 15 + ["Community Clinic"] * 15,
        "region": ["North"] * 15 + ["South"] * 13 + ["East"] * 12 + ["West"] * 10,
    })


@pytest.fixture
def scientific_df():
    return pd.DataFrame({
        "species": ["Oak"] * 10 + ["Pine"] * 10 + ["Birch"] * 8 + ["Maple"] * 7 + ["Willow"] * 5,
        "habitat": ["Forest"] * 15 + ["Wetland"] * 10 + ["Urban"] * 8 + ["Grassland"] * 7,
        "population": [1200 + i * 50 for i in range(40)],
        "observation_year": list(range(2010, 2030))[:20] + list(range(2015, 2035))[:20],
        "conservation_status": ["Least Concern"] * 15 + ["Near Threatened"] * 10 + ["Vulnerable"] * 8 + ["Endangered"] * 7,
        "region": ["Eastern"] * 12 + ["Western"] * 10 + ["Northern"] * 10 + ["Southern"] * 8,
    })


@pytest.fixture
def sales_df():
    return pd.DataFrame({
        "Order_ID": [f"ORD{i}" for i in range(1, 51)],
        "Product": [f"Product_{i % 5}" for i in range(1, 51)],
        "Sales": [100 + i * 10 for i in range(50)],
        "Profit": [20 + i * 3 for i in range(50)],
        "Region": ["East"] * 15 + ["West"] * 15 + ["Central"] * 10 + ["South"] * 10,
    })


# ---------------------------------------------------------------------------
# Multi-label category detection
# ---------------------------------------------------------------------------

class TestMultiLabelDetection:
    def test_comma_separated_detected(self):
        series = pd.Series(["Dramas, International Movies", "Comedies", "Action, Thrillers", "Dramas"])
        assert detect_multi_label_column(series) == ","

    def test_pipe_separated_detected(self):
        series = pd.Series(["Drama|Action", "Comedy", "Thriller|Horror", "Drama|Comedy"])
        assert detect_multi_label_column(series) == "|"

    def test_semicolon_separated_detected(self):
        series = pd.Series(["Python; Java", "JavaScript", "Go; Rust", "Python; Go"])
        assert detect_multi_label_column(series) == ";"

    def test_single_values_not_detected(self):
        series = pd.Series(["Drama", "Comedy", "Action", "Thriller", "Horror"])
        assert detect_multi_label_column(series) is None

    def test_long_text_not_detected(self):
        series = pd.Series([
            "This is a very long sentence that describes the plot of a movie in great detail and includes commas, "
            "and it should not be treated as a multi-label field because it is natural prose"
        ] * 5)
        assert detect_multi_label_column(series) is None

    def test_explode_multi_label(self):
        df = pd.DataFrame({"genre": ["Drama, Action", "Comedy", "Thriller, Horror"]})
        exploded = explode_multi_label(df, "genre", ",")
        values = exploded["genre"].tolist()
        assert "Drama" in values
        assert "Action" in values
        assert "Comedy" in values
        assert "Thriller" in values
        assert "Horror" in values
        assert len(values) == 5


# ---------------------------------------------------------------------------
# Count-based analysis
# ---------------------------------------------------------------------------

class TestCountBasedAnalysis:
    def test_entertainment_releases_by_year(self, entertainment_df):
        result = deterministic_investigation_fallback("Build a chart of releases by year", entertainment_df)
        trace = result["trace_metadata"]
        assert trace["dimension"] == "release_year"
        assert trace["fallback"] == "count_based_analysis"
        assert result["artifacts"]  # chart artifact exists

    def test_entertainment_genre_dominance(self, entertainment_df):
        result = deterministic_investigation_fallback("Which genres dominate the platform?", entertainment_df)
        trace = result["trace_metadata"]
        assert trace["dimension"] == "listed_in"
        assert trace["fallback"] == "count_based_analysis"
        # Multi-label: should have fewer unique values than raw (split happened)
        chart_rows = result["artifacts"][0]["content"]["rows"]
        raw_unique = entertainment_df["listed_in"].nunique()
        chart_unique = len(chart_rows)
        assert chart_unique < raw_unique, "Multi-label splitting should reduce unique values"

    def test_entertainment_country_contribution(self, entertainment_df):
        result = deterministic_investigation_fallback("Which countries contribute most content?", entertainment_df)
        assert result["trace_metadata"]["dimension"] == "country"
        assert result["trace_metadata"]["fallback"] == "count_based_analysis"

    def test_entertainment_type_comparison(self, entertainment_df):
        result = deterministic_investigation_fallback("Compare movie and TV show distributions", entertainment_df)
        assert result["trace_metadata"]["dimension"] == "type"

    def test_healthcare_diagnosis_frequency(self, healthcare_df):
        result = deterministic_investigation_fallback("Which diagnoses are most common?", healthcare_df)
        assert result["trace_metadata"]["dimension"] == "diagnosis"
        assert result["trace_metadata"]["fallback"] == "count_based_analysis"

    def test_healthcare_symptom_frequency(self, healthcare_df):
        result = deterministic_investigation_fallback("What are the most frequent symptoms?", healthcare_df)
        assert result["trace_metadata"]["dimension"] == "symptoms"
        assert result["trace_metadata"]["fallback"] == "count_based_analysis"

    def test_scientific_species_abundance(self, scientific_df):
        result = deterministic_investigation_fallback("What species are most abundant?", scientific_df)
        assert result["trace_metadata"]["dimension"] == "species"
        assert result["trace_metadata"]["fallback"] == "count_based_analysis"


# ---------------------------------------------------------------------------
# Temporal grouping: year as dimension, never as default metric
# ---------------------------------------------------------------------------

class TestTemporalGrouping:
    def test_year_not_used_as_metric_in_count_query(self, entertainment_df):
        result = deterministic_investigation_fallback("Build a chart of releases by year", entertainment_df)
        assert result["trace_metadata"]["metric"] == "record_count"
        assert "sum(release_year)" not in result["final_answer"]
        assert "average(release_year)" not in result["final_answer"]

    def test_year_like_column_detection(self):
        series = pd.Series([2018, 2019, 2020, 2021, 2022])
        assert _looks_year_like_column(series, "release_year") is True
        assert _looks_year_like_column(series, "birth_year") is True

    def test_non_year_numeric_not_flagged(self):
        series = pd.Series([100, 200, 300, 400, 500])
        assert _looks_year_like_column(series, "sales") is False

    def test_select_metric_skips_year_column(self):
        df = pd.DataFrame({"release_year": [2020, 2021, 2022], "rating": ["TV-MA"] * 3})
        metric = _select_metric_column(df, "Build a chart of releases by year")
        assert metric is None  # Should not pick release_year

    def test_explicit_metric_skips_year_for_count_query(self):
        df = pd.DataFrame({"release_year": [2020, 2021, 2022]})
        metric = _explicit_metric_column(df, "Build a chart of releases by year")
        assert metric is None


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------

class TestDurationParsing:
    def test_duration_outlier_uses_parsed_duration(self, entertainment_df):
        result = deterministic_investigation_fallback("Find unusual outliers in movie duration", entertainment_df)
        trace = result["trace_metadata"]
        # Should NOT use release_year as metric
        assert trace.get("metric") != "release_year"
        assert "release_year" not in result["final_answer"][:200] or "duration" in result["final_answer"][:200].lower()


# ---------------------------------------------------------------------------
# Domain-neutral language
# ---------------------------------------------------------------------------

class TestDomainNeutralLanguage:
    FORBIDDEN_BUSINESS_TERMS = [
        "customer journey",
        "operational optimization",
        "revenue stabilization",
        "executive prioritization",
        "transactional behavior",
    ]

    def _check_no_business_language(self, result):
        text = result["final_answer"].lower()
        for term in self.FORBIDDEN_BUSINESS_TERMS:
            assert term not in text, f"Found forbidden business term '{term}' in response"

    def test_healthcare_no_business_language(self, healthcare_df):
        result = deterministic_investigation_fallback("Which diagnoses are most common?", healthcare_df)
        self._check_no_business_language(result)

    def test_scientific_no_business_language(self, scientific_df):
        result = deterministic_investigation_fallback("What species are most abundant?", scientific_df)
        self._check_no_business_language(result)


# ---------------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------------

class TestArtifactValidation:
    def test_count_based_produces_chart_artifact(self, entertainment_df):
        result = deterministic_investigation_fallback("Build a chart of releases by year", entertainment_df)
        artifacts = result.get("artifacts", [])
        assert len(artifacts) >= 1
        chart = artifacts[0]
        assert chart["artifact_type"] == "chart"
        content = chart["content"]
        assert content["chart_type"] == "bar"
        assert content["x"] == "release_year"
        assert content["y"] == "count"
        assert content["rows"]  # has data rows

    def test_grouped_metric_produces_chart_artifact(self, sales_df):
        result = deterministic_investigation_fallback("Compare sales across regions", sales_df)
        artifacts = result.get("artifacts", [])
        assert any(a["artifact_type"] == "chart" for a in artifacts)


# ---------------------------------------------------------------------------
# Stem matching
# ---------------------------------------------------------------------------

class TestStemMatching:
    def test_diagnosis_diagnoses(self):
        assert _stem_match("diagnosis", "diagnoses") is True

    def test_symptom_symptoms(self):
        assert _stem_match("symptom", "symptoms") is True

    def test_treatment_treatments(self):
        assert _stem_match("treatment", "treatments") is True

    def test_category_categories(self):
        assert _stem_match("category", "categories") is True

    def test_unrelated_words(self):
        assert _stem_match("hospital", "diagnoses") is False
        assert _stem_match("age", "hospital") is False


# ---------------------------------------------------------------------------
# Count-based question detection
# ---------------------------------------------------------------------------

class TestCountBasedDetection:
    @pytest.mark.parametrize("question", [
        "Which genres dominate the platform?",
        "Which diagnoses are most common?",
        "What are the most frequent symptoms?",
        "How many movies by genre?",
        "Build a chart of releases by year",
        "Which countries contribute the most content?",
        "What species are most abundant?",
    ])
    def test_count_based_question_detected(self, question):
        assert _is_count_based_question(question) is True

    @pytest.mark.parametrize("question", [
        "Compare treatment effectiveness by region",
        "What is the average risk score?",
        "Show the total sales by quarter",
    ])
    def test_metric_question_not_count_based(self, question):
        assert _is_count_based_question(question) is False


# ---------------------------------------------------------------------------
# Count-based dimension resolution
# ---------------------------------------------------------------------------

class TestCountBasedDimension:
    def test_year_in_question_picks_year_column(self):
        df = pd.DataFrame({
            "type": ["Movie", "TV Show"],
            "release_year": [2020, 2021],
        })
        result = _count_based_dimension("releases by year", df, "type", None)
        assert result == "release_year"

    def test_diagnosis_plural_matches(self):
        df = pd.DataFrame({
            "diagnosis": ["Hypertension", "Diabetes"],
            "hospital": ["General", "University"],
        })
        result = _count_based_dimension("Which diagnoses are most common?", df, "hospital", None)
        assert result == "diagnosis"

    def test_symptoms_exact_match(self):
        df = pd.DataFrame({
            "symptoms": ["Fatigue", "Pain"],
            "hospital": ["General", "University"],
        })
        result = _count_based_dimension("What are the most frequent symptoms?", df, "hospital", None)
        assert result == "symptoms"


# ---------------------------------------------------------------------------
# Cross-investigation isolation
# ---------------------------------------------------------------------------

class TestCrossInvestigationIsolation:
    def test_no_stale_state_between_investigations(self, entertainment_df):
        """Two separate investigations should not share state."""
        result_a = deterministic_investigation_fallback(
            "Compare movie and TV show distributions",
            entertainment_df,
        )
        result_b = deterministic_investigation_fallback(
            "Build a chart of releases by year",
            entertainment_df,
        )
        # Investigation B should not inherit any state from A
        assert result_b["trace_metadata"]["dimension"] == "release_year"
        # And A should have its own correct dimension
        assert result_a["trace_metadata"]["dimension"] == "type"


# ---------------------------------------------------------------------------
# Sales regression check
# ---------------------------------------------------------------------------

class TestSalesRegression:
    def test_sales_by_region(self, sales_df):
        result = deterministic_investigation_fallback("Which region has the highest sales?", sales_df)
        trace = result["trace_metadata"]
        assert trace.get("metric") == "Sales" or "Sales" in result["final_answer"]

    def test_profit_comparison(self, sales_df):
        result = deterministic_investigation_fallback("Compare profit across regions", sales_df)
        assert "Profit" in result["final_answer"]
