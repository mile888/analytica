"""Tests for the Semantic Role Assignment Engine.

Validates that the engine correctly infers target/outcome, grouping,
and explanatory variables across multiple domains without any
dataset-specific hardcoding.
"""

import pandas as pd
import pytest

from source.product.semantic_role_assignment import (
    SemanticRoleAssignment,
    assign_semantic_roles,
    prevalence_analysis_response,
    association_analysis_response,
    _classify_analysis_operation,
    _detect_target_variable,
    _detect_grouping_variables,
    _is_binary_column,
)
from source.product.fallback_analysis import deterministic_investigation_fallback


# ── Test fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def healthcare_df():
    """Healthcare dataset with binary outcome and risk factors."""
    return pd.DataFrame({
        "PatientID": range(1, 21),
        "Age": [25, 55, 42, 68, 29, 35, 60, 48, 31, 72, 27, 50, 39, 65, 33, 45, 58, 41, 70, 36],
        "Sex": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "HighBP": [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        "HighChol": [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        "Smoker": [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
        "BMI": [22.1, 31.5, 27.3, 29.8, 23.4, 25.1, 28.6, 30.2, 21.8, 33.0, 22.5, 26.7, 24.9, 31.1, 23.0, 27.8, 29.5, 24.3, 32.2, 25.6],
        "PhysActivity": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1],
        "HeartDiseaseorAttack": [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    })


@pytest.fixture
def retail_df():
    """Retail dataset with churn flag."""
    return pd.DataFrame({
        "CustomerID": range(1, 11),
        "Segment": ["Consumer", "Corporate", "Consumer", "Corporate", "Home Office",
                     "Consumer", "Corporate", "Home Office", "Consumer", "Corporate"],
        "Region": ["East", "West", "South", "East", "West", "South", "East", "West", "East", "South"],
        "TotalSpend": [500.0, 1200.0, 300.0, 800.0, 450.0, 600.0, 950.0, 350.0, 700.0, 1100.0],
        "Tenure": [12, 36, 6, 24, 18, 30, 48, 9, 15, 42],
        "Churn": [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    })


@pytest.fixture
def education_df():
    """Education dataset with pass/fail outcome."""
    return pd.DataFrame({
        "StudentID": range(1, 11),
        "Attendance": [0.95, 0.60, 0.85, 0.70, 0.40, 0.90, 0.55, 0.80, 0.65, 0.75],
        "StudyHours": [15, 5, 12, 8, 3, 14, 4, 11, 7, 9],
        "ParentEducation": ["College", "High School", "College", "Graduate", "High School",
                            "College", "High School", "Graduate", "High School", "College"],
        "FreeReduced": [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        "Passed": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def hr_df():
    """HR dataset with attrition flag."""
    return pd.DataFrame({
        "EmployeeID": range(1, 11),
        "Department": ["Sales", "Engineering", "Sales", "Marketing", "Engineering",
                       "Sales", "Marketing", "Engineering", "Sales", "Marketing"],
        "YearsAtCompany": [2, 8, 1, 5, 12, 3, 7, 10, 4, 6],
        "Satisfaction": [3, 4, 2, 3, 5, 4, 2, 4, 3, 1],
        "Attrition": [0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
    })


@pytest.fixture
def movie_df():
    """Entertainment dataset — no binary outcome, generic questions should work."""
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
    })


# ── PART 1: Operation Classification ──────────────────────────────────────

class TestOperationClassification:
    def test_prevalence_detected(self):
        assert _classify_analysis_operation("compare heart disease prevalence between smokers") == "prevalence"

    def test_prevalence_rate(self):
        assert _classify_analysis_operation("what is the churn rate by segment") == "prevalence"

    def test_prevalence_how_common(self):
        assert _classify_analysis_operation("how common is heart disease among smokers") == "prevalence"

    def test_association_detected(self):
        assert _classify_analysis_operation("which factors are most associated with heart disease") == "association"

    def test_association_risk_factors(self):
        assert _classify_analysis_operation("what are the risk factors for churn") == "association"

    def test_association_predictors(self):
        assert _classify_analysis_operation("what predicts student failure") == "association"

    def test_trend_with_indicators(self):
        assert _classify_analysis_operation("which health indicators become more common among older patients") == "prevalence"

    def test_visualization_with_prevalence(self):
        assert _classify_analysis_operation("build a visualization of heart disease prevalence by age group") == "prevalence"

    def test_generic_summary_not_classified(self):
        assert _classify_analysis_operation("summarize this dataset") == ""

    def test_generic_outlier_not_classified(self):
        assert _classify_analysis_operation("find outliers in bmi") == ""

    def test_generic_chart_not_classified(self):
        assert _classify_analysis_operation("build a chart of releases by year") == ""


# ── PART 2: Binary Column Detection ──────────────────────────────────────

class TestBinaryColumnDetection:
    def test_numeric_binary(self):
        assert _is_binary_column(pd.Series([0, 1, 1, 0, 1])) is True

    def test_numeric_non_binary(self):
        assert _is_binary_column(pd.Series([0, 1, 2, 3])) is False

    def test_string_yes_no(self):
        assert _is_binary_column(pd.Series(["Yes", "No", "Yes", "No"])) is True

    def test_string_true_false(self):
        assert _is_binary_column(pd.Series(["True", "False", "True"])) is True

    def test_male_female_excluded(self):
        """Male/Female is demographic, not an outcome."""
        assert _is_binary_column(pd.Series(["Male", "Female", "Male"])) is False

    def test_many_categories_not_binary(self):
        assert _is_binary_column(pd.Series(["A", "B", "C", "D"])) is False

    def test_float_binary(self):
        assert _is_binary_column(pd.Series([0.0, 1.0, 1.0, 0.0])) is True

    def test_empty_series(self):
        assert _is_binary_column(pd.Series(dtype=object)) is False


# ── PART 3: Target Variable Detection ────────────────────────────────────

class TestTargetVariableDetection:
    def test_heart_disease_detected(self, healthcare_df):
        target = _detect_target_variable("compare heart disease prevalence between smokers", healthcare_df)
        assert target == "HeartDiseaseorAttack"

    def test_churn_detected(self, retail_df):
        target = _detect_target_variable("which segments have highest churn rate", retail_df)
        assert target == "Churn"

    def test_passed_detected(self, education_df):
        target = _detect_target_variable("which factors are associated with student failure", education_df)
        # Should detect Passed (binary outcome) even though "failure" is the opposite
        assert target == "Passed"

    def test_attrition_detected(self, hr_df):
        target = _detect_target_variable("compare attrition rates by department", hr_df)
        assert target == "Attrition"

    def test_no_target_in_movie_dataset(self, movie_df):
        target = _detect_target_variable("summarize this dataset", movie_df)
        assert target is None


# ── PART 4: Grouping Variable Detection ──────────────────────────────────

class TestGroupingVariableDetection:
    def test_smoker_grouping(self, healthcare_df):
        groups = _detect_grouping_variables(
            "compare heart disease prevalence between smokers and non smokers",
            healthcare_df,
            exclude={"HeartDiseaseorAttack"},
        )
        assert "Smoker" in groups

    def test_segment_grouping(self, retail_df):
        groups = _detect_grouping_variables(
            "which customer segments have highest churn rate",
            retail_df,
            exclude={"Churn"},
        )
        assert "Segment" in groups

    def test_department_grouping(self, hr_df):
        groups = _detect_grouping_variables(
            "compare attrition rates by department",
            hr_df,
            exclude={"Attrition"},
        )
        assert "Department" in groups


# ── PART 5: Full Role Assignment ─────────────────────────────────────────

class TestFullRoleAssignment:
    def test_healthcare_prevalence(self, healthcare_df):
        """Heart disease prevalence between smokers → target=disease, group=Smoker."""
        roles = assign_semantic_roles(
            "Compare heart disease prevalence between smokers and non-smokers.",
            healthcare_df,
        )
        assert roles is not None
        assert roles.operation == "prevalence"
        assert roles.target_variable == "HeartDiseaseorAttack"
        assert "Smoker" in roles.grouping_variables
        assert roles.confidence >= 0.5

    def test_healthcare_association(self, healthcare_df):
        """Risk factors for heart disease → association analysis."""
        roles = assign_semantic_roles(
            "Which combinations of risk factors appear most associated with heart disease?",
            healthcare_df,
        )
        assert roles is not None
        assert roles.operation == "association"
        assert roles.target_variable == "HeartDiseaseorAttack"
        assert len(roles.explanatory_variables) >= 3

    def test_healthcare_visualization(self, healthcare_df):
        """Visualization of heart disease prevalence by age and gender."""
        roles = assign_semantic_roles(
            "Build a visualization of heart disease prevalence by age group and gender.",
            healthcare_df,
        )
        assert roles is not None
        assert roles.target_variable == "HeartDiseaseorAttack"
        assert roles.operation == "prevalence"

    def test_retail_churn(self, retail_df):
        """Churn rate by segment → target=Churn, group=Segment."""
        roles = assign_semantic_roles(
            "Which customer segments have highest churn risk?",
            retail_df,
        )
        assert roles is not None
        assert roles.target_variable == "Churn"

    def test_education_failure(self, education_df):
        """Student failure factors → target=Passed, association."""
        roles = assign_semantic_roles(
            "Which factors appear most associated with student failure?",
            education_df,
        )
        assert roles is not None
        assert roles.operation == "association"
        assert roles.target_variable == "Passed"

    def test_hr_attrition(self, hr_df):
        """Attrition by department → target=Attrition, group=Department."""
        roles = assign_semantic_roles(
            "Compare attrition rates by department.",
            hr_df,
        )
        assert roles is not None
        assert roles.target_variable == "Attrition"
        assert roles.operation == "prevalence"
        assert "Department" in roles.grouping_variables

    def test_generic_question_returns_none(self, movie_df):
        """Generic questions should not trigger role assignment."""
        roles = assign_semantic_roles("Summarize this dataset.", movie_df)
        assert roles is None

    def test_outlier_question_returns_none(self, healthcare_df):
        """Outlier questions should not trigger role assignment."""
        roles = assign_semantic_roles("Find outliers in BMI.", healthcare_df)
        assert roles is None

    def test_missing_values_returns_none(self, movie_df):
        """Data quality questions should not trigger role assignment."""
        roles = assign_semantic_roles("Show missing values.", movie_df)
        assert roles is None

    def test_empty_dataframe_returns_none(self):
        roles = assign_semantic_roles("Compare churn.", pd.DataFrame())
        assert roles is None

    def test_none_dataframe_returns_none(self):
        roles = assign_semantic_roles("Compare churn.", None)
        assert roles is None


# ── PART 6: Prevalence Analysis Response ─────────────────────────────────

class TestPrevalenceAnalysisResponse:
    def test_healthcare_prevalence_response(self, healthcare_df):
        roles = assign_semantic_roles(
            "Compare heart disease prevalence between smokers and non-smokers.",
            healthcare_df,
        )
        result = prevalence_analysis_response(
            "Compare heart disease prevalence between smokers and non-smokers.",
            healthcare_df,
            roles,
        )
        assert result is not None
        summary = result["summary"].lower()
        # Must mention HeartDiseaseorAttack and prevalence
        assert "prevalence" in summary
        assert "heartdiseaseo" in summary.replace(" ", "") or "heart" in summary
        # Must NOT say mean(Smoker) or average(Smoker)
        assert "mean(smoker)" not in summary
        assert "average(smoker)" not in summary
        # Must have artifacts
        assert len(result["artifacts"]) >= 1
        # Trace metadata
        assert result["trace_metadata"]["analysis_type"] == "semantic_role_prevalence"
        assert result["trace_metadata"]["target_variable"] == "HeartDiseaseorAttack"

    def test_churn_prevalence_response(self, retail_df):
        roles = assign_semantic_roles(
            "What is the churn rate by segment?",
            retail_df,
        )
        result = prevalence_analysis_response(
            "What is the churn rate by segment?",
            retail_df,
            roles,
        )
        assert result is not None
        summary = result["summary"].lower()
        assert "churn" in summary
        assert "prevalence" in summary or "rate" in summary or "%" in summary

    def test_attrition_prevalence_response(self, hr_df):
        roles = assign_semantic_roles(
            "Compare attrition rates by department.",
            hr_df,
        )
        result = prevalence_analysis_response(
            "Compare attrition rates by department.",
            hr_df,
            roles,
        )
        assert result is not None
        summary = result["summary"].lower()
        assert "attrition" in summary


# ── PART 7: Association Analysis Response ────────────────────────────────

class TestAssociationAnalysisResponse:
    def test_healthcare_association_response(self, healthcare_df):
        roles = assign_semantic_roles(
            "Which combinations of risk factors appear most associated with heart disease?",
            healthcare_df,
        )
        result = association_analysis_response(
            "Which combinations of risk factors appear most associated with heart disease?",
            healthcare_df,
            roles,
        )
        assert result is not None
        summary = result["summary"].lower()
        # Must reference the target
        assert "heartdiseaseo" in summary.replace(" ", "") or "heart" in summary
        # Must mention association
        assert "association" in summary or "prevalence" in summary or "gap" in summary
        # Must have artifacts
        assert len(result["artifacts"]) >= 1
        # Trace metadata
        assert result["trace_metadata"]["analysis_type"] == "semantic_role_association"

    def test_education_association_response(self, education_df):
        roles = assign_semantic_roles(
            "Which factors appear most associated with student failure?",
            education_df,
        )
        result = association_analysis_response(
            "Which factors appear most associated with student failure?",
            education_df,
            roles,
        )
        assert result is not None
        assert result["trace_metadata"]["target_variable"] == "Passed"


# ── PART 8: Fallback Pipeline Integration ────────────────────────────────

class TestFallbackPipelineIntegration:
    """Verify that the semantic role intercept fires correctly in the fallback pipeline."""

    def test_prevalence_via_fallback(self, healthcare_df):
        """Prevalence question through deterministic_investigation_fallback."""
        result = deterministic_investigation_fallback(
            "Compare heart disease prevalence between smokers and non-smokers.",
            healthcare_df,
        )
        assert result is not None
        analysis_type = (result.get("trace_metadata") or {}).get("analysis_type", "")
        # Must be semantic_role_prevalence, NOT deterministic_pandas or overview
        assert analysis_type == "semantic_role_prevalence", (
            f"Expected 'semantic_role_prevalence' but got '{analysis_type}'"
        )
        summary = result["summary"].lower()
        assert "prevalence" in summary
        assert "mean(smoker)" not in summary

    def test_association_via_fallback(self, healthcare_df):
        """Association question through deterministic_investigation_fallback."""
        result = deterministic_investigation_fallback(
            "Which combinations of risk factors appear most associated with heart disease?",
            healthcare_df,
        )
        assert result is not None
        analysis_type = (result.get("trace_metadata") or {}).get("analysis_type", "")
        assert analysis_type == "semantic_role_association", (
            f"Expected 'semantic_role_association' but got '{analysis_type}'"
        )

    def test_generic_question_not_intercepted(self, healthcare_df):
        """Generic questions should bypass the role assignment intercept."""
        result = deterministic_investigation_fallback(
            "Find outliers in BMI.",
            healthcare_df,
        )
        assert result is not None
        analysis_type = (result.get("trace_metadata") or {}).get("analysis_type", "")
        assert analysis_type != "semantic_role_prevalence"
        assert analysis_type != "semantic_role_association"

    def test_churn_via_fallback(self, retail_df):
        """Churn rate question through fallback."""
        result = deterministic_investigation_fallback(
            "What is the churn rate by segment?",
            retail_df,
        )
        assert result is not None
        analysis_type = (result.get("trace_metadata") or {}).get("analysis_type", "")
        assert analysis_type == "semantic_role_prevalence"


# ── PART 9: Role Confusion Critic ────────────────────────────────────────

class TestRoleConfusionCritic:
    def test_detects_wrong_mean(self, healthcare_df):
        from source.product.grounding_critic import role_confusion_check
        wrong = "Smoker has an average of 0.45 and a median of 0.00."
        refusal = role_confusion_check(
            "Compare heart disease prevalence between smokers and non-smokers.",
            wrong,
            df=healthcare_df,
        )
        assert refusal is not None
        assert "grouping variable" in refusal

    def test_allows_correct_answer(self, healthcare_df):
        from source.product.grounding_critic import role_confusion_check
        correct = "HeartDiseaseorAttack prevalence is 40% among smokers vs 10% among non-smokers."
        refusal = role_confusion_check(
            "Compare heart disease prevalence between smokers and non-smokers.",
            correct,
            df=healthcare_df,
        )
        assert refusal is None

    def test_no_role_assignment_no_refusal(self, movie_df):
        from source.product.grounding_critic import role_confusion_check
        answer = "The dataset has 8 rows and 9 columns."
        refusal = role_confusion_check("Summarize this dataset.", answer, df=movie_df)
        assert refusal is None


# ── PART 10: Stale Context Isolation ─────────────────────────────────────

class TestStaleContextIsolation:
    def test_role_assignment_is_fresh_per_question(self, healthcare_df):
        """Role assignment must be computed fresh for each question, no contamination."""
        roles1 = assign_semantic_roles(
            "Compare heart disease prevalence between smokers.",
            healthcare_df,
        )
        roles2 = assign_semantic_roles(
            "Find outliers in BMI.",
            healthcare_df,
        )
        assert roles1 is not None
        assert roles2 is None  # Generic question → no roles

    def test_different_questions_different_roles(self, healthcare_df):
        roles1 = assign_semantic_roles(
            "Compare heart disease prevalence between smokers.",
            healthcare_df,
        )
        roles2 = assign_semantic_roles(
            "Which factors are most associated with heart disease?",
            healthcare_df,
        )
        assert roles1.operation == "prevalence"
        assert roles2.operation == "association"
        assert roles1.target_variable == roles2.target_variable  # Same target
        assert "Smoker" in roles1.grouping_variables
        assert len(roles2.explanatory_variables) > len(roles1.grouping_variables)


# ── PART 11: No Wrong Metric Contamination ───────────────────────────────

class TestNoWrongMetric:
    @pytest.mark.parametrize("question,target", [
        ("Compare heart disease prevalence between smokers and non-smokers.", "HeartDiseaseorAttack"),
        ("What is the churn rate by segment?", "Churn"),
        ("Compare attrition rates by department.", "Attrition"),
    ])
    def test_target_not_confused_with_grouping(self, question, target, healthcare_df, retail_df, hr_df):
        df_map = {
            "HeartDiseaseorAttack": healthcare_df,
            "Churn": retail_df,
            "Attrition": hr_df,
        }
        df = df_map[target]
        roles = assign_semantic_roles(question, df)
        assert roles is not None
        assert roles.target_variable == target
        # Target must not appear in grouping variables
        assert target not in roles.grouping_variables


# ── PART 12: Multi-Dataset Support ───────────────────────────────────────

class TestMultiDatasetRoles:
    def test_role_assignment_per_dataset(self, healthcare_df, movie_df):
        """Role assignment must run independently per dataset."""
        q = "Compare heart disease prevalence between smokers and non-smokers."
        roles_health = assign_semantic_roles(q, healthcare_df)
        roles_movie = assign_semantic_roles(q, movie_df)

        assert roles_health is not None
        assert roles_health.target_variable == "HeartDiseaseorAttack"
        # Movie dataset has no binary outcome column → no role assignment
        assert roles_movie is None or roles_movie.confidence < 0.5


# ── PART 13: Hardcode Audit ──────────────────────────────────────────────

def test_no_hardcoded_names_in_role_engine():
    """No dataset-specific names in the semantic role engine."""
    import inspect
    from source.product import semantic_role_assignment

    source = inspect.getsource(semantic_role_assignment)
    for forbidden in (
        "HeartDiseaseorAttack", "Smoker", "BMI", "Age",
        "Netflix", "Superstore", "Movie", "TV Show",
    ):
        assert forbidden not in source, f"'{forbidden}' found in semantic_role_assignment"
