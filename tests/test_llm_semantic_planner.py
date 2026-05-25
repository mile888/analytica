"""Tests for the LLM Semantic Planner pipeline.

All tests use mocked LLM outputs — no real API calls required.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from source.product.llm_semantic_planner import (
    SemanticPlan,
    build_schema_context,
    plan_analysis,
    _parse_plan,
    _extract_json,
    ALLOWED_OPERATIONS,
)
from source.product.plan_validator import (
    validate_plan,
    ValidatedPlan,
    PlanRejection,
)
from source.product.plan_executor import (
    execute_plan,
    EvidencePackage,
)
from source.product.grounded_synthesis import (
    synthesize,
    mechanical_synthesis,
    SynthesisResult,
)
from source.product.grounding_critic import (
    validate_synthesis,
    CriticResult,
    _extract_numbers,
)


# ── Test fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def entertainment_df():
    return pd.DataFrame({
        "title": [f"Movie {i}" for i in range(50)],
        "listed_in": ["Drama, Comedy"] * 15 + ["Action"] * 10 + ["Documentary"] * 10 + ["Horror, Thriller"] * 10 + ["Sci-Fi"] * 5,
        "release_year": list(range(2010, 2020)) * 5,
        "duration": ["90 min", "120 min", "95 min", "110 min", "88 min"] * 10,
        "country": ["US"] * 20 + ["UK"] * 10 + ["India"] * 10 + ["Japan"] * 5 + ["France"] * 5,
        "rating": ["PG-13"] * 15 + ["R"] * 15 + ["PG"] * 10 + ["G"] * 10,
    })


@pytest.fixture
def healthcare_df():
    return pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(60)],
        "diagnosis": ["Heart Disease"] * 15 + ["Diabetes"] * 15 + ["Hypertension"] * 10 + ["Asthma"] * 10 + ["Cancer"] * 10,
        "age": [45, 67, 34, 55, 72, 41, 63, 38, 50, 61] * 6,
        "treatment_cost": [5000, 12000, 3500, 8500, 15000, 4200, 9800, 2800, 7400, 11000] * 6,
        "hospital": ["City General"] * 20 + ["Memorial"] * 15 + ["St. Mary"] * 15 + ["University"] * 10,
        "smoker": ["Yes"] * 25 + ["No"] * 35,
    })


@pytest.fixture
def retail_df():
    return pd.DataFrame({
        "order_id": [f"O{i:04d}" for i in range(80)],
        "city": ["New York"] * 20 + ["Los Angeles"] * 15 + ["Chicago"] * 15 + ["Houston"] * 15 + ["Phoenix"] * 15,
        "category": ["Technology"] * 20 + ["Furniture"] * 20 + ["Office Supplies"] * 20 + ["Clothing"] * 20,
        "sales": [250.0, 180.0, 420.0, 95.0, 310.0, 540.0, 160.0, 380.0] * 10,
        "profit": [45.0, -12.0, 88.0, 15.0, 62.0, -25.0, 34.0, 71.0] * 10,
        "discount": [0.0, 0.1, 0.2, 0.3, 0.0, 0.15, 0.25, 0.05] * 10,
    })


@pytest.fixture
def scientific_df():
    return pd.DataFrame({
        "species": ["Oak"] * 15 + ["Pine"] * 12 + ["Maple"] * 10 + ["Birch"] * 8 + ["Cedar"] * 5,
        "habitat": ["Forest"] * 20 + ["Wetland"] * 10 + ["Mountain"] * 10 + ["Grassland"] * 10,
        "population": [1200, 850, 2100, 450, 3200, 780, 1500, 620, 900, 1100] * 5,
        "region": ["North"] * 15 + ["South"] * 12 + ["East"] * 13 + ["West"] * 10,
        "observation_year": [2018, 2019, 2020, 2021, 2022] * 10,
    })


def _mock_llm_plan(plan_dict: dict) -> MagicMock:
    """Create a mock LLM that returns the given plan as JSON."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(plan_dict)
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Schema context builder
# ═══════════════════════════════════════════════════════════════════════════

class TestSchemaContext:
    def test_identifies_numeric_columns(self, entertainment_df):
        schema = build_schema_context(entertainment_df)
        assert "release_year" in schema.numeric_columns

    def test_identifies_categorical_columns(self, entertainment_df):
        schema = build_schema_context(entertainment_df)
        assert "listed_in" in schema.categorical_columns
        assert "country" in schema.categorical_columns

    def test_identifies_year_like_columns(self, entertainment_df):
        schema = build_schema_context(entertainment_df)
        assert "release_year" in schema.year_like_columns

    def test_identifies_duration_columns(self, entertainment_df):
        schema = build_schema_context(entertainment_df)
        assert "duration" in schema.duration_like_columns

    def test_identifies_multi_label_columns(self, entertainment_df):
        schema = build_schema_context(entertainment_df)
        assert "listed_in" in schema.multi_label_columns

    def test_identifies_identifier_columns(self, healthcare_df):
        schema = build_schema_context(healthcare_df)
        assert "patient_id" in schema.identifier_columns

    def test_row_count(self, entertainment_df):
        schema = build_schema_context(entertainment_df)
        assert schema.row_count == 50


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Plan parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanParsing:
    def test_parses_valid_json(self, entertainment_df):
        schema = build_schema_context(entertainment_df)
        plan_json = json.dumps({
            "operation": "COUNT_DISTRIBUTION",
            "dimension": "listed_in",
            "aggregation": "count",
            "artifact_type": "bar",
            "needs_execution": True,
            "reasoning": "Count genres",
            "confidence": 0.9,
        })
        plan = _parse_plan(plan_json, schema)
        assert plan is not None
        assert plan.operation == "COUNT_DISTRIBUTION"
        assert plan.dimension == "listed_in"
        assert plan.confidence == 0.9

    def test_parses_json_in_markdown_fence(self, entertainment_df):
        schema = build_schema_context(entertainment_df)
        text = "```json\n" + json.dumps({
            "operation": "COUNT_DISTRIBUTION",
            "dimension": "country",
            "aggregation": "count",
            "confidence": 0.8,
        }) + "\n```"
        plan = _parse_plan(text, schema)
        assert plan is not None
        assert plan.operation == "COUNT_DISTRIBUTION"

    def test_rejects_invalid_operation(self, entertainment_df):
        schema = build_schema_context(entertainment_df)
        plan = _parse_plan(json.dumps({"operation": "INVALID_OP"}), schema)
        assert plan is None

    def test_rejects_garbage_text(self, entertainment_df):
        schema = build_schema_context(entertainment_df)
        plan = _parse_plan("This is not JSON at all", schema)
        assert plan is None

    def test_normalizes_aggregation(self, entertainment_df):
        schema = build_schema_context(entertainment_df)
        plan = _parse_plan(json.dumps({
            "operation": "COUNT_DISTRIBUTION",
            "aggregation": "INVALID_AGG",
            "confidence": 0.8,
        }), schema)
        assert plan is not None
        assert plan.aggregation == "count"  # defaults to count


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: Plan validator
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanValidator:
    def test_valid_count_plan(self, entertainment_df):
        plan = SemanticPlan(
            operation="COUNT_DISTRIBUTION",
            dimension="listed_in",
            aggregation="count",
            confidence=0.9,
        )
        result = validate_plan(plan, entertainment_df)
        assert isinstance(result, ValidatedPlan)

    def test_rejects_nonexistent_column(self, entertainment_df):
        plan = SemanticPlan(
            operation="COUNT_DISTRIBUTION",
            dimension="genre",  # doesn't exist
            aggregation="count",
            confidence=0.9,
        )
        result = validate_plan(plan, entertainment_df)
        assert isinstance(result, PlanRejection)
        assert any("genre" in r for r in result.reasons)

    def test_rejects_year_as_metric(self, entertainment_df):
        plan = SemanticPlan(
            operation="METRIC_AGGREGATION",
            metric="release_year",
            dimension="listed_in",
            aggregation="mean",
            confidence=0.9,
        )
        result = validate_plan(plan, entertainment_df)
        assert isinstance(result, ValidatedPlan)
        # Should repair: change to count
        assert result.plan.aggregation == "count"
        assert len(result.repairs) > 0

    def test_rejects_id_as_metric(self, healthcare_df):
        plan = SemanticPlan(
            operation="METRIC_AGGREGATION",
            metric="patient_id",
            dimension="diagnosis",
            aggregation="sum",
            confidence=0.9,
        )
        result = validate_plan(plan, healthcare_df)
        assert isinstance(result, ValidatedPlan)
        # Should repair: remove identifier metric
        assert result.plan.metric is None
        assert result.plan.aggregation == "count"

    def test_fixes_column_case(self, entertainment_df):
        plan = SemanticPlan(
            operation="COUNT_DISTRIBUTION",
            dimension="Listed_In",  # wrong case
            aggregation="count",
            confidence=0.9,
        )
        result = validate_plan(plan, entertainment_df)
        assert isinstance(result, ValidatedPlan)
        assert result.plan.dimension == "listed_in"
        assert len(result.repairs) > 0

    def test_valid_metric_aggregation(self, retail_df):
        plan = SemanticPlan(
            operation="METRIC_AGGREGATION",
            metric="sales",
            dimension="city",
            aggregation="sum",
            confidence=0.9,
        )
        result = validate_plan(plan, retail_df)
        assert isinstance(result, ValidatedPlan)
        assert result.plan.metric == "sales"

    def test_rejects_nonnumeric_metric_for_mean(self, entertainment_df):
        plan = SemanticPlan(
            operation="METRIC_AGGREGATION",
            metric="country",  # not numeric
            dimension="listed_in",
            aggregation="mean",
            confidence=0.9,
        )
        result = validate_plan(plan, entertainment_df)
        assert isinstance(result, PlanRejection)


# ═══════════════════════════════════════════════════════════════════════════
# Part 4: Plan executor
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanExecutor:
    def test_count_distribution(self, entertainment_df):
        plan = SemanticPlan(
            operation="COUNT_DISTRIBUTION",
            dimension="country",
            aggregation="count",
            confidence=0.9,
        )
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, entertainment_df, question="Which countries?")
        assert evidence.record_count == 50
        assert len(evidence.computed_tables) > 0
        assert evidence.computed_tables[0]["rows"][0]["value"] == "US"

    def test_metric_aggregation(self, retail_df):
        plan = SemanticPlan(
            operation="METRIC_AGGREGATION",
            metric="sales",
            dimension="city",
            aggregation="sum",
            confidence=0.9,
        )
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, retail_df, question="Sales by city")
        assert len(evidence.computed_tables) > 0
        rows = evidence.computed_tables[0]["rows"]
        assert all("sum" in row for row in rows)

    def test_category_mix_shift(self, entertainment_df):
        plan = SemanticPlan(
            operation="CATEGORY_MIX_SHIFT",
            time_field="release_year",
            category_field="listed_in",
            confidence=0.8,
        )
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, entertainment_df, question="How did genres change?")
        assert len(evidence.computed_tables) > 0
        table = evidence.computed_tables[0]
        assert "threshold" in table
        assert "shifts" in table

    def test_diversity_analysis(self, scientific_df):
        plan = SemanticPlan(
            operation="DIVERSITY_ANALYSIS",
            dimension="region",
            category_field="species",
            confidence=0.8,
        )
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, scientific_df, question="Species diversity by region")
        assert len(evidence.computed_tables) > 0

    def test_duration_analysis(self, entertainment_df):
        plan = SemanticPlan(
            operation="DURATION_ANALYSIS",
            duration_field="duration",
            category_field="listed_in",
            confidence=0.8,
        )
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, entertainment_df, question="Duration by genre")
        assert len(evidence.computed_tables) > 0

    def test_strategic_synthesis(self, retail_df):
        plan = SemanticPlan(
            operation="STRATEGIC_SYNTHESIS",
            category_field="category",
            confidence=0.7,
        )
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, retail_df, question="Business strategy")
        assert len(evidence.computed_tables) > 0
        assert "dataset_overview" in evidence.summary_statistics

    def test_outlier_analysis(self, retail_df):
        plan = SemanticPlan(
            operation="OUTLIER_ANALYSIS",
            metric="profit",
            confidence=0.8,
        )
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, retail_df, question="Find outliers")
        assert "outlier_count" in evidence.summary_statistics


# ═══════════════════════════════════════════════════════════════════════════
# Part 5: Grounded synthesis
# ═══════════════════════════════════════════════════════════════════════════

class TestGroundedSynthesis:
    def test_mechanical_synthesis_from_count(self, entertainment_df):
        plan = SemanticPlan(operation="COUNT_DISTRIBUTION", dimension="country", aggregation="count", confidence=0.9)
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, entertainment_df, question="Which countries?")
        result = mechanical_synthesis("Which countries?", evidence)
        assert result.answer
        assert "US" in result.answer
        assert len(result.findings) > 0

    def test_mechanical_synthesis_from_aggregation(self, retail_df):
        plan = SemanticPlan(operation="METRIC_AGGREGATION", metric="sales", dimension="city", aggregation="sum", confidence=0.9)
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, retail_df, question="Sales by city")
        result = mechanical_synthesis("Sales by city", evidence)
        assert result.answer
        assert len(result.findings) > 0

    def test_synthesis_with_mock_llm(self, entertainment_df):
        plan = SemanticPlan(operation="COUNT_DISTRIBUTION", dimension="country", aggregation="count", confidence=0.9)
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, entertainment_df, question="Which countries?")

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "answer": "The US dominates with 20 titles (40% of total).",
            "findings": [{"title": "US dominance", "evidence": "20 titles, 40% share", "importance": "high"}],
            "limitations": ["Only top 5 shown"],
            "next_steps": ["Analyze trends over time"],
        })
        mock_llm = MagicMock(return_value=mock_response)
        mock_llm.invoke = MagicMock(return_value=mock_response)

        with patch("source.llm.factory.make_llm", return_value=mock_llm):
            result = synthesize("Which countries?", evidence)
        assert "US" in result.answer
        assert len(result.findings) > 0

    def test_synthesis_fallback_on_llm_error(self, entertainment_df):
        plan = SemanticPlan(operation="COUNT_DISTRIBUTION", dimension="country", aggregation="count", confidence=0.9)
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, entertainment_df, question="Which countries?")

        with patch("source.llm.factory.make_llm", side_effect=RuntimeError("No API key")):
            result = synthesize("Which countries?", evidence)
        # Should fall back to mechanical synthesis
        assert result.answer
        assert "US" in result.answer


# ═══════════════════════════════════════════════════════════════════════════
# Part 6: Grounding critic
# ═══════════════════════════════════════════════════════════════════════════

class TestGroundingCritic:
    def test_passes_grounded_synthesis(self, entertainment_df):
        plan = SemanticPlan(operation="COUNT_DISTRIBUTION", dimension="country", aggregation="count", confidence=0.9)
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, entertainment_df, question="Which countries?")
        synthesis = mechanical_synthesis("Which countries?", evidence)
        result = validate_synthesis(synthesis, evidence)
        assert result.passed

    def test_detects_invented_number(self, entertainment_df):
        plan = SemanticPlan(operation="COUNT_DISTRIBUTION", dimension="country", aggregation="count", confidence=0.9)
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, entertainment_df, question="Which countries?")
        synthesis = SynthesisResult(
            answer="The US has 99999 titles",  # invented number
            findings=[],
        )
        result = validate_synthesis(synthesis, evidence)
        assert not result.passed
        assert len(result.issues) > 0

    def test_detects_hallucination(self, entertainment_df):
        plan = SemanticPlan(operation="COUNT_DISTRIBUTION", dimension="country", aggregation="count", confidence=0.9)
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, entertainment_df, question="Which countries?")
        synthesis = SynthesisResult(
            answer="Based on my domain knowledge, historically speaking, the US dominates with 20 titles",
            findings=[],
        )
        result = validate_synthesis(synthesis, evidence)
        assert result.has_critical_failures

    def test_number_extraction(self):
        numbers = _extract_numbers("The top 3 categories account for 45.5% of 1200 records")
        assert 3.0 in numbers
        assert 45.5 in numbers
        assert 1200.0 in numbers


# ═══════════════════════════════════════════════════════════════════════════
# Part 7: Full pipeline (end-to-end with mock LLM)
# ═══════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    def _run_pipeline(self, question, df, plan_dict, synthesis_dict=None):
        """Run the full pipeline with mocked LLM."""
        mock_responses = []

        # First call: planner
        planner_response = MagicMock()
        planner_response.content = json.dumps(plan_dict)
        mock_responses.append(planner_response)

        # Second call: synthesis (optional)
        if synthesis_dict:
            synth_response = MagicMock()
            synth_response.content = json.dumps(synthesis_dict)
            mock_responses.append(synth_response)

        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(side_effect=mock_responses)

        with patch("source.product.llm_semantic_planner.make_llm", return_value=mock_llm):
            plan = plan_analysis(question, df)

        assert plan is not None
        validated = validate_plan(plan, df)
        assert isinstance(validated, ValidatedPlan)
        evidence = execute_plan(validated, df, question=question)

        if synthesis_dict:
            with patch("source.llm.factory.make_llm", return_value=mock_llm):
                synthesis = synthesize(question, evidence)
        else:
            synthesis = mechanical_synthesis(question, evidence)

        critic = validate_synthesis(synthesis, evidence)
        return plan, validated, evidence, synthesis, critic

    def test_entertainment_genre_count(self, entertainment_df):
        plan_dict = {
            "operation": "COUNT_DISTRIBUTION",
            "dimension": "listed_in",
            "aggregation": "count",
            "artifact_type": "bar",
            "confidence": 0.9,
            "reasoning": "User asks which genres dominate",
        }
        plan, validated, evidence, synthesis, critic = self._run_pipeline(
            "Which genres dominate?", entertainment_df, plan_dict
        )
        assert plan.operation == "COUNT_DISTRIBUTION"
        assert evidence.computed_tables
        assert synthesis.findings
        assert critic.passed or not critic.has_critical_failures

    def test_healthcare_diagnosis_count(self, healthcare_df):
        plan_dict = {
            "operation": "COUNT_DISTRIBUTION",
            "dimension": "diagnosis",
            "aggregation": "count",
            "confidence": 0.9,
            "reasoning": "User asks about common diagnoses",
        }
        plan, validated, evidence, synthesis, critic = self._run_pipeline(
            "Which diagnoses are most common?", healthcare_df, plan_dict
        )
        assert "Heart Disease" in str(evidence.computed_tables) or "Diabetes" in str(evidence.computed_tables)
        assert synthesis.findings

    def test_retail_sales_by_city(self, retail_df):
        plan_dict = {
            "operation": "METRIC_AGGREGATION",
            "metric": "sales",
            "dimension": "city",
            "aggregation": "sum",
            "confidence": 0.9,
            "reasoning": "User wants sales by city",
        }
        plan, validated, evidence, synthesis, critic = self._run_pipeline(
            "Which cities have highest sales?", retail_df, plan_dict
        )
        assert evidence.computed_tables
        assert any("sum" in str(row) for row in evidence.computed_tables[0]["rows"])

    def test_scientific_species_by_habitat(self, scientific_df):
        plan_dict = {
            "operation": "COUNT_DISTRIBUTION",
            "dimension": "species",
            "aggregation": "count",
            "confidence": 0.9,
            "reasoning": "User asks about species abundance",
        }
        plan, validated, evidence, synthesis, critic = self._run_pipeline(
            "What species are most abundant?", scientific_df, plan_dict
        )
        assert evidence.computed_tables
        assert synthesis.findings

    def test_invalid_plan_falls_back(self, entertainment_df):
        """LLM proposes invalid column → validator rejects."""
        plan_dict = {
            "operation": "COUNT_DISTRIBUTION",
            "dimension": "genre",  # doesn't exist
            "aggregation": "count",
            "confidence": 0.9,
        }
        mock_llm = _mock_llm_plan(plan_dict)
        with patch("source.product.llm_semantic_planner.make_llm", return_value=mock_llm):
            plan = plan_analysis("Which genres?", entertainment_df)
        assert plan is not None
        result = validate_plan(plan, entertainment_df)
        assert isinstance(result, PlanRejection)

    def test_llm_unavailable_returns_none(self, entertainment_df):
        """LLM is unavailable → plan_analysis returns None."""
        with patch("source.product.llm_semantic_planner.make_llm", side_effect=RuntimeError("No key")):
            plan = plan_analysis("Which genres?", entertainment_df)
        assert plan is None


# ═══════════════════════════════════════════════════════════════════════════
# Part 8: Findings pipeline (adapter integration)
# ═══════════════════════════════════════════════════════════════════════════

class TestFindingsPipeline:
    def test_findings_pass_through_adapter(self, entertainment_df):
        """Findings from semantic planner output pass through adapter."""
        from source.product.adapter import agent_output_to_investigation_update

        plan = SemanticPlan(operation="COUNT_DISTRIBUTION", dimension="country", aggregation="count", confidence=0.9)
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, entertainment_df, question="Which countries?")
        synthesis = mechanical_synthesis("Which countries?", evidence)

        # Build output dict in the same format as _build_semantic_planner_output
        key_findings = []
        for finding in synthesis.findings:
            if isinstance(finding, dict):
                title = finding.get("title", "")
                ev = finding.get("evidence", "")
                key_findings.append(f"{title}: {ev}" if ev else title)

        output = {
            "summary": synthesis.answer,
            "final_answer": synthesis.answer,
            "structured_report": {
                "question": "Which countries?",
                "summary": synthesis.answer,
                "key_findings": key_findings,
                "evidence": ["Computed from 50 records"],
                "limitations": synthesis.limitations,
                "next_steps": synthesis.next_steps,
            },
            "key_findings": key_findings,
            "trace_metadata": {"analysis_type": "count_distribution", "semantic_planner": True},
            "tool_timeline": [{"tool": "semantic_planner", "status": "ok"}],
            "artifacts": [],
        }

        update = agent_output_to_investigation_update(output)
        # Must have findings
        assert len(update.findings) > 0
        # Must have a report
        assert update.report is not None
        assert update.report.key_findings

    def test_invariant_valid_analysis_has_finding(self, entertainment_df):
        """Valid analysis → at least one finding."""
        plan = SemanticPlan(operation="COUNT_DISTRIBUTION", dimension="country", aggregation="count", confidence=0.9)
        validated = ValidatedPlan(plan=plan)
        evidence = execute_plan(validated, entertainment_df, question="Which countries?")
        synthesis = mechanical_synthesis("Which countries?", evidence)
        assert len(synthesis.findings) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Part 9: Service integration (with mock)
# ═══════════════════════════════════════════════════════════════════════════

class TestServiceIntegration:
    def test_try_semantic_planner_returns_output(self, entertainment_df):
        """_try_semantic_planner produces valid output with mocked LLM."""
        plan_dict = {
            "operation": "COUNT_DISTRIBUTION",
            "dimension": "country",
            "aggregation": "count",
            "confidence": 0.9,
            "reasoning": "Count by country",
        }

        mock_planner_llm = _mock_llm_plan(plan_dict)
        synth_response = MagicMock()
        synth_response.content = json.dumps({
            "answer": "US leads with 20 titles.",
            "findings": [{"title": "US leads", "evidence": "20 titles", "importance": "high"}],
        })
        mock_synth_llm = MagicMock()
        mock_synth_llm.invoke = MagicMock(return_value=synth_response)

        with patch("source.product.llm_semantic_planner.make_llm", return_value=mock_planner_llm), \
             patch("source.llm.factory.make_llm", return_value=mock_synth_llm), \
             patch("source.product.service.SEMANTIC_PLANNER_MODE", "llm_first"):
            from source.product.service import _try_semantic_planner
            result = _try_semantic_planner("Which countries dominate?", entertainment_df, {})

        assert result is not None
        assert result["summary"]
        assert result["key_findings"]
        assert result["trace_metadata"]["semantic_planner"] is True

    def test_try_semantic_planner_returns_none_deterministic_mode(self, entertainment_df):
        """In deterministic_first mode, planner returns None."""
        with patch("source.product.service.SEMANTIC_PLANNER_MODE", "deterministic_first"):
            from source.product.service import _try_semantic_planner
            result = _try_semantic_planner("Which countries?", entertainment_df, {})
        assert result is None

    def test_try_semantic_planner_returns_none_on_llm_error(self, entertainment_df):
        """When LLM fails, planner returns None (graceful fallback)."""
        with patch("source.product.llm_semantic_planner.make_llm", side_effect=RuntimeError("No key")), \
             patch("source.product.service.SEMANTIC_PLANNER_MODE", "llm_first"):
            from source.product.service import _try_semantic_planner
            result = _try_semantic_planner("Which genres?", entertainment_df, {})
        assert result is None


class TestArchitectureGuards:
    """Verify that the LLM semantic planner output is not overridden by legacy heuristics."""

    def test_is_semantic_planner_output_detects_trace(self):
        """_is_semantic_planner_output returns True for planner output."""
        from source.product.service import _is_semantic_planner_output
        output = {"trace_metadata": {"semantic_planner": True, "analysis_type": "count_distribution"}}
        assert _is_semantic_planner_output(output) is True

    def test_is_semantic_planner_output_rejects_non_planner(self):
        """_is_semantic_planner_output returns False for non-planner output."""
        from source.product.service import _is_semantic_planner_output
        output = {"trace_metadata": {"analysis_type": "grouped_analysis"}}
        assert _is_semantic_planner_output(output) is False

    def test_is_semantic_planner_output_handles_none(self):
        """_is_semantic_planner_output handles None/missing trace."""
        from source.product.service import _is_semantic_planner_output
        assert _is_semantic_planner_output(None) is False
        assert _is_semantic_planner_output({}) is False
        assert _is_semantic_planner_output({"trace_metadata": None}) is False

    def test_apply_llm_reasoning_skipped_for_planner_output(self):
        """apply_llm_reasoning_layer should not run when output has semantic_planner trace."""
        from source.product.service import _is_semantic_planner_output
        planner_output = {
            "summary": "Drama leads with 42% share.",
            "trace_metadata": {"semantic_planner": True, "analysis_type": "count_distribution"},
        }
        assert _is_semantic_planner_output(planner_output) is True

    def test_overview_replacement_skipped_for_planner_output(self):
        """_should_replace_generic_overview should not run when output has semantic_planner trace."""
        from source.product.service import _is_semantic_planner_output
        planner_output = {
            "summary": "Drama leads with 42% share.",
            "trace_metadata": {"semantic_planner": True},
        }
        assert _is_semantic_planner_output(planner_output) is True

    def test_quality_gate_forced_valid_for_planner_output(self):
        """Quality gate must not discard validated LLM planner output."""
        from source.product.service import _is_semantic_planner_output
        planner_output = {
            "summary": "Analysis complete.",
            "trace_metadata": {"semantic_planner": True},
        }
        assert _is_semantic_planner_output(planner_output) is True

    def test_old_vocabulary_cannot_override_validated_plan(self, entertainment_df):
        """Vocabulary lists in semantic_layer should not contain dataset-specific column names."""
        from source.product.semantic_layer import _dimension_semantic_groups, _metric_semantic_groups

        dim_groups = _dimension_semantic_groups()
        metric_groups = _metric_semantic_groups()

        all_dim_terms = [term for group in dim_groups.values() for term in group]
        all_metric_terms = [term for group in metric_groups.values() for term in group]

        assert "listed_in" not in all_dim_terms
        assert "listed" not in all_dim_terms
        assert "listed in" not in all_dim_terms
        assert "movies" not in all_metric_terms
        assert "shows" not in all_metric_terms
        assert "episodes" not in all_metric_terms
        assert "patients" not in all_metric_terms
        assert "cases" not in all_metric_terms

    def test_semantic_resolution_no_dataset_columns(self):
        """SEMANTIC_CONCEPT_ALIASES should not contain dataset-specific column names."""
        from source.product.fallbacks.semantic_resolution import SEMANTIC_CONCEPT_ALIASES

        for concept, aliases in SEMANTIC_CONCEPT_ALIASES.items():
            assert "listed_in" not in aliases, f"listed_in found in {concept} aliases"

    def test_llm_reasoning_no_filler_templates(self):
        """_local_grounded_synthesis should not contain hardcoded business filler."""
        import inspect
        from source.product.llm_reasoning import _local_grounded_synthesis

        source = inspect.getsource(_local_grounded_synthesis)
        filler_phrases = [
            "optimizing the wrong lever",
            "highest decision leverage",
            "strong enough for prioritization",
            "optimize around the wrong demand signal",
            "customer-behavior concern is uneven participation",
            "revenue stability",
        ]
        for phrase in filler_phrases:
            assert phrase not in source, f"Filler phrase '{phrase}' still in _local_grounded_synthesis"
