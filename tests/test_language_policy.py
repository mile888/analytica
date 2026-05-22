from __future__ import annotations

import pandas as pd

from source.product.fallback_analysis import deterministic_investigation_fallback
from source.product.hypothesis_reasoning import HypothesisEngine
from source.product.language_policy import DetectedLanguage, ResponseLanguagePolicy
from source.product.question_suggestions import build_thread_aware_question_suggestions
from source.product.semantic_layer import build_semantic_dataset_profile
from source.product.transformation_engine import run_transformation_analysis


def test_russian_query_uses_authoritative_english_policy() -> None:
    policy = ResponseLanguagePolicy.from_message(
        "Построй график top cities by sales",
        protected_terms=["City", "Sales"],
    )

    assert policy.language == DetectedLanguage.ENGLISH


def test_detects_english_main_language_with_russian_fragment() -> None:
    policy = ResponseLanguagePolicy.from_message(
        "Show trend по месяцам for Sales",
        protected_terms=["Sales"],
    )

    assert policy.language == DetectedLanguage.ENGLISH


def test_russian_chart_request_gets_english_answer_and_preserves_columns() -> None:
    df = pd.DataFrame(
        {
            "Sales": [100.0, 300.0, 50.0, 200.0],
            "City": ["Jamestown", "Cheyenne", "Boston", "Bellingham"],
        }
    )

    result = deterministic_investigation_fallback("Построй график top cities by sales", df)
    answer = result["final_answer"]

    assert answer.startswith("`Sales` by `City` is led")
    assert "total `Sales`" in answer
    assert "`City`" in answer
    assert "`Sales`" in answer
    assert "Я сравнил" not in answer


def test_english_query_with_russian_fragment_gets_english_answer() -> None:
    df = pd.DataFrame(
        {
            "Sales": [100.0, 300.0, 50.0, 200.0],
            "City": ["Jamestown", "Cheyenne", "Boston", "Bellingham"],
        }
    )

    result = deterministic_investigation_fallback("Show top cities по Sales", df)
    answer = result["final_answer"]

    assert answer.startswith("`Sales` by `City` is led")
    assert "total `Sales`" in answer
    assert "`City`" in answer
    assert "`Sales`" in answer
    assert "Я сравнил" not in answer


def test_suggested_questions_are_english_even_after_russian_user_language() -> None:
    suggestions = build_thread_aware_question_suggestions(
        {
            "active_metric": "Sales",
            "active_dimension": "City",
            "active_business_question": "Построй график top cities by sales",
            "recent_findings": [],
            "suggested_next_questions": [],
        },
        response_language="ru",
    )

    assert suggestions
    assert suggestions[0].startswith("Does")
    assert "`Sales`" in suggestions[0]
    assert "`City`" in suggestions[0]


def test_hypothesis_answer_is_english_for_russian_user_language() -> None:
    df = pd.DataFrame(
        {
            "Ship Mode": ["Standard Class", "Standard Class", "Same Day"],
            "Sales": [100.0, 110.0, 250.0],
        }
    )
    semantic = build_semantic_dataset_profile(df=df)

    result = HypothesisEngine.validate(
        question="Гипотеза: Standard Class доминирует из-за volume, а не order value.",
        dataframe=df,
        branch_state={},
        semantic_profile=semantic,
    )
    answer = result["final_answer"]

    assert answer.startswith("The hypothesis")
    assert "`Standard Class`" in answer
    assert "`Ship Mode`" in answer
    assert "Гипотеза" not in answer


def test_transformation_followup_is_english_for_russian_user_language() -> None:
    df = pd.DataFrame(
        {
            "Sales": [100.0, 1000.0, 110.0, 120.0],
            "City": ["A", "A", "B", "B"],
        }
    )

    result = run_transformation_analysis(
        question="А если убрать extreme orders?",
        df=df,
        metric_col="Sales",
        dimension_col="City",
    )
    answer = result["final_answer"]

    assert answer.startswith("After removing")
    assert "`Sales`" in answer
    assert "`City`" in answer
    assert "После удаления" not in answer


def test_chart_title_is_english_for_russian_language() -> None:
    df = pd.DataFrame(
        {
            "Sales": [100.0, 300.0, 50.0],
            "City": ["A", "B", "C"],
        }
    )

    result = deterministic_investigation_fallback("Построй график top cities by sales", df)
    chart = next(artifact for artifact in result["artifacts"] if artifact["artifact_type"] == "chart")

    assert chart["title"] == "Sales by City"
    assert "Средний" not in chart["title"]
