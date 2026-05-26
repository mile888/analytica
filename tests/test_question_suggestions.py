from __future__ import annotations

from source.product.data_profiling import profile_dataframe
from source.product.data_sources import ColumnSemanticNote, ColumnSemanticRole, DataSourceSemanticNotes
from source.product.question_suggestions import (
    build_data_aware_question_suggestions,
    build_generic_question_suggestions,
    build_thread_aware_question_suggestions,
)
from source.product.semantic_layer import build_investigation_thread_state

import pandas as pd


def test_generic_suggestions_do_not_use_dataset_specific_business_examples() -> None:
    text = "\n".join(build_generic_question_suggestions()).lower()

    for forbidden in ["sales", "revenue", "superstore", "ship mode", "customer segment", "profit by region"]:
        assert forbidden not in text


def test_generic_fallback_works_without_profile() -> None:
    suggestions = build_data_aware_question_suggestions(limit=3)

    assert suggestions == build_generic_question_suggestions()[:3]


def test_numeric_columns_produce_numeric_suggestions() -> None:
    profile = profile_dataframe(pd.DataFrame({"metric_value": [1.0, 2.0, 10.0]}))

    suggestions = build_data_aware_question_suggestions(profile=profile)
    joined = "\n".join(suggestions)

    assert "`metric_value`" in joined
    assert "highest and lowest" in joined
    assert "outliers" in joined


def test_categorical_columns_produce_grouping_suggestions() -> None:
    profile = profile_dataframe(pd.DataFrame({"category_label": ["A", "B", "A"], "metric_value": [1, 2, 3]}))

    suggestions = build_data_aware_question_suggestions(profile=profile)
    joined = "\n".join(suggestions)

    assert "`category_label`" in joined
    assert "distributed" in joined or "vary across" in joined


def test_date_columns_produce_time_suggestions() -> None:
    profile = profile_dataframe(pd.DataFrame({"event_date": ["2026-01-01", "2026-01-02"], "metric_value": [1, 2]}))

    suggestions = build_data_aware_question_suggestions(profile=profile)
    joined = "\n".join(suggestions)

    assert "`event_date`" in joined
    assert "over time" in joined


def test_semantic_notes_affect_display_wording_and_roles() -> None:
    profile = profile_dataframe(pd.DataFrame({"raw_score": ["low", "high", "low"], "category_label": ["A", "B", "A"]}))
    notes = DataSourceSemanticNotes(
        data_source_id="ds_test",
        column_notes=[
            ColumnSemanticNote(
                column_name="raw_score",
                display_name="Outcome score",
                semantic_role=ColumnSemanticRole.TARGET,
            )
        ],
    )

    suggestions = build_data_aware_question_suggestions(profile=profile, semantic_notes=notes)
    joined = "\n".join(suggestions)

    assert "`Outcome score` (`raw_score`)" in joined
    assert "How does" in joined


def test_duplicate_suggestions_removed_and_limit_respected() -> None:
    profile = profile_dataframe(pd.DataFrame({"metric_value": [1, 2, 3]}))

    suggestions = build_data_aware_question_suggestions(profile=profile, limit=4)

    assert len(suggestions) == 4
    assert len({suggestion.lower() for suggestion in suggestions}) == len(suggestions)


def test_thread_aware_suggestions_continue_latest_chart_topic() -> None:
    thread = build_investigation_thread_state(
        {
            "latest_chart_context": {
                "metric": "Sales",
                "dimension": "City",
                "chart_type": "bar",
                "title": "Sales by City",
            },
            "latest_findings": ["Sales are concentrated in a few cities."],
        }
    )

    suggestions = build_thread_aware_question_suggestions(thread, limit=4)
    joined = "\n".join(suggestions)

    assert "Sales" in joined
    assert "City" in joined
    assert "record volume" in joined
    assert "statistical outliers" in joined
