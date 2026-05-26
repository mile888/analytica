from __future__ import annotations

import pandas as pd

from source.product.semantic_layer import (
    InvestigationThreadState,
    SemanticRole,
    build_investigation_focus,
    build_investigation_thread_state,
    build_semantic_dataset_profile,
    match_entity_or_value,
    select_dimension_column,
    select_metric_column,
    semantic_selection_text,
    thread_selection_text,
)

def test_semantic_layer_infers_generic_column_roles() -> None:
    df = pd.DataFrame(
        {
            "entity_id": ["a1", "a2", "a3", "a4"],
            "amount_value": [10.0, 12.5, 9.0, 30.0],
            "category_label": ["A", "A", "B", "C"],
            "event_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "text_note": ["short note", "another note", "review", "follow-up"],
        }
    )

    semantic = build_semantic_dataset_profile(df=df)

    assert semantic.by_name["amount_value"].role == SemanticRole.METRIC
    assert semantic.by_name["category_label"].role == SemanticRole.DIMENSION
    assert semantic.by_name["event_date"].role == SemanticRole.TIMESTAMP
    assert semantic.by_name["entity_id"].role == SemanticRole.IDENTIFIER

def test_semantic_focus_resolves_ambiguous_follow_up_to_latest_chart() -> None:
    df = pd.DataFrame(
        {
            "City": ["Moscow", "Kazan", "Sochi"],
            "Ship Mode": ["Second Class", "Same Day", "Standard Class"],
            "Sales": [1200.0, 500.0, 700.0],
            "Profit": [200.0, 50.0, 80.0],
        }
    )
    semantic = build_semantic_dataset_profile(df=df)
    focus = build_investigation_focus(
        {
            "latest_chart_context": {
                "metric": "Sales",
                "dimension": "City",
                "chart_type": "bar",
                "title": "Average Sales by City",
            }
        },
        question="Какие самые сильные?",
    )
    selection_text = semantic_selection_text("Какие самые сильные?", focus)

    assert select_metric_column(semantic, selection_text, explicit_text="Какие самые сильные?", focus=focus) == "Sales"
    assert (
        select_dimension_column(
            semantic,
            selection_text,
            explicit_text="Какие самые сильные?",
            focus=focus,
            exclude={"Sales"},
        )
        == "City"
    )

