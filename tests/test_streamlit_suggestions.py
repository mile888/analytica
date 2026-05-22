import pandas as pd
from app import infer_demo_suggestions


def test_demo_suggestions_prefer_business_metric_over_id_columns():
    df = pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "record_code": ["A-1", "A-2", "A-3"],
            "category_label": ["Group A", "Group B", "Group A"],
            "metric_value": [100.0, 250.0, 150.0],
        }
    )

    suggestions = infer_demo_suggestions(df)
    joined = "\n".join(suggestions)

    assert "metric_value" in joined
    assert "category_label" in joined
    assert "How many unique entities" in joined
