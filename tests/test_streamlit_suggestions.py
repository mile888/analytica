import pandas as pd

from app import infer_demo_suggestions


def test_demo_suggestions_prefer_business_metric_over_id_columns():
    df = pd.DataFrame(
        {
            "Row ID": [1, 2, 3],
            "Order ID": ["CA-1", "CA-2", "CA-3"],
            "Category": ["Furniture", "Technology", "Furniture"],
            "Sales": [100.0, 250.0, 150.0],
        }
    )

    suggestions = infer_demo_suggestions(df)
    joined = "\n".join(suggestions)

    assert "Sales" in joined
    assert "Category" in joined
    assert "Row ID" not in joined
