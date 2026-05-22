import pandas as pd

from source.func import preview_result_and_facts, safe_exec


def _df():
    return pd.DataFrame(
        {
            "category_label": ["A", "B", "A", "B"],
            "metric_value": [10, 20, 15, 5],
        }
    )


def test_safe_pandas_analysis_ok():
    result, err = safe_exec(
        "result = df.groupby('category_label')['metric_value'].sum().reset_index()",
        _df(),
        "pandas",
    )
    kind, preview, facts, b64 = preview_result_and_facts(result, err)

    assert err is None
    assert kind == "dataframe"
    assert "category_label" in preview
    assert not b64


def test_matplotlib_analysis_ok():
    code = """
fig, ax = plt.subplots()
df.groupby('category_label')['metric_value'].sum().plot(kind='bar', ax=ax)
result = fig
""".strip()

    result, err = safe_exec(code, _df(), "pandas")
    kind, preview, facts, b64 = preview_result_and_facts(result, err)

    assert err is None
    assert kind == "plot"
    assert preview == "<matplotlib Figure>"
    assert b64.startswith("data:image/png;base64,")


def test_executor_blocks_file_and_unsafe_imports():
    blocked_cases = {
        "open": "result = open('README.md').read()",
        "import_os": "import os\nresult = os.getcwd()",
        "read_csv": "result = pd.read_csv('data/example.csv')",
    }

    for name, code in blocked_cases.items():
        result, err = safe_exec(code, _df(), "pandas")
        assert result is None, name
        assert err, name


def test_executor_requires_result_variable():
    result, err = safe_exec("df.groupby('category_label')['metric_value'].sum()", _df(), "pandas")

    assert result is None
    assert "result" in err
