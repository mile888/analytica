import pandas as pd
from langchain.tools import ToolRuntime

from source.dataframe import validate_read_only_sql
from source.engine import create_engine
from source.runtime_context import AnalyticaContext
from source.tools.analytics_tools import build_analytics_tools

def _runtime(run_context):
    return ToolRuntime(
        state={},
        context=AnalyticaContext(run_state=run_context),
        config={},
        stream_writer=lambda _: None,
        tool_call_id=None,
        store=None,
    )

def _invoke(tool, runtime, **kwargs):
    return tool.invoke({**kwargs, "runtime": runtime})

def _analytics_tools():
    context = {
        "query": "sql test",
        "df": pd.DataFrame(
            {
                "category_label": ["A", "B", "A", "C"],
                "metric_value": [10, 20, 15, 5],
            }
        ),
        "engine": "pandas",
        "schema": "",
        "loaded_skills": [],
    }
    return context, {tool.name: tool for tool in build_analytics_tools(context)}, _runtime(context)

def test_sql_query_requires_exact_check_before_execution():
    _, tools, runtime = _analytics_tools()
    query = "SELECT category_label, SUM(metric_value) AS total FROM data GROUP BY category_label"

    first_attempt = _invoke(tools["query_dataframe_sql"], runtime, query=query)
    checked = _invoke(tools["check_dataframe_sql"], runtime, query=query)
    second_attempt = _invoke(tools["query_dataframe_sql"], runtime, query=query)

    assert "Call check_dataframe_sql" in first_attempt["exec_error"]
    assert checked["valid"] == "true"
    assert second_attempt["exec_error"] == ""
    assert "category_label" in second_attempt["result_preview"]

def test_read_only_validation_blocks_mutating_sql():
    for query in [
        "DROP TABLE data",
        "DELETE FROM data",
        "UPDATE data SET metric_value = 0",
        "INSERT INTO data VALUES ('A', 1)",
        "ALTER TABLE data ADD COLUMN x INT",
        "CREATE TABLE x (id INT)",
    ]:
        try:
            validate_read_only_sql(query)
        except ValueError as exc:
            assert "read-only" in str(exc) or "DML/DDL" in str(exc)
        else:
            raise AssertionError(f"Query was not blocked: {query}")

