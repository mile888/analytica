import pandas as pd

from source.dataframe import validate_read_only_sql
from source.tools.analytics_tools import build_analytics_tools


def _analytics_tools():
    context = {
        "query": "sql test",
        "df": pd.DataFrame(
            {
                "segment": ["A", "B", "A", "C"],
                "metric": [10, 20, 15, 5],
            }
        ),
        "engine": "pandas",
        "schema": "",
        "loaded_skills": [],
    }
    return context, {tool.__name__: tool for tool in build_analytics_tools(context)}


def test_sql_query_requires_exact_check_before_execution():
    _, tools = _analytics_tools()
    query = "SELECT segment, SUM(metric) AS total FROM data GROUP BY segment"

    first_attempt = tools["query_dataframe_sql"](query)
    checked = tools["check_dataframe_sql"](query)
    second_attempt = tools["query_dataframe_sql"](query)

    assert "Call check_dataframe_sql" in first_attempt["exec_error"]
    assert checked["valid"] == "true"
    assert second_attempt["exec_error"] == ""
    assert "segment" in second_attempt["result_preview"]


def test_sql_metadata_is_returned_and_stored():
    context, tools = _analytics_tools()
    query = "SELECT segment, SUM(metric) AS total FROM data GROUP BY segment ORDER BY total DESC"

    tools["check_dataframe_sql"](query)
    result = tools["query_dataframe_sql"](query)

    assert result["table_name"] == "data"
    assert result["query"] == query
    assert int(result["row_count"]) == 3
    assert result["truncated"] == "False"
    assert result["checked_at"]
    assert result["executed_at"]
    assert context["sql_metadata"]["row_count"] == 3
    assert context["tool_timeline"][-1]["tool"] == "query_dataframe_sql"
    assert context["artifacts"][-1]["artifact_type"] == "table"
    assert context["artifacts"][-1]["source_tool"] == "query_dataframe_sql"
    assert context["artifacts"][-1]["rows"] == 3
    assert context["artifacts"][-1]["columns"] == 2


def test_read_only_validation_blocks_mutating_sql():
    for query in [
        "DROP TABLE data",
        "DELETE FROM data",
        "UPDATE data SET metric = 0",
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


def test_check_dataframe_sql_reports_blocked_query():
    _, tools = _analytics_tools()

    result = tools["check_dataframe_sql"]("DROP TABLE data")

    assert result["valid"] == "false"
    assert result["checked_at"]
    assert result["error"]
