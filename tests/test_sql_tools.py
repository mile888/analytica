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


def test_sql_metadata_is_returned_and_stored():
    context, tools, runtime = _analytics_tools()
    query = "SELECT category_label, SUM(metric_value) AS total FROM data GROUP BY category_label ORDER BY total DESC"

    _invoke(tools["check_dataframe_sql"], runtime, query=query)
    result = _invoke(tools["query_dataframe_sql"], runtime, query=query)

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


def test_check_dataframe_sql_reports_blocked_query():
    _, tools, runtime = _analytics_tools()

    result = _invoke(tools["check_dataframe_sql"], runtime, query="DROP TABLE data")

    assert result["valid"] == "false"
    assert result["checked_at"]
    assert result["error"]


def test_pandas_engine_to_pandas_handles_product_payloads():
    engine = create_engine("pandas")

    scalar_payload = engine.to_pandas({"outlier_count": 354, "metric": "metric_value"})
    records_payload = engine.to_pandas([{"category_label": "A", "metric_value": 1}])
    empty_payload = engine.to_pandas(None)

    assert scalar_payload.to_dict("records") == [{"outlier_count": 354, "metric": "metric_value"}]
    assert records_payload.to_dict("records") == [{"category_label": "A", "metric_value": 1}]
    assert empty_payload.empty


def test_non_pandas_engines_fall_back_for_product_payloads():
    scalar_payload = {"outlier_count": 354, "metric": "metric_value"}

    try:
        polars_result = create_engine("polars").to_pandas(scalar_payload)
    except ImportError:
        polars_result = None

    if polars_result is not None:
        assert polars_result.to_dict("records") == [scalar_payload]

    spark_result = create_engine("spark").to_pandas(scalar_payload)
    assert spark_result.to_dict("records") == [scalar_payload]


def test_sql_tools_do_not_fail_on_scalar_product_payload():
    context = {
        "query": "sql test",
        "df": {"outlier_count": 354, "metric": "metric_value"},
        "engine": "pandas",
        "schema": "",
        "loaded_skills": [],
    }
    tools = {tool.name: tool for tool in build_analytics_tools(context)}
    runtime = _runtime(context)

    tables = _invoke(tools["list_dataframe_tables"], runtime)

    assert tables["tables"] == "data"
    assert context["schema"]
