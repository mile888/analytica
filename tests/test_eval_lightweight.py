import pandas as pd

import source.agent as agent_module
from source.func import safe_exec
from source.tools.analytics_tools import build_analytics_tools
from source.tools.skill_tools import build_skill_tools


class _FakeAgent:
    def invoke(self, payload, config=None):
        return {"messages": [{"role": "assistant", "content": "Краткий аналитический ответ."}]}


def test_agent_output_contains_structured_report(monkeypatch):
    run_context = {
        "query": "Опиши данные",
        "df": pd.DataFrame({"segment": ["A"], "metric": [1]}),
        "engine": "pandas",
        "code": "result = df.head()",
        "exec_error": None,
        "result_preview": "segment metric",
        "result_facts": "dataframe shape=(1, 2)",
        "result_base64": "",
        "loaded_skills": ["data_analysis"],
        "tool_timeline": [{"tool": "inspect_dataset_schema", "status": "ok"}],
        "artifacts": [{"artifact_type": "table", "title": "Analysis table"}],
    }

    monkeypatch.setattr(agent_module, "build_deep_agent", lambda *args, **kwargs: (_FakeAgent(), run_context))

    output = agent_module.run_agent(run_context["df"], "Опиши данные", engine="pandas")

    assert output["structured_report"]["summary"] == "Краткий аналитический ответ."
    assert output["structured_report"]["generated_code"] == "result = df.head()"
    assert output["structured_report"]["loaded_skills"] == ["data_analysis"]
    assert output["structured_report"]["artifacts"][0]["artifact_type"] == "table"


def test_schema_overview_query_uses_deterministic_dataframe_profile():
    df = pd.DataFrame(
        {
            "Category": ["A", "B"],
            "Sales": [10.0, 20.0],
        }
    )

    output = agent_module.run_agent(df, "Опиши структуру данных и возможные направления анализа", engine="pandas")

    assert output["exec_error"] is None
    assert output["critic_verdict"] == ""
    assert output["tool_timeline"][0]["tool"] == "inspect_dataset_schema"
    assert "2 строк" in output["final_answer"]
    assert "Category" in output["result_preview"]
    assert "Sales" in output["structured_report"]["key_findings"][0]


def test_sql_eval_trajectory_uses_check_before_query():
    context = {
        "query": "sql eval",
        "df": pd.DataFrame({"segment": ["A", "B"], "metric": [1, 2]}),
        "engine": "pandas",
        "schema": "",
        "loaded_skills": [],
    }
    tools = {tool.__name__: tool for tool in build_analytics_tools(context)}
    query = "SELECT segment, SUM(metric) AS total FROM data GROUP BY segment"

    tools["check_dataframe_sql"](query)
    result = tools["query_dataframe_sql"](query)

    timeline = [event["tool"] for event in context["tool_timeline"]]
    assert result["exec_error"] == ""
    assert timeline.index("check_dataframe_sql") < timeline.index("query_dataframe_sql")


def test_eval_unsafe_code_blocked():
    result, err = safe_exec("import os\nresult = os.getcwd()", pd.DataFrame({"x": [1]}), "pandas")

    assert result is None
    assert err


def test_eval_skill_loading_works():
    context = {"loaded_skills": []}
    tools = {tool.__name__: tool for tool in build_skill_tools(context)}

    loaded = tools["load_skill"]("sql_querying")

    assert loaded["loaded"] == "true"
    assert loaded["skill_name"] == "sql_querying"
    assert "sql_querying" in context["loaded_skills"]


def test_bar_chart_invalid_metric_returns_message_not_empty_plot():
    context = {
        "query": "plot eval",
        "df": pd.DataFrame(
            {
                "Row ID": [1, 2, 3],
                "Order ID": ["A-1", "A-2", "A-3"],
                "Sales": [10.0, 20.0, 30.0],
            }
        ),
        "engine": "pandas",
        "schema": "",
        "loaded_skills": [],
    }
    tools = {tool.__name__: tool for tool in build_analytics_tools(context)}

    result = tools["plot_bar"]("Row ID", "Order ID")

    assert result["exec_error"] == ""
    assert result["result_kind"] == "scalar"
    assert "Cannot build bar chart" in result["result_preview"]
