import pandas as pd
from pathlib import Path
from typing import get_type_hints

from langchain_core.tools import BaseTool
from langchain.tools import ToolRuntime

import source.agent as agent_module
from source.func import safe_exec
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


def _tool_map(run_context):
    return {tool.name: tool for tool in build_analytics_tools(run_context)}


def _invoke(tool, runtime, **kwargs):
    return tool.invoke({**kwargs, "runtime": runtime})


class _FakeAgent:
    def invoke(self, payload, config=None, context=None):
        return {"messages": [{"role": "assistant", "content": "Краткий аналитический ответ."}]}


def test_agent_output_contains_structured_report(monkeypatch):
    run_context = {
        "query": "Опиши данные",
        "df": pd.DataFrame({"category_label": ["A"], "metric": [1]}),
        "engine": "pandas",
        "code": "result = df.head()",
        "exec_error": None,
        "result_preview": "category_label metric",
        "result_facts": "dataframe shape=(1, 2)",
        "result_base64": "",
        "loaded_skills": ["data-analysis"],
        "tool_timeline": [{"tool": "inspect_dataset_schema", "status": "ok"}],
        "artifacts": [{"artifact_type": "table", "title": "Analysis table"}],
    }

    monkeypatch.setattr(agent_module, "build_deep_agent", lambda *args, **kwargs: (_FakeAgent(), run_context))

    output = agent_module.run_agent(run_context["df"], "Опиши данные", engine="pandas")

    assert output["structured_report"]["summary"] == "Краткий аналитический ответ."
    assert output["structured_report"]["generated_code"] == "result = df.head()"
    assert output["structured_report"]["loaded_skills"] == ["data-analysis"]
    assert output["structured_report"]["artifacts"][0]["artifact_type"] == "table"


def test_schema_overview_query_uses_deterministic_dataframe_profile():
    df = pd.DataFrame(
        {
            "category_label": ["A", "B"],
            "metric_value": [10.0, 20.0],
        }
    )

    output = agent_module.run_agent(df, "Опиши структуру данных и возможные направления анализа", engine="pandas")

    assert output["exec_error"] is None
    assert output["critic_verdict"] == ""
    assert output["tool_timeline"][0]["tool"] == "inspect_dataset_schema"
    assert "2 строк" in output["final_answer"]
    assert "category_label" in output["result_preview"]
    assert "metric_value" in output["structured_report"]["key_findings"][0]


def test_metric_question_routes_to_data_analysis():
    skills = agent_module._guess_skills("Почему значение метрики отличается по группам?")

    assert "data-analysis" in skills
    assert "csv-dataframe-analysis" in skills
    assert "reporting" in skills


def test_sql_eval_trajectory_uses_check_before_query():
    context = {
        "query": "sql eval",
        "df": pd.DataFrame({"category_label": ["A", "B"], "metric": [1, 2]}),
        "engine": "pandas",
        "schema": "",
        "loaded_skills": [],
    }
    tools = _tool_map(context)
    query = "SELECT category_label, SUM(metric) AS total FROM data GROUP BY category_label"
    runtime = _runtime(context)

    _invoke(tools["check_dataframe_sql"], runtime, query=query)
    result = _invoke(tools["query_dataframe_sql"], runtime, query=query)

    timeline = [event["tool"] for event in context["tool_timeline"]]
    assert result["exec_error"] == ""
    assert timeline.index("check_dataframe_sql") < timeline.index("query_dataframe_sql")


def test_eval_unsafe_code_blocked():
    result, err = safe_exec("import os\nresult = os.getcwd()", pd.DataFrame({"x": [1]}), "pandas")

    assert result is None
    assert err


def test_build_deep_agent_uses_official_deepagents_skills(monkeypatch):
    captured = {}

    class FakeFilesystemBackend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeStateBackend:
        pass

    class FakeCompositeBackend:
        def __init__(self, *, default, routes):
            self.default = default
            self.routes = routes

    class FakeCompiledAgent:
        pass

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return FakeCompiledAgent()

    monkeypatch.setattr(
        agent_module,
        "_get_deep_agent_runtime",
        lambda: (fake_create_deep_agent, FakeCompositeBackend, FakeStateBackend, FakeFilesystemBackend, None),
    )
    monkeypatch.setattr(agent_module, "make_llm", lambda name: "fake-model")

    agent, _ = agent_module.build_deep_agent(pd.DataFrame({"x": [1]}), "Analyze x", engine="pandas")

    tool_names = [tool.name for tool in captured["tools"]]
    assert isinstance(agent, FakeCompiledAgent)
    assert captured["model"] == "fake-model"
    assert set(captured["skills"]) >= {
        "/source/skills/data-analysis/",
        "/source/skills/csv-dataframe-analysis/",
        "/source/skills/visualization/",
        "/source/skills/business-analysis/",
        "/source/skills/reporting/",
    }
    assert captured["memory"] == ["/memories/AGENTS.md"]
    assert captured["context_schema"] is AnalyticaContext
    assert isinstance(captured["backend"], FakeCompositeBackend)
    assert isinstance(captured["backend"].default, FakeStateBackend)
    assert set(captured["backend"].routes) == {"/artifacts/", "/memories/", "/source/skills/"}
    assert captured["backend"].routes["/artifacts/"].kwargs["virtual_mode"] is True
    assert captured["backend"].routes["/memories/"].kwargs["virtual_mode"] is True
    assert captured["backend"].routes["/source/skills/"].kwargs["virtual_mode"] is True
    assert "run_python_analysis" in tool_names
    assert "create_authoritative_query_plan" in tool_names
    assert "validate_query_plan" in tool_names
    assert "route_branch_workspace" in tool_names
    assert "load_skill" not in tool_names
    assert "list_available_skills" not in tool_names


def test_analytics_tools_are_official_langchain_tools():
    tools = build_analytics_tools()

    assert tools
    assert all(isinstance(tool, BaseTool) for tool in tools)
    assert {tool.name for tool in tools} >= {
        "inspect_dataset_schema",
        "run_python_analysis",
        "check_dataframe_sql",
    }
    for tool in tools:
        assert "runtime" not in tool.args


def test_skill_frontmatter_matches_official_folder_layout():
    skill_root = Path("source/skills")
    skill_files = sorted(skill_root.glob("*/SKILL.md"))

    assert skill_files
    for skill_file in skill_files:
        text = skill_file.read_text(encoding="utf-8")
        frontmatter = text.split("---", 2)[1]
        assert f"name: {skill_file.parent.name}" in frontmatter
        assert "description:" in frontmatter
        assert "allowed-tools" not in frontmatter


def test_run_agent_passes_typed_runtime_context_to_invoke(monkeypatch):
    captured = {}
    run_context = {
        "query": "Посчитай metric",
        "df": pd.DataFrame({"category_label": ["A"], "metric": [1]}),
        "engine": "pandas",
        "code": "result = df.head()",
        "exec_error": None,
        "result_preview": "category_label metric",
        "result_facts": "dataframe shape=(1, 2)",
        "result_base64": "",
        "loaded_skills": ["data-analysis"],
        "tool_timeline": [{"tool": "inspect_dataset_schema", "status": "ok"}],
        "artifacts": [],
    }

    class FakeAgent:
        def invoke(self, payload, config=None, context=None):
            captured["payload"] = payload
            captured["config"] = config
            captured["context"] = context
            return {"messages": [{"role": "assistant", "content": "ok"}]}

    monkeypatch.setattr(agent_module, "build_deep_agent", lambda *args, **kwargs: (FakeAgent(), run_context))

    output = agent_module.run_agent(
        run_context["df"],
        "Посчитай metric",
        engine="pandas",
        thread_id="typed-context-test",
    )

    runtime_context = captured["context"]
    assert isinstance(runtime_context, AnalyticaContext)
    assert runtime_context.thread_id == "typed-context-test"
    assert runtime_context.session_id == "typed-context-test"
    assert runtime_context.memory_file == "/memories/AGENTS.md"
    assert runtime_context.run_state is run_context
    assert "typed-context-test" not in captured["payload"]["messages"][0]["content"]
    assert output["final_answer"] == "ok"


def test_analytics_tools_accept_typed_tool_runtime():
    context = {
        "query": "typed tool runtime",
        "df": pd.DataFrame({"category_label": ["A"], "metric": [1]}),
        "engine": "pandas",
        "schema": "",
        "loaded_skills": [],
    }

    tools = _tool_map(context)
    hints = get_type_hints(tools["inspect_dataset_schema"].func)

    assert "ToolRuntime" in str(hints["runtime"])
    assert "AnalyticaContext" in str(hints["runtime"])


def test_runtime_context_is_not_inserted_into_system_prompt():
    prompt = agent_module._build_system_prompt(["inspect_dataset_schema"])

    assert "user_id" not in prompt
    assert "thread_id" not in prompt
    assert "run_id" not in prompt
    assert "session_id" not in prompt


def test_deep_agent_memory_file_is_initialized(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_module, "DEEPAGENTS_MEMORY_DIR", tmp_path)
    monkeypatch.setattr(agent_module, "DEEPAGENTS_MEMORY_FILE", "/memories/AGENTS.md")

    agent_module._ensure_deep_agent_memory_file()

    memory_file = tmp_path / "AGENTS.md"
    assert memory_file.exists()
    assert "Analytica Agent Memory" in memory_file.read_text(encoding="utf-8")


def test_history_messages_are_not_trimmed_by_custom_context_logic():
    class FakeMessage:
        def __init__(self, msg_type: str, content: str):
            self.type = msg_type
            self.content = content

    messages = [FakeMessage("human", f"question {idx}") for idx in range(20)]

    output = agent_module._history_as_deepagent_messages("final question", messages)

    assert len(output) == 21
    assert output[0] == {"role": "user", "content": "question 0"}
    assert output[-1] == {"role": "user", "content": "final question"}


def test_bar_chart_invalid_metric_returns_message_not_empty_plot():
    context = {
        "query": "plot eval",
        "df": pd.DataFrame(
            {
                "Row ID": [1, 2, 3],
                "Order ID": ["A-1", "A-2", "A-3"],
                "Metric Value": [10.0, 20.0, 30.0],
            }
        ),
        "engine": "pandas",
        "schema": "",
        "loaded_skills": [],
    }
    tools = _tool_map(context)

    result = _invoke(tools["plot_bar"], _runtime(context), dimension="Row ID", metric="Order ID")

    assert result["exec_error"] == ""
    assert result["result_kind"] == "scalar"
    assert "Cannot build bar chart" in result["result_preview"]
