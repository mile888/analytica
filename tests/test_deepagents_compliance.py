from __future__ import annotations

import pandas as pd

import source.agent as agent_module
from source.config import PROJECT_ROOT
from source.product.branch_workspace import BranchWorkspace, BranchWorkspaceManager
from source.product.execution_planner import AuthoritativeExecutionPlanner
from source.product.fallback_analysis import deterministic_investigation_fallback
from source.product.investigation import InvestigationStatus
from source.product.service import InvestigationService
from source.product.store import InvestigationStore
from source.runtime_context import AnalyticaContext


def test_create_deep_agent_uses_documented_constructor_surface(monkeypatch) -> None:
    captured = {}

    class FakeFilesystemBackend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeStateBackend:
        pass

    class FakeCompositeBackend:
        def __init__(self, default, routes):
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

    agent, _ = agent_module.build_deep_agent(pd.DataFrame({"Sales": [1]}), "Analyze Sales")

    assert isinstance(agent, FakeCompiledAgent)
    assert "instructions" not in captured
    assert set(captured) <= {
        "model",
        "tools",
        "system_prompt",
        "skills",
        "memory",
        "backend",
        "context_schema",
        "checkpointer",
    }
    assert captured["model"] == "fake-model"
    assert captured["tools"]
    assert "system_prompt" in captured
    assert "skills" in captured
    assert "backend" in captured
    assert "memory" in captured
    assert captured["context_schema"] is AnalyticaContext
    assert "subagents" not in captured


def test_deep_agent_system_prompt_is_boundary_focused_not_runtime_state() -> None:
    prompt = agent_module._build_system_prompt(["inspect_dataset_schema", "run_python_analysis"])

    assert "Deep Agents" in prompt
    assert "schema" in prompt.lower()
    assert "placeholder artifacts" in prompt
    assert "stabilization" not in prompt.lower()
    assert "Salary_LPA" not in prompt
    assert "thread_id" not in prompt
    assert "user_id" not in prompt


def test_skill_sources_are_documented_deepagents_skill_directories() -> None:
    sources = agent_module._deep_agent_skill_sources()

    assert sources
    for source in sources:
        assert source.startswith("/")
        assert source.endswith("/")
        skill_dir = PROJECT_ROOT / source.lstrip("/")
        assert skill_dir.is_dir()
        assert (skill_dir / "SKILL.md").is_file()


def test_analytics_tools_have_model_safe_public_schemas() -> None:
    from langchain_core.tools import BaseTool
    from source.tools.analytics_tools import build_analytics_tools

    tools = build_analytics_tools()

    assert tools
    for tool in tools:
        assert isinstance(tool, BaseTool)
        assert tool.description
        assert "runtime" not in tool.args
        assert "run_context" not in tool.args
        assert "fallback_run_context" not in tool.args


def test_authoritative_planner_is_structured_not_final_prose() -> None:
    df = pd.DataFrame(
        {
            "Sales": [100.0, 20.0],
            "City": ["Los Angeles", "San Francisco"],
            "Customer Name": ["A", "B"],
        }
    )

    plan = AuthoritativeExecutionPlanner.plan("Which customer in Los Angeles has the lowest revenue?", df)
    payload = plan.to_payload()

    assert payload["metric"] == "Sales"
    assert payload["filters"][0]["column"] == "City"
    assert "final_answer" not in payload
    assert "summary" not in payload


def test_branch_workspace_consumes_query_plan_without_raw_text_parsing() -> None:
    df = pd.DataFrame({"Sales": [1.0, 2.0], "City": ["A", "B"]})
    plan = AuthoritativeExecutionPlanner.plan("Посчитай выручку в каждом городе", df)

    decision = BranchWorkspaceManager.route(plan, BranchWorkspace())

    assert decision.action.value == "create"
    assert decision.identity.metric == "Sales"
    assert decision.identity.dimension == "City"


def test_deterministic_fallback_marks_authoritative_plan_safety_path() -> None:
    df = pd.DataFrame({"Sales": [1.0, 2.0], "City": ["A", "B"]})

    result = deterministic_investigation_fallback("Посчитай выручку в каждом городе", df)

    assert result["trace_metadata"]["fallback"] == "authoritative_query_plan"
    assert result["trace_metadata"]["query_plan"]["metric"] == "Sales"
    assert result["trace_metadata"]["query_plan"]["dimension"] == "City"
    assert result["tool_timeline"][0]["tool"] == "deterministic_pandas_fallback"
    assert result["tool_timeline"][1]["tool"] == "authoritative_execution_planner"


def test_product_service_normal_useful_runner_output_is_not_replaced_by_fallback() -> None:
    def runner(*, question, df=None, data_context=None):
        return {
            "final_answer": "Sales increased by 12.0 with a clear average difference across the requested groups.",
            "structured_report": {
                "summary": "Sales increased by 12.0 with a clear average difference across the requested groups.",
                "key_findings": ["Sales increased by 12.0."],
                "evidence": ["Runner output."],
                "limitations": [],
                "artifacts": [],
                "next_steps": [],
                "generated_code": "",
                "loaded_skills": [],
                "tool_timeline": [{"tool": "deep_agent", "status": "ok"}],
                "sql_metadata": {},
            },
            "tool_timeline": [{"tool": "deep_agent", "status": "ok"}],
            "artifacts": [],
            "exec_error": None,
        }

    store = InvestigationStore()
    service = InvestigationService(store=store, runner=runner)
    investigation = service.create_investigation("Explain Sales differences")

    updated = service.run_investigation(
        investigation.investigation_id,
        df=pd.DataFrame({"Sales": [1.0, 2.0], "City": ["A", "B"]}),
    )

    assert updated.status == InvestigationStatus.NEEDS_REVIEW
    assert updated.report.summary.startswith("Sales increased")
    assert updated.runs[-1].trace[0]["tool"] == "deep_agent"
