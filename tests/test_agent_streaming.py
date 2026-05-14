import pandas as pd

import source.agent as agent_module


def test_run_agent_stream_falls_back_to_run_agent(monkeypatch):
    fallback_output = {
        "final_answer": "ok",
        "code": "",
        "result_preview": "",
        "result_base64": "",
        "exec_error": None,
        "engine": "pandas",
        "needs_data": True,
        "use_case": "data_analytics",
        "selected_skills": ["data-analysis"],
        "selected_tools": ["inspect_dataset_schema"],
        "loaded_skills": ["data-analysis"],
        "tool_timeline": [{"tool": "inspect_dataset_schema", "status": "ok"}],
        "sql_metadata": {},
        "structured_report": {"summary": "ok"},
        "trace_metadata": {"engine": "pandas"},
    }

    def fail_build_deep_agent(*args, **kwargs):
        raise RuntimeError("stream unavailable")

    def fake_run_agent(*args, **kwargs):
        return dict(fallback_output)

    monkeypatch.setattr(agent_module, "build_deep_agent", fail_build_deep_agent)
    monkeypatch.setattr(agent_module, "run_agent", fake_run_agent)

    events = list(
        agent_module.run_agent_stream(
            pd.DataFrame({"segment": ["A"], "metric": [1]}),
            "Опиши данные",
            engine="pandas",
        )
    )

    assert events[0]["event"] == "start"
    assert any(event["event"] == "fallback" for event in events)
    assert events[-1]["event"] == "final"
    assert events[-1]["output"]["streaming_fallback"] is True
    assert events[-1]["output"]["final_answer"] == "ok"


def test_structured_report_schema_contains_expected_fields():
    report = agent_module._structured_report(
        final_answer="Summary",
        run_context={
            "code": "result = df.head()",
            "result_facts": "Found 1 row",
            "artifacts": [{"path": "artifacts/report.md"}],
            "exec_error": None,
        },
        loaded_skills=["data-analysis"],
        tool_timeline=[{"tool": "inspect_dataset_schema", "status": "ok"}],
        sql_metadata={"row_count": 1},
    )

    assert set(report) == {
        "summary",
        "key_findings",
        "limitations",
        "artifacts",
        "next_steps",
        "generated_code",
        "loaded_skills",
        "tool_timeline",
        "sql_metadata",
    }
    assert report["summary"] == "Summary"
    assert report["generated_code"] == "result = df.head()"
    assert report["loaded_skills"] == ["data-analysis"]
