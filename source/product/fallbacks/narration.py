from __future__ import annotations

from typing import Any


def _timeline(tool: str, status: str = "ok", **metadata: Any) -> list[dict[str, Any]]:
    event = {"tool": tool, "status": status}
    event.update({key: value for key, value in metadata.items() if value not in (None, "")})
    return [event]


def _output(
    *,
    question: str,
    summary: str,
    findings: list[str],
    evidence: list[str],
    limitations: list[str],
    next_steps: list[str],
    code: str,
    result_preview: str,
    timeline: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    trace_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "final_answer": summary,
        "code": code,
        "result_preview": result_preview,
        "result_base64": "",
        "exec_error": None,
        "engine": "pandas",
        "needs_data": True,
        "use_case": "data_analytics",
        "loaded_skills": ["data-analysis", "csv-dataframe-analysis", "reporting"],
        "tool_timeline": timeline,
        "sql_metadata": {},
        "structured_report": {
            "question": question,
            "summary": summary,
            "key_findings": findings,
            "evidence": evidence,
            "limitations": limitations,
            "artifacts": [],
            "next_steps": next_steps,
            "generated_code": code,
            "loaded_skills": ["data-analysis", "csv-dataframe-analysis", "reporting"],
            "tool_timeline": timeline,
            "sql_metadata": {},
        },
        "trace_metadata": trace_metadata,
        "critic_verdict": "",
        "critic_feedback": "",
        "artifacts": artifacts,
    }
