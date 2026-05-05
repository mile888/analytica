"""Compatibility output types for the active Deep Agents runtime.

The old planner/codegen/critic LangGraph pipeline was moved out of the active
runtime. The application now executes through `source.agent.run_agent`, which
uses `deepagents.create_deep_agent`.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict


class DeepAgentRunState(TypedDict, total=False):
    """Normalized metadata returned by the Deep Agent adapter to UI/CLI callers."""

    query: str
    final_answer: str
    code: str
    exec_error: Optional[str]
    result_preview: str
    result_base64: str
    result_kind: Literal["plot", "dataframe", "series", "scalar", "error", ""]
    result_facts: str
    engine: str
    needs_data: Optional[bool]
    use_case: str
    selected_skills: list[str]
    selected_tools: list[str]
    critic_verdict: str
    critic_feedback: str
    raw_result: Any


# Backward-compatible alias for old imports in archived modules.
AgentState = DeepAgentRunState
