"""
Analytics agent graph — clean pipeline architecture.

Flow:
  router → (data path)    → planner → codegen → exec → route_after_exec
                                                         ├─ exec error + retries left → codegen
                                                         ├─ exec error + exhausted    → reporter
                                                         └─ success                   → critic
                                                                                        ├─ OK    → reporter
                                                                                        └─ RETRY → codegen
         → (business path) → business → END
"""
from typing import List, Optional, Any
import json

import pandas as pd
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from source.prompts import (ROUTER_PROMPT, BUSINESS_PROMPT, PLANNER_PROMPT, CODEGEN_PROMPT, CRITIC_PROMPT, REPORTER_PROMPT)

from source.func import (
    detect_engine,
    extract_code_block,
    safe_exec,
    preview_result_and_facts,
    _is_bar_command,
    _bar_codegen,
)
from source.llm.factory import make_llm
from source.engine import create_engine

from source.state import AgentState


# ═══════════════════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════════════════

def _parse_json_loose(text: str) -> Optional[dict]:
    """Try to parse JSON from text — first raw, then find first {...} block."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Nodes
# ═══════════════════════════════════════════════════════════════════════

def business_or_data_node(state: AgentState) -> AgentState:
    query = state["query"]

    llm = make_llm("router")
    msg = llm.invoke(ROUTER_PROMPT.format_messages(query=query))

    data = _parse_json_loose(msg.content)
    if data:
        needs_data = bool(data.get("needs_data", True))
    else:
        # If router fails to produce valid JSON, default to data path (safer)
        needs_data = True

    if needs_data:
        eng_name = detect_engine(state.get("df"), state.get("engine"))
        eng = create_engine(eng_name)
        table = eng.ensure_table(state.get("df"))
        schema = eng.schema_text(table)
        return {"needs_data": True, "use_case": "data_analytics", "engine": eng_name, "df": table, "schema": schema}

    return {"needs_data": False, "use_case": "business_analytics", "engine": "pandas", "schema": ""}


def route_after_router(state: AgentState) -> str:
    return "planner" if state.get("needs_data") else "business"


def business_node(state: AgentState) -> AgentState:
    llm = make_llm("reporter")
    history_text = "\n".join(
        [f"{m.type}: {getattr(m, 'content', '')}" for m in state.get("messages", [])][-6:]
    ) or "(пусто)"
    msg = llm.invoke(BUSINESS_PROMPT.format_messages(
        query=state["query"],
        messages=history_text,
    ))
    return {"final_answer": msg.content}


def planner_node(state: AgentState) -> AgentState:
    llm = make_llm("planner")
    history_text = "\n".join(
        [f"{m.type}: {getattr(m, 'content', '')}" for m in state.get("messages", [])][-6:]
    ) or "(пусто)"
    schema = state.get("schema") or create_engine(state.get("engine", "pandas")).schema_text(state["df"])
    plan_msg = llm.invoke(PLANNER_PROMPT.format_messages(
        query=state["query"],
        messages=history_text,
        schema=schema,
        engine=state.get("engine", "pandas"),
    ))
    return {"plan": plan_msg.content, "schema": schema}


def codegen_node(state: AgentState) -> AgentState:
    if _is_bar_command(state["query"]):
        code = _bar_codegen(state["query"])
        return {"code": code, "schema": state.get("schema") or ""}

    llm = make_llm("codegen")
    schema = state.get("schema") or create_engine(state.get("engine", "pandas")).schema_text(state["df"])
    msg = llm.invoke(CODEGEN_PROMPT.format_messages(
        query=state["query"],
        plan=state.get("plan", ""),
        critic_feedback=state.get("critic_feedback", ""),
        exec_error=state.get("exec_error") or "",
        schema=schema,
        engine=state.get("engine", "pandas"),
    ))
    code = extract_code_block(msg.content)
    if not code.strip():
        code = (msg.content or "").strip()
    return {"code": code, "schema": schema}


def exec_node(state: AgentState) -> AgentState:
    engine = state.get("engine", "pandas")
    result, err = safe_exec(state["code"], state["df"], engine)
    kind, rp, facts, b64 = preview_result_and_facts(result, err)

    attempts = int(state.get("attempts", 0)) + 1

    return {
        "result": result,
        "exec_error": err,
        "result_kind": kind,
        "result_preview": rp,
        "result_facts": facts,
        "result_base64": b64,
        "attempts": attempts,
    }


def route_after_exec(state: AgentState) -> str:
    """Route based on execution outcome.

    - exec error + retries left → codegen (direct retry with error context)
    - exec error + exhausted    → reporter (give up gracefully)
    - success                   → critic  (quality check)
    """
    if state.get("exec_error"):
        attempts = int(state.get("attempts", 0))
        max_attempts = int(state.get("max_attempts", 2))
        if attempts < max_attempts:
            return "codegen"
        return "reporter"
    return "critic"


def critic_node(state: AgentState) -> AgentState:
    llm = make_llm("critic")
    msg = llm.invoke(CRITIC_PROMPT.format_messages(
        query=state["query"],
        plan=state.get("plan", ""),
        code=state.get("code", ""),
        result_kind=state.get("result_kind", "scalar"),
        result_facts=state.get("result_facts", state.get("result_preview", "")),
        exec_error=state.get("exec_error", None),
    ))

    data = _parse_json_loose(msg.content)
    if data:
        verdict = data.get("verdict", "OK")
        feedback = data.get("feedback", "")
    else:
        raw = (msg.content or "").strip()
        if "RETRY" in raw:
            verdict = "RETRY"
            feedback = raw[:500]
        else:
            verdict = "OK"
            feedback = raw[:500]

    return {"critic_verdict": verdict, "critic_feedback": feedback}


def route_after_critic(state: AgentState) -> str:
    if state.get("critic_verdict") == "OK":
        return "reporter"

    attempts = int(state.get("attempts", 0))
    max_attempts = int(state.get("max_attempts", 2))
    if attempts >= max_attempts:
        return "reporter"
    return "codegen"


def reporter_node(state: AgentState) -> AgentState:
    llm = make_llm("reporter")
    msg = llm.invoke(REPORTER_PROMPT.format_messages(
        query=state["query"],
        plan=state.get("plan", ""),
        result_kind=state.get("result_kind", "scalar"),
        result_facts=state.get("result_facts", state.get("result_preview", "")),
    ))
    return {"final_answer": msg.content}


# ═══════════════════════════════════════════════════════════════════════
#  Graph
# ═══════════════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router", business_or_data_node)
    g.add_node("business", business_node)

    g.add_node("planner", planner_node)
    g.add_node("codegen", codegen_node)
    g.add_node("exec", exec_node)
    g.add_node("critic", critic_node)
    g.add_node("reporter", reporter_node)

    g.set_entry_point("router")

    g.add_conditional_edges(
        "router",
        route_after_router,
        {"planner": "planner", "business": "business"},
    )

    g.add_edge("business", END)

    g.add_edge("planner", "codegen")
    g.add_edge("codegen", "exec")

    # After exec: error → retry codegen or give up; success → critic
    g.add_conditional_edges(
        "exec",
        route_after_exec,
        {"codegen": "codegen", "critic": "critic", "reporter": "reporter"},
    )

    # After critic: OK → reporter; RETRY → codegen
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {"codegen": "codegen", "reporter": "reporter"},
    )

    g.add_edge("reporter", END)
    return g.compile()


def run_once(df: Any, query: str, messages: Optional[List[BaseMessage]] = None, engine: str = "auto"):
    app = build_graph()
    init_state: AgentState = {
        "query": query,
        "df": df,
        "engine": engine,
        "messages": messages or [],
        "attempts": 0,
        "max_attempts": 2,
        "critic_feedback": "",
    }
    out = app.invoke(init_state)

    return {
        "final_answer": out.get("final_answer", ""),
        "result_preview": out.get("result_preview", ""),
        "result_base64": out.get("result_base64", ""),
        "code": out.get("code", ""),
        "critic_verdict": out.get("critic_verdict", ""),
        "critic_feedback": out.get("critic_feedback", ""),
        "exec_error": out.get("exec_error", None),
        "use_case": out.get("use_case", ""),
        "engine": out.get("engine", ""),
        "needs_data": out.get("needs_data", None),
    }
