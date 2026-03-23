from typing import List, Optional, Any
import json

import pandas as pd
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

from source.prompts import (
    ROUTER_PROMPT,
    BUSINESS_PROMPT,
    PLANNER_PROMPT,
    CODEGEN_PROMPT,
    CRITIC_PROMPT,
    REPORTER_PROMPT,
)

from source.func import (
    make_llm,
    detect_engine,
    ensure_engine_table,
    df_schema_text,
    extract_code_block,
    safe_exec,
    preview_result_and_facts,
    _is_bar_command,
    _bar_codegen,
)

from source.state import AgentState


def business_or_data_node(state: AgentState) -> AgentState:
    query = state["query"]

    llm = make_llm("router")
    msg = llm.invoke(ROUTER_PROMPT.format_messages(query=query))

    needs_data = True
    use_case = "data_analytics"
    reason = ""

    raw = (msg.content or "").strip()
    try:
        data = json.loads(raw)
    except Exception:
        q = query.lower()
        needs_data = any(k in q for k in ["топ", "sum", "сумм", "средн", "avg", "mean", "график", "bar", "таблиц", "посчитай", "сколько", "корреляц", "доля"])
        use_case = "data_analytics" if needs_data else "business_analytics"
        reason = "heuristic"
    else:
        needs_data = bool(data.get("needs_data", True))
        use_case = data.get("use_case", "data_analytics")
        reason = data.get("reason", "")

    if needs_data:
        eng = detect_engine(state.get("df"), state.get("engine"))
        table = ensure_engine_table(state.get("df"), eng)
        schema = df_schema_text(table, eng)
        return {"needs_data": True, "use_case": "data_analytics", "engine": eng, "df": table, "schema": schema}

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
    schema = state.get("schema") or df_schema_text(state["df"], state.get("engine", "pandas"))
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
        return {"code": code}

    llm = make_llm("codegen")
    schema = state.get("schema") or df_schema_text(state["df"], state.get("engine", "pandas"))
    critic_fb = state.get("critic_feedback", "")
    msg = llm.invoke(CODEGEN_PROMPT.format_messages(
        query=state["query"],
        plan=state.get("plan", ""),
        critic_feedback=critic_fb,
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
    kind, rp, facts = preview_result_and_facts(result, err)
    return {
        "result": result,
        "exec_error": err,
        "result_kind": kind,
        "result_preview": rp,
        "result_facts": facts,
    }


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

    verdict = "RETRY"
    feedback = "Не удалось разобрать ответ критика."

    raw = (msg.content or "").strip()
    try:
        data = json.loads(raw)
        verdict = data.get("verdict", "RETRY")
        feedback = data.get("feedback", "")
        return {"critic_verdict": verdict, "critic_feedback": feedback}
    except Exception:
        pass

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            chunk = raw[start:end+1]
            data = json.loads(chunk)
            verdict = data.get("verdict", "RETRY")
            feedback = data.get("feedback", "")
            return {"critic_verdict": verdict, "critic_feedback": feedback}
    except Exception:
        pass

    txt = raw
    if "OK" in txt and "RETRY" not in txt:
        verdict = "OK"
        feedback = txt[:500]
    else:
        verdict = "RETRY"
        feedback = txt[:500]

    return {"critic_verdict": verdict, "critic_feedback": feedback}


def reporter_node(state: AgentState) -> AgentState:
    llm = make_llm("reporter")
    msg = llm.invoke(REPORTER_PROMPT.format_messages(
        query=state["query"],
        plan=state.get("plan", ""),
        result_kind=state.get("result_kind", "scalar"),
        result_facts=state.get("result_facts", state.get("result_preview", "")),
    ))
    return {"final_answer": msg.content}


def attempt_guard_node(state: AgentState) -> AgentState:
    if state.get("critic_verdict") == "RETRY":
        attempts = int(state.get("attempts", 0)) + 1
        return {"attempts": attempts}
    return {"attempts": int(state.get("attempts", 0))}


def route_after_critic(state: AgentState) -> str:
    if state.get("critic_verdict") == "OK":
        return "reporter"

    attempts = int(state.get("attempts", 0))
    max_attempts = int(state.get("max_attempts", 2))
    if attempts >= max_attempts:
        return "reporter"
    return "codegen"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router", business_or_data_node)
    g.add_node("business", business_node)

    g.add_node("planner", planner_node)
    g.add_node("codegen", codegen_node)
    g.add_node("exec", exec_node)
    g.add_node("critic", critic_node)
    g.add_node("attempt_guard", attempt_guard_node)
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
    g.add_edge("exec", "critic")
    g.add_edge("critic", "attempt_guard")

    g.add_conditional_edges(
        "attempt_guard",
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
        "code": out.get("code", ""),
        "critic_verdict": out.get("critic_verdict", ""),
        "critic_feedback": out.get("critic_feedback", ""),
        "exec_error": out.get("exec_error", None),
        "use_case": out.get("use_case", ""),
        "engine": out.get("engine", ""),
        "needs_data": out.get("needs_data", None),
    }
