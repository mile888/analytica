from typing import List, Optional
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END


from source.prompts import (PLANNER_PROMPT, 
                            CODEGEN_PROMPT, 
                            CRITIC_PROMPT, 
                            REPORTER_PROMPT)

from source.func import (make_llm,
                        df_schema_text, 
                        extract_code_block, 
                        safe_exec_pandas, 
                        preview_result_and_facts,
                        _is_bar_command,
                        _bar_codegen)

from source.state import AgentState


def planner_node(state: AgentState) -> AgentState:
    llm = make_llm()
    history_text = "\n".join(
        [f"{m.type}: {getattr(m, 'content', '')}" for m in state.get("chat_history", [])][-6:]
    ) or "(пусто)"
    schema = df_schema_text(state["df"])
    plan_msg = llm.invoke(PLANNER_PROMPT.format_messages(
        query=state["query"],
        history=history_text,
        schema=schema,
    ))
    return {"plan": plan_msg.content}

def codegen_node(state: AgentState) -> AgentState:
    if _is_bar_command(state["query"]):
        code = _bar_codegen(state["query"])
        return {"code": code}

    llm = make_llm()
    schema = df_schema_text(state["df"])
    critic_fb = state.get("critic_feedback", "")
    msg = llm.invoke(CODEGEN_PROMPT.format_messages(
        query=state["query"],
        plan=state["plan"],
        critic_feedback=critic_fb,
        schema=schema,
    ))
    code = extract_code_block(msg.content)
    if not code.strip():
        code = msg.content.strip()
    return {"code": code}

def exec_node(state: AgentState) -> AgentState:
    result, err = safe_exec_pandas(state["code"], state["df"])
    kind, rp, facts = preview_result_and_facts(result, err)
    return {
        "result": result,
        "exec_error": err,
        "result_kind": kind,
        "result_preview": rp,
        "result_facts": facts,
    }


def critic_node(state: AgentState) -> AgentState:
    import json
    llm = make_llm()
    msg = llm.invoke(CRITIC_PROMPT.format_messages(
        query=state["query"],
        plan=state["plan"],
        code=state["code"],
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
    llm = make_llm()
    msg = llm.invoke(REPORTER_PROMPT.format_messages(
        query=state["query"],
        plan=state["plan"],
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

    g.add_node("planner", planner_node)
    g.add_node("codegen", codegen_node)
    g.add_node("exec", exec_node)
    g.add_node("critic", critic_node)
    g.add_node("attempt_guard", attempt_guard_node)
    g.add_node("reporter", reporter_node)

    g.set_entry_point("planner")

    g.add_edge("planner", "codegen")
    g.add_edge("codegen", "exec")
    g.add_edge("exec", "critic")
    g.add_edge("critic", "attempt_guard")

    g.add_conditional_edges(
        "attempt_guard",
        route_after_critic,
        {"codegen": "codegen", "reporter": "reporter"}
    )

    g.add_edge("reporter", END)
    return g.compile()


def run_once(df: pd.DataFrame, query: str, chat_history: Optional[List[BaseMessage]] = None):
    app = build_graph()
    init_state: AgentState = {
        "query": query,
        "df": df,
        "chat_history": chat_history or [],
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
    }
