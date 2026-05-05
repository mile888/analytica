from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver

from source.llm.factory import make_llm
from source.tools.analytics_tools import build_analytics_tools


SKILL_TOOL_REGISTRY: dict[str, list[str]] = {
    "business-analysis": [],
    "data-analysis": ["inspect_dataset_schema", "top_n", "find_sales_drops", "run_python_analysis"],
    "visualization": ["inspect_dataset_schema", "top_n", "plot_bar", "run_bar_command", "run_python_analysis"],
    "reporting": [],
}

SKILL_DIR = Path(__file__).resolve().parent / "skills"
DEFAULT_THREAD_ID = "analytica-streamlit-thread"
_DEEP_AGENT_CHECKPOINTER = MemorySaver()

ANALYTICS_SYSTEM_PROMPT = """Ты deep agent для аналитики данных и бизнес-анализа.

У тебя есть tools:
- `inspect_dataset_schema`: используй первым для любых запросов, где нужны данные.
- `top_n`: используй для топ-N агрегаций по dimension/metric.
- `plot_bar`: используй для bar chart по dimension/metric.
- `find_sales_drops`: используй для запросов про аномалии, падения, резкие провалы продаж по датам.
- `run_python_analysis`: используй только если универсальных tools недостаточно; сам напиши безопасный код и передай его tool.

Правила:
1. Для запросов по данным сначала вызови `inspect_dataset_schema`.
2. Не выдумывай колонки и результаты.
3. Предпочитай `top_n` и `plot_bar` вместо написания кода.
3a. Для поиска резких падений по дням используй `find_sales_drops`, не пиши произвольный код.
4. Если нужен код, в `run_python_analysis` передавай только Python без import, файлов, print; итог присваивай в `result`.
5. Для `/bar`, `/barh`, `/bar_share`, `/bar_stacked`, `/bar_grouped` результат должен быть графиком.
6. Финальный ответ пиши сам, на русском, основываясь только на фактах из tools.
7. Skills лежат в `/skills/`; если skill подходит к запросу, прочитай его инструкции и соблюдай раздел `Tools to Use`.
"""


def _history_as_deepagent_messages(
    query: str,
    messages: Optional[list[BaseMessage]],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in (messages or [])[-8:]:
        msg_type = getattr(msg, "type", "")
        role = "assistant" if msg_type == "ai" else "user"
        content = str(getattr(msg, "content", "") or "")
        if content:
            out.append({"role": role, "content": content})
    out.append({"role": "user", "content": query})
    return out


def _skill_files(create_file_data) -> dict[str, Any]:
    files: dict[str, Any] = {}
    if not SKILL_DIR.exists():
        return files

    for path in sorted(SKILL_DIR.rglob("SKILL.md")):
        if not path.is_file():
            continue
        rel = path.relative_to(SKILL_DIR).as_posix()
        files[f"/skills/{rel}"] = create_file_data(path.read_text(encoding="utf-8"))
    return files


def _guess_skills(query: str) -> list[str]:
    q = (query or "").lower()
    visual_markers = ("график", "диаграм", "визуал", "chart", "plot", "/bar")
    data_markers = (
        "посчитай",
        "рассчитай",
        "сравни",
        "топ",
        "сумм",
        "средн",
        "агрег",
        "таблиц",
        "колон",
        "данн",
        "dataset",
        "аномал",
        "упал",
        "упали",
        "паден",
        "резко",
    )

    needs_visual = any(marker in q for marker in visual_markers)
    needs_data = needs_visual or any(marker in q for marker in data_markers)
    if not needs_data:
        return ["business-analysis", "reporting"]

    skills = ["data-analysis"]
    if needs_visual:
        skills.append("visualization")
    skills.append("reporting")
    return skills


def _tools_for_skills(skills: list[str]) -> list[str]:
    tools: list[str] = []
    for skill in skills:
        for tool in SKILL_TOOL_REGISTRY.get(skill, []):
            if tool not in tools:
                tools.append(tool)
    return tools


def _message_content(message: Any) -> str:
    if hasattr(message, "content"):
        return str(message.content or "")
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    return str(message or "")


def _is_quota_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "resource_exhausted" in text or "quota" in text or "429" in text


def _looks_like_failed_agent_answer(text: str) -> bool:
    lowered = (text or "").lower()
    failed_markers = (
        "i can't help",
        "i cannot help",
        "can't help you",
        "unfortunately",
        "не могу помочь",
    )
    return any(marker in lowered for marker in failed_markers)


def _error_response(
    query: str,
    engine: str,
    selected_skills: list[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "final_answer": (
            "Не удалось выполнить запрос через Deep Agent runtime. "
            "Подробности сохранены в блоке Exec error ниже."
        ),
        "code": "",
        "result_preview": "",
        "result_base64": "",
        "exec_error": reason,
        "engine": engine,
        "needs_data": "data-analysis" in selected_skills or "visualization" in selected_skills,
        "use_case": "data_analytics" if "data-analysis" in selected_skills else "business_analytics",
        "selected_skills": selected_skills,
        "selected_tools": _tools_for_skills(selected_skills),
        "critic_verdict": "ERROR",
        "critic_feedback": reason,
    }


def build_deep_agent(df: Any, query: str, engine: str = "auto"):
    """Create a Deep Agent plus a mutable run context for compatibility output."""
    run_context: dict[str, Any] = {
        "query": query,
        "df": df,
        "engine": engine,
        "schema": "",
        "plan": "",
        "code": "",
        "exec_error": None,
        "result_kind": "",
        "result_preview": "",
        "result_facts": "",
        "result_base64": "",
    }

    tools = build_analytics_tools(run_context)

    agent = create_deep_agent(
        model=make_llm("deep_agent"),
        tools=tools,
        system_prompt=ANALYTICS_SYSTEM_PROMPT,
        skills=["/skills/"],
        backend=StateBackend(),
        checkpointer=_DEEP_AGENT_CHECKPOINTER,
    )
    return agent, run_context


def run_agent(
    df: Any,
    query: str,
    messages: Optional[List[BaseMessage]] = None,
    engine: str = "auto",
    thread_id: str = DEFAULT_THREAD_ID,
):
    selected_skills = _guess_skills(query)

    try:
        agent, run_context = build_deep_agent(df=df, query=query, engine=engine)

        input_messages = (
            [{"role": "user", "content": query}]
            if thread_id
            else _history_as_deepagent_messages(query, messages)
        )
        result = agent.invoke(
            {
                "messages": input_messages,
                "files": _skill_files(create_file_data),
            },
            config={"configurable": {"thread_id": thread_id or DEFAULT_THREAD_ID}},
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if _is_quota_error(exc):
            reason = "LLM quota/rate limit exhausted. " + reason
        return _error_response(query=query, engine=engine, selected_skills=selected_skills, reason=reason)

    result_messages = result.get("messages", []) if isinstance(result, dict) else []
    final_answer = _message_content(result_messages[-1]) if result_messages else ""
    needs_data = "data-analysis" in selected_skills or "visualization" in selected_skills
    no_tool_result = not run_context.get("code") and not run_context.get("result_preview") and not run_context.get("result_base64")
    if (needs_data and no_tool_result) or _looks_like_failed_agent_answer(final_answer):
        reason = "Deep Agent did not produce an analytics tool result."
        return _error_response(query=query, engine=run_context.get("engine", engine), selected_skills=selected_skills, reason=reason)

    return {
        "final_answer": final_answer,
        "code": run_context.get("code", ""),
        "result_preview": run_context.get("result_preview", ""),
        "result_base64": run_context.get("result_base64", ""),
        "exec_error": run_context.get("exec_error"),
        "engine": run_context.get("engine", engine),
        "needs_data": needs_data,
        "use_case": "data_analytics" if "data-analysis" in selected_skills else "business_analytics",
        "selected_skills": selected_skills,
        "selected_tools": _tools_for_skills(selected_skills),
        "critic_verdict": "",
        "critic_feedback": "",
    }


def run_once(df, query, messages=None, engine="auto", thread_id: str = DEFAULT_THREAD_ID):
    """Compatibility wrapper for the old interface."""
    return run_agent(
        df=df,
        query=query,
        messages=messages,
        engine=engine,
        thread_id=thread_id,
    )
