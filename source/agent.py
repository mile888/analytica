from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Generator, List, Optional
from uuid import uuid4
import warnings

import pandas as pd

from source.checkpointing import get_checkpointer
from source.config import (
    ANALYTICA_THREAD_ID,
    ARTIFACT_DIR,
    DEEPAGENTS_MEMORY_DIR,
    DEEPAGENTS_MEMORY_FILE,
    LANGSMITH_TRACING_ENABLED,
    PROJECT_ROOT,
    SKILL_DIR,
    THREAD_PREFIX,
)
from source.dataframe import dataframe_profile
from source.engine import create_engine
from source.llm.factory import make_llm
from source.llm.llm_config import load_llm_config
from source.runtime_context import AnalyticaContext
from source.tools.analytics_tools import build_analytics_tools

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
else:
    BaseMessage = Any


def _get_deep_agent_runtime():
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Deep Agent dependencies are not installed. Run project setup first "
            "(`make setup` or install requirements.txt), then retry."
        ) from exc

    return create_deep_agent, CompositeBackend, StateBackend, FilesystemBackend, get_checkpointer()


def _tool_name(tool: Any) -> str:
    return getattr(tool, "name", None) or getattr(tool, "__name__", type(tool).__name__)


def _deep_agent_skill_sources() -> list[str]:
    relative = SKILL_DIR.relative_to(PROJECT_ROOT)
    root = PROJECT_ROOT / relative
    skill_dirs = sorted(path for path in root.iterdir() if path.is_dir() and (path / "SKILL.md").exists())
    if not skill_dirs:
        return ["/" + relative.as_posix().rstrip("/") + "/"]
    return ["/" + path.relative_to(PROJECT_ROOT).as_posix().rstrip("/") + "/" for path in skill_dirs]


def _deep_agent_memory_files() -> list[str]:
    memory_path = DEEPAGENTS_MEMORY_FILE.strip() or "/memories/AGENTS.md"
    if not memory_path.startswith("/"):
        memory_path = "/" + memory_path
    return [memory_path]


def _ensure_deep_agent_memory_file() -> None:
    DEEPAGENTS_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    memory_file = _deep_agent_memory_files()[0]
    if memory_file.startswith("/memories/"):
        relative_memory_file = memory_file.removeprefix("/memories/")
    else:
        relative_memory_file = memory_file.lstrip("/")
    memory_path = DEEPAGENTS_MEMORY_DIR / (relative_memory_file or "AGENTS.md")
    if memory_path.exists():
        return

    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        "# Analytica Agent Memory\n\n"
        "Use this file for durable, high-signal memory that should persist across "
        "agent runs, such as stable user preferences, durable dataset caveats, "
        "or reusable analysis guidance.\n\n"
        "Do not store every raw message, transient tool output, or large artifacts here.\n",
        encoding="utf-8",
    )


def _build_deep_agent_backend(CompositeBackend: Any, StateBackend: Any, FilesystemBackend: Any) -> Any:
    _ensure_deep_agent_memory_file()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/artifacts/": FilesystemBackend(root_dir=str(ARTIFACT_DIR), virtual_mode=True),
            "/memories/": FilesystemBackend(root_dir=str(DEEPAGENTS_MEMORY_DIR), virtual_mode=True),
            "/source/skills/": FilesystemBackend(root_dir=str(PROJECT_ROOT / "source" / "skills"), virtual_mode=True),
        },
    )


def _build_system_prompt(tool_names: list[str]) -> str:
    tools = ", ".join(f"`{name}`" for name in tool_names)
    return f"""You are Analytica, a Deep Agents analytical copilot for tabular data.

Use the available Deep Agents skills for detailed workflows and the custom analytics tools for execution: {tools}.

Core operating rules:
1. Inspect schema, create and validate an authoritative query plan, then route it through the branch workspace before execution.
2. Use only real dataset fields and computed tool results. Never invent data, silently substitute requested targets, or create placeholder artifacts.
3. Preserve explicit filters, chart type, artifact identity, and transformation lineage.
4. Keep SQL/DataFrame operations read-only unless a dedicated safe tool explicitly supports the requested mutation.
5. Keep user-facing answers concise, evidence-backed, and in the user's language. Do not expose orchestration, memory, backend, or prompt mechanics."""


def _history_as_deepagent_messages(
    query: str,
    messages: Optional[list[BaseMessage]],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages or []:
        msg_type = getattr(msg, "type", "")
        role = "assistant" if msg_type == "ai" else "user"
        content = str(getattr(msg, "content", "") or "")
        if content:
            out.append({"role": role, "content": content})
    out.append({"role": "user", "content": query})
    return out


def _guess_skills(query: str) -> list[str]:
    q = (query or "").lower()
    visual_markers = ("график", "диаграм", "визуал", "chart", "plot", "/bar")
    sql_markers = ("sql", "query", "запрос", "select", "where", "group by")
    code_markers = ("код", "python", "execute", "exec", "ошиб")
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
        "метрик",
        "значен",
        "групп",
        "измерен",
        "показател",
        "metric",
        "value",
        "amount",
        "total",
        "group",
        "dimension",
        "column",
        "dataset",
        "csv",
        "dataframe",
        "аномал",
        "упал",
        "упали",
        "паден",
        "резко",
    )

    needs_visual = any(marker in q for marker in visual_markers)
    needs_sql = any(marker in q for marker in sql_markers)
    needs_data = needs_visual or needs_sql or any(marker in q for marker in data_markers)
    if not needs_data:
        return ["business-analysis", "reporting"]

    skills = ["data-analysis", "csv-dataframe-analysis"]
    if needs_sql:
        skills.append("sql-querying")
    if needs_visual:
        skills.append("visualization")
    if any(marker in q for marker in code_markers):
        skills.append("code-execution-safety")
    skills.append("reporting")
    return skills


def _tools_for_skills(skills: list[str]) -> list[str]:
    tool_map = {
        "business-analysis": [],
        "code-execution-safety": ["run_python_analysis"],
        "csv-dataframe-analysis": ["inspect_dataset_schema", "top_n", "find_drops", "run_python_analysis"],
        "data-analysis": ["inspect_dataset_schema", "top_n", "find_drops", "run_python_analysis", "write_report_artifact"],
        "reporting": ["write_report_artifact"],
        "sql-querying": ["list_dataframe_tables", "describe_dataframe_table", "check_dataframe_sql", "query_dataframe_sql"],
        "visualization": ["inspect_dataset_schema", "plot_bar", "run_bar_command", "run_python_analysis", "write_report_artifact"],
    }
    out: list[str] = []
    for skill in skills:
        for tool in tool_map.get(skill, []):
            if tool not in out:
                out.append(tool)
    return out


def _query_type(selected_skills: list[str]) -> str:
    if "visualization" in selected_skills:
        return "visualization"
    if "sql-querying" in selected_skills:
        return "sql"
    if "data-analysis" in selected_skills:
        return "data-analysis"
    return "business-analysis"


def _llm_trace_info() -> dict[str, str]:
    try:
        cfg = load_llm_config()
        provider = cfg.defaults.provider
        override = cfg.node_overrides.get("deep_agent")
        provider = (override.provider if override and override.provider else None) or provider
        provider_cfg = cfg.providers.get(provider)
        model = (
            override.model
            if override and getattr(override, "model", None)
            else getattr(provider_cfg, "model", None) if provider_cfg else None
        ) or cfg.defaults.model
        return {
            "provider": provider,
            "model": model,
            "fallback_provider": cfg.defaults.fallback_provider,
            "fallback_model": cfg.defaults.fallback_model,
        }
    except Exception:
        return {"provider": "", "model": ""}


def _trace_metadata(
    df: Any,
    engine: str,
    selected_skills: list[str],
    used_tools: list[str],
) -> dict[str, Any]:
    llm = _llm_trace_info()
    return {
        "dataset_shape": tuple(getattr(df, "shape", ()) or ()),
        "engine": engine,
        "loaded_skills": selected_skills,
        "used_tools": used_tools,
        "query_type": _query_type(selected_skills),
        "provider": llm.get("provider", ""),
        "model": llm.get("model", ""),
        "fallback_provider": llm.get("fallback_provider", ""),
        "fallback_model": llm.get("fallback_model", ""),
    }


def _agent_config(thread_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    if LANGSMITH_TRACING_ENABLED:
        config["tags"] = ["analytica", "data-analysis", metadata.get("query_type", "analysis")]
        config["metadata"] = metadata
    return config


def _agent_runtime_context(
    run_context: dict[str, Any],
    thread_id: str,
    *,
    user_id: str = "local",
    session_id: str = "",
    run_id: str = "",
) -> AnalyticaContext:
    return AnalyticaContext(
        user_id=user_id,
        thread_id=thread_id,
        session_id=session_id or thread_id,
        run_id=run_id,
        artifact_dir=str(ARTIFACT_DIR),
        memory_file=_deep_agent_memory_files()[0],
        run_state=run_context,
    )


def _stage_for_tool(tool_name: str) -> str:
    mapping = {
        "inspect_dataset_schema": "schema_inspection",
        "list_dataframe_tables": "schema_inspection",
        "describe_dataframe_table": "schema_inspection",
        "check_dataframe_sql": "sql_check",
        "query_dataframe_sql": "sql_query",
        "run_python_analysis": "code_execution",
        "plot_bar": "code_execution",
        "run_bar_command": "code_execution",
        "write_report_artifact": "report_generation",
    }
    return mapping.get(tool_name, "tool")


def _structured_report(
    final_answer: str,
    run_context: dict[str, Any],
    loaded_skills: list[str],
    tool_timeline: list[dict[str, Any]],
    sql_metadata: dict[str, Any],
) -> dict[str, Any]:
    limitations: list[str] = []
    if run_context.get("exec_error"):
        limitations.append(str(run_context.get("exec_error")))
    return {
        "summary": final_answer,
        "key_findings": [run_context.get("result_facts", "")] if run_context.get("result_facts") else [],
        "limitations": limitations,
        "artifacts": run_context.get("artifacts", []),
        "next_steps": [],
        "generated_code": run_context.get("code", ""),
        "loaded_skills": loaded_skills,
        "tool_timeline": tool_timeline,
        "sql_metadata": sql_metadata,
    }


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


def _is_schema_overview_query(query: str) -> bool:
    q = (query or "").lower()
    markers = (
        "опиши структуру",
        "структур",
        "схем",
        "колон",
        "тип",
        "пропуск",
        "направления анализа",
        "что ты можешь сказать",
        "что можешь сказать",
        "об этом датасет",
        "про этот датасет",
        "какие тут данные",
        "summarize this dataset",
        "summarize dataset",
        "describe data",
        "describe dataset",
        "schema",
        "overview",
    )
    return any(marker in q for marker in markers)


def _to_pandas(df: Any, engine: str) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    return create_engine(engine).to_pandas(df)


def _schema_fields_from_profile(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric_columns: list[str] = []
    date_columns: list[str] = []
    categorical_columns: list[str] = []
    for col in df.columns:
        col_name = str(col)
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(col_name)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(series.dropna().head(50), errors="coerce", dayfirst=True)
        if len(parsed) and parsed.notna().mean() >= 0.7:
            date_columns.append(col_name)
        else:
            categorical_columns.append(col_name)
    return {
        "numeric_columns": numeric_columns,
        "date_columns": date_columns,
        "categorical_columns": categorical_columns,
    }


def _schema_overview_response(
    df: Any,
    query: str,
    engine: str,
    selected_skills: list[str],
    selected_tools: list[str],
    thread_id: str,
) -> dict[str, Any]:
    pdf = _to_pandas(df, "pandas" if engine == "auto" else engine)
    profile = dataframe_profile(pdf)
    fields = _schema_fields_from_profile(pdf)
    missing_total = sum(profile["missing"].values())
    numeric = fields["numeric_columns"]
    dates = fields["date_columns"]
    categorical = fields["categorical_columns"]

    directions = [
        "проверить пропуски, типы данных и дубликаты",
        "посчитать агрегаты по ключевым категориям",
        "найти топ-N значений по числовым метрикам",
    ]
    if dates and numeric:
        directions.append("проанализировать динамику и падения метрик по датам")
    if categorical and numeric:
        directions.append("сравнить сегменты/категории по бизнес-метрикам")
    if numeric:
        directions.append("построить визуализации распределений и bar chart по выбранной метрике")

    final_answer = "\n".join(
        [
            f"Я вижу основу для аналитического расследования: {profile['rows']} строк и {profile['columns_count']} колонок.",
            f"Лучшие количественные точки входа: {', '.join(numeric[:6]) if numeric else 'явные числовые метрики не обнаружены'}.",
            f"Для сегментации подойдут: {', '.join(categorical[:8]) if categorical else 'явные категориальные разрезы не обнаружены'}.",
            f"Для динамики можно проверить: {', '.join(dates) if dates else 'явные временные колонки не обнаружены'}.",
            f"Качество данных: найдено {missing_total} пропусков.",
            "Я бы продолжил так: " + "; ".join(directions) + ".",
        ]
    )
    result_preview = "\n".join(
        [
            f"shape: ({profile['rows']}, {profile['columns_count']})",
            f"columns: {profile['columns']}",
            f"dtypes: {profile['dtypes']}",
            f"missing: {profile['missing']}",
        ]
    )
    result_facts = (
        f"The dataset has {profile['rows']} rows and {profile['columns_count']} columns. "
        f"Useful quantitative fields include {', '.join(numeric[:6]) if numeric else 'no clear numeric metric yet'}. "
        f"Useful segmentation fields include {', '.join(categorical[:6]) if categorical else 'no clear category yet'}. "
        f"Data quality review should account for {missing_total} missing values."
    )
    tool_timeline = [
        {
            "tool": "inspect_dataset_schema",
            "status": "ok",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "engine": "pandas" if engine == "auto" else engine,
            "columns": profile["columns_count"],
        }
    ]
    loaded_skills = [skill for skill in ("data-analysis", "csv-dataframe-analysis", "reporting") if skill in selected_skills]
    run_context = {
        "query": query,
        "code": "",
        "exec_error": None,
        "result_preview": result_preview,
        "result_facts": result_facts,
        "result_base64": "",
        "artifacts": [],
    }
    trace_metadata = _trace_metadata(
        pdf,
        "pandas" if engine == "auto" else engine,
        selected_skills,
        ["inspect_dataset_schema"],
    )
    return {
        "final_answer": final_answer,
        "code": "",
        "result_preview": result_preview,
        "result_base64": "",
        "exec_error": None,
        "engine": "pandas" if engine == "auto" else engine,
        "needs_data": True,
        "use_case": "data_analytics",
        "selected_skills": selected_skills,
        "selected_tools": selected_tools,
        "loaded_skills": loaded_skills,
        "tool_timeline": tool_timeline,
        "sql_metadata": {},
        "structured_report": _structured_report(final_answer, run_context, loaded_skills, tool_timeline, {}),
        "trace_metadata": trace_metadata,
        "critic_verdict": "",
        "critic_feedback": "",
        "artifacts": [],
        "thread_id": thread_id,
    }


def _error_response(
    query: str,
    engine: str,
    selected_skills: list[str],
    reason: str,
) -> dict[str, Any]:
    run_context: dict[str, Any] = {
        "code": "",
        "exec_error": reason,
        "result_facts": "",
        "artifacts": [],
    }
    structured_report = _structured_report(
        final_answer="Не удалось выполнить запрос через Deep Agent runtime.",
        run_context=run_context,
        loaded_skills=[],
        tool_timeline=[],
        sql_metadata={},
    )
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
        "loaded_skills": [],
        "tool_timeline": [],
        "sql_metadata": {},
        "structured_report": structured_report,
        "critic_verdict": "ERROR",
        "critic_feedback": reason,
    }


def build_deep_agent(df: Any, query: str, engine: str = "auto"):
    """Create a Deep Agent plus a mutable run context for compatibility output."""
    create_deep_agent, CompositeBackend, StateBackend, FilesystemBackend, checkpointer = _get_deep_agent_runtime()
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
        "loaded_skills": [],
        "tool_timeline": [],
    }

    tools = build_analytics_tools()
    tool_names = [_tool_name(tool) for tool in tools]

    agent = create_deep_agent(
        model=make_llm("deep_agent"),
        tools=tools,
        system_prompt=_build_system_prompt(tool_names),
        skills=_deep_agent_skill_sources(),
        memory=_deep_agent_memory_files(),
        backend=_build_deep_agent_backend(CompositeBackend, StateBackend, FilesystemBackend),
        context_schema=AnalyticaContext,
        checkpointer=checkpointer,
    )
    return agent, run_context


def run_agent(
    df: Any,
    query: str,
    messages: Optional[List[BaseMessage]] = None,
    engine: str = "auto",
    thread_id: str | None = None,
):
    selected_skills = _guess_skills(query)
    resolved_thread_id = thread_id or ANALYTICA_THREAD_ID or f"{THREAD_PREFIX}-{uuid4().hex}"
    selected_tools = _tools_for_skills(selected_skills)
    trace_metadata = _trace_metadata(df, engine, selected_skills, selected_tools)

    if _is_schema_overview_query(query):
        return _schema_overview_response(
            df=df,
            query=query,
            engine=engine,
            selected_skills=selected_skills,
            selected_tools=selected_tools,
            thread_id=resolved_thread_id,
        )

    try:
        agent, run_context = build_deep_agent(df=df, query=query, engine=engine)

        input_messages = [{"role": "user", "content": query}]
        if messages and not thread_id:
            input_messages = _history_as_deepagent_messages(query, messages)
        result = agent.invoke(
            {"messages": input_messages},
            config=_agent_config(resolved_thread_id, trace_metadata),
            context=_agent_runtime_context(run_context, resolved_thread_id),
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

    loaded_skills = run_context.get("loaded_skills", [])
    tool_timeline = run_context.get("tool_timeline", [])
    sql_metadata = run_context.get("sql_metadata", {})
    structured_report = _structured_report(
        final_answer=final_answer,
        run_context=run_context,
        loaded_skills=loaded_skills,
        tool_timeline=tool_timeline,
        sql_metadata=sql_metadata,
    )
    trace_metadata = _trace_metadata(
        df,
        run_context.get("engine", engine),
        selected_skills,
        [event.get("tool", "") for event in tool_timeline] or selected_tools,
    )

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
        "selected_tools": selected_tools,
        "loaded_skills": loaded_skills,
        "tool_timeline": tool_timeline,
        "sql_metadata": sql_metadata,
        "structured_report": structured_report,
        "trace_metadata": trace_metadata,
        "critic_verdict": "",
        "critic_feedback": "",
        "artifacts": run_context.get("artifacts", []),
        "thread_id": resolved_thread_id,
    }


def _emit_tool_events(
    run_context: dict[str, Any],
    emitted_count: int,
) -> tuple[int, list[dict[str, Any]]]:
    timeline = run_context.get("tool_timeline", []) or []
    events: list[dict[str, Any]] = []
    for event in timeline[emitted_count:]:
        tool_name = str(event.get("tool", ""))
        events.append(
            {
                "event": "tool",
                "stage": _stage_for_tool(tool_name),
                "message": tool_name,
                "tool_event": event,
            }
        )
    return len(timeline), events


def run_agent_stream(
    df: Any,
    query: str,
    messages: Optional[List[BaseMessage]] = None,
    engine: str = "auto",
    thread_id: str | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Stream coarse agent progress events and end with the normal run_agent payload.

    If the Deep Agent runtime does not expose streaming or streaming fails before a
    final state is available, the generator falls back to the stable run_agent path.
    """
    selected_skills = _guess_skills(query)
    selected_tools = _tools_for_skills(selected_skills)
    resolved_thread_id = thread_id or ANALYTICA_THREAD_ID or f"{THREAD_PREFIX}-{uuid4().hex}"
    trace_metadata = _trace_metadata(df, engine, selected_skills, selected_tools)

    yield {
        "event": "start",
        "stage": "agent_start",
        "message": "Agent started",
        "selected_skills": selected_skills,
        "selected_tools": selected_tools,
        "trace_metadata": trace_metadata,
    }

    if _is_schema_overview_query(query):
        output = _schema_overview_response(
            df=df,
            query=query,
            engine=engine,
            selected_skills=selected_skills,
            selected_tools=selected_tools,
            thread_id=resolved_thread_id,
        )
        for event in output.get("tool_timeline", []) or []:
            tool_name = str(event.get("tool", ""))
            yield {
                "event": "tool",
                "stage": _stage_for_tool(tool_name),
                "message": tool_name,
                "tool_event": event,
            }
        yield {"event": "final", "stage": "done", "message": "Agent completed", "output": output}
        return

    try:
        agent, run_context = build_deep_agent(df=df, query=query, engine=engine)
        input_messages = [{"role": "user", "content": query}]
        if messages and not thread_id:
            input_messages = _history_as_deepagent_messages(query, messages)

        if not hasattr(agent, "stream"):
            raise RuntimeError("Deep Agent runtime has no stream() method.")

        config = _agent_config(resolved_thread_id, trace_metadata)
        emitted_count = 0
        result: Any = None

        try:
            runtime_context = _agent_runtime_context(run_context, resolved_thread_id)
            chunks = agent.stream(
                {"messages": input_messages},
                config=config,
                context=runtime_context,
                stream_mode="values",
            )
        except TypeError:
            chunks = agent.stream(
                {"messages": input_messages},
                config=config,
                context=_agent_runtime_context(run_context, resolved_thread_id),
            )

        for chunk in chunks:
            if isinstance(chunk, dict):
                result = chunk
            yield {"event": "chunk", "stage": "agent", "message": "Agent step received"}
            emitted_count, tool_events = _emit_tool_events(run_context, emitted_count)
            for event in tool_events:
                yield event

        emitted_count, tool_events = _emit_tool_events(run_context, emitted_count)
        for event in tool_events:
            yield event

        if not isinstance(result, dict) or not result.get("messages"):
            raise RuntimeError("Streaming completed without final agent messages.")

        result_messages = result.get("messages", [])
        final_answer = _message_content(result_messages[-1]) if result_messages else ""
        needs_data = "data-analysis" in selected_skills or "visualization" in selected_skills
        no_tool_result = not run_context.get("code") and not run_context.get("result_preview") and not run_context.get("result_base64")
        if (needs_data and no_tool_result) or _looks_like_failed_agent_answer(final_answer):
            raise RuntimeError("Deep Agent stream did not produce an analytics tool result.")

        loaded_skills = run_context.get("loaded_skills", [])
        tool_timeline = run_context.get("tool_timeline", [])
        sql_metadata = run_context.get("sql_metadata", {})
        trace_metadata = _trace_metadata(
            df,
            run_context.get("engine", engine),
            selected_skills,
            [event.get("tool", "") for event in tool_timeline] or selected_tools,
        )
        output = {
            "final_answer": final_answer,
            "code": run_context.get("code", ""),
            "result_preview": run_context.get("result_preview", ""),
            "result_base64": run_context.get("result_base64", ""),
            "exec_error": run_context.get("exec_error"),
            "engine": run_context.get("engine", engine),
            "needs_data": needs_data,
            "use_case": "data_analytics" if "data-analysis" in selected_skills else "business_analytics",
            "selected_skills": selected_skills,
            "selected_tools": selected_tools,
            "loaded_skills": loaded_skills,
            "tool_timeline": tool_timeline,
            "sql_metadata": sql_metadata,
            "structured_report": _structured_report(final_answer, run_context, loaded_skills, tool_timeline, sql_metadata),
            "trace_metadata": trace_metadata,
            "critic_verdict": "",
            "critic_feedback": "",
            "artifacts": run_context.get("artifacts", []),
            "thread_id": resolved_thread_id,
        }
        yield {"event": "final", "stage": "done", "message": "Agent completed", "output": output}
    except Exception as exc:
        yield {
            "event": "fallback",
            "stage": "fallback",
            "message": "Streaming fallback used. Stable run_agent will produce the final result.",
            "detail": f"{type(exc).__name__}: {exc}",
        }
        output = run_agent(df=df, query=query, messages=messages, engine=engine, thread_id=resolved_thread_id)
        for event in output.get("tool_timeline", []) or []:
            tool_name = str(event.get("tool", ""))
            yield {
                "event": "tool",
                "stage": _stage_for_tool(tool_name),
                "message": tool_name,
                "tool_event": event,
            }
        output["streaming_fallback"] = True
        yield {"event": "final", "stage": "done", "message": "Agent completed via fallback", "output": output}


def run_once(df, query, messages=None, engine="auto", thread_id: str | None = None):
    """Compatibility wrapper for the old interface."""
    return run_agent(
        df=df,
        query=query,
        messages=messages,
        engine=engine,
        thread_id=thread_id,
    )
