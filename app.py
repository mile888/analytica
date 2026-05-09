from __future__ import annotations

import io
import re
import uuid
import warnings
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from source.config import (
    BAR_CHART_SUGGESTION_TEMPLATE,
    DEFAULT_DATA_PATH,
    SCHEMA_SUGGESTION_PROMPT,
    SQL_TOP_N_SUGGESTION_TEMPLATE,
    THREAD_PREFIX,
    TOP_N_SUGGESTION_TEMPLATE,
    ANALYTICA_THREAD_ID,
)
from source.dataframe import read_csv_dataset
from source.agent import run_agent_stream, run_once
from source.engine import create_engine


def inject_styles():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
          .title { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin: 0.2rem 0 0.2rem 0;}
          .subtitle { color: #6b7280; margin-bottom: 0.9rem; }
          .pill {
            display:inline-block; padding: 3px 10px; border-radius: 999px;
            border: 1px solid rgba(0,0,0,.10); background: rgba(0,0,0,.03);
            font-size: 0.85rem; margin-right: 6px; margin-bottom: 6px;
          }
          .card {
            border: 1px solid rgba(0,0,0,.08);
            border-radius: 18px;
            padding: 14px 16px;
            background: rgba(255,255,255,.75);
            box-shadow: 0 10px 25px rgba(0,0,0,.06);
          }
          .muted { color: #6b7280; }
          .mono textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
          .small { font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str):
    st.markdown(f"<span class='pill'>{text}</span>", unsafe_allow_html=True)


def card(title: str, body_html: str, icon: str = "🧩"):
    st.markdown(
        f"""
        <div class="card">
          <div style="font-weight:700; margin-bottom:.35rem;">{icon} {title}</div>
          <div class="small" style="color:#374151; line-height:1.4;">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


BROKEN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img[^>]*>", re.IGNORECASE)


def load_default_df() -> Optional[pd.DataFrame]:
    if DEFAULT_DATA_PATH.exists():
        try:
            return read_csv_dataset(DEFAULT_DATA_PATH)
        except Exception:
            return None
    return None


def make_thread_id() -> str:
    if ANALYTICA_THREAD_ID:
        return ANALYTICA_THREAD_ID
    return f"{THREAD_PREFIX}-streamlit-{uuid.uuid4().hex}"


def load_uploaded_csv(uploaded_file, sep: str, encoding: str) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    try:
        return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding)
    except pd.errors.ParserError:
        return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding, escapechar=chr(92))


@dataclass
class TurnMeta:
    timestamp: str
    query: str
    final_answer: str
    code: str
    result_preview: str
    result_base64: str
    exec_error: Optional[str]
    critic_verdict: str
    critic_feedback: str
    engine: str
    needs_data: Optional[bool]
    use_case: str
    loaded_skills: List[str]
    tool_timeline: List[Dict[str, Any]]
    sql_metadata: Dict[str, Any]
    structured_report: Dict[str, Any]
    trace_metadata: Dict[str, Any]
    artifacts: List[Dict[str, Any]]
    streaming_fallback: bool


def init_session():
    if "df" not in st.session_state:
        st.session_state.df = load_default_df()
    if "chat" not in st.session_state:
        st.session_state.chat: List[BaseMessage] = []
    if "turns" not in st.session_state:
        st.session_state.turns: List[TurnMeta] = []
    if "last_raw" not in st.session_state:
        st.session_state.last_raw: Dict[str, Any] = {}
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = make_thread_id()


def append_turn(out: Dict[str, Any], query: str):
    meta = TurnMeta(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        query=query,
        final_answer=out.get("final_answer", "") or "",
        code=out.get("code", "") or "",
        result_preview=out.get("result_preview", "") or "",
        result_base64=out.get("result_base64", "") or "",
        exec_error=out.get("exec_error", None),
        critic_verdict=out.get("critic_verdict", "") or "",
        critic_feedback=out.get("critic_feedback", "") or "",
        engine=out.get("engine", "") or "",
        needs_data=out.get("needs_data", None),
        use_case=out.get("use_case", "") or "",
        loaded_skills=list(out.get("loaded_skills", []) or []),
        tool_timeline=list(out.get("tool_timeline", []) or []),
        sql_metadata=dict(out.get("sql_metadata", {}) or {}),
        structured_report=dict(out.get("structured_report", {}) or {}),
        trace_metadata=dict(out.get("trace_metadata", {}) or {}),
        artifacts=list(out.get("artifacts", []) or out.get("structured_report", {}).get("artifacts", []) or []),
        streaming_fallback=bool(out.get("streaming_fallback", False)),
    )
    st.session_state.turns.append(meta)
    st.session_state.last_raw = out


def clean_chat_answer(text: str) -> str:
    cleaned = BROKEN_IMAGE_RE.sub("", text or "")

    def replace_local_markdown_link(match: re.Match[str]) -> str:
        label = match.group(1)
        target = match.group(2).strip()
        if target.startswith(("http://", "https://", "mailto:")):
            return match.group(0)
        return label

    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace_local_markdown_link, cleaned)
    cleaned = re.sub(
        r"<a\s+[^>]*href=[\"'](?!https?://|mailto:)[^\"']+[\"'][^>]*>(.*?)</a>",
        r"\1",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = cleaned.replace("по следующей ссылке:", "ниже:")
    return cleaned.strip()


def render_artifact_downloads(artifacts: List[Dict[str, Any]]) -> None:
    if not artifacts:
        st.info("artifacts пустой")
        return

    for idx, artifact in enumerate(artifacts, start=1):
        title = str(artifact.get("title") or artifact.get("path") or f"Artifact {idx}")
        path_value = artifact.get("path")
        artifact_type = str(artifact.get("artifact_type") or artifact.get("type") or "artifact")
        if path_value:
            path = Path(str(path_value))
            if path.exists() and path.is_file():
                st.download_button(
                    label=f"Скачать: {title}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="text/markdown" if path.suffix.lower() == ".md" else "application/octet-stream",
                    key=f"artifact-download-{idx}-{path.name}",
                    width="stretch",
                )
                st.caption(f"{artifact_type}: {path}")
            else:
                st.warning(f"{artifact_type}: файл не найден: {path}")
        else:
            st.json(artifact)


def infer_demo_suggestions(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []

    numeric_cols = [str(c) for c in df.select_dtypes(include="number").columns]
    non_numeric_cols = [str(c) for c in df.columns if str(c) not in numeric_cols]
    date_cols: list[str] = []
    for col in non_numeric_cols:
        sample = df[col].dropna().head(50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
        if len(parsed) and parsed.notna().mean() >= 0.7:
            date_cols.append(str(col))

    def score_dimension(col: str) -> tuple[int, int]:
        lowered = col.lower()
        bad_name = (
            lowered == "id"
            or lowered.endswith(" id")
            or lowered.endswith("_id")
            or "code" in lowered
            or "name" in lowered
        )
        cardinality = int(df[col].nunique(dropna=True)) if col in df else 0
        too_unique = cardinality > max(30, int(len(df) * 0.5))
        return (1 if bad_name or too_unique else 0, cardinality)

    def score_metric(col: str) -> tuple[int, int]:
        lowered = col.lower()
        bad_name = lowered == "id" or lowered.endswith(" id") or lowered.endswith("_id") or "code" in lowered or "postal" in lowered
        preferred = any(marker in lowered for marker in ("sales", "revenue", "profit", "amount", "price", "metric", "value"))
        return (0 if preferred else 1, 1 if bad_name else 0)

    dimension_candidates = [c for c in non_numeric_cols if c not in date_cols]
    metric_candidates = list(numeric_cols)
    dimension = min(dimension_candidates, key=score_dimension) if dimension_candidates else ""
    metric = min(metric_candidates, key=score_metric) if metric_candidates else ""
    date_col = date_cols[0] if date_cols else ""

    suggestions = [SCHEMA_SUGGESTION_PROMPT]
    if dimension and metric:
        suggestions.append(TOP_N_SUGGESTION_TEMPLATE.format(dimension=dimension, metric=metric))
        suggestions.append(BAR_CHART_SUGGESTION_TEMPLATE.format(dimension=dimension, metric=metric))
        suggestions.append(SQL_TOP_N_SUGGESTION_TEMPLATE.format(dimension=dimension, metric=metric))
    if date_col and metric:
        suggestions.append(f"Найди самые сильные падения `{metric}` по датам из `{date_col}`")
    suggestions.append("Сделай краткий отчёт по найденным фактам")
    return suggestions


def render_assistant_turn(t: TurnMeta):
    answer = clean_chat_answer(t.final_answer)
    st.markdown(answer if answer.strip() else "*(пустой ответ)*")
    if t.result_base64:
        st.image(t.result_base64, width="stretch")
    if t.artifacts:
        render_artifact_downloads(t.artifacts)


def run_with_streaming_progress(df: pd.DataFrame, prompt: str, engine: str) -> Dict[str, Any]:
    stage_labels = {
        "agent_start": "Запуск агента",
        "skill_loading": "Skill loading",
        "schema_inspection": "Schema inspection",
        "sql_check": "SQL check",
        "sql_query": "SQL query",
        "code_execution": "Code execution",
        "report_generation": "Report generation",
        "fallback": "Fallback",
        "done": "Готово",
    }
    progress_rows: list[dict[str, Any]] = []
    output: Dict[str, Any] = {}

    status = st.status("Агент запускается...", expanded=True)
    timeline_slot = st.empty()

    try:
        for event in run_agent_stream(
            df,
            prompt,
            messages=st.session_state.chat[:-1],
            engine=engine,
            thread_id=st.session_state.thread_id,
        ):
            stage = str(event.get("stage", "agent"))
            label = stage_labels.get(stage, stage)
            message = str(event.get("message", label))
            progress_rows.append(
                {
                    "stage": label,
                    "event": event.get("event", ""),
                    "message": message,
                }
            )
            status.update(label=f"{label}: {message}", state="running")
            timeline_slot.dataframe(pd.DataFrame(progress_rows), width="stretch", height=220)
            if event.get("event") == "final":
                output = dict(event.get("output", {}) or {})
    except Exception as e:
        status.update(label="Streaming fallback: использую стабильный run_once", state="running")
        output = run_once(
            df,
            prompt,
            messages=st.session_state.chat[:-1],
            engine=engine,
            thread_id=st.session_state.thread_id,
        )
        output["streaming_fallback"] = True
        progress_rows.append({"stage": "Fallback", "event": "fallback", "message": f"{type(e).__name__}: {e}"})
        timeline_slot.dataframe(pd.DataFrame(progress_rows), width="stretch", height=220)

    if not output:
        output = run_once(
            df,
            prompt,
            messages=st.session_state.chat[:-1],
            engine=engine,
            thread_id=st.session_state.thread_id,
        )
        output["streaming_fallback"] = True
        progress_rows.append({"stage": "Fallback", "event": "fallback", "message": "No final streaming output"})
        timeline_slot.dataframe(pd.DataFrame(progress_rows), width="stretch", height=220)

    if output.get("streaming_fallback"):
        status.update(label="Готово: использован стабильный fallback run_agent", state="complete")
    else:
        status.update(label="Готово", state="complete")
    return output


def build_markdown_report() -> str:
    lines: List[str] = []
    lines.append("# Analytica — отчёт\n")
    if st.session_state.df is not None:
        df: pd.DataFrame = st.session_state.df
        lines.append("## Датасет\n")
        lines.append(f"- shape: `{df.shape}`\n")
        lines.append(f"- columns: `{list(df.columns)}`\n")

    lines.append("\n---\n")
    lines.append("## Диалог\n")
    for i, t in enumerate(st.session_state.turns, start=1):
        lines.append(f"\n### {i}. {t.timestamp}\n")
        lines.append(f"**Запрос:** {t.query}\n")
        lines.append("\n**Ответ:**\n")
        lines.append(t.final_answer.strip() or "(пусто)")
        lines.append("\n")
        lines.append(f"\n**use_case:** `{t.use_case}`  \n**engine:** `{t.engine}`  \n**needs_data:** `{t.needs_data}`\n")
        if t.loaded_skills:
            lines.append(f"\n**loaded_skills:** `{t.loaded_skills}`\n")
        if t.tool_timeline:
            lines.append(f"\n**tool_timeline:** `{t.tool_timeline}`\n")
        if t.sql_metadata:
            lines.append(f"\n**sql_metadata:** `{t.sql_metadata}`\n")
        if t.structured_report:
            lines.append(f"\n**structured_report:** `{t.structured_report}`\n")
        if t.trace_metadata:
            lines.append(f"\n**trace_metadata:** `{t.trace_metadata}`\n")
        if t.artifacts:
            lines.append(f"\n**artifacts:** `{t.artifacts}`\n")
        if t.critic_verdict or t.critic_feedback:
            lines.append(f"\n**critic_verdict:** `{t.critic_verdict}`\n")
            if t.critic_feedback:
                lines.append("\n**critic_feedback:**\n")
                lines.append(t.critic_feedback.strip())
                lines.append("\n")
        if t.exec_error:
            lines.append("\n**exec_error:**\n")
            lines.append(f"```\n{t.exec_error}\n```\n")
        if t.result_preview:
            lines.append("\n**result_preview:**\n")
            lines.append(f"```\n{t.result_preview}\n```\n")
        if t.code:
            lines.append("\n**code:**\n")
            lines.append(f"```python\n{t.code}\n```\n")
        lines.append("\n---\n")
    return "\n".join(lines).strip() + "\n"


st.set_page_config(page_title="Analytica • Demo", layout="wide")
inject_styles()
init_session()

st.markdown("<div class='title'>Analytica Demo</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='subtitle'>Загрузи CSV (или используй <code>{DEFAULT_DATA_PATH}</code>) → задай вопрос → агент построит план, код и ответ.</div>",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("⚙️ Настройки")

    st.subheader("Датасет")
    uploaded = st.file_uploader("Загрузить CSV", type=["csv"])

    sep = st.selectbox("Разделитель", [",", ";", "\t"], index=0)
    encoding = st.selectbox("Encoding", ["utf-8", "utf-8-sig", "cp1251"], index=0)

    if uploaded is not None:
        try:
            st.session_state.df = load_uploaded_csv(uploaded, sep=sep, encoding=encoding)
            st.success("CSV загружен")
        except Exception as e:
            st.error(f"Не удалось прочитать CSV: {e}")

    if st.button("Очистить диалог", width="stretch"):
        st.session_state.chat = []
        st.session_state.turns = []
        st.session_state.last_raw = {}
        st.session_state.thread_id = make_thread_id()

    st.divider()

    st.subheader("Движок")
    engine = st.selectbox("engine для run_once()", ["auto", "pandas", "polars", "spark"], index=0)
    st.caption("Если выберешь polars/spark — убедись, что зависимости установлены и агент умеет их использовать.")


    st.divider()

    report_md = build_markdown_report()
    st.download_button(
        "Скачать отчёт (md)",
        data=report_md.encode("utf-8"),
        file_name="analytica_report.md",
        mime="text/markdown",
        width="stretch",
        disabled=(len(st.session_state.turns) == 0),
    )


df = st.session_state.df

top_left, top_right = st.columns([1.15, 0.85], gap="large")

with top_left:
    if df is None:
        card(
            "Нет данных",
            f"Положи файл <code>{DEFAULT_DATA_PATH}</code> или загрузи CSV в сайдбаре.",
            icon="📁",
        )
    else:
        pill(f"rows: {df.shape[0]}")
        pill(f"cols: {df.shape[1]}")
        st.subheader("Данные (превью)")
        st.dataframe(df.head(50), width="stretch", height=360)

        with st.expander("Схема (как видит агент)"):
            try:
                schema_engine = engine if engine != "auto" else "pandas"
                st.code(create_engine(schema_engine).schema_text(df), language="text")
            except Exception as e:
                st.error(f"Не удалось построить схему: {e}")

with top_right:
    st.subheader("Чат")

    with st.expander("Подсказки для демо", expanded=(len(st.session_state.turns) == 0)):
        for suggestion in infer_demo_suggestions(df):
            st.markdown(f"- {suggestion}")

    if st.session_state.turns:
        for t in st.session_state.turns:
            with st.chat_message("user"):
                st.markdown(t.query)
            with st.chat_message("assistant"):
                render_assistant_turn(t)
    else:
        for m in st.session_state.chat:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(getattr(m, "content", "") or "")

    prompt = st.chat_input("Спроси что-нибудь про данные…", disabled=(df is None))
    if prompt and df is not None:
        st.session_state.chat.append(HumanMessage(content=prompt))

        try:
            out = run_with_streaming_progress(df, prompt, engine)
        except Exception as e:
            out = {
                "final_answer": (
                    "Не удалось выполнить запрос: модель или агент вернули ошибку. "
                    "Подробности сохранены в блоке Exec error ниже."
                ),
                "code": "",
                "result_preview": "",
                "result_base64": "",
                "exec_error": f"{type(e).__name__}: {e}",
                "critic_verdict": "ERROR",
                "critic_feedback": "",
                "engine": engine,
                "needs_data": None,
                "use_case": "error",
            }
        answer = out.get("final_answer", "") or ""

        st.session_state.chat.append(AIMessage(content=answer))
        append_turn(out, prompt)
        st.rerun()


st.divider()
st.subheader("Детали последнего запуска")

if len(st.session_state.turns) == 0:
    card("Пока пусто", "Спроси что-нибудь — здесь появятся код, превью результата и вердикт критика.")
else:
    t = st.session_state.turns[-1]

    meta_cols = st.columns(5)
    with meta_cols[0]:
        pill(f"time: {t.timestamp}")
    with meta_cols[1]:
        pill(f"use_case: {t.use_case or '-'}")
    with meta_cols[2]:
        pill(f"engine: {t.engine or '-'}")
    with meta_cols[3]:
        pill(f"needs_data: {t.needs_data}")
    with meta_cols[4]:
        pill(f"critic: {t.critic_verdict or '-'}")

    if t.loaded_skills:
        pill(f"loaded_skills: {', '.join(t.loaded_skills)}")
    if t.streaming_fallback:
        pill("streaming: fallback")

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        with st.expander("Critic feedback", expanded=bool(t.critic_feedback)):
            if t.critic_feedback:
                st.markdown(t.critic_feedback)
            else:
                st.info("critic_feedback пустой")

        with st.expander("Exec error", expanded=bool(t.exec_error)):
            if t.exec_error:
                st.error(t.exec_error)
            else:
                st.success("exec_error пустой")

        with st.expander("Result preview", expanded=True):
            if t.result_preview:
                st.code(t.result_preview)
            else:
                st.info("result_preview пустой")

        with st.expander("Tool timeline", expanded=bool(t.tool_timeline)):
            if t.tool_timeline:
                st.dataframe(pd.DataFrame(t.tool_timeline), width="stretch", height=220)
            else:
                st.info("tool_timeline пустой")

    with c2:
        with st.expander("Generated code", expanded=bool(t.code)):
            if t.code:
                st.code(t.code, language="python")
            else:
                st.info("code пустой")

        with st.expander("Результат", expanded=True):
            if t.result_base64:
                st.image(t.result_base64, width="stretch")
            elif t.result_preview:
                st.code(t.result_preview)
            else:
                st.info("Нет результата для отображения.")

        with st.expander("SQL metadata", expanded=bool(t.sql_metadata)):
            if t.sql_metadata:
                st.json(t.sql_metadata)
            else:
                st.info("sql_metadata пустой")

        with st.expander("Artifacts", expanded=bool(t.artifacts)):
            render_artifact_downloads(t.artifacts)

        with st.expander("Structured report", expanded=False):
            if t.structured_report:
                st.json(t.structured_report)
            else:
                st.info("structured_report пустой")

        with st.expander("Trace metadata", expanded=False):
            if t.trace_metadata:
                st.json(t.trace_metadata)
            else:
                st.info("trace_metadata пустой")
