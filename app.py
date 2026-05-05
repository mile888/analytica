from __future__ import annotations

import io
import os
import re
import uuid
from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from source.agent import run_once
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


DEFAULT_TRAIN = os.path.join("data", "train.csv")
BROKEN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img[^>]*>", re.IGNORECASE)


def load_default_df() -> Optional[pd.DataFrame]:
    if os.path.exists(DEFAULT_TRAIN):
        try:
            return pd.read_csv(DEFAULT_TRAIN)
        except Exception:
            return None
    return None


def load_uploaded_csv(uploaded_file, sep: str, encoding: str) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding)


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
        st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex}"


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
    )
    st.session_state.turns.append(meta)
    st.session_state.last_raw = out


def clean_chat_answer(text: str) -> str:
    cleaned = BROKEN_IMAGE_RE.sub("", text or "").strip()
    return cleaned


def render_assistant_turn(t: TurnMeta):
    answer = clean_chat_answer(t.final_answer)
    st.markdown(answer if answer.strip() else "*(пустой ответ)*")
    if t.result_base64:
        st.image(t.result_base64, use_container_width=True)


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
    "<div class='subtitle'>Загрузи CSV (или используй <code>data/train.csv</code>) → задай вопрос → агент построит план, код и ответ.</div>",
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

    if st.button("Очистить диалог", use_container_width=True):
        st.session_state.chat = []
        st.session_state.turns = []
        st.session_state.last_raw = {}
        st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex}"

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
        use_container_width=True,
        disabled=(len(st.session_state.turns) == 0),
    )


df = st.session_state.df

top_left, top_right = st.columns([1.15, 0.85], gap="large")

with top_left:
    if df is None:
        card(
            "Нет данных",
            "Положи файл <code>data/train.csv</code> или загрузи CSV в сайдбаре.",
            icon="📁",
        )
    else:
        pill(f"rows: {df.shape[0]}")
        pill(f"cols: {df.shape[1]}")
        st.subheader("Данные (превью)")
        st.dataframe(df.head(50), use_container_width=True, height=360)

        with st.expander("Схема (как видит агент)"):
            try:
                schema_engine = engine if engine != "auto" else "pandas"
                st.code(create_engine(schema_engine).schema_text(df), language="text")
            except Exception as e:
                st.error(f"Не удалось построить схему: {e}")

with top_right:
    st.subheader("Чат")

    with st.expander("Подсказки для демо", expanded=(len(st.session_state.turns) == 0)):
        st.markdown(
            """
            - Построй топ-5 категорий по выручке и покажи график
            - Найди аномалии по дням: где продажи резко упали?
            - Сделай срез по региону и месяцу, сравни динамику
            - `/bar category revenue` — быстрый bar-командой
            """
        )

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

        with st.spinner("Агент думает…"):
            try:
                out = run_once(
                    df,
                    prompt,
                    messages=st.session_state.chat[:-1],
                    engine=engine,
                    thread_id=st.session_state.thread_id,
                )
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

    with c2:
        with st.expander("Generated code", expanded=bool(t.code)):
            if t.code:
                st.code(t.code, language="python")
            else:
                st.info("code пустой")

        with st.expander("Результат", expanded=True):
            if t.result_base64:
                st.image(t.result_base64, use_container_width=True)
            elif t.result_preview:
                st.code(t.result_preview)
            else:
                st.info("Нет результата для отображения.")
