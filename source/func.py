from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Literal
import os
import re
import shlex

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

EngineName = Literal["pandas", "polars", "spark"]


def make_llm(purpose: str = "general"):
    """
    purpose: router | planner | codegen | critic | reporter | general
    Мягкая стабилизация: для кода/критика/роутера температура ниже.
    """
    model = os.getenv("ANALYTICA_MODEL", "gemini-2.5-flash")
    purpose = (purpose or "general").lower().strip()

    temperature = 0.7
    if purpose in {"codegen", "critic", "router"}:
        temperature = float(os.getenv("ANALYTICA_TEMP_LOW", "0.2"))
    elif purpose in {"reporter"}:
        temperature = float(os.getenv("ANALYTICA_TEMP_REPORTER", "0.4"))
    elif purpose in {"planner"}:
        temperature = float(os.getenv("ANALYTICA_TEMP_PLANNER", "0.7"))

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


def detect_engine(df: Any, engine: Optional[str] = None) -> EngineName:
    """
    Выбор движка по use-case с мягкими дефолтами:
      <= 1m   -> pandas
      <= 15m  -> polars
      > 15m   -> spark
    Можно зафиксировать через ANALYTICA_ENGINE=pandas|polars|spark
    """
    forced = (engine or os.getenv("ANALYTICA_ENGINE", "auto")).lower().strip()
    if forced in {"pandas", "polars", "spark"}:
        return forced

    pandas_max = int(os.getenv("ANALYTICA_PANDAS_MAX_ROWS", "1000000"))
    polars_max = int(os.getenv("ANALYTICA_POLARS_MAX_ROWS", "15000000"))

    try:
        n = len(df)
    except Exception:
        return "pandas"

    if n <= pandas_max:
        return "pandas"
    if n <= polars_max:
        return "polars"
    return "spark"


def _to_polars(df: Any):
    import polars as pl
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df).lazy()
    if hasattr(df, "collect") and hasattr(df, "schema"):
        return df
    raise TypeError("Cannot convert df to polars")


def _to_spark(df: Any):
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    if isinstance(df, pd.DataFrame):
        return spark.createDataFrame(df)
    if hasattr(df, "select") and hasattr(df, "schema") and hasattr(df, "limit"):
        return df
    raise TypeError("Cannot convert df to spark")


def ensure_engine_table(df: Any, engine: EngineName) -> Any:
    """
    Приводим таблицу к выбранному движку, по возможности.
    Это мягкий мост (в идеале большие данные надо читать сразу нужным движком).
    """
    if engine == "pandas":
        if isinstance(df, pd.DataFrame):
            return df
        return to_pandas(df)

    if engine == "polars":
        return _to_polars(df)

    if engine == "spark":
        return _to_spark(df)

    return df


def df_schema_text(df: Any, engine: Optional[str] = None) -> str:
    eng = (engine or "pandas").lower().strip()

    if eng == "pandas":
        if not isinstance(df, pd.DataFrame):
            try:
                df = to_pandas(df)
            except Exception:
                return "schema: <unknown>"
        dtypes = {c: str(t) for c, t in df.dtypes.items()}
        return (
            f"engine=pandas\n"
            f"rows={len(df)}, cols={len(df.columns)}\n"
            f"columns={list(df.columns)}\n"
            f"dtypes={dtypes}\n"
            f"missing={df.isna().sum().to_dict()}"
        )

    if eng == "polars":
        try:
            import polars as pl
            lf = df
            if isinstance(df, pl.DataFrame):
                lf = df.lazy()
            schema = getattr(lf, "schema", None)
            cols = list(schema.keys()) if schema else []
            dtypes = {k: str(v) for k, v in (schema or {}).items()}
            return (
                "engine=polars\n"
                f"cols={len(cols)}\n"
                f"columns={cols}\n"
                f"dtypes={dtypes}"
            )
        except Exception:
            return "engine=polars\nschema: <unavailable>"

    if eng == "spark":
        try:
            sdf = df
            fields = getattr(sdf.schema, "fields", [])
            cols = [f.name for f in fields]
            dtypes = {f.name: f.dataType.simpleString() for f in fields}
            return (
                "engine=spark\n"
                f"cols={len(cols)}\n"
                f"columns={cols}\n"
                f"dtypes={dtypes}"
            )
        except Exception:
            return "engine=spark\nschema: <unavailable>"

    return "schema: <unknown>"


def preview_result(result: Any, max_chars: int = 800) -> str:
    try:
        if isinstance(result, pd.DataFrame):
            return result.head(10).to_string(index=False)[:max_chars]
        if isinstance(result, pd.Series):
            return result.head(10).to_string()[:max_chars]
        return str(result)[:max_chars]
    except Exception as e:
        return f"<preview_error: {e}>"


def _describe_matplotlib_figure(fig: Any, max_bars: int = 10, max_chars: int = 800) -> str:
    """
    Пытаемся извлечь семантическое описание графика:
    title/xlabel/ylabel + бары (label -> height), если это bar chart.
    Работает без сохранения в файл.
    """
    try:
        axes = getattr(fig, "axes", None)
        if not axes:
            return "plot: Figure (no axes found)"

        ax = axes[0]
        title = ""
        xlabel = ""
        ylabel = ""
        try:
            title = ax.get_title() or ""
            xlabel = ax.get_xlabel() or ""
            ylabel = ax.get_ylabel() or ""
        except Exception:
            pass

        bars = []
        try:
            patches = getattr(ax, "patches", [])
            for p in patches[:max_bars]:
                x = getattr(p, "get_x", lambda: None)()
                w = getattr(p, "get_width", lambda: None)()
                h = getattr(p, "get_height", lambda: None)()
                if h is None:
                    continue
                bars.append(float(h))
        except Exception:
            pass

        bars_txt = ""
        if bars:
            bars_txt = f"; top_bars_heights={bars[:max_bars]}"

        out = f"plot: title={title!r}, xlabel={xlabel!r}, ylabel={ylabel!r}{bars_txt}"
        return out[:max_chars]
    except Exception as e:
        return f"plot: <describe_error {e}>"[:max_chars]


def preview_result_and_facts(result: Any, err: Optional[str]) -> tuple[str, str, str]:
    if err:
        return "error", "", err

    try:
        import matplotlib.figure as mplfig
        if isinstance(result, mplfig.Figure):
            return "plot", "<matplotlib Figure>", _describe_matplotlib_figure(result)
    except Exception:
        pass

    if isinstance(result, pd.DataFrame):
        rp = preview_result(result)
        facts = f"dataframe shape={result.shape}; columns={list(result.columns)}; preview=\n{rp}"
        return "dataframe", rp, facts

    if isinstance(result, pd.Series):
        rp = preview_result(result)
        facts = f"series len={len(result)}; name={result.name}; preview=\n{rp}"
        return "series", rp, facts

    rp = preview_result(result)
    facts = f"scalar type={type(result).__name__}; value_preview={rp}"
    return "scalar", rp, facts


_CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_code_block(text: str) -> str:
    m = _CODE_BLOCK_RE.search(text or "")
    return (m.group(1) if m else "").strip()


SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "sorted": sorted,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "str": str,
}

def to_pandas(df: Any, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Мост для визуализаций/простых операций.
    Для больших данных лучше делать агрегации в нативном движке,
    но для bar-команд берём только нужные колонки.
    """
    if isinstance(df, pd.DataFrame):
        return df[cols].copy() if cols else df.copy()

    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            out = df.select(cols) if cols else df
            return out.to_pandas()
        if hasattr(df, "collect"):  # LazyFrame
            lf = df.select(cols) if cols else df
            return lf.collect().to_pandas()
    except Exception:
        pass

    try:
        if hasattr(df, "toPandas"):
            sdf = df.select(*cols) if cols else df
            return sdf.toPandas()
    except Exception:
        pass

    raise TypeError("Unsupported table type for to_pandas")


def safe_exec(code: str, df: Any, engine: EngineName) -> tuple[Any, Optional[str]]:
    banned = ["import ", "open(", "read_csv", "read_excel", "to_csv", "to_excel", "eval(", "exec("]
    lowered = (code or "").lower()
    if any(b.lower() in lowered for b in banned):
        return None, "Unsafe code blocked by policy (imports/files/exec/eval)."

    env: Dict[str, Any] = {"df": df, "pd": pd, "engine": engine, "to_pandas": to_pandas}

    try:
        import matplotlib.pyplot as plt
        env["plt"] = plt
    except Exception:
        env["plt"] = None

    if engine == "polars":
        try:
            import polars as pl
            env["pl"] = pl
            env["pl_df"] = df
        except Exception:
            pass

    if engine == "spark":
        try:
            from pyspark.sql import functions as F
            env["F"] = F
            env["spark_df"] = df
        except Exception:
            pass

    try:
        exec(code, {"__builtins__": SAFE_BUILTINS}, env)
        if "result" not in env:
            return None, "Code executed but did not assign variable `result`."
        return env["result"], None
    except Exception as e:
        return None, f"Execution error: {e}"


def safe_exec_pandas(code: str, df: pd.DataFrame) -> tuple[Any, Optional[str]]:
    """
    Backward-compatible wrapper.
    """
    return safe_exec(code, df, "pandas")


def _is_bar_command(query: str) -> bool:
    q = (query or "").strip().lower()
    return q.startswith(("/bar", "/barh", "/bar_share", "/bar_stacked", "/bar_grouped"))


def _safe_int(x: str, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _parse_kv_args(query: str) -> Dict[str, str]:
    """
    Парсер вида: /bar x=col y=val title="Top 10"
    """
    kv: Dict[str, str] = {}
    parts = shlex.split(query)
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k.strip()] = v.strip().strip('"').strip("'")
    return kv


def _bar_codegen(query: str) -> str:
    cmd = (query or "").strip().split()[0].lower()
    kv = _parse_kv_args(query or "")

    x = kv.get("x", "")
    y = kv.get("y", "")
    agg = kv.get("agg", "sum").lower()
    top = _safe_int(kv.get("top", "10"), 10)
    title = kv.get("title", "")

    group = kv.get("group", "")
    stack = kv.get("stack", "")

    if not x or not y:
        return "result = 'CommandBar error: нужно указать x=<col> и y=<col>'"

    if agg not in {"sum", "mean", "median", "min", "max", "count"}:
        agg = "sum"

    cols = [x, y]
    if cmd == "/bar_grouped" and group:
        cols.append(group)
    if cmd == "/bar_stacked" and stack:
        cols.append(stack)

    prelude = f"""
df2 = to_pandas(df, cols={cols!r})
df2 = df2.dropna(subset={list(dict.fromkeys(cols))!r})
df2[{y!r}] = pd.to_numeric(df2[{y!r}], errors="coerce")
df2 = df2.dropna(subset=[{y!r}])
""".strip()

    fig_block = """fig, ax = plt.subplots(figsize=(6, 4))""".strip()

    if cmd == "/bar":
        code = f"""
{prelude}
s = df2.groupby({x!r})[{y!r}].{agg}().sort_values(ascending=False).head({top})
{fig_block}
ax.bar(s.index.astype(str), s.values)
ax.set_xlabel({x!r})
ax.set_ylabel(f\"{agg}({y})\")
ax.set_title({title!r} if {bool(title)} else "Bar chart")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
result = fig
""".strip()
        return code

    if cmd == "/barh":
        code = f"""
{prelude}
s = df2.groupby({x!r})[{y!r}].{agg}().sort_values(ascending=True).head({top})
{fig_block}
ax.barh(s.index.astype(str), s.values)
ax.set_xlabel(f\"{agg}({y})\")
ax.set_ylabel({x!r})
ax.set_title({title!r} if {bool(title)} else "Horizontal bar chart")
fig.tight_layout()
result = fig
""".strip()
        return code

    if cmd == "/bar_share":
        code = f"""
{prelude}
s = df2.groupby({x!r})[{y!r}].{agg}().sort_values(ascending=False)
s = (s / s.sum()).sort_values(ascending=False).head({top})
{fig_block}
ax.bar(s.index.astype(str), s.values)
ax.set_xlabel({x!r})
ax.set_ylabel("share")
ax.set_title({title!r} if {bool(title)} else "Share bar chart")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
result = fig
""".strip()
        return code

    if cmd == "/bar_stacked":
        if not stack:
            return "result = 'CommandBar error: /bar_stacked требует stack=<col>'"
        code = f"""
{prelude}
pt = df2.pivot_table(index={x!r}, columns={stack!r}, values={y!r}, aggfunc={agg}, fill_value=0.0)
pt = pt.assign(__total__=pt.sum(axis=1)).sort_values("__total__", ascending=False).head({top}).drop(columns="__total__")
{fig_block}
pt.plot(kind="bar", stacked=True, ax=ax)
ax.set_xlabel({x!r})
ax.set_ylabel(f\"{agg}({y})\")
ax.set_title({title!r} if {bool(title)} else "Stacked bar chart")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
result = fig
""".strip()
        return code

    if cmd == "/bar_grouped":
        if not group:
            return "result = 'CommandBar error: /bar_grouped требует group=<col>'"
        code = f"""
{prelude}
pt = df2.pivot_table(index={x!r}, columns={group!r}, values={y!r}, aggfunc={agg}, fill_value=0.0)
pt = pt.assign(__total__=pt.sum(axis=1)).sort_values("__total__", ascending=False).head({top}).drop(columns="__total__")
{fig_block}
pt.plot(kind="bar", ax=ax)
ax.set_xlabel({x!r})
ax.set_ylabel(f\"{agg}({y})\")
ax.set_title({title!r} if {bool(title)} else "Grouped bar chart")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
result = fig
""".strip()
        return code

    return "result = 'CommandBar error: неизвестная bar-команда'"
