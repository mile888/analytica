from typing import Any, Dict, Optional
import pandas as pd
from source.llm.factory import make_llm

def df_schema_text(df: pd.DataFrame) -> str:
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    return (
        f"rows={len(df)}, cols={len(df.columns)}\n"
        f"columns={list(df.columns)}\n"
        f"dtypes={dtypes}\n"
        f"missing={df.isna().sum().to_dict()}"
    )

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
            patches = getattr(ax, "patches", []) or []
            # ticks as labels:
            xticklabels = []
            try:
                xticklabels = [t.get_text() for t in ax.get_xticklabels()]
                xticklabels = [x for x in xticklabels if x is not None]
            except Exception:
                xticklabels = []

            for i, p in enumerate(patches[:max_bars]):
                h = getattr(p, "get_height", lambda: None)()
                lbl = ""
                if i < len(xticklabels) and xticklabels[i]:
                    lbl = xticklabels[i]
                else:
                    x = getattr(p, "get_x", lambda: None)()
                    lbl = str(x)
                bars.append((lbl, h))
        except Exception:
            bars = []

        parts = []
        parts.append("plot: matplotlib Figure")
        if title:
            parts.append(f"title={title}")
        if xlabel:
            parts.append(f"xlabel={xlabel}")
        if ylabel:
            parts.append(f"ylabel={ylabel}")
        if bars:
            cleaned = [(l.strip(), v) for l, v in bars]
            cleaned = [(l if l else f"bar_{i+1}", v) for i, (l, v) in enumerate(cleaned)]
            bars_txt = ", ".join([f"{l}={v:.4g}" if isinstance(v, (int, float)) else f"{l}={v}" for l, v in cleaned])
            parts.append(f"bars(top {min(len(bars), max_bars)}): {bars_txt}")

        s = " | ".join(parts)
        return s[:max_chars]
    except Exception as e:
        return f"plot: Figure (describe_error: {e})"[:max_chars]


def preview_result_and_facts(result: Any, exec_error: Optional[str]) -> tuple[str, str, str]:
    """
    Возвращает (result_kind, result_preview, result_facts)
    result_preview — коротко (для логов),
    result_facts — максимально полезно для critic/reporter.
    """
    if exec_error is not None:
        return "error", "", f"Execution failed: {exec_error}"

    try:
        # matplotlib Figure
        try:
            from matplotlib.figure import Figure
            if isinstance(result, Figure):
                facts = _describe_matplotlib_figure(result)
                return "plot", "Figure", facts
        except Exception:
            # если matplotlib не доступен как класс — fallback ниже
            pass

        if isinstance(result, pd.DataFrame):
            prev = result.head(10).to_string(index=False)[:800]
            facts = f"dataframe shape={result.shape}, columns={list(result.columns)}; head:\n{prev}"
            return "dataframe", prev, facts

        if isinstance(result, pd.Series):
            prev = result.head(10).to_string()[:800]
            facts = f"series name={result.name}, len={len(result)}; head:\n{prev}"
            return "series", prev, facts

        s = str(result)[:800]
        return "scalar", s, f"scalar: {s}"
    except Exception as e:
        return "scalar", "", f"preview_error: {e}"

def extract_code_block(text: str) -> str:
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()



def _strip_quotes(v: str) -> str:
    v = v.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    return v

def _parse_kv_args(query: str) -> Dict[str, str]:
    parts = query.strip().split()
    kv: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        kv[k.strip().lower()] = _strip_quotes(v.strip())
    return kv

def _safe_int(x: str, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _is_bar_command(query: str) -> bool:
    q = query.strip().lower()
    return q.startswith(("/bar", "/barh", "/bar_share", "/bar_stacked", "/bar_grouped"))

def _bar_codegen(query: str) -> str:
    cmd = query.strip().split()[0].lower()
    kv = _parse_kv_args(query)

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

    if cmd in ("/bar_stacked", "/bar_grouped"):
        third = stack if cmd == "/bar_stacked" else group
        if not third:
            need = "stack" if cmd == "/bar_stacked" else "group"
            return f"result = 'CommandBar error: {cmd} требует {need}=<col>'"

        prelude = f"""
df2 = df.copy()
df2 = df2.dropna(subset={[x, third, y]!r})
df2[{y!r}] = pd.to_numeric(df2[{y!r}], errors="coerce")
df2 = df2.dropna(subset={[y]!r})
""".strip()
    else:
        prelude = f"""
df2 = df.copy()
df2 = df2.dropna(subset={[x, y]!r})
df2[{y!r}] = pd.to_numeric(df2[{y!r}], errors="coerce")
df2 = df2.dropna(subset={[y]!r})
""".strip()

    fig_block = """
fig, ax = plt.subplots(figsize=(6, 4))
""".strip()

    if cmd == "/bar":
        code = f"""
{prelude}
s = df2.groupby({x!r})[{y!r}].{agg}().sort_values(ascending=False).head({top})
{fig_block}
ax.bar(s.index.astype(str), s.values)
ax.set_xlabel({x!r})
ax.set_ylabel(f"{agg}({y})")
ax.set_title({title!r} if {title!r} else "Bar chart")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
result = fig
""".strip()
        return code

    if cmd == "/barh":
        code = f"""
{prelude}
s = df2.groupby({x!r})[{y!r}].{agg}().sort_values(ascending=False).head({top})
{fig_block}
ax.barh(s.index.astype(str), s.values)
ax.set_ylabel({x!r})
ax.set_xlabel(f"{agg}({y})")
ax.set_title({title!r} if {bool(title)} else "Horizontal bar chart")
fig.tight_layout()
result = fig
""".strip()
        return code

    if cmd == "/bar_share":
        code = f"""
{prelude}
s = df2.groupby({x!r})[{y!r}].{agg}().sort_values(ascending=False).head({top})
total = float(s.sum()) if float(s.sum()) != 0.0 else 1.0
share = (s / total) * 100.0
{fig_block}
ax.bar(share.index.astype(str), share.values)
ax.set_xlabel({x!r})
ax.set_ylabel("share (%)")
ax.set_title({title!r} if {bool(title)} else "Bar share (%)")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
result = fig
""".strip()
        return code

    if cmd == "/bar_stacked":
        code = f"""
{prelude}
pt = df2.pivot_table(index={x!r}, columns={stack!r}, values={y!r}, aggfunc={agg}, fill_value=0.0)
pt = pt.assign(__total__=pt.sum(axis=1)).sort_values("__total__", ascending=False).head({top}).drop(columns="__total__")
{fig_block}
pt.plot(kind="bar", stacked=True, ax=ax)
ax.set_xlabel({x!r})
ax.set_ylabel(f"{agg}({y})")
ax.set_title({title!r} if {bool(title)} else "Stacked bar chart")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
result = fig
""".strip()
        return code

    if cmd == "/bar_grouped":
        code = f"""
{prelude}
pt = df2.pivot_table(index={x!r}, columns={group!r}, values={y!r}, aggfunc={agg}, fill_value=0.0)
pt = pt.assign(__total__=pt.sum(axis=1)).sort_values("__total__", ascending=False).head({top}).drop(columns="__total__")
{fig_block}
pt.plot(kind="bar", ax=ax)
ax.set_xlabel({x!r})
ax.set_ylabel(f"{agg}({y})")
ax.set_title({title!r} if {bool(title)} else "Grouped bar chart")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
result = fig
""".strip()
        return code

    return "result = 'CommandBar error: неизвестная bar-команда'"



def safe_exec_pandas(code: str, df: pd.DataFrame) -> tuple[Any, Optional[str]]:
    banned = ["import ", "open(", "read_csv", "read_excel", "to_csv", "to_excel", "eval(", "exec("]
    lowered = code.lower()
    if any(b.lower() in lowered for b in banned):
        return None, "Unsafe code blocked by policy (imports/files/exec/eval)."

    env: Dict[str, Any] = {"df": df, "pd": pd}


    try:
        import matplotlib.pyplot as plt
        env["plt"] = plt
    except Exception:
        env["plt"] = None

    try:
        exec(code, {}, env)
        if "result" not in env:
            return None, "Code executed but did not assign variable `result`."
        return env["result"], None
    except Exception as e:
        return None, f"Execution error: {e}"
