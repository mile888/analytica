from __future__ import annotations

from typing import Any, Dict, Optional
import ast
import os
import re

import pandas as pd
from dotenv import load_dotenv

from source.config import ALLOWED_CODE_IMPORTS, MPLBACKEND, MPLCONFIGDIR
from source.llm.llm_config import EngineName
from source.engine import BaseEngine, create_engine

load_dotenv()
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", MPLBACKEND)

from source.llm.factory import _get_config


def detect_engine(df: Any, engine: Optional[str] = None) -> EngineName:
    """
    Resolve the compute engine.

    Priority:
      1. Explicit `engine` argument (if not "auto").
      2. `defaults.engine` from llm_config.yaml.
      3. Falls back to "pandas".
    """
    if engine and engine.strip().lower() in {"pandas", "polars", "spark"}:
        return engine.strip().lower()

    cfg = _get_config()
    return cfg.defaults.engine


def get_engine(df: Any, engine: Optional[str] = None) -> BaseEngine:
    """Convenience: detect engine name and return the engine instance."""
    name = detect_engine(df, engine)
    return create_engine(name)


def preview_result(result: Any, max_chars: int = 800) -> str:
    try:
        if isinstance(result, pd.DataFrame):
            return result.head(10).to_string(index=False)[:max_chars]
        if isinstance(result, pd.Series):
            return result.head(10).to_string()[:max_chars]
        return str(result)[:max_chars]
    except Exception as e:
        return f"<preview_error: {e}>"


def figure_to_base64(fig: Any, fmt: str = "png", dpi: int = 150) -> str:
    """Render a matplotlib Figure to a base64-encoded PNG string.

    Returns a data-URI ready string: 'data:image/png;base64,...'
    Returns empty string on failure.
    """
    try:
        import base64
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        buf.close()
        return f"data:image/{fmt};base64,{b64}"
    except Exception:
        return ""


def _describe_matplotlib_figure(fig: Any, max_bars: int = 10, max_chars: int = 800) -> str:
    """
    Extract a semantic description of a matplotlib figure:
    title/xlabel/ylabel + bar heights (if bar chart).
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


def preview_result_and_facts(result: Any, err: Optional[str]) -> tuple[str, str, str, str]:
    """Inspect an execution result and return (kind, preview, facts, base64).

    base64 is a data-URI string for plot results, empty string otherwise.
    """
    if err:
        return "error", "", err, ""

    try:
        import matplotlib.figure as mplfig
        if isinstance(result, mplfig.Figure):
            b64 = figure_to_base64(result)
            return "plot", "<matplotlib Figure>", _describe_matplotlib_figure(result), b64
    except Exception:
        pass

    if isinstance(result, pd.DataFrame):
        rp = preview_result(result)
        facts = f"dataframe shape={result.shape}; columns={list(result.columns)}; preview=\n{rp}"
        return "dataframe", rp, facts, ""

    if isinstance(result, pd.Series):
        rp = preview_result(result)
        facts = f"series len={len(result)}; name={result.name}; preview=\n{rp}"
        return "series", rp, facts, ""

    rp = preview_result(result)
    facts = f"scalar type={type(result).__name__}; value_preview={rp}"
    return "scalar", rp, facts, ""


_CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code_block(text: str) -> str:
    m = _CODE_BLOCK_RE.search(text or "")
    if m:
        return m.group(1).strip()
    return (text or "").strip()

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
        prelude = f'''
df2 = to_pandas(df) if callable(to_pandas) else df.copy()
df2 = df2.dropna(subset=[{x!r}, {third!r}, {y!r}])
df2[{y!r}] = pd.to_numeric(df2[{y!r}], errors="coerce")
df2 = df2.dropna(subset=[{y!r}])
'''.strip()
    else:
        prelude = f'''
df2 = to_pandas(df) if callable(to_pandas) else df.copy()
df2 = df2.dropna(subset=[{x!r}, {y!r}])
df2[{y!r}] = pd.to_numeric(df2[{y!r}], errors="coerce")
df2 = df2.dropna(subset=[{y!r}])
'''.strip()

    fig_block = '''
fig, ax = plt.subplots(figsize=(6, 4))
'''.strip()

    if cmd == "/bar":
        code = f'''
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
'''.strip()
        return code

    if cmd == "/barh":
        code = f'''
{prelude}
s = df2.groupby({x!r})[{y!r}].{agg}().sort_values(ascending=False).head({top})
{fig_block}
ax.barh(s.index.astype(str), s.values)
ax.set_ylabel({x!r})
ax.set_xlabel(f"{agg}({y})")
ax.set_title({title!r} if {bool(title)} else "Horizontal bar chart")
fig.tight_layout()
result = fig
'''.strip()
        return code

    if cmd == "/bar_share":
        code = f'''
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
'''.strip()
        return code

    if cmd == "/bar_stacked":
        code = f'''
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
'''.strip()
        return code

    if cmd == "/bar_grouped":
        code = f'''
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
'''.strip()
        return code

    return "result = 'CommandBar error: unknown'"


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
    "callable": callable,
    "float": float,
    "int": int,
    "str": str,
}


def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    allowed_roots = {item.split(".", 1)[0] for item in ALLOWED_CODE_IMPORTS}
    allowed_exact = set(ALLOWED_CODE_IMPORTS)
    if name in allowed_exact or root in allowed_roots:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import {name!r} is not allowed in analysis code.")


def _validate_analysis_ast(code: str) -> Optional[str]:
    """Reject file/network/process side effects while allowing normal pandas analysis."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return f"Syntax error: {exc}"

    blocked_calls = {
        "open",
        "eval",
        "exec",
        "compile",
        "input",
        "__import__",
        "read_csv",
        "read_excel",
        "read_parquet",
        "to_csv",
        "to_excel",
        "to_parquet",
        "to_sql",
        "remove",
        "unlink",
        "rmdir",
        "mkdir",
        "makedirs",
        "system",
        "popen",
        "run",
        "call",
        "check_call",
        "check_output",
    }
    blocked_import_roots = {"os", "sys", "subprocess", "pathlib", "shutil", "socket", "requests"}
    allowed_import_roots = {item.split(".", 1)[0] for item in ALLOWED_CODE_IMPORTS}

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif node.module:
                names = [node.module]
            for name in names:
                root = name.split(".", 1)[0]
                if root in blocked_import_roots:
                    return f"Unsafe import blocked by policy: {name}"
                if root not in allowed_import_roots:
                    return f"Import {name!r} is not allowed in analysis code."

        if isinstance(node, ast.Call):
            func = node.func
            call_name = ""
            if isinstance(func, ast.Name):
                call_name = func.id
            elif isinstance(func, ast.Attribute):
                call_name = func.attr
            if call_name in blocked_calls:
                return f"Unsafe call blocked by policy: {call_name}()"

        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "Dunder attribute access is blocked in analysis code."

    return None


def safe_exec(code: str, df: Any, engine: BaseEngine | EngineName) -> tuple[Any, Optional[str]]:
    """Execute generated code in a sandboxed environment.

    `engine` can be either a BaseEngine instance or an EngineName string.
    """
    policy_error = _validate_analysis_ast(code)
    if policy_error:
        return None, policy_error

    eng: BaseEngine = engine if isinstance(engine, BaseEngine) else create_engine(engine)
    env = eng.exec_env(df)
    builtins = dict(SAFE_BUILTINS)
    builtins["__import__"] = _limited_import

    try:
        exec(code, {"__builtins__": builtins}, env)
        if "result" not in env:
            return None, "Code executed but did not assign variable `result`."
        return env["result"], None
    except Exception as e:
        return None, f"Execution error: {e}"
