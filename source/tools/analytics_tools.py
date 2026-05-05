from __future__ import annotations

from typing import Any, Callable

from source.func import _bar_codegen
from source.tools.code_tools import execute_code_tool
from source.tools.data_tools import inspect_schema_tool


def build_analytics_tools(run_context: dict[str, Any]) -> list[Callable[..., dict[str, str]]]:
    """Create Deep Agent analytics tools bound to one run context."""

    def inspect_dataset_schema() -> dict[str, str]:
        """Inspect the dataset schema and resolve the compute engine."""
        data_info = inspect_schema_tool(run_context["df"], run_context["engine"])
        run_context["df"] = data_info["df"]
        run_context["engine"] = data_info["engine"]
        run_context["schema"] = data_info["schema"]
        return {
            "engine": run_context["engine"],
            "schema": run_context["schema"],
        }

    def run_python_analysis(code: str) -> dict[str, str]:
        """Execute safe Python code. Code must assign the final value to `result`."""
        if not run_context["schema"]:
            inspect_dataset_schema()
        run_context["code"] = code
        exec_out = execute_code_tool(code, run_context["df"], run_context["engine"])
        run_context.update(exec_out)
        return {
            "exec_error": str(exec_out.get("exec_error") or ""),
            "result_kind": str(exec_out.get("result_kind", "")),
            "result_preview": str(exec_out.get("result_preview", "")),
            "result_facts": str(exec_out.get("result_facts", "")),
            "has_result_base64": str(bool(exec_out.get("result_base64"))),
        }

    def top_n(
        dimension: str,
        metric: str,
        n: int = 5,
        agg: str = "sum",
        ascending: bool = False,
    ) -> dict[str, str]:
        """Aggregate metric by dimension and return top N rows."""
        if agg not in {"sum", "mean", "median", "min", "max", "count"}:
            agg = "sum"
        n = max(1, min(int(n), 50))
        code = f"""
df2 = to_pandas(df) if callable(to_pandas) else df.copy()
df2 = df2.dropna(subset=[{dimension!r}, {metric!r}])
df2[{metric!r}] = pd.to_numeric(df2[{metric!r}], errors="coerce")
df2 = df2.dropna(subset=[{metric!r}])
result = (
    df2.groupby({dimension!r})[{metric!r}]
    .{agg}()
    .sort_values(ascending={bool(ascending)!r})
    .head({n})
    .reset_index(name="{agg}_{metric}")
)
""".strip()
        return run_python_analysis(code)

    def plot_bar(
        dimension: str,
        metric: str,
        n: int = 5,
        agg: str = "sum",
        title: str = "",
        horizontal: bool = False,
    ) -> dict[str, str]:
        """Create a bar chart for metric aggregated by dimension."""
        if agg not in {"sum", "mean", "median", "min", "max", "count"}:
            agg = "sum"
        n = max(1, min(int(n), 50))
        plot_call = "ax.barh(s.index.astype(str), s.values)" if horizontal else "ax.bar(s.index.astype(str), s.values)"
        tick_line = "" if horizontal else 'ax.tick_params(axis="x", rotation=35)'
        code = f"""
df2 = to_pandas(df) if callable(to_pandas) else df.copy()
df2 = df2.dropna(subset=[{dimension!r}, {metric!r}])
df2[{metric!r}] = pd.to_numeric(df2[{metric!r}], errors="coerce")
df2 = df2.dropna(subset=[{metric!r}])
s = df2.groupby({dimension!r})[{metric!r}].{agg}().sort_values(ascending=False).head({n})
fig, ax = plt.subplots(figsize=(7, 4))
{plot_call}
ax.set_xlabel({metric!r} if {horizontal!r} else {dimension!r})
ax.set_ylabel({dimension!r} if {horizontal!r} else "{agg}({metric})")
ax.set_title({title!r} if {bool(title)!r} else "Top {n}: {dimension} by {agg}({metric})")
{tick_line}
fig.tight_layout()
result = fig
""".strip()
        return run_python_analysis(code)

    def find_sales_drops(
        date_col: str = "Order Date",
        metric: str = "Sales",
        n: int = 10,
        date_format: str = "",
    ) -> dict[str, str]:
        """Find the largest day-over-day drops in a numeric metric by date."""
        n = max(1, min(int(n), 50))
        parse_format = f", format={date_format!r}" if date_format else ""
        code = f"""
df2 = to_pandas(df) if callable(to_pandas) else df.copy()
df2 = df2.dropna(subset=[{date_col!r}, {metric!r}])
df2[{date_col!r}] = pd.to_datetime(df2[{date_col!r}]{parse_format}, errors="coerce", dayfirst=True)
df2[{metric!r}] = pd.to_numeric(df2[{metric!r}], errors="coerce")
df2 = df2.dropna(subset=[{date_col!r}, {metric!r}])
daily = (
    df2.groupby(df2[{date_col!r}].dt.date)[{metric!r}]
    .sum()
    .sort_index()
    .reset_index(name="daily_{metric}")
)
daily["previous_day_{metric}"] = daily["daily_{metric}"].shift(1)
daily["absolute_drop"] = daily["daily_{metric}"] - daily["previous_day_{metric}"]
daily["drop_pct"] = (daily["absolute_drop"] / daily["previous_day_{metric}"]) * 100.0
result = (
    daily.dropna(subset=["absolute_drop"])
    .sort_values("absolute_drop", ascending=True)
    .head({n})
)
""".strip()
        return run_python_analysis(code)

    def run_bar_command(command: str) -> dict[str, str]:
        """Execute a supported /bar command string."""
        code = _bar_codegen(command)
        return run_python_analysis(code)

    return [
        inspect_dataset_schema,
        top_n,
        plot_bar,
        find_sales_drops,
        run_bar_command,
        run_python_analysis,
    ]
