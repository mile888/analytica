from __future__ import annotations

import re
import warnings
from datetime import datetime
from typing import Any

import pandas as pd
from langchain.tools import ToolRuntime, tool

from source.config import ALLOWED_AGGREGATIONS, ARTIFACT_DIR, DEFAULT_TOP_N, MAX_TOOL_ROWS, MAX_TOP_N
from source.dataframe import (
    dataframe_profile,
    list_sql_tables,
    sql_table_schema,
    validate_read_only_sql,
    run_dataframe_sql_with_metadata,
)
from source.engine import create_engine
from source.func import _bar_codegen, detect_engine, preview_result_and_facts, safe_exec
from source.runtime_context import AnalyticaContext
from source.product.branch_workspace import BranchWorkspaceManager
from source.product.execution_planner import (
    AuthoritativeExecutionPlanner,
    FilterConstraintValidator,
    TemporalSanityValidator,
)


def _to_pandas_safe(value: Any, engine: str = "pandas") -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return create_engine(engine).to_pandas(value)
    except Exception:
        try:
            return create_engine("pandas").to_pandas(value)
        except Exception:
            return pd.DataFrame()


def _safe_agg(agg: str) -> str:
    return agg if agg in set(ALLOWED_AGGREGATIONS) else (ALLOWED_AGGREGATIONS[0] if ALLOWED_AGGREGATIONS else "sum")


def _safe_n(n: int | str | None) -> int:
    try:
        value = int(n if n is not None else DEFAULT_TOP_N)
    except (TypeError, ValueError):
        value = DEFAULT_TOP_N
    return max(1, min(value, MAX_TOP_N))


def _quote(value: str) -> str:
    return repr(value)


def _inspect_schema(df: Any, engine: str = "auto") -> dict[str, Any]:
    engine_name = detect_engine(df, engine)
    engine_impl = create_engine(engine_name)
    try:
        table = engine_impl.ensure_table(df)
    except Exception:
        engine_name = "pandas"
        engine_impl = create_engine(engine_name)
        table = engine_impl.to_pandas(df)
    return {
        "engine": engine_name,
        "df": table,
        "schema": engine_impl.schema_text(table),
    }


def _execute_code(code: str, df: Any, engine: str) -> dict[str, Any]:
    result, err = safe_exec(code, df, engine)
    kind, preview, facts, b64 = preview_result_and_facts(result, err)
    return {
        "result": result,
        "exec_error": err,
        "result_kind": kind,
        "result_preview": preview,
        "result_facts": facts,
        "result_base64": b64,
    }


def _infer_schema_fields(df: Any, engine: str) -> dict[str, list[str]]:
    try:
        data = df if isinstance(df, pd.DataFrame) else _inspect_schema(df, engine)["df"]
        pdf = data if isinstance(data, pd.DataFrame) else pd.DataFrame()
        if not isinstance(data, pd.DataFrame):
            pdf = _to_pandas_safe(data, engine)
    except Exception:
        pdf = pd.DataFrame()

    numeric_cols = []
    date_cols = []
    categorical_cols = []
    for col in pdf.columns:
        series = pdf[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(str(col))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(series.dropna().head(50), errors="coerce", dayfirst=True)
        if len(parsed) and parsed.notna().mean() >= 0.7:
            date_cols.append(str(col))
        else:
            categorical_cols.append(str(col))
    return {
        "numeric_columns": numeric_cols,
        "date_columns": date_cols,
        "categorical_columns": categorical_cols,
    }


def _slug(text: str, fallback: str = "artifact") -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip().lower()).strip("-")
    return slug[:80] or fallback


def _record_tool_event(run_context: dict[str, Any], tool_name: str, status: str, **metadata: Any) -> None:
    event = {
        "tool": tool_name,
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    event.update({key: value for key, value in metadata.items() if value not in (None, "")})
    run_context.setdefault("tool_timeline", []).append(event)


def _figure_title(result: Any) -> str:
    try:
        axes = getattr(result, "axes", None) or []
        if axes:
            title = axes[0].get_title()
            if title:
                return str(title)
    except Exception:
        pass
    return "Chart artifact"


def _analysis_artifact_metadata(exec_out: dict[str, Any], source_tool: str) -> dict[str, Any] | None:
    result = exec_out.get("result")
    result_kind = str(exec_out.get("result_kind", ""))
    created_at = datetime.now().isoformat(timespec="seconds")

    if result_kind == "plot":
        return {
            "artifact_type": "chart",
            "title": _figure_title(result),
            "source_tool": source_tool,
            "format": "png;base64",
            "created_at": created_at,
        }

    if isinstance(result, pd.DataFrame):
        return {
            "artifact_type": "table",
            "title": "Analysis table",
            "source_tool": source_tool,
            "rows": int(result.shape[0]),
            "columns": int(result.shape[1]),
            "format": "dataframe_preview",
            "created_at": created_at,
        }

    if isinstance(result, pd.Series):
        return {
            "artifact_type": "table",
            "title": "Analysis series",
            "source_tool": source_tool,
            "rows": int(len(result)),
            "columns": 1,
            "format": "series_preview",
            "created_at": created_at,
        }

    return None


def _append_artifact_metadata(run_context: dict[str, Any], artifact: dict[str, Any] | None) -> None:
    if artifact:
        run_context.setdefault("artifacts", []).append(artifact)


def _runtime_run_state(
    runtime: ToolRuntime[AnalyticaContext],
    fallback_run_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if runtime is not None and getattr(runtime, "context", None) is not None:
        return runtime.context.run_state
    if fallback_run_context is not None:
        return fallback_run_context
    raise RuntimeError("Analytica runtime context is required for analytics tools.")


def build_analytics_tools(fallback_run_context: dict[str, Any] | None = None) -> list[Any]:
    """Create DeepAgents analytics tools that read run data from ToolRuntime."""

    @tool
    def create_authoritative_query_plan(question: str, runtime: ToolRuntime[AnalyticaContext]) -> dict[str, Any]:
        """Parse the user query into the single authoritative analytical execution plan."""
        run_context = _runtime_run_state(runtime, fallback_run_context)
        pdf = run_context.get("df")
        if not isinstance(pdf, pd.DataFrame):
            pdf = _to_pandas_safe(pdf, run_context.get("engine", "pandas"))
        plan = AuthoritativeExecutionPlanner.plan(question, pdf, active_branch=run_context.get("branch_workspace"))
        run_context["last_query_plan"] = plan.to_payload()
        _record_tool_event(run_context, "create_authoritative_query_plan", "ok", intent=plan.intent, metric=plan.metric, dimension=plan.dimension)
        return plan.to_payload()

    @tool
    def validate_query_plan(runtime: ToolRuntime[AnalyticaContext]) -> dict[str, Any]:
        """Validate the current authoritative query plan for filters and forbidden substitutions."""
        run_context = _runtime_run_state(runtime, fallback_run_context)
        raw_plan = run_context.get("last_query_plan")
        if not isinstance(raw_plan, dict):
            return {"valid": False, "errors": ["no query plan available"]}
        pdf = run_context.get("df")
        if not isinstance(pdf, pd.DataFrame):
            pdf = _to_pandas_safe(pdf, run_context.get("engine", "pandas"))
        plan = AuthoritativeExecutionPlanner.plan(str(raw_plan.get("raw_question") or run_context.get("query") or ""), pdf)
        errors = FilterConstraintValidator.validate(plan, pdf)
        if plan.forbidden_substitutions:
            errors.extend([f"forbidden substitution target: {item}" for item in plan.forbidden_substitutions])
        _record_tool_event(run_context, "validate_query_plan", "error" if errors else "ok", errors=len(errors))
        return {"valid": not errors, "errors": errors, "plan": plan.to_payload()}

    @tool
    def route_branch_workspace(runtime: ToolRuntime[AnalyticaContext]) -> dict[str, Any]:
        """Route the current query plan into the analytical branch workspace."""
        run_context = _runtime_run_state(runtime, fallback_run_context)
        raw_plan = run_context.get("last_query_plan")
        if not isinstance(raw_plan, dict):
            return {"error": "no query plan available"}
        pdf = run_context.get("df")
        if not isinstance(pdf, pd.DataFrame):
            pdf = _to_pandas_safe(pdf, run_context.get("engine", "pandas"))
        plan = AuthoritativeExecutionPlanner.plan(str(raw_plan.get("raw_question") or run_context.get("query") or ""), pdf)
        workspace = BranchWorkspaceManager.from_payload(run_context.get("branch_workspace"))
        decision = BranchWorkspaceManager.route(plan, workspace)
        workspace.active_branch_id = decision.branch_id
        run_context["branch_workspace"] = workspace.to_payload()
        _record_tool_event(run_context, "route_branch_workspace", "ok", action=decision.action.value)
        return decision.to_payload()

    @tool
    def validate_delivery_delay_sanity(
        order_date_col: str,
        ship_date_col: str,
        runtime: ToolRuntime[AnalyticaContext],
    ) -> dict[str, Any]:
        """Validate delivery delay date calculations before shipping-delay analysis."""
        run_context = _runtime_run_state(runtime, fallback_run_context)
        pdf = run_context.get("df")
        if not isinstance(pdf, pd.DataFrame):
            pdf = _to_pandas_safe(pdf, run_context.get("engine", "pandas"))
        issues = TemporalSanityValidator.delivery_delay_issues(pdf, order_date_col, ship_date_col)
        _record_tool_event(run_context, "validate_delivery_delay_sanity", "error" if issues else "ok", issues=len(issues))
        return {"valid": not issues, "issues": issues}

    @tool
    def inspect_dataset_schema(runtime: ToolRuntime[AnalyticaContext]) -> dict[str, str]:
        """Inspect the dataset schema and resolve the compute engine."""
        run_context = _runtime_run_state(runtime, fallback_run_context)
        data_info = _inspect_schema(run_context["df"], run_context["engine"])
        run_context["df"] = data_info["df"]
        run_context["engine"] = data_info["engine"]
        run_context["schema"] = data_info["schema"]
        fields = _infer_schema_fields(run_context["df"], run_context["engine"])
        run_context["schema_fields"] = fields
        try:
            pdf = run_context["df"] if isinstance(run_context["df"], pd.DataFrame) else pd.DataFrame()
            if not isinstance(run_context["df"], pd.DataFrame):
                pdf = _to_pandas_safe(run_context["df"], run_context["engine"])
            run_context["data_profile"] = dataframe_profile(pdf)
        except Exception as exc:
            run_context["data_profile"] = {"profile_error": str(exc)}
        _record_tool_event(
            run_context,
            "inspect_dataset_schema",
            "ok",
            engine=run_context["engine"],
            columns=len(run_context.get("data_profile", {}).get("columns", []))
            if isinstance(run_context.get("data_profile"), dict)
            else None,
        )
        return {
            "engine": run_context["engine"],
            "schema": run_context["schema"],
            "profile": str(run_context.get("data_profile", {})),
            **fields,
        }

    @tool
    def run_python_analysis(code: str, runtime: ToolRuntime[AnalyticaContext]) -> dict[str, str]:
        """Execute Python analysis code against the current dataset.

        The code runs with the current table as `df`, pandas as `pd`, optional matplotlib
        as `plt`, and a helper `to_pandas`. Assign the final value to `result`.
        """
        run_context = _runtime_run_state(runtime, fallback_run_context)
        if not run_context["schema"]:
            inspect_dataset_schema.func(runtime)
        run_context["code"] = code
        exec_out = _execute_code(code, run_context["df"], run_context["engine"])
        run_context.update(exec_out)
        if not exec_out.get("exec_error"):
            _append_artifact_metadata(
                run_context,
                _analysis_artifact_metadata(exec_out, "run_python_analysis"),
            )
        _record_tool_event(
            run_context,
            "run_python_analysis",
            "error" if exec_out.get("exec_error") else "ok",
            result_kind=exec_out.get("result_kind", ""),
        )
        return {
            "exec_error": str(exec_out.get("exec_error") or ""),
            "result_kind": str(exec_out.get("result_kind", "")),
            "result_preview": str(exec_out.get("result_preview", "")),
            "result_facts": str(exec_out.get("result_facts", "")),
            "has_result_base64": str(bool(exec_out.get("result_base64"))),
        }

    @tool
    def top_n(
        dimension: str,
        metric: str,
        runtime: ToolRuntime[AnalyticaContext],
        n: int = DEFAULT_TOP_N,
        agg: str = "sum",
        ascending: bool = False,
    ) -> dict[str, str]:
        """Aggregate metric by dimension and return top N rows."""
        agg = _safe_agg(agg)
        n = _safe_n(n)
        code = f"""
df2 = to_pandas(df) if callable(to_pandas) else df.copy()
df2 = df2.dropna(subset=[{_quote(dimension)}, {_quote(metric)}])
df2[{_quote(metric)}] = pd.to_numeric(df2[{_quote(metric)}], errors="coerce")
df2 = df2.dropna(subset=[{_quote(metric)}])
if df2.empty:
    result = "Cannot aggregate: metric column has no numeric values after conversion. Choose a numeric metric from the inspected schema."
else:
    result = (
        df2.groupby({_quote(dimension)})[{_quote(metric)}]
        .{agg}()
        .sort_values(ascending={bool(ascending)!r})
        .head({n})
        .reset_index(name={_quote(f"{agg}_{metric}")})
    )
""".strip()
        return run_python_analysis.func(code, runtime)

    @tool
    def plot_bar(
        dimension: str,
        metric: str,
        runtime: ToolRuntime[AnalyticaContext],
        n: int = DEFAULT_TOP_N,
        agg: str = "sum",
        title: str = "",
        horizontal: bool = False,
    ) -> dict[str, str]:
        """Create a bar chart for metric aggregated by dimension."""
        agg = _safe_agg(agg)
        n = _safe_n(n)
        plot_call = "ax.barh(s.index.astype(str), s.values)" if horizontal else "ax.bar(s.index.astype(str), s.values)"
        tick_line = "" if horizontal else 'ax.tick_params(axis="x", rotation=35)'
        code = f"""
df2 = to_pandas(df) if callable(to_pandas) else df.copy()
df2 = df2.dropna(subset=[{_quote(dimension)}, {_quote(metric)}])
df2[{_quote(metric)}] = pd.to_numeric(df2[{_quote(metric)}], errors="coerce")
df2 = df2.dropna(subset=[{_quote(metric)}])
if df2.empty:
    result = "Cannot build bar chart: metric column has no numeric values after conversion. Choose a numeric metric from the inspected schema."
else:
    s = df2.groupby({_quote(dimension)})[{_quote(metric)}].{agg}().sort_values(ascending=False).head({n})
    if s.empty:
        result = "Cannot build bar chart: no grouped rows available for the selected columns."
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        {plot_call}
        ax.set_xlabel({_quote(metric)} if {horizontal!r} else {_quote(dimension)})
        ax.set_ylabel({_quote(dimension)} if {horizontal!r} else {_quote(f"{agg}({metric})")})
        ax.set_title({_quote(title)} if {bool(title)!r} else {_quote(f"Top {n}: {dimension} by {agg}({metric})")})
        {tick_line}
        fig.tight_layout()
        result = fig
""".strip()
        return run_python_analysis.func(code, runtime)

    @tool
    def find_drops(
        runtime: ToolRuntime[AnalyticaContext],
        date_col: str = "",
        metric: str = "",
        n: int = DEFAULT_TOP_N,
        date_format: str = "",
    ) -> dict[str, str]:
        """Find the largest period-over-period drops in a numeric metric by date.

        Args:
            date_col: Date-like column from the dataset. Leave empty to use the first inferred date column.
            metric: Numeric metric column. Leave empty to use the first inferred numeric column.
            n: Number of drops to return.
            date_format: Optional pandas datetime format if automatic parsing is not enough.
        """
        run_context = _runtime_run_state(runtime, fallback_run_context)
        if not run_context["schema"]:
            inspect_dataset_schema.func(runtime)
        fields = run_context.get("schema_fields", {})
        date_col = date_col or next(iter(fields.get("date_columns", [])), "")
        metric = metric or next(iter(fields.get("numeric_columns", [])), "")
        if not date_col or not metric:
            return {
                "exec_error": "Could not infer date_col or metric. Call inspect_dataset_schema and pass explicit columns.",
                "result_kind": "error",
                "result_preview": "",
                "result_facts": str(fields),
                "has_result_base64": "False",
            }
        n = _safe_n(n)
        parse_format = f", format={date_format!r}" if date_format else ""
        code = f"""
df2 = to_pandas(df) if callable(to_pandas) else df.copy()
df2 = df2.dropna(subset=[{_quote(date_col)}, {_quote(metric)}])
df2[{_quote(date_col)}] = pd.to_datetime(df2[{_quote(date_col)}]{parse_format}, errors="coerce", dayfirst=True)
df2[{_quote(metric)}] = pd.to_numeric(df2[{_quote(metric)}], errors="coerce")
df2 = df2.dropna(subset=[{_quote(date_col)}, {_quote(metric)}])
daily = (
    df2.groupby(df2[{_quote(date_col)}].dt.date)[{_quote(metric)}]
    .sum()
    .sort_index()
    .reset_index(name={_quote(f"daily_{metric}")})
)
daily[{_quote(f"previous_{metric}")}] = daily[{_quote(f"daily_{metric}")}].shift(1)
daily["absolute_drop"] = daily[{_quote(f"daily_{metric}")}] - daily[{_quote(f"previous_{metric}")}]
daily["drop_pct"] = (daily["absolute_drop"] / daily[{_quote(f"previous_{metric}")}]) * 100.0
result = (
    daily.dropna(subset=["absolute_drop"])
    .sort_values("absolute_drop", ascending=True)
    .head({n})
)
""".strip()
        return run_python_analysis.func(code, runtime)

    @tool
    def list_dataframe_tables(runtime: ToolRuntime[AnalyticaContext]) -> dict[str, str]:
        """List SQL tables available for read-only DataFrame querying."""
        run_context = _runtime_run_state(runtime, fallback_run_context)
        if not run_context["schema"]:
            inspect_dataset_schema.func(runtime)
        pdf = run_context["df"] if isinstance(run_context["df"], pd.DataFrame) else None
        if pdf is None:
            pdf = _to_pandas_safe(run_context["df"], run_context["engine"])
        tables = list_sql_tables(pdf)
        _record_tool_event(run_context, "list_dataframe_tables", "ok", tables=", ".join(tables))
        return {"tables": ", ".join(tables)}

    @tool
    def describe_dataframe_table(
        runtime: ToolRuntime[AnalyticaContext],
        table_name: str = "",
    ) -> dict[str, str]:
        """Return SQLite schema and sample rows for the current DataFrame table."""
        run_context = _runtime_run_state(runtime, fallback_run_context)
        if not run_context["schema"]:
            inspect_dataset_schema.func(runtime)
        pdf = run_context["df"] if isinstance(run_context["df"], pd.DataFrame) else None
        if pdf is None:
            pdf = _to_pandas_safe(run_context["df"], run_context["engine"])
        schema = sql_table_schema(pdf, table_name or None)
        run_context["sql_schema"] = schema
        _record_tool_event(run_context, "describe_dataframe_table", "ok", table_name=table_name or None)
        return {"schema": schema}

    @tool
    def check_dataframe_sql(query: str, runtime: ToolRuntime[AnalyticaContext]) -> dict[str, str]:
        """Validate a read-only SQL query before execution."""
        run_context = _runtime_run_state(runtime, fallback_run_context)
        try:
            checked = validate_read_only_sql(query)
            checked_at = datetime.now().isoformat(timespec="seconds")
            run_context["last_checked_sql"] = checked
            run_context["last_checked_sql_at"] = checked_at
            _record_tool_event(run_context, "check_dataframe_sql", "ok", query=checked)
            return {"valid": "true", "query": checked, "checked_at": checked_at, "error": ""}
        except Exception as exc:
            checked_at = datetime.now().isoformat(timespec="seconds")
            _record_tool_event(run_context, "check_dataframe_sql", "error", error=str(exc))
            return {"valid": "false", "query": query, "checked_at": checked_at, "error": str(exc)}

    @tool
    def query_dataframe_sql(
        query: str,
        runtime: ToolRuntime[AnalyticaContext],
        table_name: str = "",
    ) -> dict[str, str]:
        """Execute a read-only SQL query against the current DataFrame.

        Use `list_dataframe_tables`, `describe_dataframe_table`, and
        `check_dataframe_sql` before this tool, following the LangChain SQL-agent
        pattern. The default table name comes from configuration.
        """
        run_context = _runtime_run_state(runtime, fallback_run_context)
        if not run_context["schema"]:
            inspect_dataset_schema.func(runtime)
        pdf = run_context["df"] if isinstance(run_context["df"], pd.DataFrame) else None
        if pdf is None:
            pdf = _to_pandas_safe(run_context["df"], run_context["engine"])
        try:
            checked = validate_read_only_sql(query)
            if run_context.get("last_checked_sql") != checked:
                raise ValueError(
                    "Call check_dataframe_sql with this exact query before query_dataframe_sql."
                )
            executed_at = datetime.now().isoformat(timespec="seconds")
            sql_result = run_dataframe_sql_with_metadata(pdf, checked, table_name or None)
            result = sql_result.data
        except Exception as exc:
            err = f"SQL execution error: {exc}"
            run_context["exec_error"] = err
            run_context["result_kind"] = "error"
            run_context["result_preview"] = ""
            run_context["result_facts"] = err
            _record_tool_event(run_context, "query_dataframe_sql", "error", error=str(exc))
            return {
                "exec_error": err,
                "result_kind": "error",
                "result_preview": "",
                "result_facts": err,
            }

        preview = result.head(MAX_TOOL_ROWS).to_string(index=False)
        run_context["code"] = checked
        run_context["exec_error"] = None
        run_context["result_kind"] = "dataframe"
        run_context["result_preview"] = preview
        run_context["sql_metadata"] = {
            "checked_at": run_context.get("last_checked_sql_at", ""),
            "executed_at": executed_at,
            "row_count": sql_result.row_count,
            "truncated": sql_result.truncated,
            "table_name": sql_result.table_name,
            "query": checked,
        }
        _append_artifact_metadata(
            run_context,
            {
                "artifact_type": "table",
                "title": "SQL query result",
                "source_tool": "query_dataframe_sql",
                "rows": int(sql_result.row_count),
                "columns": int(result.shape[1]),
                "format": "dataframe_preview",
                "created_at": executed_at,
            },
        )
        run_context["result_facts"] = (
            f"SQL result shape={result.shape}; row_count={sql_result.row_count}; "
            f"truncated={sql_result.truncated}; table={sql_result.table_name}; "
            f"columns={list(result.columns)}; preview=\n{preview}"
        )
        _record_tool_event(
            run_context,
            "query_dataframe_sql",
            "ok",
            table_name=sql_result.table_name,
            row_count=sql_result.row_count,
            truncated=sql_result.truncated,
        )
        return {
            "exec_error": "",
            "result_kind": "dataframe",
            "result_preview": preview,
            "result_facts": run_context["result_facts"],
            "checked_at": run_context["sql_metadata"]["checked_at"],
            "executed_at": executed_at,
            "row_count": str(sql_result.row_count),
            "truncated": str(sql_result.truncated),
            "table_name": sql_result.table_name,
            "query": checked,
        }

    @tool
    def run_bar_command(command: str, runtime: ToolRuntime[AnalyticaContext]) -> dict[str, str]:
        """Execute a supported /bar command string."""
        code = _bar_codegen(command)
        return run_python_analysis.func(code, runtime)

    @tool
    def write_report_artifact(
        title: str,
        markdown: str,
        runtime: ToolRuntime[AnalyticaContext],
        file_name: str = "",
    ) -> dict[str, str]:
        """Save a markdown report artifact outside the code execution sandbox.

        Args:
            title: Human-readable report title.
            markdown: Complete markdown content to save.
            file_name: Optional file name. If empty, a safe name is generated.
        """
        run_context = _runtime_run_state(runtime, fallback_run_context)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = _slug(file_name or title, fallback="report")
        if not safe_name.endswith(".md"):
            safe_name = f"{safe_name}.md"
        path = ARTIFACT_DIR / safe_name
        if path.exists():
            stem = path.stem
            suffix = path.suffix
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = ARTIFACT_DIR / f"{stem}-{stamp}{suffix}"
        body = markdown.strip()
        if title and not body.startswith("#"):
            body = f"# {title.strip()}\n\n{body}"
        metadata = [
            "",
            "---",
            f"generated_at: {datetime.now().isoformat(timespec='seconds')}",
            f"engine: {run_context.get('engine', '')}",
            f"query: {run_context.get('query', '')}",
            "---",
            "",
        ]
        body = body + "\n" + "\n".join(metadata)
        path.write_text(body + "\n", encoding="utf-8")
        artifact = {
            "type": "markdown_report",
            "artifact_type": "report",
            "title": title,
            "path": str(path),
            "source_tool": "write_report_artifact",
            "format": "markdown",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        run_context.setdefault("artifacts", []).append(artifact)
        run_context["result_kind"] = "artifact"
        run_context["result_preview"] = str(path)
        run_context["result_facts"] = f"Saved markdown report artifact: {path}"
        _record_tool_event(run_context, "write_report_artifact", "ok", path=str(path))
        return {
            "artifact_path": str(path),
            "result_kind": "artifact",
            "result_preview": str(path),
            "result_facts": run_context["result_facts"],
        }

    return [
        create_authoritative_query_plan,
        validate_query_plan,
        route_branch_workspace,
        validate_delivery_delay_sanity,
        inspect_dataset_schema,
        list_dataframe_tables,
        describe_dataframe_table,
        check_dataframe_sql,
        query_dataframe_sql,
        top_n,
        plot_bar,
        find_drops,
        run_bar_command,
        run_python_analysis,
        write_report_artifact,
    ]
