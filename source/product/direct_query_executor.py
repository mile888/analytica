from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from source.product.branch_workspace import BranchWorkspaceManager
from source.product.execution_planner import AuthoritativeQueryPlan, ExtremumScope, QueryFilter, TemporalSanityValidator
from source.product.fallbacks.artifact_builders import build_histogram_artifact
from source.product.fallbacks.semantic_resolution import CUSTOMER_LIKE_MARKERS, DELIVERY_DATE_MARKERS, ORDER_DATE_MARKERS, SALES_LIKE_MARKERS, SHIPPING_METHOD_MARKERS, resolve_categorical_value, resolve_dimension_column
from source.product.fallbacks.shipping_analysis import build_delivery_delay_artifacts, compute_delivery_delay_analysis


class DirectQueryExecutor:
    @classmethod
    def execute(
        cls,
        *,
        question: str,
        df: pd.DataFrame,
        plan: AuthoritativeQueryPlan,
        conversation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        context = conversation_context or {}
        if plan.intent == "constrained_aggregation":
            return cls._constrained_aggregation(question, df, plan)
        if plan.intent == "extremum":
            return cls._extremum(question, df, plan)
        if plan.intent in {"temporal_trend", "chart_request"}:
            return cls._temporal_trend(question, df, plan)
        if plan.intent == "histogram":
            return cls._histogram(question, df, plan)
        if plan.intent == "binning":
            return cls._create_bins(question, df, plan)
        if plan.intent == "bin_question":
            return cls._bins_followup(question, df, plan, context)
        if plan.intent == "shipping_delay":
            return cls._shipping_delay(question, df, plan)
        comparison = cls._distribution_comparison(question, df, plan, context)
        if comparison:
            return comparison
        return None

    @staticmethod
    def _constrained_aggregation(question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan) -> dict[str, Any] | None:
        if not plan.metric or plan.metric not in df.columns or not plan.filters:
            return None
        working = _apply_filters(df, plan.filters)
        values = pd.to_numeric(working[plan.metric], errors="coerce").dropna()
        if values.empty:
            return _result(question, "No rows with numeric values remain after applying the requested filter.", plan, [], analysis_type="constrained_aggregation")
        aggregation = plan.aggregation or "mean"
        value = float(values.mean() if aggregation == "mean" else values.sum() if aggregation == "sum" else values.count())
        filter_text = _filter_text(plan.filters)
        agg_label = "average" if aggregation == "mean" else "total" if aggregation == "sum" else "record count for"
        skew_note = ""
        if aggregation == "mean" and values.mean() > values.median() * 1.5:
            skew_note = " Median is lower, so larger orders pull the average upward."
        summary = (
            f"For records with {filter_text}, {agg_label} `{plan.metric}` is {value:.2f} across {len(values):,} rows. "
            f"Median is {float(values.median()):.2f}; range is {float(values.min()):.2f} to {float(values.max()):.2f}.{skew_note}"
        )
        return _result(
            question,
            summary,
            plan,
            [{"artifact_type": "table", "title": f"Filtered {plan.metric} summary", "content": [{
                "metric": plan.metric,
                "aggregation": aggregation,
                "value": value,
                "n": int(len(values)),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
                "filters": [item.to_payload() for item in plan.filters],
            }], "visibility": "user", "metadata": {"query_plan": plan.to_payload(), "branch_type": "filtered_metric"}}],
            analysis_type="constrained_aggregation",
        )

    @classmethod
    def _extremum(cls, question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan) -> dict[str, Any] | None:
        if not plan.metric or plan.metric not in df.columns:
            return None
        working = _apply_filters(df, plan.filters)
        values = pd.to_numeric(working[plan.metric], errors="coerce")
        working = working.loc[values.notna()].copy()
        working[plan.metric] = values.dropna()
        if working.empty:
            return _result(question, "No rows with numeric values remain after applying the requested filters.", plan, [], analysis_type="extremum")

        dimension = _extremum_dimension(question, df, plan)
        ascending = plan.ranking_direction == "ascending"
        direction_word = "lowest" if ascending else "highest"
        filter_clause = f" with {_filter_text(plan.filters)}" if plan.filters else ""

        if dimension and dimension in working.columns and plan.extremum_scope == ExtremumScope.GROUP_AGGREGATE:
            return cls.execute_group_aggregate_extremum(question, working, plan, dimension, direction_word, filter_clause)
        return cls.execute_row_level_extremum(question, working, plan, dimension, direction_word, filter_clause)

    @staticmethod
    def execute_row_level_extremum(
        question: str,
        working: pd.DataFrame,
        plan: AuthoritativeQueryPlan,
        dimension: str | None,
        direction_word: str,
        filter_clause: str,
    ) -> dict[str, Any] | None:
        if not plan.metric:
            return None
        ascending = plan.ranking_direction == "ascending"
        selector = working[plan.metric].idxmin() if ascending else working[plan.metric].idxmax()
        row = working.loc[selector]
        entity_text = ""
        content = row.to_dict()
        if dimension and dimension in working.columns:
            entity = str(row[dimension])
            entity_text = f" the `{dimension}` is `{entity}`, with"
            content = {
                "target_entity": dimension,
                "target_value": entity,
                "metric": plan.metric,
                "metric_value": float(row[plan.metric]),
                "row_index": str(selector),
                "filters": [item.to_payload() for item in plan.filters],
                "row": _row_context(row),
            }
        scope_text = f"In records{filter_clause}" if filter_clause else "Across all records"
        summary = (
            f"{scope_text},{entity_text} the {direction_word} single `{plan.metric}` record has `{plan.metric}` = {float(row[plan.metric]):.2f}. "
            f"I interpreted this as the {direction_word} single `{plan.metric}` record after applying the requested filters."
        )
        if dimension and dimension in working.columns and _ambiguous_extremum_phrase(question):
            grouped = working.groupby(dimension, dropna=False)[plan.metric].agg(total="sum", n="count").reset_index()
            alt = grouped.loc[grouped["total"].idxmin() if ascending else grouped["total"].idxmax()]
            summary += f" If you meant {direction_word} total `{plan.metric}` by `{dimension}`, that result is `{alt[dimension]}` ({float(alt['total']):.2f})."
        if dimension and plan.filters:
            filter_columns = ", ".join(f"`{item.column}`" for item in plan.filters)
            entity_label = "customer" if _is_customer_like(dimension) else f"`{dimension}`"
            summary += f" The {filter_columns} filter is preserved; this is not a global {entity_label} ranking."

        return _result(
            question,
            summary,
            plan,
            [{"artifact_type": "table", "title": f"{direction_word.title()} {plan.metric}", "content": [content], "visibility": "user", "metadata": {"query_plan": plan.to_payload(), "branch_type": "extremum", "extremum_scope": "row_level"}}],
            analysis_type="extremum",
        )

    @staticmethod
    def execute_group_aggregate_extremum(
        question: str,
        working: pd.DataFrame,
        plan: AuthoritativeQueryPlan,
        dimension: str,
        direction_word: str,
        filter_clause: str,
    ) -> dict[str, Any] | None:
        if not plan.metric:
            return None
        aggregation = plan.aggregate_function or plan.aggregation or "sum"
        agg_col = "aggregate_value"
        if aggregation == "mean":
            grouped = working.groupby(dimension, dropna=False)[plan.metric].agg(**{agg_col: "mean"}, n="count").reset_index()
            agg_label = "average"
        elif aggregation == "median":
            grouped = working.groupby(dimension, dropna=False)[plan.metric].agg(**{agg_col: "median"}, n="count").reset_index()
            agg_label = "median"
        elif aggregation == "count":
            grouped = working.groupby(dimension, dropna=False)[plan.metric].agg(**{agg_col: "count"}, n="count").reset_index()
            agg_label = "count of"
        else:
            grouped = working.groupby(dimension, dropna=False)[plan.metric].agg(**{agg_col: "sum"}, n="count").reset_index()
            agg_label = "total"
            aggregation = "sum"
        selector = grouped[agg_col].idxmin() if plan.ranking_direction == "ascending" else grouped[agg_col].idxmax()
        row = grouped.loc[selector]
        entity = str(row[dimension])
        summary = (
            f"The `{dimension}`{filter_clause} with the {direction_word} {agg_label} `{plan.metric}` is `{entity}`: "
            f"{agg_label} {float(row[agg_col]):.2f} across {int(row['n'])} rows."
        )
        content = [{
            dimension: entity,
            "aggregate_function": aggregation,
            "aggregate_value": float(row[agg_col]),
            "n": int(row["n"]),
            "metric": plan.metric,
            "direction": direction_word,
            "filters": [item.to_payload() for item in plan.filters],
        }]
        return _result(
            question,
            summary,
            plan,
            [{"artifact_type": "table", "title": f"{direction_word.title()} {agg_label.title()} {plan.metric}", "content": content, "visibility": "user", "metadata": {"query_plan": plan.to_payload(), "branch_type": "extremum", "extremum_scope": "group_aggregate"}}],
            analysis_type="extremum",
        )

    @staticmethod
    def _temporal_trend(question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan) -> dict[str, Any] | None:
        if not plan.metric or plan.metric not in df.columns or not plan.time_axis or plan.time_axis not in df.columns:
            return None
        working = _apply_filters(df, plan.filters)
        values = pd.to_numeric(working[plan.metric], errors="coerce")
        dates = pd.to_datetime(working[plan.time_axis], errors="coerce")
        working = working.loc[values.notna() & dates.notna()].copy()
        if working.empty:
            return _result(question, "No rows with valid dates and numeric values remain for the requested time trend.", plan, [], analysis_type="temporal_trend")
        working[plan.metric] = pd.to_numeric(working[plan.metric], errors="coerce")
        working["_period"] = _period_labels(pd.to_datetime(working[plan.time_axis], errors="coerce"), plan.time_grain or "month")
        aggregation = plan.aggregation or "mean"
        grouped = (
            working.groupby("_period", sort=True)[plan.metric]
            .agg(value=aggregation if aggregation in {"mean", "sum", "count"} else "mean", n="count")
            .reset_index()
        )
        rows = [
            {"period": str(row["_period"]), "value": float(row["value"]), "n": int(row["n"])}
            for _, row in grouped.iterrows()
        ]
        first = rows[0]["value"]
        last = rows[-1]["value"]
        change = last - first
        agg_label = "average" if aggregation == "mean" else "total" if aggregation == "sum" else "count of"
        grain_label = {"day": "daily", "month": "monthly", "year": "yearly"}.get(plan.time_grain or "month", "monthly")
        insight = _temporal_insight_text(rows, metric=plan.metric, aggregation_label=agg_label)
        if plan.intent == "chart_request" or plan.chart_type == "line":
            summary = (
                f"Created a line chart of {agg_label} `{plan.metric}` over `{plan.time_axis}` using {grain_label} aggregation. "
                f"{insight}"
            )
        else:
            summary = (
                f"{agg_label.title()} `{plan.metric}` over `{plan.time_axis}` uses {grain_label} periods. {insight}"
            )
        artifact = {
            "artifact_type": "chart",
            "title": f"Average {plan.metric} over {plan.time_axis}" if aggregation == "mean" else f"{plan.metric} trend over {plan.time_axis}",
            "content": {
                "chart_type": "line",
                "visualization_type": "trend",
                "metric": plan.metric,
                "time_axis": plan.time_axis,
                "timestamp": plan.time_axis,
                "time_grain": plan.time_grain or "month",
                "aggregation": aggregation,
                "filters": [item.to_payload() for item in plan.filters],
                "x": "period",
                "y": "value",
                "rows": rows,
            },
            "visibility": "user",
            "pinned": True,
            "metadata": {"branch_type": "temporal", "query_plan": plan.to_payload(), "metric": plan.metric, "time_axis": plan.time_axis, "aggregation": aggregation, "chart_type": "line", "filters": [item.to_payload() for item in plan.filters]},
        }
        return _result(question, summary, plan, [artifact], analysis_type="temporal_trend")

    @staticmethod
    def _histogram(question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan) -> dict[str, Any] | None:
        if not plan.metric or plan.metric not in df.columns:
            return None
        working = _apply_filters(df, plan.filters)
        values = pd.to_numeric(working[plan.metric], errors="coerce").dropna()
        if values.empty:
            return _result(question, "The histogram cannot be built because no numeric values remain after filtering.", plan, [], analysis_type="histogram")
        bins = _histogram_bins(values)
        filter_suffix = _filter_suffix(plan.filters)
        title = f"{plan.metric} distribution{filter_suffix}"
        summary = (
            f"`{plan.metric}` distribution{filter_suffix} uses {len(values):,} rows. "
            f"Median is {float(values.median()):.2f}, mean is {float(values.mean()):.2f}, and the range is {float(values.min()):.2f} to {float(values.max()):.2f}. "
            f"Filter: {_filter_text(plan.filters) if plan.filters else 'none'}."
        )
        artifact = build_histogram_artifact(
            metric=plan.metric,
            bins=bins,
            filters=plan.filters,
            title=title,
            query_plan=plan.to_payload(),
            row_count=int(len(values)),
        )
        result = _result(question, summary, plan, [artifact], analysis_type="histogram")
        result["trace_metadata"]["distribution_state"] = {
            "branch_type": "distribution",
            "metric": plan.metric,
            "filters": [item.to_payload() for item in plan.filters],
            "chart_type": "histogram",
            "bins": bins,
            "row_count": int(len(values)),
        }
        return result

    @staticmethod
    def _distribution_comparison(question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan, context: dict[str, Any]) -> dict[str, Any] | None:
        normalized = _norm(question)
        if not any(marker in normalized for marker in ("compare against", "compare with", "сравн")):
            return None
        state = _distribution_state(context)
        metric = str(state.get("metric") or "")
        filters = state.get("filters") if isinstance(state.get("filters"), list) else []
        base = next((item for item in filters if isinstance(item, dict) and item.get("column") in df.columns), None)
        if not metric or metric not in df.columns or not base:
            return None
        column = str(base["column"])
        base_value = str(base["value"])
        target_value = _match_value(question, df, column, exclude={base_value})
        if not target_value:
            return None
        rows = [_distribution_stats(df, metric, column, base_value), _distribution_stats(df, metric, column, target_value)]
        summary = (
            f"Compared `{metric}` distributions for `{base_value}` and `{target_value}`. "
            f"{base_value}: n={rows[0]['n']}, mean={rows[0]['mean']:.2f}, median={rows[0]['median']:.2f}, range {rows[0]['min']:.2f} to {rows[0]['max']:.2f}. "
            f"{target_value}: n={rows[1]['n']}, mean={rows[1]['mean']:.2f}, median={rows[1]['median']:.2f}, range {rows[1]['min']:.2f} to {rows[1]['max']:.2f}. "
            "The comparison is distributional: read the median, mean, and range together to separate center shift from tail effects."
        )
        return _result(
            question,
            summary,
            plan,
            [{"artifact_type": "table", "title": f"{metric} distribution comparison", "content": rows, "visibility": "user", "metadata": {"branch_type": "distribution_comparison", "query_plan": plan.to_payload()}}],
            analysis_type="distribution_comparison",
        )

    @staticmethod
    def _create_bins(question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan) -> dict[str, Any] | None:
        metric = plan.metric if plan.metric in df.columns else _metric_from_question(question, df)
        if not metric:
            return None
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if values.empty:
            return None
        binned = pd.qcut(values, q=min(4, max(2, values.nunique())), duplicates="drop")
        rows = binned.value_counts().sort_index().reset_index()
        rows.columns = [f"{metric}_bin", "record_count"]
        derived = {
            "derived_field": f"{metric}_bin",
            "source_metric": metric,
            "method": "quantile",
            "labels": [str(item) for item in rows[f"{metric}_bin"].tolist()],
        }
        summary = f"Created `{metric}_bin` from `{metric}` using quantile bins. The bins are ready for contribution, sparsity, and volume-vs-value checks."
        result = _result(question, summary, plan, [{"artifact_type": "table", "title": f"{metric} automatic bins", "content": rows.to_dict(orient="records"), "visibility": "user", "metadata": {"derived_field": derived, "branch_type": "distribution", "query_plan": plan.to_payload()}}], analysis_type="binning")
        result["trace_metadata"]["derived_field"] = derived
        return result

    @staticmethod
    def _bins_followup(question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan, context: dict[str, Any]) -> dict[str, Any] | None:
        state = context.get("conversation_state") if isinstance(context.get("conversation_state"), dict) else {}
        derived = state.get("derived_field") if isinstance(state.get("derived_field"), dict) else {}
        if not derived:
            return None
        metric = str(derived.get("source_metric") or "")
        derived_field = str(derived.get("derived_field") or "")
        if not metric or metric not in df.columns or not derived_field:
            return None
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        bins = pd.qcut(values, q=min(4, max(2, values.nunique())), duplicates="drop")
        binned = df.loc[values.index].copy()
        binned[derived_field] = bins.astype(str)
        grouped = binned.groupby(derived_field, dropna=False)[metric].agg(record_count="count", total="sum", average="mean").reset_index()
        rows = grouped.to_dict(orient="records")
        summary, analysis_type = _bin_followup_summary(question, rows, derived_field, metric)
        result = _result(question, summary, plan, [{"artifact_type": "table", "title": f"{derived_field} analysis", "content": rows, "visibility": "user", "metadata": {"derived_field": derived, "branch_type": "distribution", "query_plan": plan.to_payload(), "analysis_type": analysis_type}}], analysis_type=analysis_type)
        result["trace_metadata"]["derived_field"] = derived
        result["trace_metadata"]["dimension"] = derived_field
        return result

    @staticmethod
    def _shipping_delay(question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan) -> dict[str, Any] | None:
        metric = plan.metric if plan.metric in df.columns else _metric_from_question(question, df)
        order_col = _column_by_markers(df, ORDER_DATE_MARKERS)
        ship_col = _column_by_markers(df, DELIVERY_DATE_MARKERS)
        if not metric or not order_col or not ship_col:
            return None
        issues = TemporalSanityValidator.delivery_delay_issues(df, order_col, ship_col)
        method_col = _column_by_markers(df, SHIPPING_METHOD_MARKERS)
        analysis = compute_delivery_delay_analysis(df, metric=metric, order_column=order_col, delivery_column=ship_col, method_column=method_col)
        if not analysis:
            return None
        relationship = _growth_delay_relationship(question, df, metric, order_col, ship_col)
        if relationship:
            limitations = relationship.get("limitations", [])
            artifacts = build_delivery_delay_artifacts(analysis, query_plan=plan.to_payload())
            artifacts.append(
                {
                    "artifact_type": "table",
                    "title": f"{metric} growth and delivery delay",
                    "content": relationship["rows"],
                    "visibility": "user",
                    "metadata": {
                        "branch_type": "shipping_delay",
                        "analysis_type": "growth_delay_relationship",
                        "metric": metric,
                        "time_axis": order_col,
                        "derived_field": "delivery_delay_days",
                        "query_plan": plan.to_payload(),
                    },
                }
            )
            return _result(question, relationship["summary"], plan, artifacts, analysis_type="growth_delay_relationship", limitations=limitations)
        delayed_share = analysis.delayed_count / analysis.row_count * 100.0 if analysis.row_count else 0.0
        largest_bucket = max(analysis.bucket_rows, key=lambda row: int(row.get("record_count") or 0)) if analysis.bucket_rows else {}
        method_text = ""
        if method_col and analysis.method_rows:
            top_method = analysis.method_rows[0]
            method_text = f" Largest `{method_col}` group: `{top_method[method_col]}` with {top_method['record_count']} rows and average delay {top_method['average_delay_days']:.2f} days."
        summary = (
            f"Delivery-delay analysis derived `delivery_delay_days` from `{ship_col}` minus `{order_col}` across {analysis.row_count:,} rows. "
            f"Average delay is {analysis.average_delay_days:.2f} days; maximum delay is {analysis.max_delay_days:.2f} days. "
            f"Late/delayed records with positive delay: {analysis.delayed_count:,} ({delayed_share:.1f}%). "
            f"Largest delay bucket: `{largest_bucket.get('delay_bucket')}` with {int(largest_bucket.get('record_count') or 0):,} rows."
            f"{method_text}"
        )
        limitations = []
        if issues:
            warning = "Some parsed delays look unusual: " + "; ".join(issues) + ". Interpret the delay summary cautiously."
            summary += " " + warning
            limitations = issues
        return _result(question, summary, plan, build_delivery_delay_artifacts(analysis, query_plan=plan.to_payload()), analysis_type="shipping_delay", limitations=limitations)


def _result(question: str, summary: str, plan: AuthoritativeQueryPlan, artifacts: list[dict[str, Any]], *, analysis_type: str, limitations: list[str] | None = None) -> dict[str, Any]:
    timeline = (
        [
            {"tool": "trend_check", "status": "ok", "intent": plan.intent},
            {"tool": "direct_query_executor", "status": "ok", "intent": plan.intent},
        ]
        if analysis_type == "temporal_trend"
        else [
            {"tool": "deterministic_pandas_fallback", "status": "ok", "intent": plan.intent},
            {"tool": "direct_query_executor", "status": "ok", "intent": plan.intent},
        ]
    )
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        content = artifact.get("content") if isinstance(artifact.get("content"), dict) else {}
        metadata.setdefault("metric", plan.metric)
        metadata.setdefault("dimension", plan.dimension)
        metadata.setdefault("aggregation", plan.aggregation)
        metadata.setdefault("filters", [item.to_payload() for item in plan.filters])
        metadata.setdefault("chart_type", content.get("chart_type") or plan.chart_type)
        row_count = content.get("row_count") if isinstance(content, dict) else None
        if row_count is None and isinstance(artifact.get("content"), list):
            row_count = len(artifact["content"])
        metadata.setdefault("row_count", row_count)
        metadata.setdefault("artifact_type", artifact.get("artifact_type") or artifact.get("type"))
        metadata.setdefault("query_plan", plan.to_payload())
        artifact["metadata"] = metadata
    return {
        "final_answer": summary,
        "code": "",
        "result_preview": "",
        "result_base64": "",
        "exec_error": None,
        "engine": "pandas",
        "needs_data": True,
        "use_case": "data_analytics",
        "loaded_skills": ["data-analysis", "csv-dataframe-analysis", "reporting"],
        "tool_timeline": timeline,
        "sql_metadata": {},
        "structured_report": {
            "question": question,
            "summary": summary,
            "key_findings": [summary],
            "evidence": ["Executed directly from the authoritative query plan."],
            "limitations": limitations or [],
            "artifacts": [],
            "next_steps": [],
            "generated_code": "",
            "loaded_skills": ["data-analysis", "csv-dataframe-analysis", "reporting"],
            "tool_timeline": timeline,
            "sql_metadata": {},
        },
        "trace_metadata": {
            "fallback": "direct_query_executor",
            "analysis_type": analysis_type,
            "metric": plan.metric,
            "dimension": plan.dimension,
            "chart_type": plan.chart_type,
            "query_plan": plan.to_payload(),
            "filters": [item.to_payload() for item in plan.filters],
            "extremum_scope": plan.extremum_scope.value if isinstance(plan.extremum_scope, ExtremumScope) else plan.extremum_scope,
        },
        "critic_verdict": "",
        "critic_feedback": "",
        "artifacts": artifacts,
    }


def _apply_filters(df: pd.DataFrame, filters: list[QueryFilter]) -> pd.DataFrame:
    working = df.copy()
    for item in filters:
        if item.column in working.columns and item.operator == "equals":
            working = working[working[item.column].astype(str).str.casefold() == str(item.value).casefold()]
    return working


def _filter_text(filters: list[QueryFilter]) -> str:
    return ", ".join(f"`{item.column}` = `{item.value}`" for item in filters)


def _filter_suffix(filters: list[QueryFilter]) -> str:
    if not filters:
        return ""
    if len(filters) == 1:
        return f" in {filters[0].value}"
    return " with " + _filter_text(filters)


def _histogram_bins(values: pd.Series) -> list[dict[str, Any]]:
    counts = pd.cut(values, bins=min(10, max(3, int(values.nunique())))).value_counts().sort_index()
    return [
        {"left": float(interval.left), "right": float(interval.right), "label": f"{interval.left:.2f} to {interval.right:.2f}", "count": int(count)}
        for interval, count in counts.items()
    ]


def _distribution_state(context: dict[str, Any]) -> dict[str, Any]:
    state = context.get("conversation_state") if isinstance(context.get("conversation_state"), dict) else {}
    if isinstance(state.get("distribution_state"), dict):
        return state["distribution_state"]
    chart = context.get("latest_chart_context") if isinstance(context.get("latest_chart_context"), dict) else {}
    if chart.get("chart_type") == "histogram":
        filters = chart.get("filters") if isinstance(chart.get("filters"), list) else []
        return {"metric": chart.get("metric"), "filters": filters}
    recent = context.get("recent_artifacts") if isinstance(context.get("recent_artifacts"), list) else []
    for artifact in reversed(recent):
        content = artifact.get("content") if isinstance(artifact, dict) else None
        if isinstance(content, dict) and content.get("chart_type") == "histogram":
            return {"metric": content.get("metric") or content.get("x"), "filters": content.get("filters") or []}
    return {}


def _distribution_stats(df: pd.DataFrame, metric: str, column: str, value: str) -> dict[str, Any]:
    subset = df[df[column].astype(str).str.casefold() == value.casefold()]
    values = pd.to_numeric(subset[metric], errors="coerce").dropna()
    if values.empty:
        return {column: value, "n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {column: value, "n": int(len(values)), "mean": float(values.mean()), "median": float(values.median()), "min": float(values.min()), "max": float(values.max())}


def _extremum_dimension(question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan) -> str | None:
    normalized = _norm(question)
    if plan.dimension and plan.dimension in df.columns:
        return plan.dimension
    if any(marker in normalized for marker in ("customer", "client", "клиент")):
        return resolve_dimension_column(question, [str(col) for col in df.columns], metric=plan.metric, df=df) or _column_by_markers(df, CUSTOMER_LIKE_MARKERS)
    if "city" in normalized or "город" in normalized:
        return _column_by_markers(df, ("city", "город", "town", "location"))
    if "order" in normalized:
        return _column_by_markers(df, ("order id", "order_id", "order", "заказ"))
    return None


def _row_context(row: pd.Series) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for column in row.index:
        if _is_context_column(str(column)):
            value = row[column]
            context[column] = str(value) if not isinstance(value, (int, float, bool)) else value
    return context


def _growth_delay_relationship(question: str, df: pd.DataFrame, metric: str, order_col: str, ship_col: str) -> dict[str, Any] | None:
    text = _norm(question)
    if not any(marker in text for marker in ("growth", "trend", "related", "relationship", "correlat", "рост", "тренд", "связан")):
        return None
    working = df[[metric, order_col, ship_col]].copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    working[order_col] = pd.to_datetime(working[order_col], errors="coerce")
    working[ship_col] = pd.to_datetime(working[ship_col], errors="coerce")
    working = working.dropna(subset=[metric, order_col, ship_col])
    if working.empty:
        return None
    working["delivery_delay_days"] = (working[ship_col] - working[order_col]).dt.days
    working["period"] = _period_labels(working[order_col], "month")
    grouped = (
        working.groupby("period", dropna=False)
        .agg(
            metric_value=(metric, "sum"),
            average_delay_days=("delivery_delay_days", "mean"),
            record_count=(metric, "count"),
        )
        .reset_index()
        .sort_values("period")
    )
    if len(grouped) < 3:
        return None
    grouped["metric_change"] = grouped["metric_value"].diff()
    comparable = grouped.dropna(subset=["metric_change"]).copy()
    if comparable.empty:
        return None
    comparable["growth_state"] = comparable["metric_change"].apply(lambda value: "growth" if value > 0 else "decline_or_flat")
    growth_rows = comparable[comparable["growth_state"] == "growth"]
    other_rows = comparable[comparable["growth_state"] != "growth"]
    growth_delay = float(growth_rows["average_delay_days"].mean()) if not growth_rows.empty else 0.0
    other_delay = float(other_rows["average_delay_days"].mean()) if not other_rows.empty else 0.0
    gap = growth_delay - other_delay
    corr = comparable["metric_change"].corr(comparable["average_delay_days"]) if len(comparable) >= 3 else 0.0
    corr_value = float(corr) if pd.notna(corr) else 0.0
    strength = "strong" if abs(corr_value) >= 0.6 else "moderate" if abs(corr_value) >= 0.3 else "weak"
    direction = "longer" if gap > 0 else "shorter" if gap < 0 else "similar"
    rows = [
        {
            "period": str(row["period"]),
            "metric_value": float(row["metric_value"]),
            "metric_change": float(row["metric_change"]) if pd.notna(row["metric_change"]) else None,
            "average_delay_days": float(row["average_delay_days"]),
            "record_count": int(row["record_count"]),
            "growth_state": str(row.get("growth_state") or "baseline"),
        }
        for _, row in comparable.iterrows()
    ]
    summary = (
        f"`{metric}` growth is compared with derived `delivery_delay_days` by `{order_col}` period. "
        f"Growth periods have average delay {growth_delay:.2f} days versus {other_delay:.2f} days in flat/declining periods, so delays are {direction} during growth by {abs(gap):.2f} days. "
        f"The period-level relationship is {strength} (correlation {corr_value:.2f}), so treat it as directional evidence rather than proof of causality."
    )
    limitations = []
    if comparable["record_count"].min() < 3:
        limitations.append("Some periods have very small row counts, so period-level delay comparisons are sensitive to individual records.")
    return {"summary": summary, "rows": rows, "limitations": limitations}


def _ambiguous_extremum_phrase(question: str) -> bool:
    normalized = _norm(question)
    if any(marker in normalized for marker in ("total", "average", "mean", "median", "суммар", "средн", "медиан", "всего", "по сумме")):
        return False
    return any(marker in normalized for marker in ("customer with", "which customer", "какой customer", "клиент"))


def _period_labels(dates: pd.Series, grain: str) -> pd.Series:
    if grain == "day":
        return dates.dt.to_period("D").astype(str)
    if grain == "year":
        return dates.dt.to_period("Y").astype(str)
    return dates.dt.to_period("M").astype(str)


def _temporal_insight_text(rows: list[dict[str, Any]], *, metric: str, aggregation_label: str) -> str:
    values = [float(row.get("value") or 0.0) for row in rows]
    if len(values) < 2:
        return f"There is only one valid period, so `{metric}` does not yet show a directional pattern."
    first = values[0]
    last = values[-1]
    change = last - first
    pct = (change / abs(first) * 100.0) if first else 0.0
    direction = "rises" if change > 0 else "declines" if change < 0 else "stays flat"
    deltas = [values[idx] - values[idx - 1] for idx in range(1, len(values))]
    abs_deltas = [abs(item) for item in deltas]
    avg_abs_delta = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
    value_range = max(values) - min(values)
    baseline = abs(sum(values) / len(values)) or 1.0
    volatility_ratio = max(avg_abs_delta / baseline, value_range / baseline)
    volatility = "volatile rather than smooth" if volatility_ratio >= 0.35 else "fairly steady" if volatility_ratio <= 0.12 else "moderately uneven"
    largest_up_idx = max(range(len(deltas)), key=lambda idx: deltas[idx]) if deltas else None
    largest_down_idx = min(range(len(deltas)), key=lambda idx: deltas[idx]) if deltas else None
    strongest_up = ""
    strongest_down = ""
    if largest_up_idx is not None and deltas[largest_up_idx] > 0:
        strongest_up = f" Largest positive jump: {rows[largest_up_idx]['period']} to {rows[largest_up_idx + 1]['period']} ({deltas[largest_up_idx]:+.2f})."
    if largest_down_idx is not None and deltas[largest_down_idx] < 0:
        strongest_down = f" Largest decline: {rows[largest_down_idx]['period']} to {rows[largest_down_idx + 1]['period']} ({deltas[largest_down_idx]:+.2f})."
    magnitude = f"{abs(change):.2f}" if abs(pct) < 0.1 else f"{abs(change):.2f} ({pct:+.1f}%)"
    return (
        f"{aggregation_label.title()} `{metric}` {direction} overall, moving from {first:.2f} to {last:.2f} "
        f"across {len(rows)} periods; the net change is {magnitude}. "
        f"The series is {volatility}, so period-to-period movement matters as much as the endpoint change."
        f"{strongest_up}{strongest_down}"
    )


def _match_value(question: str, df: pd.DataFrame, column: str, exclude: set[str] | None = None) -> str:
    resolved = resolve_categorical_value(question, df, preferred_columns=[column], exclude=exclude)
    return resolved.value if resolved else ""


def _norm(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _column_by_markers(df: pd.DataFrame, markers: tuple[str, ...]) -> str | None:
    best: tuple[int, str] | None = None
    for column in df.columns:
        normalized = _norm(str(column))
        score = sum(20 + len(marker) for marker in markers if marker in normalized)
        if score and (best is None or score > best[0]):
            best = (score, str(column))
    return best[1] if best else None


def _metric_from_question(question: str, df: pd.DataFrame) -> str | None:
    normalized = _norm(question)
    marker_groups = (
        SALES_LIKE_MARKERS,
        ("profit", "прибыл", "margin"),
        ("cost", "expense", "затрат", "расход"),
        ("price", "цена"),
        ("score", "rating", "оцен", "рейтинг"),
    )
    for markers in marker_groups:
        if any(marker in normalized for marker in markers):
            match = _column_by_markers(df, markers)
            if match:
                return match
    numeric = [str(column) for column in df.select_dtypes(include="number").columns]
    return numeric[0] if numeric else None


def _bin_followup_summary(question: str, rows: list[dict[str, Any]], derived_field: str, metric: str) -> tuple[str, str]:
    normalized = _norm(question)
    sorted_total = sorted(rows, key=lambda row: float(row.get("total") or 0), reverse=True)
    sorted_count = sorted(rows, key=lambda row: int(row.get("record_count") or 0), reverse=True)
    sorted_avg = sorted(rows, key=lambda row: float(row.get("average") or 0), reverse=True)
    top_total = sorted_total[0] if sorted_total else {}
    top_count = sorted_count[0] if sorted_count else {}
    top_avg = sorted_avg[0] if sorted_avg else {}
    if any(marker in normalized for marker in ("contribute", "contributes", "most revenue", "most sales", "largest total", "biggest total", "вклад")):
        return (
            f"`{top_total.get(derived_field)}` contributes the most `{metric}` with total {float(top_total.get('total') or 0):.2f} across {int(top_total.get('record_count') or 0)} rows.",
            "bins_contribution",
        )
    if any(marker in normalized for marker in ("driven", "explain", "volume", "order value", "average", "связ", "объяс", "объем", "объём")):
        return (
            f"Across `{derived_field}`, the largest total is `{top_total.get(derived_field)}` ({float(top_total.get('total') or 0):.2f}), the largest record volume is `{top_count.get(derived_field)}` (n={int(top_count.get('record_count') or 0)}), and the highest average `{metric}` is `{top_avg.get(derived_field)}` ({float(top_avg.get('average') or 0):.2f}). If the total leader matches the volume leader, volume is the main explanation; if it matches the average leader, order value is the stronger driver.",
            "bins_volume_value_driver",
        )
    if any(marker in normalized for marker in ("sparse", "small sample", "low sample", "few records", "малень", "редк")):
        counts = [int(row.get("record_count") or 0) for row in rows]
        if counts and max(counts) - min(counts) <= 1:
            return (f"No `{derived_field}` bins are materially sparse; the quantile strategy produced nearly balanced groups with {min(counts)} to {max(counts)} records per bin.", "bins_sparsity_balanced")
        sparse = sorted(rows, key=lambda row: int(row.get("record_count") or 0))[:3]
        bits = ", ".join(f"`{row.get(derived_field)}` n={int(row.get('record_count') or 0)}" for row in sparse)
        return (f"Sparsest `{derived_field}` bins are {bits}. These bins need caution before comparing `{metric}` averages.", "bins_sparsity")
    spread = float(top_avg.get("average") or 0) - float(sorted_avg[-1].get("average") or 0) if len(sorted_avg) > 1 else 0.0
    return (
        f"`{derived_field}` bins differ in both volume and average `{metric}`. Record-volume leader: `{top_count.get(derived_field)}` (n={int(top_count.get('record_count') or 0)}); average-value leader: `{top_avg.get(derived_field)}` ({float(top_avg.get('average') or 0):.2f}); average spread is {spread:.2f}.",
        "bins_comparison",
    )


def _is_context_column(column: str) -> bool:
    normalized = _norm(column)
    return any(
        marker in normalized
        for marker in (
            "id",
            "name",
            "customer",
            "client",
            "product",
            "city",
            "region",
            "country",
            "date",
            "time",
            "клиент",
            "продукт",
            "город",
            "дата",
        )
    )


def _is_customer_like(column: str) -> bool:
    normalized = _norm(column)
    return any(marker in normalized for marker in ("customer", "client", "клиент", "buyer"))
