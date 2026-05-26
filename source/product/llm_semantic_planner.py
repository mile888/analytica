"""Build semantic analysis plans from user questions and dataset schemas."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from source.llm.factory import make_llm

ALLOWED_OPERATIONS = frozenset({
    "COUNT_DISTRIBUTION",
    "METRIC_AGGREGATION",
    "GROUPED_AGGREGATION",
    "CATEGORICAL_OUTCOME_BREAKDOWN",
    "ORDERED_OUTCOME_RISK",
    "TEMPORAL_TREND",
    "CATEGORY_MIX_SHIFT",
    "DIVERSITY_ANALYSIS",
    "DURATION_ANALYSIS",
    "OUTLIER_ANALYSIS",
    "COMPARISON_ANALYSIS",
    "STRATEGIC_SYNTHESIS",
    "CORRELATION_ANALYSIS",
    "DATA_QUALITY_CHECK",
    "DATASET_OVERVIEW",
})

ALLOWED_AGGREGATIONS = frozenset({
    "count", "sum", "mean", "median", "min", "max",
    "share", "unique_count", "entropy", "none",
})

ALLOWED_ARTIFACT_TYPES = frozenset({
    "bar", "histogram", "line", "table", "heatmap", "none",
})


@dataclass
class SemanticPlan:
    """Stores the operation, columns, filters, and locked user constraints."""

    operation: str
    metric: str | None = None
    dimension: str | None = None
    metric_columns: list[str] = field(default_factory=list)
    grouping_columns: list[str] = field(default_factory=list)
    time_field: str | None = None
    category_field: str | None = None
    duration_field: str | None = None
    target_column: str | None = None
    target_values: list[str] = field(default_factory=list)
    ordered_column: str | None = None
    filters: list[dict[str, Any]] = field(default_factory=list)
    aggregation: str = "count"
    sorting: str | None = None
    limit: int | None = None
    artifact_type: str = "none"
    visualization_requested: bool = False
    artifact_required: bool = False
    constraints_locked: dict[str, bool] = field(default_factory=dict)
    forbidden_aggregations: list[str] = field(default_factory=list)
    needs_execution: bool = True
    reasoning: str = ""
    confidence: float = 0.0
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SchemaContext:
    """Stores dataset schema metadata without raw rows."""

    columns: list[dict[str, Any]]
    row_count: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    year_like_columns: list[str]
    date_like_columns: list[str]
    identifier_columns: list[str]
    duration_like_columns: list[str]
    multi_label_columns: list[str]
    ordered_range_columns: list[str] = field(default_factory=list)


def build_schema_context(df: pd.DataFrame) -> SchemaContext:
    """Build a schema profile for planning."""

    columns: list[dict[str, Any]] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    year_like: list[str] = []
    date_like: list[str] = []
    identifier_cols: list[str] = []
    duration_like: list[str] = []
    multi_label: list[str] = []
    ordered_range: list[str] = []

    for col in df.columns:
        col_name = str(col)
        series = df[col]
        dtype_str = str(series.dtype)
        nunique = int(series.nunique())
        non_null = int(series.notna().sum())
        sample_values = [str(v) for v in series.dropna().head(5).tolist()]

        col_info: dict[str, Any] = {
            "name": col_name,
            "dtype": dtype_str,
            "nunique": nunique,
            "non_null_count": non_null,
            "sample_values": sample_values,
        }

        is_numeric = pd.api.types.is_numeric_dtype(series)
        if is_numeric:
            numeric_cols.append(col_name)
            desc = series.describe()
            col_info["stats"] = {
                "min": _safe_num(desc.get("min")),
                "max": _safe_num(desc.get("max")),
                "mean": _safe_num(desc.get("mean")),
                "median": _safe_num(series.median()),
            }
            smin, smax = _safe_num(desc.get("min")), _safe_num(desc.get("max"))
            if smin is not None and smax is not None and 1900 <= smin <= 2100 and 1900 <= smax <= 2100:
                year_like.append(col_name)
                col_info["semantic_hint"] = "year"
        else:
            categorical_cols.append(col_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sample = series.dropna().head(30)
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if len(parsed) and parsed.notna().mean() >= 0.7:
                        date_like.append(col_name)
                        col_info["semantic_hint"] = "date"
                except Exception:
                    pass

            if _is_identifier(col_name, nunique, len(df)):
                identifier_cols.append(col_name)
                col_info["semantic_hint"] = "identifier"

            if _is_duration_like(sample_values):
                duration_like.append(col_name)
                col_info["semantic_hint"] = "duration"

            if _is_multi_label(sample_values):
                multi_label.append(col_name)
                col_info["semantic_hint"] = "multi_label"

            if _is_ordered_range_like(sample_values):
                ordered_range.append(col_name)
                existing_hint = str(col_info.get("semantic_hint") or "")
                col_info["semantic_hint"] = "ordered_range" if not existing_hint else f"{existing_hint},ordered_range"

        columns.append(col_info)

    return SchemaContext(
        columns=columns,
        row_count=len(df),
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        year_like_columns=year_like,
        date_like_columns=date_like,
        identifier_columns=identifier_cols,
        duration_like_columns=duration_like,
        multi_label_columns=multi_label,
        ordered_range_columns=ordered_range,
    )


def plan_analysis(
    question: str,
    df: pd.DataFrame,
    context: dict[str, Any] | None = None,
) -> SemanticPlan | None:
    """Use the LLM to build a plan, or return None for fallback planning."""
    schema = build_schema_context(df)
    explicit_plan = plan_explicit_constraints(question, df, schema=schema, context=context)
    if explicit_plan is not None:
        return explicit_plan
    prompt = _build_planner_prompt(question, schema, context)

    try:
        llm = make_llm("semantic_planner")
        response = llm.invoke(prompt)
        text = _response_text(response)
    except Exception:
        return None

    return _parse_plan(text, schema)


def plan_explicit_constraints(
    question: str,
    df: pd.DataFrame,
    *,
    schema: SchemaContext | None = None,
    context: dict[str, Any] | None = None,
) -> SemanticPlan | None:
    """Build a plan from explicit metric, group, chart, and aggregation rules."""

    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    schema = schema or build_schema_context(df)
    text = _normalize_text(question)
    columns = [str(col) for col in df.columns]
    mentioned_columns = _mentioned_schema_columns(question, columns)
    aggregation, forbidden = _requested_aggregation(text)
    limit = _requested_limit(text)
    sorting = "asc" if _has_any(text, ("lowest", "smallest", "ascending", "least", "bottom ")) else "desc" if _has_any(text, ("top ", "highest", "largest", "descending", "most", "strongest")) else None
    artifact_type = _requested_artifact_type(text)
    visualization_requested = artifact_type != "none"

    explicit_group = _explicit_grouping_column(question, columns)
    explicit_metric = _explicit_metric_column(question, columns)
    if not explicit_group and not explicit_metric and _has_any(text, ("histogram", "distribution", "trend", "over time", "time series")):
        return None
    metric = explicit_metric or _metric_from_mentioned_columns(text, mentioned_columns, df, aggregation)
    dimension = explicit_group or _dimension_from_mentioned_columns(text, mentioned_columns, df, exclude={metric} if metric else set())

    target_column, target_values = _target_from_value_mentions(question, df)
    ordered_column = _ordered_column_from_question(text, mentioned_columns, schema)

    if _looks_like_ordered_outcome_risk(text) and target_column and target_values and ordered_column:
        groups = _requested_or_candidate_groupings(
            question,
            df,
            mentioned_columns,
            exclude={target_column, ordered_column},
            prefer_demographic=True,
        )
        if groups:
            return SemanticPlan(
                operation="ORDERED_OUTCOME_RISK",
                grouping_columns=groups,
                dimension=groups[0],
                target_column=target_column,
                target_values=target_values,
                ordered_column=ordered_column,
                aggregation="share",
                sorting=sorting or "desc",
                limit=limit,
                artifact_type=artifact_type if visualization_requested else "table",
                visualization_requested=visualization_requested,
                artifact_required=visualization_requested,
                constraints_locked={
                    "target_column": target_column in mentioned_columns,
                    "ordered_column": ordered_column in mentioned_columns or _ordered_concept_mentioned(text, ordered_column),
                    "grouping_columns": bool(set(groups) & set(mentioned_columns)),
                },
                reasoning="Detected an ordered outcome-risk question with an ordinal/range segmentation field.",
                confidence=0.86,
                warnings=_range_warnings([ordered_column], schema),
            )

    if visualization_requested and artifact_type != "histogram" and target_column and len(target_values) >= 2:
        groups = _requested_or_candidate_groupings(
            question,
            df,
            mentioned_columns,
            exclude={target_column, ordered_column or ""},
            prefer_demographic=False,
        )
        if groups:
            return SemanticPlan(
                operation="CATEGORICAL_OUTCOME_BREAKDOWN",
                grouping_columns=groups,
                dimension=groups[0],
                category_field=groups[1] if len(groups) > 1 else None,
                target_column=target_column,
                target_values=target_values,
                aggregation="share",
                artifact_type=artifact_type if artifact_type != "none" else "heatmap",
                visualization_requested=True,
                artifact_required=True,
                constraints_locked={
                    "target_column": target_column in mentioned_columns,
                    "grouping_columns": bool(set(groups) & set(mentioned_columns)),
                    "chart_type": True,
                },
                reasoning="Detected a visualization of categorical outcome values by one or more grouping fields.",
                confidence=0.9,
            )

    analytical_markers = (
        "top ",
        "rank",
        "ranking",
        "highest",
        "lowest",
        "total",
        "sum",
        "average",
        "mean",
        "median",
        "group by",
        "grouping column",
        "metric",
        " by ",
    )
    has_locked_metric_or_group = bool(explicit_group or explicit_metric)
    has_grouped_metric_intent = bool(
        metric
        and dimension
        and has_locked_metric_or_group
        and (aggregation or limit or _has_any(text, analytical_markers))
    )
    if has_grouped_metric_intent:
        agg = aggregation or "sum"
        warnings_list = _range_warnings([metric], schema)
        limitations = []
        if metric in schema.ordered_range_columns and agg in {"sum", "mean", "median"}:
            limitations.append(f"`{metric}` is an ordered/range categorical field; execution will use approximate numeric midpoints.")
        return SemanticPlan(
            operation="GROUPED_AGGREGATION",
            metric=metric,
            dimension=dimension,
            metric_columns=[metric],
            grouping_columns=[dimension],
            aggregation=agg,
            sorting=sorting or "desc",
            limit=limit,
            artifact_type=artifact_type if visualization_requested else "bar",
            visualization_requested=visualization_requested,
            artifact_required=visualization_requested or bool(limit),
            constraints_locked={
                "grouping_columns": bool(explicit_group or dimension in mentioned_columns),
                "metric_columns": bool(explicit_metric or metric in mentioned_columns),
                "aggregation": bool(aggregation or forbidden),
                "sorting": bool(sorting),
                "limit": limit is not None,
                "chart_type": visualization_requested,
            },
            forbidden_aggregations=forbidden,
            needs_execution=True,
            reasoning="Detected explicit grouped aggregation constraints before heuristic metric selection.",
            confidence=0.92 if (explicit_group or explicit_metric) else 0.78,
            limitations=limitations,
            warnings=warnings_list,
        )

    return None


# ── Prompt construction ────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
You are an analytical planner. Your job is to decide WHAT analysis to perform \
on a dataset, NOT to perform the analysis yourself.

RULES:
1. Output ONLY valid JSON matching the schema below. No prose, no markdown.
2. Use ONLY columns that exist in the dataset schema provided.
3. Do NOT invent column names.
4. Year/date fields are TIME DIMENSIONS, never metrics to sum/average.
5. ID/code/key fields are IDENTIFIERS, never metrics.
5a. If the user explicitly names metric, grouping, aggregation, chart type,
top N, sorting, or filters, mark that field in constraints_locked and do not
substitute another field.
5b. If the user says not to use an aggregation, include it in
forbidden_aggregations and choose the requested alternative if present.
6. When the question asks about frequency, distribution, popularity, or \
dominance WITHOUT referencing a numeric metric, use operation=COUNT_DISTRIBUTION \
with aggregation=count.
7. When the question asks about strategy, insights, executive summary, or \
business implications, use operation=STRATEGIC_SYNTHESIS.
8. When the question asks how something changed over time or after a date, \
use operation=CATEGORY_MIX_SHIFT or TEMPORAL_TREND.
9. When the question asks about diversity or variety across groups, \
use operation=DIVERSITY_ANALYSIS.
10. When the question mentions duration, length, or time-span by category, \
use operation=DURATION_ANALYSIS.
11. If you are uncertain which columns to use, set confidence below 0.5 and \
explain in the reasoning field.
12. Match user concepts to actual column names using domain knowledge. \
For example, "genre" might map to a column called "category" or "classification".

OUTPUT SCHEMA:
{
  "operation": "COUNT_DISTRIBUTION | METRIC_AGGREGATION | GROUPED_AGGREGATION | \
CATEGORICAL_OUTCOME_BREAKDOWN | ORDERED_OUTCOME_RISK | TEMPORAL_TREND | \
CATEGORY_MIX_SHIFT | DIVERSITY_ANALYSIS | DURATION_ANALYSIS | OUTLIER_ANALYSIS | \
COMPARISON_ANALYSIS | STRATEGIC_SYNTHESIS | CORRELATION_ANALYSIS | \
DATA_QUALITY_CHECK | DATASET_OVERVIEW",
  "metric": "column_name or null",
  "dimension": "column_name or null",
  "metric_columns": ["column_name"],
  "grouping_columns": ["column_name"],
  "time_field": "column_name or null",
  "category_field": "column_name or null",
  "duration_field": "column_name or null",
  "target_column": "column_name or null",
  "target_values": ["value"],
  "ordered_column": "column_name or null",
  "filters": [],
  "aggregation": "count | sum | mean | median | min | max | share | \
unique_count | entropy | none",
  "sorting": "asc | desc | null",
  "limit": 10,
  "artifact_type": "bar | histogram | line | table | heatmap | none",
  "visualization_requested": false,
  "artifact_required": false,
  "constraints_locked": {"grouping_columns": false, "metric_columns": false, "aggregation": false},
  "forbidden_aggregations": [],
  "needs_execution": true,
  "reasoning": "brief explanation of your plan",
  "confidence": 0.0,
  "limitations": [],
  "warnings": []
}
"""


def _build_planner_prompt(
    question: str,
    schema: SchemaContext,
    context: dict[str, Any] | None = None,
) -> str:
    schema_text = _format_schema_for_prompt(schema)
    parts = [
        _PLANNER_SYSTEM,
        f"\n\nDATASET SCHEMA ({schema.row_count} rows):\n{schema_text}",
        f"\n\nUSER QUESTION: {question}",
    ]
    if context:
        ctx_str = json.dumps(
            {k: v for k, v in context.items() if k in ("active_metric", "active_dimension", "latest_findings")},
            ensure_ascii=False,
            default=str,
        )
        parts.append(f"\n\nINVESTIGATION CONTEXT: {ctx_str}")
    parts.append("\n\nOUTPUT (JSON only, no markdown):")
    return "".join(parts)


def _format_schema_for_prompt(schema: SchemaContext) -> str:
    lines: list[str] = []
    for col in schema.columns:
        hint = col.get("semantic_hint", "")
        hint_str = f" [{hint}]" if hint else ""
        stats = col.get("stats")
        stats_str = ""
        if stats:
            stats_str = f" (min={stats.get('min')}, max={stats.get('max')}, mean={stats.get('mean'):.1f})" if stats.get("mean") is not None else ""
        samples = ", ".join(f'"{v}"' for v in col.get("sample_values", [])[:4])
        lines.append(
            f"  - {col['name']} ({col['dtype']}, {col['nunique']} unique){hint_str}{stats_str}"
            f"  samples: [{samples}]"
        )
    sections = ["\n".join(lines)]
    if schema.year_like_columns:
        sections.append(f"Year-like columns (time dimensions, NOT metrics): {schema.year_like_columns}")
    if schema.identifier_columns:
        sections.append(f"Identifier columns (NOT metrics): {schema.identifier_columns}")
    if schema.duration_like_columns:
        sections.append(f"Duration columns (contain parseable durations): {schema.duration_like_columns}")
    if schema.multi_label_columns:
        sections.append(f"Multi-label columns (comma-separated values): {schema.multi_label_columns}")
    if schema.ordered_range_columns:
        sections.append(f"Ordered/range categorical columns (may require numeric midpoint parsing): {schema.ordered_range_columns}")
    return "\n".join(sections)


# ── Response parsing ────────────────────────────────────────────────────────

def _parse_plan(text: str, schema: SchemaContext) -> SemanticPlan | None:
    """Parse LLM response text into a SemanticPlan. Returns None on failure."""
    json_str = _extract_json(text)
    if not json_str:
        return None
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    operation = str(data.get("operation", "")).upper()
    if operation not in ALLOWED_OPERATIONS:
        return None

    aggregation = str(data.get("aggregation", "count")).lower()
    if aggregation not in ALLOWED_AGGREGATIONS:
        aggregation = "count"

    artifact_type = str(data.get("artifact_type", "none")).lower()
    if artifact_type not in ALLOWED_ARTIFACT_TYPES:
        artifact_type = "none"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    limit = _int_or_none(data.get("limit"))
    constraints = data.get("constraints_locked") if isinstance(data.get("constraints_locked"), dict) else {}
    forbidden = data.get("forbidden_aggregations") if isinstance(data.get("forbidden_aggregations"), list) else []

    return SemanticPlan(
        operation=operation,
        metric=_nullable_str(data.get("metric")),
        dimension=_nullable_str(data.get("dimension")),
        metric_columns=_string_list(data.get("metric_columns")),
        grouping_columns=_string_list(data.get("grouping_columns")),
        time_field=_nullable_str(data.get("time_field")),
        category_field=_nullable_str(data.get("category_field")),
        duration_field=_nullable_str(data.get("duration_field")),
        target_column=_nullable_str(data.get("target_column")),
        target_values=_string_list(data.get("target_values")),
        ordered_column=_nullable_str(data.get("ordered_column")),
        filters=data.get("filters") if isinstance(data.get("filters"), list) else [],
        aggregation=aggregation,
        sorting=_nullable_str(data.get("sorting")),
        limit=limit,
        artifact_type=artifact_type,
        visualization_requested=bool(data.get("visualization_requested", artifact_type != "none")),
        artifact_required=bool(data.get("artifact_required", False)),
        constraints_locked={str(k): bool(v) for k, v in constraints.items()},
        forbidden_aggregations=[str(item).lower() for item in forbidden],
        needs_execution=bool(data.get("needs_execution", True)),
        reasoning=str(data.get("reasoning", "")),
        confidence=confidence,
        limitations=data.get("limitations") if isinstance(data.get("limitations"), list) else [],
        warnings=data.get("warnings") if isinstance(data.get("warnings"), list) else [],
    )


def _extract_json(text: str) -> str | None:
    """Extract JSON object from LLM response (may contain markdown fences)."""
    # Try direct parse first
    stripped = text.strip()
    if stripped.startswith("{"):
        return stripped

    # Try markdown code fence
    match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*\n?```", stripped, re.DOTALL)
    if match:
        return match.group(1)

    # Try to find any JSON object
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", stripped, re.DOTALL)
    if match:
        return match.group(0)

    return None


# ── Utility helpers ─────────────────────────────────────────────────────────

def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold())).strip()


def _tokens(value: str) -> list[str]:
    return [token for token in _normalize_text(value).split() if token]


def _stem_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _stemmed_tokens(value: str) -> list[str]:
    return [_stem_token(token) for token in _tokens(value)]


def _column_mentioned(question: str, column: str) -> bool:
    question_norm = f" {_normalize_text(question)} "
    column_norm = _normalize_text(column)
    if column_norm and f" {column_norm} " in question_norm:
        return True
    col_tokens = _stemmed_tokens(column)
    q_tokens = set(_stemmed_tokens(question))
    return bool(col_tokens) and set(col_tokens).issubset(q_tokens)


def _mentioned_schema_columns(question: str, columns: list[str]) -> list[str]:
    return [column for column in columns if _column_mentioned(question, column)]


def _resolve_column_phrase(raw: str, columns: list[str]) -> str | None:
    phrase = str(raw or "").strip(" `.,:;")
    if not phrase:
        return None
    phrase_norm = _normalize_text(phrase)
    exact = next((column for column in columns if _normalize_text(column) == phrase_norm), None)
    if exact:
        return exact
    phrase_tokens = set(_stemmed_tokens(phrase))
    if not phrase_tokens:
        return None
    matches = [
        column
        for column in columns
        if set(_stemmed_tokens(column)).issubset(phrase_tokens) or phrase_tokens.issubset(set(_stemmed_tokens(column)))
    ]
    if matches:
        return sorted(matches, key=lambda item: abs(len(_tokens(item)) - len(_tokens(phrase))))[0]
    return None


def _explicit_grouping_column(question: str, columns: list[str]) -> str | None:
    patterns = (
        r"use\s+(.+?)\s+as\s+(?:the\s+)?(?:grouping\s+column|grouping|group|dimension)",
        r"group\s+by\s+(.+?)(?:[,.]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE | re.DOTALL)
        if match:
            resolved = _resolve_column_phrase(match.group(1), columns)
            if resolved:
                return resolved
    return None


def _explicit_metric_column(question: str, columns: list[str]) -> str | None:
    normalized = _normalize_text(question)
    for column in columns:
        column_norm = _normalize_text(column)
        if column_norm and re.search(rf"\b{re.escape(column_norm)}\b\s+as\s+(?:the\s+)?(?:numeric\s+metric|metric|measure|value)\b", normalized):
            return column
    patterns = (
        r"use\s+(.+?)\s+as\s+(?:the\s+)?(?:numeric\s+metric|metric|measure|value)",
        r"(?:metric|measure)\s+(?:column\s+)?(?:is|=)\s+(.+?)(?:[,.]|$)",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, question, flags=re.IGNORECASE | re.DOTALL))
        for match in reversed(matches):
            resolved = _resolve_column_phrase(match.group(1), columns)
            if resolved:
                return resolved
    return None


def _metric_from_mentioned_columns(
    normalized_question: str,
    mentioned_columns: list[str],
    df: pd.DataFrame,
    aggregation: str | None,
) -> str | None:
    if not mentioned_columns:
        return None
    if aggregation in {"sum", "mean", "median", "min", "max"} or _has_any(normalized_question, ("metric", "income", "revenue", "sales", "profit", "amount", "score", "duration", "cost", "price", "salary", "value")):
        numeric_or_range = [
            column for column in mentioned_columns
            if pd.api.types.is_numeric_dtype(df[column]) or _is_ordered_range_like([str(v) for v in df[column].dropna().head(20).tolist()])
        ]
        if numeric_or_range:
            # Prefer the mentioned column closest to an aggregation marker.
            return numeric_or_range[-1]
    numeric = [column for column in mentioned_columns if pd.api.types.is_numeric_dtype(df[column])]
    return numeric[-1] if numeric else None


def _dimension_from_mentioned_columns(
    normalized_question: str,
    mentioned_columns: list[str],
    df: pd.DataFrame,
    *,
    exclude: set[str] | None = None,
) -> str | None:
    exclude = exclude or set()
    candidates = [column for column in mentioned_columns if column not in exclude and not pd.api.types.is_numeric_dtype(df[column])]
    if candidates:
        return candidates[0]
    candidates = [column for column in mentioned_columns if column not in exclude]
    return candidates[0] if candidates else None


def _requested_aggregation(normalized_question: str) -> tuple[str | None, list[str]]:
    marker_map = {
        "sum": ("total", "sum", "summed", "overall", "cumulative"),
        "mean": ("average", "averages", "mean", "avg"),
        "median": ("median",),
        "count": ("count", "counts", "number of", "volume"),
    }
    matches: list[tuple[int, str]] = []
    forbidden: list[str] = []
    for aggregation, markers in marker_map.items():
        for marker in markers:
            start = 0
            while True:
                idx = normalized_question.find(marker, start)
                if idx < 0:
                    break
                if _marker_is_negated(normalized_question, idx):
                    if aggregation not in forbidden:
                        forbidden.append(aggregation)
                else:
                    matches.append((idx, aggregation))
                start = idx + len(marker)
    matches.sort(key=lambda item: item[0])
    selected = matches[-1][1] if matches else None
    if selected in forbidden:
        selected = None
    return selected, forbidden


def _marker_is_negated(normalized_question: str, idx: int) -> bool:
    prefix = normalized_question[max(0, idx - 32):idx]
    return any(marker in prefix for marker in ("do not use", "don t use", "dont use", "not use", "not ", "no "))


def _requested_limit(normalized_question: str) -> int | None:
    match = re.search(r"\btop\s+(\d{1,3})\b", normalized_question)
    if not match:
        return None
    try:
        return max(1, min(int(match.group(1)), 100))
    except ValueError:
        return None


def _requested_artifact_type(normalized_question: str) -> str:
    if "heatmap" in normalized_question:
        return "heatmap"
    if "line chart" in normalized_question or "line graph" in normalized_question:
        return "line"
    if "histogram" in normalized_question:
        return "histogram"
    if _has_any(normalized_question, ("visualization", "visualisation", "chart", "graph", "plot", "visual analysis")):
        return "bar"
    return "none"


def _target_from_value_mentions(question: str, df: pd.DataFrame) -> tuple[str | None, list[str]]:
    text = _normalize_text(question)
    best_column: str | None = None
    best_values: list[str] = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        values = []
        for value in df[column].dropna().unique().tolist()[:100]:
            value_text = str(value)
            value_norm = _normalize_text(value_text)
            if value_norm and f" {value_norm} " in f" {text} ":
                values.append(value_text)
        if len(values) > len(best_values):
            best_column = str(column)
            best_values = values
    return best_column, best_values


def _ordered_column_from_question(
    normalized_question: str,
    mentioned_columns: list[str],
    schema: SchemaContext,
) -> str | None:
    for column in schema.ordered_range_columns:
        if column in mentioned_columns or _ordered_concept_mentioned(normalized_question, column):
            return column
    return mentioned_columns[0] if mentioned_columns and mentioned_columns[0] in schema.ordered_range_columns else None


def _ordered_concept_mentioned(normalized_question: str, column: str) -> bool:
    column_tokens = set(_stemmed_tokens(column))
    q_tokens = set(_stemmed_tokens(normalized_question))
    return bool(column_tokens & q_tokens)


def _requested_or_candidate_groupings(
    question: str,
    df: pd.DataFrame,
    mentioned_columns: list[str],
    *,
    exclude: set[str],
    prefer_demographic: bool,
) -> list[str]:
    groups = [
        column for column in mentioned_columns
        if column not in exclude and not pd.api.types.is_numeric_dtype(df[column]) and not _looks_identifier_name(column)
    ]
    if groups:
        return groups[:3]
    if not prefer_demographic and len(mentioned_columns) > 1:
        return [
            column for column in mentioned_columns
            if column not in exclude and not pd.api.types.is_numeric_dtype(df[column])
        ][:3]
    candidates = []
    for column in df.columns:
        name = str(column)
        if name in exclude or pd.api.types.is_numeric_dtype(df[column]) or _looks_identifier_name(name):
            continue
        nunique = int(df[column].nunique(dropna=True))
        if 2 <= nunique <= min(50, max(3, len(df) // 2)):
            candidates.append(name)
    return candidates[:3]


def _looks_like_ordered_outcome_risk(normalized_question: str) -> bool:
    return _has_any(normalized_question, ("risk", "rate", "prevalence", "probability", "share")) and _has_any(
        normalized_question,
        ("increase", "decrease", "decreases", "as ", "strongest"),
    )


def _range_warnings(columns: list[str | None], schema: SchemaContext) -> list[str]:
    warnings_list: list[str] = []
    for column in columns:
        if column and column in schema.ordered_range_columns:
            warnings_list.append(f"{column} appears to be an ordered/range category and may require midpoint parsing.")
    return warnings_list


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _looks_identifier_name(name: str) -> bool:
    norm = _normalize_text(name)
    return any(marker in norm.split() or marker in norm for marker in ("id", "uuid", "guid", "key", "code", "index"))


def _is_ordered_range_like(sample_values: list[str]) -> bool:
    if len(sample_values) < 2:
        return False
    parsed = [parse_ordered_range_value(value) for value in sample_values]
    parse_rate = sum(value is not None for value in parsed) / max(len(sample_values), 1)
    range_markers = ("to", "below", "under", "less than", "more than", "above", "over", "-", "–", "—", "<", ">")
    marker_rate = sum(any(marker in str(value).casefold() for marker in range_markers) for value in sample_values) / max(len(sample_values), 1)
    return parse_rate >= 0.6 and marker_rate >= 0.3


def parse_ordered_range_value(value: Any) -> float | None:
    """Parse an ordinal/range category into an approximate numeric score.

    The parser is intentionally generic: it understands "below/under",
    "more than/above", explicit numeric ranges, zero/none categories, and
    low/medium/high labels. Callers must present this as an approximation.
    """

    text = str(value or "").casefold().strip()
    if not text:
        return None
    normalized = re.sub(r"[,₹$€£]", "", text)
    normalized = re.sub(r"\b(?:rs|usd|eur|gbp|inr|rub)\.?\b", "", normalized)
    if re.search(r"\b(no|none|zero|nil)\b", normalized):
        return 0.0
    ordinal_words = {
        "very low": 0.5,
        "very high": 4.0,
        "medium": 2.0,
        "moderate": 2.0,
        "low": 1.0,
        "high": 3.0,
    }
    for marker, score in ordinal_words.items():
        if re.search(rf"\b{re.escape(marker)}\b", normalized):
            return score
    numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", normalized)]
    if len(numbers) >= 2:
        return sum(numbers[:2]) / 2.0
    if len(numbers) == 1:
        number = numbers[0]
        if any(marker in normalized for marker in ("below", "under", "less than", "<")):
            return number / 2.0
        if any(marker in normalized for marker in ("more than", "above", "over", "greater than", ">")):
            return number * 1.25
        return number
    return None

def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return " ".join(str(item.get("text") if isinstance(item, dict) else item) for item in content)
    return str(content or "")


def _nullable_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.lower() != "null" else None


def _safe_num(value: Any) -> float | None:
    try:
        result = float(value)
        if result != result:  # NaN
            return None
        return result
    except (TypeError, ValueError):
        return None


def _is_identifier(col_name: str, nunique: int, row_count: int) -> bool:
    """Check if a column looks like an identifier (high uniqueness, _id suffix, etc.)."""
    norm = col_name.lower().replace(" ", "_")
    id_markers = ("_id", "id_", "uuid", "guid", "key", "code", "index")
    if any(marker in norm for marker in id_markers):
        return True
    if row_count > 10 and nunique > row_count * 0.8:
        return True
    return False


def _is_duration_like(sample_values: list[str]) -> bool:
    """Check if sample values look like durations."""
    duration_pattern = re.compile(r"\d+\s*(min|hour|hr|sec|season|ep|episode|day)", re.IGNORECASE)
    matches = sum(1 for v in sample_values if duration_pattern.search(v))
    return len(sample_values) >= 2 and matches >= len(sample_values) * 0.5


def _is_multi_label(sample_values: list[str]) -> bool:
    """Check if sample values look like comma-separated multi-label entries."""
    if len(sample_values) < 3:
        return False
    comma_count = sum(1 for v in sample_values if "," in v)
    return comma_count >= len(sample_values) * 0.3
