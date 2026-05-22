from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from source.product.fallbacks.semantic_resolution import (
    CUSTOMER_LIKE_MARKERS,
    LOCATION_LIKE_MARKERS,
    ORDER_DATE_MARKERS,
    PRODUCT_LIKE_MARKERS,
    SALES_LIKE_MARKERS,
    column_by_markers,
    resolve_dimension_column,
    resolve_categorical_value,
)


@dataclass(frozen=True)
class QueryFilter:
    column: str
    operator: str
    value: Any
    source: str = "explicit"

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafeAlias:
    phrase: str
    resolved_to: str
    alias_type: str

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


class ExtremumScope(str, Enum):
    ROW_LEVEL = "row_level"
    GROUP_AGGREGATE = "group_aggregate"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class AuthoritativeQueryPlan:
    raw_question: str
    intent: str
    metric: str | None = None
    metric_source: str | None = None
    metric_alias_used: str | None = None
    dimension: str | None = None
    dimension_source: str | None = None
    filters: list[QueryFilter] = field(default_factory=list)
    time_axis: str | None = None
    time_grain: str | None = None
    aggregation: str | None = None
    aggregation_intent: str | None = None
    extremum_scope: ExtremumScope | None = None
    target_entity: str | None = None
    aggregate_function: str | None = None
    ranking_direction: str | None = None
    chart_type: str | None = None
    transformation: str | None = None
    hypothesis: str | None = None
    quality_target: str | None = None
    scope: str = "new_task"
    requires_new_branch: bool = True
    can_use_active_branch: bool = False
    missing_required_fields: list[str] = field(default_factory=list)
    safe_aliases: list[SafeAlias] = field(default_factory=list)
    forbidden_substitutions: list[str] = field(default_factory=list)
    requested_dimension_type: str | None = None
    requires_user_confirmation: bool = False
    confirmation_reason: str | None = None
    available_dimension_alternatives: list[str] = field(default_factory=list)
    confidence: str = "medium"
    clarification_question: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["filters"] = [item.to_payload() for item in self.filters]
        payload["safe_aliases"] = [item.to_payload() for item in self.safe_aliases]
        payload["extremum_scope"] = self.extremum_scope.value if isinstance(self.extremum_scope, ExtremumScope) else self.extremum_scope
        return payload


class SafeBusinessAliasResolver:
    METRIC_ALIAS_GROUPS = {
        "sales_like": SALES_LIKE_MARKERS,
        "profit_like": ("profit", "margin", "прибыл"),
        "cost_like": ("cost", "expense", "затрат", "расход"),
        "price_like": ("price", "цена"),
        "compensation_like": ("salary", "compensation", "pay", "wage", "income", "earnings", "зарплат", "оклад"),
        "score_like": ("score", "rating", "оцен", "рейтинг", "балл"),
    }
    FORBIDDEN = {
        "place": {"unrequested geography column"},
        "category": {"unrequested segment column"},
        "customer": {"unrequested region column"},
        "heatmap": {"different chart family"},
        "bin": {"unrelated categorical column"},
        "shipping behavior": {"raw timestamp column"},
    }

    @classmethod
    def resolve_metric(cls, question: str, columns: list[str], df: pd.DataFrame | None = None) -> tuple[str | None, str | None, list[SafeAlias]]:
        normalized = _normalize(question)
        aliases: list[SafeAlias] = []
        metric_columns = _metric_candidate_columns(columns, df)
        for column in metric_columns:
            if _normalize(column) in normalized:
                return column, None, aliases
        for alias_group in cls.METRIC_ALIAS_GROUPS.values():
            phrase = next((marker for marker in alias_group if marker in normalized), None)
            if not phrase:
                continue
            target = column_by_markers(metric_columns, alias_group) or column_by_markers(columns, alias_group)
            if target:
                alias_phrase = _matched_question_phrase(normalized, alias_group) or phrase
                aliases.append(SafeAlias(phrase=alias_phrase, resolved_to=target, alias_type="metric"))
                return target, alias_phrase, aliases
            return None, phrase, aliases
        numeric = metric_columns or [col for col in columns if _is_numeric_like(col)]
        return (numeric[0] if numeric else None), None, aliases

    @classmethod
    def forbidden_for(cls, question: str) -> list[str]:
        normalized = _normalize(question)
        forbidden: list[str] = []
        for phrase, substitutions in cls.FORBIDDEN.items():
            if phrase in normalized:
                forbidden.extend(sorted(substitutions))
        return forbidden


class NonAnalyticalUtteranceClassifier:
    MARKERS = (
        "stateful heuristics",
        "heuristics fighting",
        "this is broken",
        "why did it do that",
        "branch manager",
        "agent is confused",
        "orchestration",
        "routing is weird",
        "system is confused",
    )

    @classmethod
    def is_non_analytical(cls, question: str) -> bool:
        normalized = _normalize(question)
        if any(marker in normalized for marker in cls.MARKERS):
            return True
        has_operation = any(
            marker in normalized
            for marker in (
                "show",
                "build",
                "calculate",
                "compare",
                "rank",
                "top",
                "lowest",
                "highest",
                "histogram",
                "heatmap",
                "trend",
                "growth",
                "sales",
                "revenue",
                "выруч",
                "продаж",
                "посч",
                "сравн",
                "постр",
            )
        )
        return not has_operation and any(marker in normalized for marker in ("broken", "weird", "confused", "fighting"))


class AuthoritativeExecutionPlanner:
    @classmethod
    def plan(
        cls,
        question: str,
        df: pd.DataFrame | None,
        *,
        active_branch: dict[str, Any] | None = None,
    ) -> AuthoritativeQueryPlan:
        columns = [str(col) for col in getattr(df, "columns", [])] if isinstance(df, pd.DataFrame) else []
        normalized = _normalize(question)
        if NonAnalyticalUtteranceClassifier.is_non_analytical(question):
            return AuthoritativeQueryPlan(
                raw_question=question,
                intent="non_analytical",
                scope="meta",
                requires_new_branch=False,
                can_use_active_branch=False,
                confidence="high",
            )
        metric, metric_alias, aliases = SafeBusinessAliasResolver.resolve_metric(question, columns, df if isinstance(df, pd.DataFrame) else None)
        dimension = _resolve_dimension(normalized, columns, metric, df=df if isinstance(df, pd.DataFrame) else None)
        requested_dimension = _requested_dimension_type(normalized)
        confirmation_required, confirmation_reason = _dimension_confirmation_state(
            requested_dimension,
            dimension,
            columns,
            metric,
            df if isinstance(df, pd.DataFrame) else None,
        )
        filters = _resolve_filters(question, df) if isinstance(df, pd.DataFrame) else []
        chart_type = ChartIntentPlanner.resolve(normalized)
        time_axis = _resolve_time_axis(normalized, columns)
        intent = _resolve_intent(normalized, chart_type)
        transformation = _resolve_transformation(normalized)
        aggregation, aggregation_intent = AggregationIntentResolver.resolve(normalized, intent, chart_type)
        extremum_scope = _resolve_extremum_scope(normalized, intent, dimension)
        aggregate_function = aggregation if extremum_scope == ExtremumScope.GROUP_AGGREGATE else None
        if intent == "general" and metric and aggregation in {"mean", "sum", "count"} and filters:
            intent = "constrained_aggregation"
        ranking_direction = "ascending" if _has_minimum_intent(normalized) else "descending"
        missing: list[str] = []
        if intent in {"rank_groups", "filtered_minmax", "constrained_aggregation"} and not metric:
            missing.append("metric")
        if intent == "rank_groups" and not dimension:
            missing.append("dimension")
        if intent in {"histogram", "seasonality_heatmap", "growth", "shipping_delay", "temporal_trend", "chart_request", "extremum"} and not metric:
            missing.append("metric")
        if intent in {"growth", "seasonality_heatmap", "shipping_delay", "temporal_trend", "chart_request"} and not time_axis:
            missing.append("time_axis")
        scope = "follow_up" if _is_followup(normalized) else "new_task"
        requires_new = scope == "new_task"
        return AuthoritativeQueryPlan(
            raw_question=question,
            intent=intent,
            metric=metric,
            metric_source="safe_alias" if metric_alias else ("explicit_or_inferred" if metric else None),
            metric_alias_used=metric_alias,
            dimension=dimension,
            dimension_source="explicit" if dimension and _normalize(dimension) in normalized else ("semantic" if dimension else None),
            filters=filters,
            time_axis=time_axis,
            time_grain=_resolve_time_grain(normalized, intent),
            aggregation=aggregation,
            aggregation_intent=aggregation_intent,
            extremum_scope=extremum_scope,
            target_entity=dimension,
            aggregate_function=aggregate_function,
            ranking_direction=ranking_direction,
            chart_type=chart_type,
            transformation=transformation,
            hypothesis=question if "hypothesis" in normalized or "гипотез" in normalized else None,
            quality_target="duplicates" if any(marker in normalized for marker in ("duplicate", "дублик")) else None,
            scope=scope,
            requires_new_branch=requires_new,
            can_use_active_branch=not requires_new,
            missing_required_fields=missing,
            safe_aliases=aliases,
            forbidden_substitutions=SafeBusinessAliasResolver.forbidden_for(question),
            requested_dimension_type=requested_dimension,
            requires_user_confirmation=confirmation_required,
            confirmation_reason=confirmation_reason,
            available_dimension_alternatives=_available_dimension_alternatives(columns, metric, df if isinstance(df, pd.DataFrame) else None),
            confidence="low" if missing or confirmation_required else "high",
            clarification_question=_clarification_for(missing),
        )


class ChartIntentPlanner:
    CHART_MARKERS = {
        "seasonality_heatmap": ("seasonality heatmap", "сезонность heatmap", "теплов"),
        "histogram": ("histogram", "гистограмм", "distribution", "распредел"),
        "heatmap": ("heatmap",),
        "scatter": ("scatter",),
        "boxplot": ("boxplot", "box plot"),
        "treemap": ("treemap",),
        "line": ("line chart", "динамик", "линия", "тренд"),
        "waterfall": ("waterfall",),
        "correlation_matrix": ("correlation matrix",),
        "anomaly_timeline": ("anomaly timeline",),
    }

    @classmethod
    def resolve(cls, normalized_question: str) -> str | None:
        for chart, markers in cls.CHART_MARKERS.items():
            if any(marker in normalized_question for marker in markers):
                return chart
        if any(marker in normalized_question for marker in ("chart", "graph", "plot", "visualize", "график", "построй график", "нарисуй", "визуализируй", "диаграмма")):
            return "line" if _has_temporal_intent(normalized_question) else None
        return None


class FilterConstraintValidator:
    @staticmethod
    def validate(plan: AuthoritativeQueryPlan, df: pd.DataFrame | None = None) -> list[str]:
        errors: list[str] = []
        if plan.filters and not isinstance(df, pd.DataFrame):
            errors.append("filters require a dataframe")
        if isinstance(df, pd.DataFrame):
            for item in plan.filters:
                if item.column not in df.columns:
                    errors.append(f"filter column missing: {item.column}")
                elif item.operator == "equals":
                    values = df[item.column].astype(str).str.casefold()
                    if str(item.value).casefold() not in set(values.dropna()):
                        errors.append(f"filter value not found: {item.column}={item.value}")
        return errors


class TemporalSanityValidator:
    @staticmethod
    def delivery_delay_issues(df: pd.DataFrame, order_col: str, ship_col: str) -> list[str]:
        if order_col not in df.columns or ship_col not in df.columns:
            return ["order or ship date column missing"]
        order = pd.to_datetime(df[order_col], errors="coerce", dayfirst=False)
        ship = pd.to_datetime(df[ship_col], errors="coerce", dayfirst=False)
        valid = pd.DataFrame({"order": order, "ship": ship}).dropna()
        if valid.empty:
            return ["order and ship dates could not be parsed"]
        delay = (valid["ship"] - valid["order"]).dt.days
        issues: list[str] = []
        if (delay < 0).any():
            issues.append("negative delivery delays found")
        if (delay > 60).mean() > 0.05:
            issues.append("more than 5% of parsed delivery delays exceed 60 days")
        if delay.max() > 365:
            issues.append("delivery delay above one year found")
        return issues


def _resolve_intent(normalized: str, chart_type: str | None) -> str:
    if any(marker in normalized for marker in ("late shipment", "late shipments", "delayed shipment", "delayed shipments", "late event", "late events", "delayed event", "delayed events", "shipping delay", "shipping delays", "delivery delay", "delivery delays", "задержк", "поздн")):
        return "shipping_delay"
    if any(marker in normalized for marker in ("growth", "trend", "рост", "динамик")) and any(marker in normalized for marker in ("shipping", "delay", "ship", "delivery")):
        return "shipping_delay"
    if chart_type and _has_temporal_intent(normalized):
        return "chart_request"
    if _has_temporal_intent(normalized):
        return "temporal_trend"
    if _has_extremum_intent(normalized):
        return "extremum"
    if any(marker in normalized for marker in ("customer", "customers", "client", "clients", "клиент")) and any(
        marker in normalized for marker in ("revenue", "выруч", "sales", "продаж", "top", "lowest", "highest")
    ):
        return "extremum" if _has_extremum_intent(normalized) else "rank_groups"
    if chart_type == "histogram":
        return "histogram"
    if chart_type in {"seasonality_heatmap", "heatmap"}:
        return "seasonality_heatmap"
    if "bin" in normalized or "bins" in normalized:
        return "binning" if any(marker in normalized for marker in ("create", "automatic", "automatically", "созд")) else "bin_question"
    if any(marker in normalized for marker in ("lowest", "highest", "top", "rank", "каждом", "each", "by city", "by ", "по город", "по клиент", "по категор")):
        return "rank_groups"
    if any(marker in normalized for marker in ("growth", "trend", "рост", "where is strongest growth")):
        return "growth"
    return "meta" if "business question" in normalized or "what fields" in normalized else "general"


def _resolve_dimension(normalized: str, columns: list[str], metric: str | None = None, df: pd.DataFrame | None = None) -> str | None:
    return resolve_dimension_column(normalized, columns, metric=metric, df=df)


def _requested_dimension_type(normalized: str) -> str | None:
    top_match = re.search(r"\b(?:top|lowest|highest|leading|biggest|largest|rank)\s+(.+?)\s+by\s+.+$", normalized)
    if top_match:
        return _clean_requested_dimension(top_match.group(1))
    by_match = re.search(r"\b(?:by|across|per|по)\s+(.+)$", normalized)
    if by_match:
        return _clean_requested_dimension(by_match.group(1))
    return None


def _clean_requested_dimension(value: str) -> str | None:
    words = [
        word
        for word in re.sub(r"[^\w\s]", " ", value).split()
        if word
        and word
        not in {
            "the",
            "a",
            "an",
            "average",
            "avg",
            "mean",
            "total",
            "sum",
            "count",
            "of",
            "for",
            "from",
            "with",
            "using",
        }
    ]
    return " ".join(words) or None


def _dimension_confirmation_state(
    requested: str | None,
    resolved: str | None,
    columns: list[str],
    metric: str | None,
    df: pd.DataFrame | None,
) -> tuple[bool, str | None]:
    if not requested:
        return False, None
    if _requested_dimension_has_schema_match(requested, columns):
        return False, None
    if not resolved:
        return True, "requested_dimension_missing"
    if resolve_dimension_column(requested, columns, metric=metric, df=df) == resolved:
        return False, None
    if _dimension_terms_compatible(requested, resolved):
        return False, None
    return True, "resolved_dimension_semantic_mismatch"


def _requested_dimension_has_schema_match(requested: str, columns: list[str]) -> bool:
    requested_terms = _dimension_terms(requested)
    if not requested_terms:
        return False
    for column in columns:
        column_terms = _dimension_terms(column)
        if requested_terms & column_terms:
            return True
    return False


def _dimension_terms_compatible(requested: str, resolved: str) -> bool:
    requested_terms = _dimension_terms(requested)
    resolved_terms = _dimension_terms(resolved)
    return bool(requested_terms and resolved_terms and requested_terms & resolved_terms)


def _dimension_terms(value: str) -> set[str]:
    terms: set[str] = set()
    for raw in _normalize(value).split():
        if len(raw) < 3:
            continue
        terms.add(raw)
        if raw.endswith("ies") and len(raw) > 4:
            terms.add(raw[:-3] + "y")
        if raw.endswith("s") and len(raw) > 3:
            terms.add(raw[:-1])
    return terms


def _available_dimension_alternatives(columns: list[str], metric: str | None, df: pd.DataFrame | None) -> list[str]:
    alternatives: list[str] = []
    for column in columns:
        if column == metric:
            continue
        if isinstance(df, pd.DataFrame) and column in df.columns:
            series = df[column]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
                continue
            if int(series.nunique(dropna=True)) <= 0:
                continue
        alternatives.append(column)
    return alternatives[:8]


def _metric_candidate_columns(columns: list[str], df: pd.DataFrame | None) -> list[str]:
    if not isinstance(df, pd.DataFrame):
        return [column for column in columns if _is_numeric_like(column)]
    candidates: list[str] = []
    for column in columns:
        if column not in df.columns:
            continue
        series = df[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            candidates.append(column)
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            candidates.append(column)
    return candidates


def _resolve_filters(question: str, df: pd.DataFrame) -> list[QueryFilter]:
    normalized = _normalize(question)
    filters: list[QueryFilter] = []
    location_columns = [column for column in [str(col) for col in df.columns] if column_by_markers([column], LOCATION_LIKE_MARKERS)]
    ordered_columns = location_columns + [str(col) for col in df.columns if str(col) not in set(location_columns)]
    resolved = resolve_categorical_value(question, df, preferred_columns=ordered_columns)
    if resolved:
        return [QueryFilter(column=resolved.column, operator="equals", value=resolved.value, source=resolved.source)]
    for column in ordered_columns:
        if not pd.api.types.is_object_dtype(df[column]) and not pd.api.types.is_string_dtype(df[column]):
            continue
        values = df[column].dropna().astype(str).unique()
        for value in values:
            value_norm = _normalize(value)
            if len(value_norm) >= 3 and value_norm in normalized:
                filters.append(QueryFilter(column=str(column), operator="equals", value=str(value)))
                break
    return filters


def _resolve_time_axis(normalized: str, columns: list[str]) -> str | None:
    for column in columns:
        if _normalize(column) in normalized and _is_time_like_name(column):
            return column
    if any(marker in normalized for marker in ("order", "заказ", "event", "transaction")):
        match = column_by_markers(columns, ORDER_DATE_MARKERS)
        if match:
            return match
    match = column_by_markers(columns, ("date", "time", "timestamp", "дата", "время"))
    if match:
        return match
    return None


def _resolve_transformation(normalized: str) -> str | None:
    if any(marker in normalized for marker in ("remove extreme", "remove outlier", "filter outlier")):
        return "remove_extreme_records"
    if "median" in normalized:
        return "median"
    return None


class AggregationIntentResolver:
    AVERAGE_MARKERS = ("average", "mean", "avg", "typical", "median", "средн", "медиан")
    REVENUE_MARKERS = (
        "revenue",
        "sales",
        "выручк",
        "доход",
        "продаж",
        "оборот",
        "принос",
    )
    TOTAL_MARKERS = ("total", "summed", "sum ", "общая выручка", "суммар", "сумма", "всего", "по сумме", "общий sales", "aggregate")
    RANKING_MARKERS = ("top", "leading", "strongest", "biggest", "largest", "топ", "лидер", "сильн", "больше всего")

    @classmethod
    def resolve(cls, normalized: str, intent: str, chart_type: str | None) -> tuple[str | None, str | None]:
        if any(marker in normalized for marker in ("average", "mean", "avg", "typical", "средн")):
            return "mean", "explicit_average_requested"
        if any(marker in normalized for marker in ("median", "медиан")):
            return "median", "explicit_average_requested"
        if "count" in normalized or "volume" in normalized:
            return "count", "explicit_count_requested"
        if any(marker in normalized for marker in cls.TOTAL_MARKERS):
            return "sum", "explicit_total_requested"
        if intent == "rank_groups" and any(marker in normalized for marker in cls.REVENUE_MARKERS):
            return "sum", "business_revenue_default"
        if intent == "shipping_delay":
            return "sum", "business_revenue_default"
        if intent == "rank_groups" and any(marker in normalized for marker in cls.RANKING_MARKERS):
            return "sum", "business_revenue_default"
        if intent == "rank_groups":
            return "mean", "default_group_average"
        if chart_type == "histogram":
            return "distribution", "distribution"
        return None, None


def _resolve_aggregation(normalized: str, intent: str, chart_type: str | None) -> tuple[str | None, str | None]:
    return AggregationIntentResolver.resolve(normalized, intent, chart_type)


def _resolve_time_grain(normalized: str, intent: str) -> str | None:
    if intent not in {"growth", "seasonality_heatmap", "shipping_delay", "temporal_trend", "chart_request"}:
        return None
    if any(marker in normalized for marker in ("по год", "yearly", "annual", "by year")):
        return "year"
    if any(marker in normalized for marker in ("по дня", "daily", "by day", "по дат", "by date", "over dates")):
        return "day"
    return "month"


def _resolve_extremum_scope(normalized: str, intent: str, dimension: str | None) -> ExtremumScope | None:
    if intent != "extremum":
        return None
    aggregate_markers = (
        "total sales",
        "total revenue",
        "summed sales",
        "average sales",
        "mean sales",
        "median sales",
        "by total",
        "by average",
        "aggregate sales",
        "revenue by customer",
        "суммар",
        "всего",
        "общая выручка",
        "сумма",
        "по сумме",
        "в сумме",
        "общий sales",
        "средн",
        "average",
        "mean",
        "медиан",
        "median",
        "aggregate",
    )
    if dimension and any(marker in normalized for marker in aggregate_markers):
        return ExtremumScope.GROUP_AGGREGATE
    if dimension:
        return ExtremumScope.ROW_LEVEL
    return ExtremumScope.AMBIGUOUS


def _has_temporal_intent(normalized: str) -> bool:
    return any(
        marker in normalized
        for marker in (
            "по дат",
            "по времени",
            "динамик",
            "тренд",
            "во времени",
            "по месяц",
            "по дня",
            "по год",
            "over time",
            "trend",
            "by date",
            "eventdate",
            "transactiondate",
            "timestamp",
            "over dates",
            "monthly",
            "daily",
            "yearly",
            "time series",
        )
    )


def _has_extremum_intent(normalized: str) -> bool:
    return _has_minimum_intent(normalized) or any(
        marker in normalized
        for marker in (
            "highest",
            "maximum",
            "max ",
            "largest",
            "наибольш",
            "максим",
            "самый высокий",
        )
    )


def _has_minimum_intent(normalized: str) -> bool:
    return any(
        marker in normalized
        for marker in (
            "lowest",
            "minimum",
            "min ",
            "smallest",
            "наименьш",
            "миним",
            "самый низк",
        )
    )


def _clarification_for(missing: list[str]) -> str | None:
    if not missing:
        return None
    return "Missing required fields: " + ", ".join(missing)


def _is_followup(normalized: str) -> bool:
    return any(marker in normalized for marker in ("what supports", "what contradict", "how validate", "what changed", "remain leaders"))


def _is_numeric_like(column: str) -> bool:
    low = _normalize(column)
    return any(marker in low for marker in ("sales", "sale", "revenue", "profit", "amount", "value", "cost", "price", "metric", "salary", "compensation", "pay", "wage", "income", "earnings", "выруч", "продаж", "доход", "прибыл", "цена", "зарплат", "оклад", "score", "rating"))


def _is_time_like_name(column: str) -> bool:
    low = _normalize(column)
    return any(marker in low for marker in ("date", "time", "timestamp", "дата", "время"))


def _best_column_for_markers(columns: list[str], markers: tuple[str, ...]) -> str | None:
    return column_by_markers(columns, markers)


def _matched_question_phrase(normalized_question: str, markers: tuple[str, ...]) -> str | None:
    words = normalized_question.split()
    for marker in sorted(markers, key=len, reverse=True):
        for word in words:
            if marker in word:
                return word
    return None


def _normalize(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").casefold().split())
