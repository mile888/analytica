from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
import warnings

import pandas as pd

from source.product.data_sources import (
    ColumnInferredRole,
    ColumnSemanticRole,
    DataSourceProfile,
    DataSourceSemanticNotes,
    DataSourceUsageContext,
)


class SemanticRole(StrEnum):
    METRIC = "metric"
    DIMENSION = "dimension"
    TIMESTAMP = "timestamp"
    IDENTIFIER = "identifier"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SemanticColumnProfile:
    name: str
    dtype: str = ""
    role: SemanticRole = SemanticRole.UNKNOWN
    role_confidence: float = 0.0
    semantic_tags: list[str] = field(default_factory=list)
    examples: list[Any] = field(default_factory=list)
    cardinality: int | None = None
    nullable: bool = False
    missing_count: int = 0
    sparsity: float = 0.0
    distribution_hints: list[str] = field(default_factory=list)
    business_hints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SemanticDatasetProfile:
    row_count: int = 0
    column_count: int = 0
    columns: list[SemanticColumnProfile] = field(default_factory=list)

    @property
    def by_name(self) -> dict[str, SemanticColumnProfile]:
        return {column.name: column for column in self.columns}

    @property
    def metrics(self) -> list[SemanticColumnProfile]:
        return [column for column in self.columns if column.role == SemanticRole.METRIC]

    @property
    def dimensions(self) -> list[SemanticColumnProfile]:
        return [column for column in self.columns if column.role == SemanticRole.DIMENSION]

    @property
    def timestamps(self) -> list[SemanticColumnProfile]:
        return [column for column in self.columns if column.role == SemanticRole.TIMESTAMP]

    @property
    def identifiers(self) -> list[SemanticColumnProfile]:
        return [column for column in self.columns if column.role == SemanticRole.IDENTIFIER]

    @property
    def texts(self) -> list[SemanticColumnProfile]:
        return [column for column in self.columns if column.role == SemanticRole.TEXT]


@dataclass(frozen=True)
class AnalyticalEntity:
    kind: str
    name: str
    column: str | None = None
    confidence: float = 0.0
    source: str = "inferred"


@dataclass(frozen=True)
class CategoryValueMatch:
    matched_column: str
    matched_value: str
    confidence: float = 0.0
    semantic_role: str = "dimension_value"


@dataclass(frozen=True)
class AnalyticalRelationship:
    relationship_type: str
    metric: str | None = None
    dimension: str | None = None
    time_axis: str | None = None
    confidence: float = 0.0
    rationale: str = ""


@dataclass(frozen=True)
class HypothesisBranch:
    hypothesis: str
    supporting_evidence: list[str] = field(default_factory=list)
    confidence: str = "medium"
    uncertainty: str = ""
    possible_drivers: list[str] = field(default_factory=list)
    conflicting_signals: list[str] = field(default_factory=list)
    suggested_validation: str = ""
    next_questions: list[str] = field(default_factory=list)
    related_findings: list[str] = field(default_factory=list)
    related_charts: list[str] = field(default_factory=list)
    branch_type: str = "driver_analysis"


@dataclass(frozen=True)
class InvestigationFocus:
    active_metric: str | None = None
    active_dimension: str | None = None
    active_time_axis: str | None = None
    active_chart: str | None = None
    active_analysis_type: str | None = None
    active_entities: list[AnalyticalEntity] = field(default_factory=list)
    active_question: str | None = None
    active_hypothesis: str | None = None
    recent_insights: list[str] = field(default_factory=list)
    recent_outputs: list[str] = field(default_factory=list)

    def as_selection_text(self) -> str:
        values = [
            self.active_question,
            self.active_metric,
            self.active_dimension,
            self.active_time_axis,
            self.active_chart,
            self.active_analysis_type,
            self.active_hypothesis,
            *(entity.name for entity in self.active_entities),
            *self.recent_insights,
            *self.recent_outputs,
        ]
        return " ".join(str(value).strip() for value in values if str(value or "").strip())


@dataclass(frozen=True)
class InvestigationThreadState:
    active_metric: str | None = None
    active_dimension: str | None = None
    active_time_axis: str | None = None
    active_segments: list[str] = field(default_factory=list)
    active_chart_type: str | None = None
    active_business_question: str | None = None
    active_hypothesis: str | None = None
    recent_findings: list[str] = field(default_factory=list)
    recent_charts: list[dict[str, Any]] = field(default_factory=list)
    recent_entities: list[AnalyticalEntity] = field(default_factory=list)
    recent_relationships: list[AnalyticalRelationship] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    suggested_next_questions: list[str] = field(default_factory=list)

    def as_selection_text(self) -> str:
        chart_text = []
        for chart in self.recent_charts:
            if isinstance(chart, dict):
                chart_text.extend(str(chart.get(key) or "") for key in ("title", "chart_type", "metric", "dimension", "time_axis"))
        relationship_text = [
            " ".join(
                str(value or "")
                for value in (item.relationship_type, item.metric, item.dimension, item.time_axis, item.rationale)
                if str(value or "").strip()
            )
            for item in self.recent_relationships
        ]
        values = [
            self.active_business_question,
            self.active_metric,
            self.active_dimension,
            self.active_time_axis,
            self.active_chart_type,
            self.active_hypothesis,
            *self.active_segments,
            *(entity.name for entity in self.recent_entities),
            *self.recent_findings,
            *chart_text,
            *relationship_text,
            *self.unresolved_questions,
        ]
        return " ".join(str(value).strip() for value in values if str(value or "").strip())


def build_semantic_dataset_profile(
    *,
    df: pd.DataFrame | None = None,
    profile: DataSourceProfile | None = None,
    usage_context: DataSourceUsageContext | None = None,
    semantic_notes: DataSourceSemanticNotes | None = None,
) -> SemanticDatasetProfile:
    """Create one normalized semantic view for charting, suggestions, and analysis."""

    row_count = int(len(df)) if isinstance(df, pd.DataFrame) else int(
        (usage_context.schema_summary.get("row_count") if usage_context else None)
        or (profile.row_count if profile else 0)
        or 0
    )
    notes_by_column = {
        note.column_name: note
        for note in (semantic_notes.column_notes if semantic_notes else [])
    }
    columns: list[SemanticColumnProfile] = []
    if isinstance(df, pd.DataFrame):
        for name in df.columns:
            series = df[name]
            missing_count = int(series.isna().sum())
            sample_values = [value for value in series.dropna().head(5).tolist()]
            unique_count = int(series.nunique(dropna=True))
            note = notes_by_column.get(str(name))
            explicit_role = _semantic_role_from_note(note.semantic_role) if note else None
            role, confidence = _infer_role(
                str(name),
                str(series.dtype),
                row_count=row_count,
                unique_count=unique_count,
                sample_values=sample_values,
                explicit_role=explicit_role,
            )
            columns.append(
                SemanticColumnProfile(
                    name=str(name),
                    dtype=str(series.dtype),
                    role=role,
                    role_confidence=confidence,
                    semantic_tags=_semantic_tags(str(name), str(series.dtype), sample_values),
                    examples=sample_values,
                    cardinality=unique_count,
                    nullable=bool(series.isna().any()),
                    missing_count=missing_count,
                    sparsity=(missing_count / row_count) if row_count else 0.0,
                    distribution_hints=_distribution_hints(str(name), str(series.dtype), unique_count, row_count),
                    business_hints=_business_hints(str(name), role),
                )
            )
    elif usage_context:
        missing = usage_context.missing_summary or {}
        for column in usage_context.column_summaries:
            note = notes_by_column.get(column.name)
            explicit_role = _semantic_role_from_note(note.semantic_role) if note else _semantic_role_from_note(column.inferred_role)
            role, confidence = _infer_role(
                column.name,
                column.dtype,
                row_count=row_count,
                unique_count=column.unique_count,
                sample_values=column.sample_values,
                explicit_role=explicit_role,
            )
            missing_count = int(missing.get(column.name) or 0)
            columns.append(
                SemanticColumnProfile(
                    name=column.name,
                    dtype=column.dtype,
                    role=role,
                    role_confidence=confidence,
                    semantic_tags=_semantic_tags(column.name, column.dtype, column.sample_values),
                    examples=list(column.sample_values or [])[:5],
                    cardinality=column.unique_count,
                    nullable=column.nullable,
                    missing_count=missing_count,
                    sparsity=(missing_count / row_count) if row_count else 0.0,
                    distribution_hints=_distribution_hints(column.name, column.dtype, column.unique_count, row_count),
                    business_hints=_business_hints(column.name, role),
                )
            )
    elif profile:
        missing = profile.missing_summary or {}
        for column in profile.columns:
            note = notes_by_column.get(column.name)
            explicit_role = _semantic_role_from_note(note.semantic_role) if note else None
            role, confidence = _infer_role(
                column.name,
                column.dtype,
                row_count=row_count,
                unique_count=column.unique_count,
                sample_values=column.sample_values,
                explicit_role=explicit_role,
            )
            missing_count = int(missing.get(column.name) or 0)
            columns.append(
                SemanticColumnProfile(
                    name=column.name,
                    dtype=column.dtype,
                    role=role,
                    role_confidence=confidence,
                    semantic_tags=_semantic_tags(column.name, column.dtype, column.sample_values),
                    examples=list(column.sample_values or [])[:5],
                    cardinality=column.unique_count,
                    nullable=column.nullable,
                    missing_count=missing_count,
                    sparsity=(missing_count / row_count) if row_count else 0.0,
                    distribution_hints=_distribution_hints(column.name, column.dtype, column.unique_count, row_count),
                    business_hints=_business_hints(column.name, role),
                )
            )
    return SemanticDatasetProfile(row_count=row_count, column_count=len(columns), columns=columns)


def build_investigation_focus(
    conversation_context: dict[str, Any] | None,
    *,
    question: str | None = None,
) -> InvestigationFocus:
    thread_state = build_investigation_thread_state(conversation_context, question=question)
    analysis_type = _analysis_type_from_chart(thread_state.active_chart_type)
    return InvestigationFocus(
        active_metric=thread_state.active_metric,
        active_dimension=thread_state.active_dimension,
        active_time_axis=thread_state.active_time_axis,
        active_chart=_clean_optional(thread_state.recent_charts[0].get("title")) if thread_state.recent_charts else None,
        active_analysis_type=analysis_type,
        active_entities=thread_state.recent_entities,
        active_question=question,
        active_hypothesis=thread_state.active_hypothesis,
        recent_insights=thread_state.recent_findings,
        recent_outputs=[
            str(chart.get("title") or "")
            for chart in thread_state.recent_charts
            if str(chart.get("title") or "").strip()
        ][:6],
    )


def build_investigation_thread_state(
    conversation_context: dict[str, Any] | None,
    *,
    question: str | None = None,
) -> InvestigationThreadState:
    context = conversation_context if isinstance(conversation_context, dict) else {}
    chart_context = context.get("latest_chart_context") if isinstance(context.get("latest_chart_context"), dict) else {}
    focus_context = context.get("focus") if isinstance(context.get("focus"), dict) else {}
    stored_thread = context.get("thread_state") if isinstance(context.get("thread_state"), dict) else {}
    recent_findings = [
        str(item)
        for item in (context.get("latest_findings") or [])
        if str(item).strip()
    ][:6]
    if not recent_findings and isinstance(stored_thread.get("recent_findings"), list):
        recent_findings = [str(item) for item in stored_thread.get("recent_findings", []) if str(item).strip()][:6]
    artifact_titles = [
        str(item)
        for item in (context.get("artifact_titles") or [])
        if str(item).strip()
    ][:8]
    memory_items = context.get("memory") if isinstance(context.get("memory"), list) else []
    unresolved_questions = [
        str(item.get("content") or "")
        for item in memory_items
        if isinstance(item, dict)
        and str(item.get("type") or "").lower() in {"open_question", "question"}
        and str(item.get("status") or "").lower() != "archived"
        and str(item.get("content") or "").strip()
    ][:5]

    active_metric = (
        _clean_optional(chart_context.get("metric"))
        or _clean_optional(focus_context.get("active_metric"))
        or _clean_optional(stored_thread.get("active_metric"))
    )
    active_dimension = (
        _clean_optional(chart_context.get("dimension"))
        or _clean_optional(focus_context.get("active_dimension"))
        or _clean_optional(stored_thread.get("active_dimension"))
    )
    active_time_axis = (
        _clean_optional(chart_context.get("time_axis"))
        or _clean_optional(focus_context.get("active_time_axis"))
        or _clean_optional(stored_thread.get("active_time_axis"))
    )
    chart_type = (
        _clean_optional(chart_context.get("chart_type"))
        or _clean_optional(focus_context.get("active_chart_type"))
        or _clean_optional(stored_thread.get("active_chart_type"))
    )
    active_question = _clean_optional(question) or _clean_optional(context.get("initial_question")) or _clean_optional(stored_thread.get("active_business_question"))
    active_hypothesis = _infer_active_hypothesis(active_metric, active_dimension, recent_findings, active_question) or _clean_optional(stored_thread.get("active_hypothesis"))

    recent_charts = []
    if chart_context:
        recent_charts.append(
            {
                "title": _clean_optional(chart_context.get("title")) or "",
                "chart_type": chart_type or "",
                "metric": active_metric or "",
                "dimension": active_dimension or "",
                "time_axis": active_time_axis or "",
            }
        )
    for title in artifact_titles:
        if title and not any(title == chart.get("title") for chart in recent_charts):
            recent_charts.append({"title": title, "chart_type": "", "metric": "", "dimension": "", "time_axis": ""})

    entities: list[AnalyticalEntity] = []
    for kind, value in (("metric", active_metric), ("dimension", active_dimension), ("time", active_time_axis)):
        if value:
            entities.append(AnalyticalEntity(kind=kind, name=value, column=value, confidence=0.9, source="thread_state"))

    relationships: list[AnalyticalRelationship] = []
    if active_metric and active_dimension:
        relationships.append(
            AnalyticalRelationship(
                relationship_type="grouped_comparison",
                metric=active_metric,
                dimension=active_dimension,
                time_axis=active_time_axis,
                confidence=0.86,
                rationale=f"Current analytical thread compares {active_metric} across {active_dimension}.",
            )
        )

    suggested = _thread_suggestions(active_metric, active_dimension, active_time_axis, recent_findings)
    return InvestigationThreadState(
        active_metric=active_metric,
        active_dimension=active_dimension,
        active_time_axis=active_time_axis,
        active_segments=[],
        active_chart_type=chart_type,
        active_business_question=active_question,
        active_hypothesis=active_hypothesis,
        recent_findings=recent_findings,
        recent_charts=recent_charts[:6],
        recent_entities=entities,
        recent_relationships=relationships,
        unresolved_questions=unresolved_questions,
        suggested_next_questions=suggested,
    )


def select_metric_column(
    semantic_profile: SemanticDatasetProfile,
    text: str,
    *,
    explicit_text: str | None = None,
    focus: InvestigationFocus | None = None,
) -> str | None:
    return _select_column(
        semantic_profile.metrics,
        text,
        explicit_text=explicit_text,
        focus_value=focus.active_metric if focus else None,
        semantic_groups=_metric_semantic_groups(),
    )


def select_dimension_column(
    semantic_profile: SemanticDatasetProfile,
    text: str,
    *,
    explicit_text: str | None = None,
    focus: InvestigationFocus | None = None,
    exclude: set[str] | None = None,
) -> str | None:
    candidates = [column for column in semantic_profile.dimensions if column.name not in (exclude or set())]
    return _select_column(
        candidates,
        text,
        explicit_text=explicit_text,
        focus_value=focus.active_dimension if focus else None,
        semantic_groups=_dimension_semantic_groups(),
        prefer_low_cardinality=True,
    )


def select_timestamp_column(
    semantic_profile: SemanticDatasetProfile,
    text: str,
    *,
    explicit_text: str | None = None,
    focus: InvestigationFocus | None = None,
) -> str | None:
    return _select_column(
        semantic_profile.timestamps,
        text,
        explicit_text=explicit_text,
        focus_value=focus.active_time_axis if focus else None,
        semantic_groups=_time_semantic_groups(),
    )


def match_entity_or_value(
    user_text: str,
    semantic_profile: SemanticDatasetProfile,
    *,
    df: pd.DataFrame | None = None,
) -> CategoryValueMatch | None:
    text = _normalize(user_text)
    if not text:
        return None
    best: CategoryValueMatch | None = None
    for column in semantic_profile.columns:
        if column.role not in {SemanticRole.DIMENSION, SemanticRole.TEXT}:
            continue
        if column.cardinality is not None and semantic_profile.row_count and column.cardinality > max(200, int(semantic_profile.row_count * 0.75)):
            continue
        values = []
        if isinstance(df, pd.DataFrame) and column.name in df.columns:
            series = df[column.name].dropna()
            if int(series.nunique(dropna=True)) <= max(200, int(len(df) * 0.75)):
                values = [str(value) for value in series.astype(str).unique().tolist()[:500]]
        if not values:
            values = [str(value) for value in column.examples if str(value).strip()]
        for value in values:
            normalized_value = _normalize(value)
            if not normalized_value or len(normalized_value) < 2:
                continue
            confidence = 0.0
            if normalized_value in text:
                confidence = 0.95
            else:
                value_tokens = set(normalized_value.split())
                text_tokens = set(text.split())
                if len(value_tokens) >= 2 and value_tokens <= text_tokens:
                    confidence = 0.9
            if confidence and (best is None or confidence > best.confidence):
                best = CategoryValueMatch(
                    matched_column=column.name,
                    matched_value=value,
                    confidence=confidence,
                    semantic_role=column.role.value,
                )
    return best


def resolve_entity_or_value(
    user_text: str,
    semantic_profile: SemanticDatasetProfile,
    dataframe: pd.DataFrame | None = None,
) -> CategoryValueMatch | None:
    return match_entity_or_value(user_text, semantic_profile, df=dataframe)


def semantic_selection_text(question: str, focus: InvestigationFocus | None) -> str:
    if not focus:
        return question
    return " ".join(part for part in [question, focus.as_selection_text()] if str(part or "").strip())


def thread_selection_text(question: str, thread_state: InvestigationThreadState | None) -> str:
    if not thread_state:
        return question
    return " ".join(part for part in [question, thread_state.as_selection_text()] if str(part or "").strip())


def _select_column(
    candidates: list[SemanticColumnProfile],
    text: str,
    *,
    explicit_text: str | None,
    focus_value: str | None,
    semantic_groups: dict[str, tuple[str, ...]],
    prefer_low_cardinality: bool = False,
) -> str | None:
    if not candidates:
        return None
    explicit = explicit_text or text
    for source_text in (explicit, text):
        mentioned = _mentioned_columns(candidates, source_text)
        if mentioned:
            return mentioned[0].name
        semantic = _semantic_column_match(candidates, source_text, semantic_groups)
        if semantic:
            return semantic.name
    if focus_value and any(column.name == focus_value for column in candidates):
        return focus_value
    if prefer_low_cardinality:
        useful = [
            column for column in candidates
            if (column.cardinality or 0) > 1
            and column.role != SemanticRole.IDENTIFIER
            and "identifier" not in column.semantic_tags
        ]
        if useful:
            return min(useful, key=lambda column: column.cardinality or 10**9).name
        return None
    return max(candidates, key=lambda column: column.role_confidence).name


def _infer_role(
    name: str,
    dtype: str,
    *,
    row_count: int,
    unique_count: int | None,
    sample_values: list[Any],
    explicit_role: SemanticRole | None,
) -> tuple[SemanticRole, float]:
    if explicit_role and explicit_role != SemanticRole.UNKNOWN:
        return explicit_role, 0.98
    normalized = _normalize(name)
    dtype_lower = dtype.lower()
    unique = int(unique_count or 0)
    if _contains_any(normalized, _time_semantic_groups()["time"]) or "datetime" in dtype_lower:
        return SemanticRole.TIMESTAMP, 0.9
    if _looks_identifier_name(normalized, unique, row_count):
        return SemanticRole.IDENTIFIER, 0.82
    if any(token in dtype_lower for token in ("int", "float", "decimal", "number")):
        if _looks_identifier_name(normalized, unique, row_count):
            return SemanticRole.IDENTIFIER, 0.72
        confidence = 0.86 if _semantic_group_match(normalized, _metric_semantic_groups()) else 0.74
        return SemanticRole.METRIC, confidence
    if _sample_values_look_dates(sample_values):
        return SemanticRole.TIMESTAMP, 0.78
    if _semantic_group_match(normalized, _dimension_semantic_groups()):
        return SemanticRole.DIMENSION, 0.86
    if unique and row_count and unique / max(row_count, 1) > 0.75:
        if _contains_any(normalized, ("name", "email", "code", "uuid", "id")):
            return SemanticRole.IDENTIFIER, 0.74
        return SemanticRole.TEXT, 0.6
    if not row_count or unique <= max(30, int(row_count * 0.5)):
        return SemanticRole.DIMENSION, 0.68
    return SemanticRole.TEXT, 0.55


def _semantic_role_from_note(role: Any) -> SemanticRole | None:
    value = getattr(role, "value", role)
    mapping = {
        ColumnSemanticRole.METRIC.value: SemanticRole.METRIC,
        ColumnSemanticRole.TARGET.value: SemanticRole.METRIC,
        ColumnSemanticRole.DIMENSION.value: SemanticRole.DIMENSION,
        ColumnSemanticRole.TIMESTAMP.value: SemanticRole.TIMESTAMP,
        ColumnSemanticRole.IDENTIFIER.value: SemanticRole.IDENTIFIER,
        ColumnSemanticRole.TEXT.value: SemanticRole.TEXT,
        ColumnInferredRole.METRIC.value: SemanticRole.METRIC,
        ColumnInferredRole.DIMENSION.value: SemanticRole.DIMENSION,
        ColumnInferredRole.TIMESTAMP.value: SemanticRole.TIMESTAMP,
        ColumnInferredRole.IDENTIFIER.value: SemanticRole.IDENTIFIER,
        ColumnInferredRole.TEXT.value: SemanticRole.TEXT,
    }
    return mapping.get(str(value)) if value is not None else None


def _metric_semantic_groups() -> dict[str, tuple[str, ...]]:
    return {
        "metric": ("metric", "value", "amount", "score", "rating", "measure", "index"),
        "money": (
            "revenue", "sales", "sale", "salary", "price", "cost", "profit", "income", "pay", "compensation",
            "spend", "выруч", "продаж", "продажи", "доход", "зарплат", "цена", "прибыл",
        ),
        "volume": ("count", "quantity", "volume", "duration", "openings", "applicants", "users", "records", "колич", "число"),
        "quality": ("rating", "score", "rank", "оцен", "рейтинг", "балл"),
    }


def _dimension_semantic_groups() -> dict[str, tuple[str, ...]]:
    return {
        "role": (
            "role", "roles", "profession", "professions", "job", "jobs", "title", "occupation", "position",
            "професс", "профессии", "профессиям", "должн", "роль", "ролям",
        ),
        "category": (
            "category", "categories", "segment", "segments", "type", "types", "class", "group", "groups",
            "категор", "сегмент", "сегменты", "сегментам", "тип", "типы", "групп",
        ),
        "organization": (
            "company", "vendor", "customer", "client", "product", "industry", "department", "team",
            "компан", "клиент", "продукт", "индустр", "отрасл",
        ),
        "place": (
            "region", "city", "country", "location", "market", "area", "state",
            "город", "города", "городам", "городах", "городов", "городе",
            "регион", "регионам", "регионах", "страна", "странам", "локац",
        ),
    }


def _time_semantic_groups() -> dict[str, tuple[str, ...]]:
    return {
        "time": (
            "date", "time", "timestamp", "created", "updated", "month", "year", "period", "posted",
            "дата", "время", "месяц", "год", "период", "динамик",
        )
    }


def _semantic_tags(name: str, dtype: str, examples: list[Any]) -> list[str]:
    normalized = _normalize(name)
    tags = []
    for tag, group in {
        **_metric_semantic_groups(),
        **_dimension_semantic_groups(),
        **_time_semantic_groups(),
    }.items():
        if _contains_any(normalized, group):
            tags.append(tag)
    if "%" in name or "percent" in normalized or "rate" in normalized:
        tags.append("percentage")
    if "rank" in normalized or "rating" in normalized or "score" in normalized:
        tags.append("ranking")
    if _sample_values_look_dates(examples) or "datetime" in dtype.lower():
        tags.append("time")
    return sorted(set(tags))


def _distribution_hints(name: str, dtype: str, unique_count: int | None, row_count: int) -> list[str]:
    hints = []
    if unique_count is not None:
        hints.append(f"{unique_count} unique values")
        if row_count and unique_count / max(row_count, 1) < 0.05:
            hints.append("low-cardinality grouping candidate")
        if row_count and unique_count / max(row_count, 1) > 0.8:
            hints.append("high-cardinality field")
    if any(token in dtype.lower() for token in ("int", "float", "decimal", "number")):
        hints.append("numeric distribution available")
    if _contains_any(_normalize(name), _time_semantic_groups()["time"]):
        hints.append("time-based analysis candidate")
    return hints


def _business_hints(name: str, role: SemanticRole) -> list[str]:
    normalized = _normalize(name)
    if role == SemanticRole.METRIC:
        if _semantic_group_match(normalized, {"money": _metric_semantic_groups()["money"]}):
            return ["business performance or value metric"]
        if _semantic_group_match(normalized, {"quality": _metric_semantic_groups()["quality"]}):
            return ["quality or ranking metric"]
        return ["quantitative comparison candidate"]
    if role == SemanticRole.DIMENSION:
        if _semantic_group_match(normalized, {"place": _dimension_semantic_groups()["place"]}):
            return ["geographic segmentation candidate"]
        return ["segmentation and comparison candidate"]
    if role == SemanticRole.TIMESTAMP:
        return ["trend and change-over-time candidate"]
    if role == SemanticRole.IDENTIFIER:
        return ["entity key; useful for counting, not averaging"]
    return []


def _mentioned_columns(candidates: list[SemanticColumnProfile], text: str) -> list[SemanticColumnProfile]:
    normalized = _normalize(text)
    matches = []
    for column in candidates:
        name = _normalize(column.name)
        tokens = [token for token in name.replace("_", " ").split() if token]
        if name and name in normalized:
            matches.append(column)
        elif tokens and all(token in normalized for token in tokens):
            matches.append(column)
    return matches


def _semantic_column_match(
    candidates: list[SemanticColumnProfile],
    text: str,
    semantic_groups: dict[str, tuple[str, ...]],
) -> SemanticColumnProfile | None:
    normalized = _normalize(text)
    requested_groups = [group for group, markers in semantic_groups.items() if _contains_any(normalized, markers)]
    if not requested_groups:
        return None
    scored: list[tuple[int, float, SemanticColumnProfile]] = []
    for column in candidates:
        column_name = _normalize(column.name)
        score = 0
        score += _explicit_entity_priority(normalized, column_name)
        for group in requested_groups:
            for marker in semantic_groups[group]:
                if marker in normalized and marker in column_name:
                    score += 24
            if _contains_any(column_name, semantic_groups[group]):
                score += 10
            if group in column.semantic_tags:
                score += 6
        if score:
            scored.append((score, column.role_confidence, column))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def _explicit_entity_priority(question: str, column_name: str) -> int:
    pairs = (
        (("город", "города", "городов", "городам", "городах", "city", "cities"), ("city", "город")),
        (("страна", "страны", "country", "countries"), ("country", "страна")),
        (("регион", "регионы", "region", "regions"), ("region", "регион")),
        (("сегмент", "сегменты", "segment", "segments"), ("segment", "сегмент")),
        (("продукт", "товар", "product", "products"), ("product", "продукт", "товар")),
        (("професс", "должн", "роль", "job title", "role", "profession"), ("job", "title", "role", "profession", "должн", "професс")),
    )
    for question_markers, column_markers in pairs:
        if _contains_any(question, question_markers) and _contains_any(column_name, column_markers):
            return 60
    return 0


def _semantic_group_match(text: str, groups: dict[str, tuple[str, ...]]) -> bool:
    return any(_contains_any(text, markers) for markers in groups.values())


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _looks_identifier_name(normalized_name: str, unique_count: int, row_count: int) -> bool:
    id_like = (
        normalized_name in {"id", "uuid"}
        or normalized_name.endswith("_id")
        or " id" in normalized_name
        or _contains_any(normalized_name, ("postal", "zip", "postcode", "zipcode", "code"))
    )
    if id_like:
        return True
    return bool(row_count and unique_count / max(row_count, 1) >= 0.8 and _contains_any(normalized_name, ("id", "uuid", "code", "key")))


def _sample_values_look_dates(values: list[Any]) -> bool:
    if not values:
        return False
    parsed = 0
    for value in values[:5]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pd.to_datetime(value, errors="raise")
            parsed += 1
        except Exception:
            continue
    return parsed >= max(1, min(3, len(values)))


def _analysis_type_from_chart(chart_type: str | None) -> str | None:
    if chart_type == "bar":
        return "grouped_metric"
    if chart_type == "line":
        return "trend"
    if chart_type == "histogram":
        return "distribution"
    if chart_type == "scatter":
        return "correlation"
    return chart_type


def _clean_optional(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _infer_active_hypothesis(
    metric: str | None,
    dimension: str | None,
    findings: list[str],
    question: str | None,
) -> str | None:
    if findings:
        return findings[0]
    if metric and dimension:
        return f"{metric} may vary meaningfully across {dimension}."
    if metric and question:
        return f"{metric} is the current analytical target."
    return None


def _thread_suggestions(
    metric: str | None,
    dimension: str | None,
    time_axis: str | None,
    findings: list[str],
) -> list[str]:
    suggestions: list[str] = []
    if metric and dimension:
        suggestions.extend(
            [
                f"Does record volume explain the `{metric}` differences across `{dimension}`?",
                f"Which `{dimension}` groups are statistical outliers for `{metric}`?",
                f"Which `{dimension}` groups are strongest by `{metric}`?",
                f"Could small sample size distort the `{dimension}` ranking?",
                f"Are high-`{metric}` `{dimension}` groups driven by average value or volume?",
            ]
        )
        if time_axis:
            suggestions.append(f"Does this `{metric}` by `{dimension}` pattern change over time?")
        else:
            suggestions.append(f"What could explain the `{metric}` gap across `{dimension}`?")
            suggestions.append(f"Does the `{metric}` pattern persist across another segment?")
    elif metric:
        suggestions.extend(
            [
                f"Which records or groups have unusual `{metric}` values?",
                f"What appears most associated with `{metric}`?",
                f"Which variables explain most variance in `{metric}`?",
            ]
        )
    if findings:
        suggestions.append("Which current findings need stronger evidence?")
    return _dedupe_text(suggestions)


def _dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = " ".join(str(item).split()).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _normalize(value: str) -> str:
    return str(value or "").replace("`", "").replace("-", " ").replace("_", " ").lower()
