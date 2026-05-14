from __future__ import annotations

from typing import Any

from source.product.data_sources import (
    ColumnInferredRole,
    ColumnSemanticNote,
    ColumnSemanticRole,
    ColumnUsageSummary,
    DataSource,
    DataSourceColumn,
    DataSourceProfile,
    DataSourceSemanticNotes,
    DataSourceStatus,
    DataSourceUsageContext,
)
from source.product.investigation import utc_now


def build_data_source_usage_context(store: Any, data_source_id: str) -> DataSourceUsageContext:
    source = store.get_data_source(data_source_id)
    profile = _get_profile_or_none(store, data_source_id)
    semantic_notes = _get_semantic_notes_or_default(store, data_source_id)
    previous_questions = _linked_previous_questions(store, source.linked_investigation_ids)
    description = _context_description(source, semantic_notes)

    if profile is None:
        return DataSourceUsageContext(
            data_source_id=source.data_source_id,
            name=source.name,
            type=source.data_source_type,
            status=source.status,
            description=description,
            tags=list(source.tags),
            schema_summary={"row_count": None, "column_count": None, "columns": []},
            freshness=_freshness(source, None),
            linked_investigation_ids=list(source.linked_investigation_ids),
            previous_questions=previous_questions,
            caveats=_source_caveats(source, None) + _semantic_caveats(semantic_notes) + ["No profile is available for this data source."],
            generated_at=utc_now(),
        )

    notes_by_column = {note.column_name: note for note in semantic_notes.column_notes}
    column_summaries = [
        _column_usage_summary(
            column,
            profile.row_count,
            profile.categorical_summary.get(column.name, {}),
            notes_by_column.get(column.name),
        )
        for column in profile.columns
    ]
    caveats = _source_caveats(source, profile)
    caveats.extend(_profile_caveats(profile, column_summaries))
    caveats.extend(_semantic_caveats(semantic_notes))

    return DataSourceUsageContext(
        data_source_id=source.data_source_id,
        name=source.name,
        type=source.data_source_type,
        status=source.status,
        description=description,
        tags=list(source.tags),
        schema_summary=_schema_summary(profile, column_summaries),
        column_summaries=column_summaries,
        sample_rows=list(profile.sampled_rows[:10]),
        missing_summary=dict(profile.missing_summary),
        numeric_summary=dict(profile.numeric_summary),
        categorical_summary=dict(profile.categorical_summary),
        freshness=_freshness(source, profile),
        linked_investigation_ids=list(source.linked_investigation_ids),
        previous_questions=previous_questions,
        caveats=caveats,
        generated_at=utc_now(),
    )


def usage_context_to_prompt(contexts: list[DataSourceUsageContext]) -> str:
    if not contexts:
        return ""
    sections = ["Data source usage context:"]
    for context in contexts:
        sections.append(f"- {context.name} ({context.type.value}, {context.status.value})")
        summary = context.schema_summary
        if summary:
            sections.append(
                f"  Shape: {summary.get('row_count')} rows, {summary.get('column_count')} columns."
            )
            roles = summary.get("roles", {})
            if roles:
                role_bits = [f"{role}: {', '.join(names[:8])}" for role, names in roles.items() if names]
                if role_bits:
                    sections.append("  Inferred roles: " + "; ".join(role_bits))
        if context.description:
            sections.append(f"  Description: {context.description}")
        column_notes = [
            f"{column.name} ({column.inferred_role.value}): {'; '.join(column.notes[:4])}"
            for column in context.column_summaries
            if column.notes
        ]
        if column_notes:
            sections.append("  Column notes: " + " | ".join(column_notes[:8]))
        if context.previous_questions:
            sections.append("  Previous questions: " + "; ".join(context.previous_questions[:5]))
        if context.caveats:
            sections.append("  Caveats: " + "; ".join(context.caveats[:8]))
    return "\n".join(sections)


def _get_profile_or_none(store: Any, data_source_id: str) -> DataSourceProfile | None:
    try:
        return store.get_data_source_profile(data_source_id)
    except KeyError:
        return None


def _get_semantic_notes_or_default(store: Any, data_source_id: str) -> DataSourceSemanticNotes:
    try:
        return store.get_data_source_semantic_notes(data_source_id)
    except (AttributeError, KeyError):
        return DataSourceSemanticNotes(data_source_id=data_source_id)


def _linked_previous_questions(store: Any, investigation_ids: list[str]) -> list[str]:
    questions: list[str] = []
    for investigation_id in investigation_ids:
        try:
            investigation = store.get_investigation(investigation_id)
        except KeyError:
            continue
        if investigation.user_question and investigation.user_question not in questions:
            questions.append(investigation.user_question)
    return questions


def _schema_summary(profile: DataSourceProfile, columns: list[ColumnUsageSummary]) -> dict[str, Any]:
    roles: dict[str, list[str]] = {}
    for column in columns:
        roles.setdefault(column.inferred_role.value, []).append(column.name)
    return {
        "row_count": profile.row_count,
        "column_count": profile.column_count,
        "columns": [column.name for column in columns],
        "roles": roles,
    }


def _column_usage_summary(
    column: DataSourceColumn,
    row_count: int,
    categorical_summary: dict[str, Any],
    note: ColumnSemanticNote | None = None,
) -> ColumnUsageSummary:
    role, notes = _infer_column_role(column, row_count, categorical_summary)
    if note:
        role, notes = _apply_semantic_note(role, notes, note)
    return ColumnUsageSummary(
        name=column.name,
        dtype=column.dtype,
        nullable=column.nullable,
        unique_count=column.unique_count,
        sample_values=list(column.sample_values),
        inferred_role=role,
        notes=notes,
    )


def _apply_semantic_note(
    deterministic_role: ColumnInferredRole,
    notes: list[str],
    note: ColumnSemanticNote,
) -> tuple[ColumnInferredRole | ColumnSemanticRole, list[str]]:
    enriched = list(notes)
    if note.semantic_role != ColumnSemanticRole.UNKNOWN:
        enriched.append(f"User semantic role overrides deterministic role {deterministic_role.value}.")
        role: ColumnInferredRole | ColumnSemanticRole = note.semantic_role
    else:
        role = deterministic_role
    if note.display_name:
        enriched.append(f"Display name: {note.display_name}.")
    if note.description:
        enriched.append(f"Description: {note.description}.")
    if note.business_meaning:
        enriched.append(f"Business meaning: {note.business_meaning}.")
    for caveat in note.caveats[:3]:
        enriched.append(f"Caveat: {caveat}.")
    if note.examples:
        enriched.append("Examples: " + ", ".join(note.examples[:5]) + ".")
    return role, enriched


def _infer_column_role(
    column: DataSourceColumn,
    row_count: int,
    categorical_summary: dict[str, Any],
) -> tuple[ColumnInferredRole, list[str]]:
    name = column.name.lower()
    dtype = column.dtype.lower()
    unique_count = column.unique_count or 0
    sample_values = [str(value) for value in column.sample_values if value is not None]
    notes: list[str] = []

    if "datetime" in dtype or "date" in dtype or "time" in dtype or "date" in name or name.endswith("_at"):
        return ColumnInferredRole.TIMESTAMP, ["Looks datetime-like from dtype or name."]

    if name in {"id", "uuid"} or name.endswith("_id") or " id" in name or "identifier" in name:
        return ColumnInferredRole.IDENTIFIER, ["Looks identifier-like from column name."]

    high_cardinality = row_count > 0 and unique_count / max(row_count, 1) >= 0.8
    if high_cardinality and any(token in name for token in ["id", "name", "email", "sku", "code"]):
        return ColumnInferredRole.IDENTIFIER, ["High cardinality and identifier-like name."]

    if any(token in dtype for token in ["int", "float", "decimal", "number"]):
        return ColumnInferredRole.METRIC, ["Numeric dtype."]

    if _looks_like_long_text(sample_values):
        return ColumnInferredRole.TEXT, ["Sample values look like long text."]

    top_values = categorical_summary.get("top_values", {}) if isinstance(categorical_summary, dict) else {}
    if "object" in dtype or "string" in dtype or "category" in dtype or top_values:
        if row_count == 0 or unique_count <= max(30, int(row_count * 0.5)):
            return ColumnInferredRole.DIMENSION, ["Categorical or low-cardinality text."]
        return ColumnInferredRole.TEXT, ["High-cardinality text."]

    notes.append("No confident role inferred.")
    return ColumnInferredRole.UNKNOWN, notes


def _looks_like_long_text(values: list[str]) -> bool:
    if not values:
        return False
    long_values = [value for value in values if len(value) > 80 or len(value.split()) > 12]
    return len(long_values) >= max(1, len(values) // 2)


def _source_caveats(source: DataSource, profile: DataSourceProfile | None) -> list[str]:
    caveats: list[str] = []
    if source.status in {DataSourceStatus.ARCHIVED, DataSourceStatus.STALE, DataSourceStatus.ERROR}:
        caveats.append(f"Source status is {source.status.value}.")
    if profile is not None and profile.row_count == 0:
        caveats.append("Profile has zero rows.")
    return caveats


def _semantic_caveats(notes: DataSourceSemanticNotes) -> list[str]:
    return [str(item) for item in notes.global_caveats if str(item).strip()]


def _context_description(source: DataSource, notes: DataSourceSemanticNotes) -> str | None:
    parts = [
        item
        for item in [notes.source_description, notes.business_context, source.description]
        if item and str(item).strip()
    ]
    return "\n\n".join(str(item).strip() for item in parts) or None


def _profile_caveats(profile: DataSourceProfile, columns: list[ColumnUsageSummary]) -> list[str]:
    caveats: list[str] = []
    if not profile.columns:
        caveats.append("Profile has no columns.")
    if not profile.sampled_rows:
        caveats.append("No sample rows are available.")
    for column_name, missing_count in profile.missing_summary.items():
        if profile.row_count > 0 and missing_count / profile.row_count >= 0.5:
            caveats.append(f"Column {column_name} has high missingness.")
    unknown_columns = [column.name for column in columns if column.inferred_role == ColumnInferredRole.UNKNOWN]
    if unknown_columns:
        caveats.append("Some columns have unknown inferred roles: " + ", ".join(unknown_columns[:8]) + ".")
    return caveats


def _freshness(source: DataSource, profile: DataSourceProfile | None) -> dict[str, Any]:
    return {
        "source_status": source.status.value,
        "source_updated_at": source.updated_at,
        "profile_generated_at": profile.generated_at if profile else None,
    }
