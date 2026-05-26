from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from source.product.data_sources import (
    ColumnSemanticNote,
    ColumnSemanticRole,
    DataSourceColumn,
    DataSourceProfile,
    DataSourceSemanticNotes,
)
from source.product.investigation import (
    DecisionMetadata,
    DecisionStatus,
    ReportSection,
    SectionReviewStatus,
    new_id,
    utc_now,
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _section_to_dict(section: ReportSection) -> dict[str, Any]:
    return {
        "id": section.section_id,
        "title": section.title,
        "content": section.content,
        "order": section.order,
        "section_type": section.section_type,
        "source_branch_id": section.source_branch_id,
        "source_artifact_ids": section.source_artifact_ids,
        "source_question_ids": section.source_question_ids,
        "artifact_ids": section.artifact_ids,
        "edited_by_user": section.edited_by_user,
        "created_by": section.created_by,
        "version": section.version,
        "review_status": section.review_status.value,
        "updated_at": _dt(section.updated_at),
        "edit_history": section.edit_history,
        "metadata": section.metadata,
    }


def _section_from_dict(data: dict[str, Any]) -> ReportSection:
    return ReportSection(
        section_id=str(data.get("id") or data.get("section_id") or new_id("section")),
        title=str(data.get("title") or ""),
        content=str(data.get("content") or ""),
        order=int(data.get("order") or 0),
        section_type=str(data.get("section_type") or data.get("type") or "narrative"),
        source_branch_id=data.get("source_branch_id"),
        source_artifact_ids=list(data.get("source_artifact_ids") or data.get("artifact_ids") or []),
        source_question_ids=list(data.get("source_question_ids") or []),
        artifact_ids=list(data.get("artifact_ids") or []),
        edited_by_user=bool(data.get("edited_by_user", False)),
        created_by=str(data.get("created_by") or "ai"),
        version=int(data.get("version") or 1),
        review_status=SectionReviewStatus(data.get("review_status") or "draft"),
        updated_at=_parse_dt(data["updated_at"]) if data.get("updated_at") else utc_now(),
        edit_history=list(data.get("edit_history") or []),
        metadata=dict(data.get("metadata") or {}),
    )


def _decision_metadata_to_dict(metadata: DecisionMetadata) -> dict[str, Any]:
    return {
        "tags": metadata.tags,
        "owner": metadata.owner,
        "audience": metadata.audience,
        "business_area": metadata.business_area,
        "decision_date": metadata.decision_date,
        "decision_status": metadata.decision_status.value,
        "short_description": metadata.short_description,
    }


def _decision_metadata_from_dict(data: dict[str, Any]) -> DecisionMetadata:
    data = data or {}
    return DecisionMetadata(
        tags=list(data.get("tags") or []),
        owner=data.get("owner"),
        audience=data.get("audience"),
        business_area=data.get("business_area"),
        decision_date=data.get("decision_date"),
        decision_status=DecisionStatus(data.get("decision_status") or "unknown"),
        short_description=data.get("short_description"),
    )


def _profile_to_dict(profile: DataSourceProfile) -> dict[str, Any]:
    return {
        "row_count": profile.row_count,
        "column_count": profile.column_count,
        "columns": [
            {
                "name": column.name,
                "dtype": column.dtype,
                "nullable": column.nullable,
                "unique_count": column.unique_count,
                "sample_values": column.sample_values,
            }
            for column in profile.columns
        ],
        "missing_summary": profile.missing_summary,
        "numeric_summary": profile.numeric_summary,
        "categorical_summary": profile.categorical_summary,
        "sampled_rows": profile.sampled_rows,
        "generated_at": _dt(profile.generated_at),
    }


def _profile_from_dict(data: dict[str, Any]) -> DataSourceProfile:
    data = data or {}
    return DataSourceProfile(
        row_count=int(data.get("row_count") or 0),
        column_count=int(data.get("column_count") or 0),
        columns=[
            DataSourceColumn(
                name=str(item.get("name") or ""),
                dtype=str(item.get("dtype") or ""),
                nullable=bool(item.get("nullable", False)),
                unique_count=item.get("unique_count"),
                sample_values=list(item.get("sample_values") or []),
            )
            for item in data.get("columns", [])
        ],
        missing_summary=dict(data.get("missing_summary") or {}),
        numeric_summary=dict(data.get("numeric_summary") or {}),
        categorical_summary=dict(data.get("categorical_summary") or {}),
        sampled_rows=list(data.get("sampled_rows") or []),
        generated_at=_parse_dt(data["generated_at"]) if data.get("generated_at") else utc_now(),
    )


def _semantic_notes_to_dict(notes: DataSourceSemanticNotes) -> dict[str, Any]:
    return {
        "data_source_id": notes.data_source_id,
        "source_description": notes.source_description,
        "business_context": notes.business_context,
        "global_caveats": notes.global_caveats,
        "column_notes": [_column_semantic_note_to_dict(note) for note in notes.column_notes],
        "updated_at": _dt(notes.updated_at),
    }


def _column_semantic_note_to_dict(note: ColumnSemanticNote) -> dict[str, Any]:
    return {
        "column_name": note.column_name,
        "display_name": note.display_name,
        "description": note.description,
        "business_meaning": note.business_meaning,
        "semantic_role": note.semantic_role.value,
        "caveats": note.caveats,
        "examples": note.examples,
    }


def _semantic_notes_from_dict(data: dict[str, Any], data_source_id: str | None = None) -> DataSourceSemanticNotes:
    data = data or {}
    return DataSourceSemanticNotes(
        data_source_id=str(data.get("data_source_id") or data_source_id or ""),
        source_description=data.get("source_description"),
        business_context=data.get("business_context"),
        global_caveats=list(data.get("global_caveats") or []),
        column_notes=[_column_semantic_note_from_dict(item) for item in data.get("column_notes", [])],
        updated_at=_parse_dt(data["updated_at"]) if data.get("updated_at") else utc_now(),
    )


def _column_semantic_note_from_dict(data: dict[str, Any]) -> ColumnSemanticNote:
    return ColumnSemanticNote(
        column_name=str(data.get("column_name") or ""),
        display_name=data.get("display_name"),
        description=data.get("description"),
        business_meaning=data.get("business_meaning"),
        semantic_role=ColumnSemanticRole(data.get("semantic_role") or "unknown"),
        caveats=list(data.get("caveats") or []),
        examples=list(data.get("examples") or []),
    )


def _normalize_tags(tags: list[str]) -> list[str]:
    normalized: list[str] = []
    for tag in tags:
        clean = " ".join(str(tag or "").strip().lower().split())
        if clean and clean not in normalized:
            normalized.append(clean)
    return normalized


def _row_value(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    return row[key] if key in row.keys() else default
