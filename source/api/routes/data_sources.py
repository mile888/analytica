from __future__ import annotations

from contextlib import suppress
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from source.api.deps import get_store
from source.api.serialization import to_jsonable
from source.dataframe import CsvLoadError, read_csv_dataset
from source.product.data_context import build_data_source_usage_context
from source.product.execution_context import mark_profile_only_runtime, persist_dataset_runtime
from source.product.data_profiling import profile_csv
from source.product.data_sources import (
    ColumnSemanticNote,
    ColumnSemanticRole,
    DataSource,
    DataSourceSemanticNotes,
    DataSourceStatus,
    DataSourceType,
)
from source.product.file_storage import save_uploaded_csv
from source.product.question_suggestions import build_data_aware_question_suggestions


router = APIRouter(prefix="/data-sources", tags=["data-sources"])


class DataSourceCreate(BaseModel):
    name: str
    type: str = "unknown"
    location: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class DataSourceUpdate(BaseModel):
    name: str | None = None
    status: str | None = None
    location: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    metadata: dict | None = None


class ColumnSemanticNotePayload(BaseModel):
    display_name: str | None = None
    description: str | None = None
    business_meaning: str | None = None
    semantic_role: str = "unknown"
    caveats: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class DataSourceSemanticNotesPayload(BaseModel):
    source_description: str | None = None
    business_context: str | None = None
    global_caveats: list[str] = Field(default_factory=list)
    column_notes: list[dict] = Field(default_factory=list)


@router.get("")
def list_data_sources(status: str | None = None):
    store = get_store()
    sources = store.list_data_sources(status=status)
    return [to_jsonable(item) for item in sources]


@router.post("")
def create_data_source(payload: DataSourceCreate):
    source = DataSource(
        name=payload.name,
        data_source_type=DataSourceType(payload.type),
        location=payload.location,
        description=payload.description,
        tags=payload.tags,
        metadata=payload.metadata,
    )
    store = get_store()
    created = store.create_data_source(source)
    if not created.location:
        mark_profile_only_runtime(store, created.data_source_id)
    return to_jsonable(store.get_data_source(created.data_source_id))


@router.post("/upload-csv")
async def upload_csv_data_source(
    file: Annotated[UploadFile, File()],
    name: Annotated[str | None, Form()] = None,
    description: Annotated[str | None, Form()] = None,
    tags: Annotated[list[str] | None, Form()] = None,
):
    import logging

    logger = logging.getLogger("analytica.upload")
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only .csv files are supported.")

    contents = await file.read()
    if not contents.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    source = DataSource(
        name=(name or filename.rsplit(".", 1)[0] or "Uploaded CSV").strip(),
        data_source_type=DataSourceType.CSV,
        description=description,
        tags=_parse_form_tags(tags),
        metadata={"original_filename": filename, "upload_kind": "csv"},
    )

    try:
        location = save_uploaded_csv(contents, filename, source.data_source_id)
    except Exception as exc:
        logger.exception("File save failed for %s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded file: {type(exc).__name__}",
        ) from exc

    source.location = location
    store = get_store()
    created = store.create_data_source(source)
    warnings: list[str] = []
    profile_status = "complete"

    profile = None
    try:
        profile = profile_csv(location)
    except (pd.errors.ParserError, UnicodeDecodeError, CsvLoadError) as exc:
        created.status = DataSourceStatus.ERROR
        if isinstance(exc, CsvLoadError):
            user_detail = exc.user_message
        elif isinstance(exc, UnicodeDecodeError):
            user_detail = (
                "Could not decode the uploaded CSV file. "
                "The file may use a non-standard encoding. "
                "Try re-saving it as UTF-8 CSV from Excel or Google Sheets."
            )
        else:
            user_detail = f"Could not parse CSV: {exc}"
        created.metadata["profile_error"] = str(exc)
        store.update_data_source(created)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=user_detail) from exc
    except Exception as exc:
        logger.exception("Profiling failed for %s (non-fatal)", filename)
        warnings.append(f"Profiling partially failed: {type(exc).__name__}: {exc}")
        profile_status = "partial"

    if profile is not None:
        try:
            store.save_data_source_profile(created.data_source_id, profile)
        except Exception as exc:
            logger.exception("Profile save failed for %s (non-fatal)", created.data_source_id)
            warnings.append(f"Profile could not be saved: {type(exc).__name__}")
            profile_status = "partial"

    try:
        df = _read_uploaded_runtime_csv(location)
        if df.empty and len(df.columns) == 0:
            mark_profile_only_runtime(store, created.data_source_id, reason="empty_dataframe")
        else:
            # Sanitize column names to avoid SQLite errors with empty/blank names
            sanitized_columns = []
            for i, col in enumerate(df.columns):
                col_str = str(col).strip()
                if not col_str or col_str.startswith("Unnamed:"):
                    col_str = f"column_{i}"
                sanitized_columns.append(col_str)
            if len(set(sanitized_columns)) < len(sanitized_columns):
                seen: dict[str, int] = {}
                deduped: list[str] = []
                for col_name in sanitized_columns:
                    if col_name in seen:
                        seen[col_name] += 1
                        deduped.append(f"{col_name}_{seen[col_name]}")
                    else:
                        seen[col_name] = 0
                        deduped.append(col_name)
                sanitized_columns = deduped
            df.columns = pd.Index(sanitized_columns)
            persist_dataset_runtime(store, created.data_source_id, df)
    except Exception as exc:
        logger.exception("Runtime persistence failed for %s (non-fatal)", created.data_source_id)
        warnings.append(f"Runtime persistence failed: {type(exc).__name__}: {exc}")
        try:
            mark_profile_only_runtime(store, created.data_source_id, reason=f"runtime_failed:{type(exc).__name__}")
        except Exception:
            pass

    try:
        data_source_payload = to_jsonable(store.get_data_source(created.data_source_id))
    except Exception as exc:
        logger.exception("Data source serialization failed (non-fatal)")
        data_source_payload = {"data_source_id": created.data_source_id, "name": created.name, "status": "active"}

    profile_payload = None
    if profile is not None:
        try:
            profile_payload = to_jsonable(profile)
        except Exception as exc:
            logger.exception("Profile serialization failed (non-fatal)")
            warnings.append(f"Profile serialization failed: {type(exc).__name__}")
            profile_status = "partial"
            profile_payload = {
                "row_count": getattr(profile, "row_count", 0),
                "column_count": getattr(profile, "column_count", 0),
                "columns": [],
                "missing_summary": {},
                "numeric_summary": {},
                "categorical_summary": {},
                "sampled_rows": [],
            }

    response: dict[str, object] = {
        "data_source": data_source_payload,
        "profile": profile_payload,
    }
    if warnings:
        response["profile_status"] = profile_status
        response["warnings"] = warnings
    return response


@router.get("/{data_source_id}")
def get_data_source(data_source_id: str):
    try:
        return to_jsonable(get_store().get_data_source(data_source_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{data_source_id}/profile")
def get_data_source_profile(data_source_id: str):
    try:
        return to_jsonable(get_store().get_data_source_profile(data_source_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{data_source_id}/usage-context")
def get_data_source_usage_context(data_source_id: str):
    try:
        return to_jsonable(build_data_source_usage_context(get_store(), data_source_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{data_source_id}/suggested-questions")
def get_data_source_suggested_questions(data_source_id: str, limit: int = 8):
    store = get_store()
    try:
        profile = None
        notes = None
        with suppress(KeyError):
            profile = store.get_data_source_profile(data_source_id)
        with suppress(KeyError):
            notes = store.get_data_source_semantic_notes(data_source_id)
        context = build_data_source_usage_context(store, data_source_id)
        return {
            "suggestions": build_data_aware_question_suggestions(
                profile=profile,
                usage_context=context,
                semantic_notes=notes,
                limit=max(1, min(limit, 20)),
            )
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{data_source_id}/semantic-notes")
def get_data_source_semantic_notes(data_source_id: str):
    try:
        return to_jsonable(get_store().get_data_source_semantic_notes(data_source_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{data_source_id}/semantic-notes")
def save_data_source_semantic_notes(data_source_id: str, payload: DataSourceSemanticNotesPayload):
    try:
        notes = DataSourceSemanticNotes(
            data_source_id=data_source_id,
            source_description=payload.source_description,
            business_context=payload.business_context,
            global_caveats=payload.global_caveats,
            column_notes=[_column_note_from_payload(item) for item in payload.column_notes],
        )
        return to_jsonable(get_store().save_data_source_semantic_notes(notes))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/{data_source_id}/semantic-notes/columns/{column_name}")
def update_column_semantic_note(data_source_id: str, column_name: str, payload: ColumnSemanticNotePayload):
    try:
        note = ColumnSemanticNote(
            column_name=column_name,
            display_name=payload.display_name,
            description=payload.description,
            business_meaning=payload.business_meaning,
            semantic_role=ColumnSemanticRole(payload.semantic_role),
            caveats=payload.caveats,
            examples=payload.examples,
        )
        return to_jsonable(get_store().update_column_semantic_note(data_source_id, column_name, note))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{data_source_id}/semantic-notes/columns/{column_name}")
def delete_column_semantic_note(data_source_id: str, column_name: str):
    try:
        return to_jsonable(get_store().delete_column_semantic_note(data_source_id, column_name))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/{data_source_id}")
def update_data_source(data_source_id: str, payload: DataSourceUpdate):
    try:
        source = get_store().get_data_source(data_source_id)
        if payload.name is not None:
            source.name = payload.name
        if payload.status is not None:
            source.status = DataSourceStatus(payload.status)
        if payload.location is not None:
            source.location = payload.location
        if payload.description is not None:
            source.description = payload.description
        if payload.tags is not None:
            source.tags = payload.tags
        if payload.metadata is not None:
            source.metadata = payload.metadata
        return to_jsonable(get_store().update_data_source(source))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{data_source_id}")
def delete_data_source(data_source_id: str, delete_file: bool = True):
    try:
        file_deleted = get_store().delete_data_source(data_source_id, delete_file=delete_file)
        return {"deleted": True, "id": data_source_id, "file_deleted": file_deleted}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _column_note_from_payload(payload: dict) -> ColumnSemanticNote:
    return ColumnSemanticNote(
        column_name=str(payload.get("column_name") or ""),
        display_name=payload.get("display_name"),
        description=payload.get("description"),
        business_meaning=payload.get("business_meaning"),
        semantic_role=ColumnSemanticRole(payload.get("semantic_role") or "unknown"),
        caveats=list(payload.get("caveats") or []),
        examples=list(payload.get("examples") or []),
    )


def _parse_form_tags(tags: list[str] | None) -> list[str]:
    values: list[str] = []
    for item in tags or []:
        values.extend(part.strip() for part in str(item).split(","))
    return [value for value in values if value]


def _read_uploaded_runtime_csv(location: str) -> pd.DataFrame:
    try:
        return read_csv_dataset(location)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except (pd.errors.ParserError, UnicodeDecodeError):
        try:
            return read_csv_dataset(location, escapechar=chr(92))
        except (pd.errors.ParserError, UnicodeDecodeError):
            return read_csv_dataset(location, escapechar=chr(92), on_bad_lines="skip")
