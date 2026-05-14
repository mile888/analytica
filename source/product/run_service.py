from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from source.product.data_context import build_data_source_usage_context, usage_context_to_prompt
from source.product.data_sources import DataSourceType
from source.product.investigation import (
    InvestigationRun,
    InvestigationRunEvent,
    InvestigationRunEventSeverity,
    InvestigationRunEventType,
    InvestigationRunStage,
    InvestigationRunStatus,
    InvestigationStatus,
    utc_now,
)
from source.product.run_validation import validate_investigation_result
from source.product.service import InvestigationService
from source.product.store import InvestigationStore


class InvestigationRunService:
    def __init__(
        self,
        store: InvestigationStore | None = None,
        investigation_service: InvestigationService | None = None,
    ) -> None:
        self.store = store or InvestigationStore()
        self.investigation_service = investigation_service or InvestigationService(self.store)

    def run_investigation(
        self,
        investigation_id: str,
        data_source_ids: list[str] | None = None,
        force_refresh_context: bool = False,
        df: Any = None,
    ) -> InvestigationRun:
        investigation = self.store.get_investigation(investigation_id)
        resolved_source_ids = _unique_ids(list(investigation.linked_data_source_ids) + list(data_source_ids or []))
        run = InvestigationRun(
            investigation_id=investigation_id,
            status=InvestigationRunStatus.QUEUED,
            current_stage=InvestigationRunStage.PREPARING_DATA,
            data_source_ids=resolved_source_ids,
            started_at=None,
            metadata={"force_refresh_context": force_refresh_context},
        )
        self.store.create_investigation_run(run)

        try:
            self._transition(run, InvestigationRunStatus.RUNNING, InvestigationRunStage.PREPARING_DATA)
            for data_source_id in resolved_source_ids:
                self.store.get_data_source(data_source_id)
                self.store.link_data_source_to_investigation(investigation_id, data_source_id)
            self._emit(
                run,
                InvestigationRunEventType.INFO,
                "Prepared data sources for this run.",
                metadata={"data_source_ids": resolved_source_ids},
            )

            self._transition(run, InvestigationRunStatus.RUNNING, InvestigationRunStage.BUILDING_CONTEXT)
            usage_contexts = [build_data_source_usage_context(self.store, item) for item in resolved_source_ids]
            run.run_context_summary = _build_run_context_summary(usage_contexts)
            self.store.update_investigation_run(run)
            self._emit(
                run,
                InvestigationRunEventType.INFO,
                f"Built usage context for {len(usage_contexts)} data source(s).",
                metadata={"data_sources_count": len(usage_contexts)},
            )
            for context in usage_contexts:
                if not context.sample_rows:
                    self._emit(
                        run,
                        InvestigationRunEventType.WARNING,
                        f"{context.name}: no sample rows are available.",
                        severity=InvestigationRunEventSeverity.WARNING,
                        metadata={"data_source_id": context.data_source_id},
                    )
                if not any(
                    "Business meaning:" in note or "Display name:" in note
                    for column in context.column_summaries
                    for note in column.notes
                ):
                    self._emit(
                        run,
                        InvestigationRunEventType.WARNING,
                        f"{context.name}: no semantic column notes were found.",
                        severity=InvestigationRunEventSeverity.WARNING,
                        metadata={"data_source_id": context.data_source_id},
                    )
                for caveat in context.caveats[:8]:
                    self._emit(
                        run,
                        InvestigationRunEventType.WARNING,
                        f"{context.name}: {caveat}",
                        severity=InvestigationRunEventSeverity.WARNING,
                        metadata={"data_source_id": context.data_source_id},
                    )

            self._transition(run, InvestigationRunStatus.RUNNING, InvestigationRunStage.RUNNING_ANALYSIS)
            analysis_df = df if df is not None else self._load_dataframe_for_run(resolved_source_ids, run)
            updated = self.investigation_service.run_investigation(
                investigation_id,
                df=analysis_df,
                data_context={
                    "data_source_ids": resolved_source_ids,
                    "product_run_id": run.run_id,
                    "data_source_usage_contexts": [_jsonable(context) for context in usage_contexts],
                    "data_context_prompt": usage_context_to_prompt(usage_contexts),
                },
            )

            self._transition(run, InvestigationRunStatus.RUNNING, InvestigationRunStage.VALIDATING_RESULTS)
            validation = validate_investigation_result(updated, run_id=run.run_id)
            run.run_context_summary["validation"] = validation
            run.artifact_ids = [artifact.artifact_id for artifact in updated.artifacts if artifact.run_id == run.run_id]
            for artifact_id in run.artifact_ids:
                self._emit(
                    run,
                    InvestigationRunEventType.ARTIFACT_CREATED,
                    "Created an investigation artifact.",
                    metadata={"artifact_id": artifact_id},
                )
            for key in ["findings_count", "artifacts_count", "reports_generated", "execution_error_present"]:
                severity = (
                    InvestigationRunEventSeverity.ERROR
                    if key == "execution_error_present" and validation.get(key)
                    else InvestigationRunEventSeverity.INFO
                )
                self._emit(
                    run,
                    InvestigationRunEventType.VALIDATION_CHECK,
                    f"Validation check: {key} = {validation.get(key)}.",
                    severity=severity,
                    metadata={"check": key, "value": validation.get(key)},
                )
            for warning in validation.get("warnings", []):
                self._emit(
                    run,
                    InvestigationRunEventType.VALIDATION_CHECK,
                    warning,
                    severity=InvestigationRunEventSeverity.WARNING,
                    metadata={"kind": "validation_warning"},
                )

            self._transition(run, InvestigationRunStatus.RUNNING, InvestigationRunStage.GENERATING_REPORT)
            if updated.report and updated.report.run_id == run.run_id:
                run.report_ids = [updated.report.report_id]
                self._emit(
                    run,
                    InvestigationRunEventType.REPORT_CREATED,
                    "Generated a DecisionReport.",
                    metadata={"report_id": updated.report.report_id},
                )
            if updated.status == InvestigationStatus.FAILED:
                latest_error = updated.runs[-1].error if updated.runs else None
                raise RuntimeError(latest_error or "Investigation run failed.")

            self._emit_stage_completed(run, run.current_stage)
            run.status = InvestigationRunStatus.COMPLETED
            run.current_stage = InvestigationRunStage.COMPLETED
            run.completed_at = utc_now()
            run.finished_at = run.completed_at
            self._emit(
                run,
                InvestigationRunEventType.STAGE_COMPLETED,
                "Run completed.",
                stage=InvestigationRunStage.COMPLETED,
            )
            return self.store.update_investigation_run(run)
        except Exception as exc:
            run.status = InvestigationRunStatus.FAILED
            run.error_message = f"{type(exc).__name__}: {exc}"
            run.error = run.error_message
            run.completed_at = utc_now()
            run.finished_at = run.completed_at
            self._emit(
                run,
                InvestigationRunEventType.ERROR,
                "Investigation run failed.",
                severity=InvestigationRunEventSeverity.ERROR,
                metadata={"error": run.error_message},
            )
            self.store.update_investigation_run(run)
            return self.store.get_investigation_run(run.run_id)

    def _load_dataframe_for_run(self, data_source_ids: list[str], run: InvestigationRun) -> Any:
        for data_source_id in data_source_ids:
            try:
                source = self.store.get_data_source(data_source_id)
            except KeyError:
                continue
            if source.data_source_type != DataSourceType.CSV or not source.location:
                continue
            path = Path(source.location)
            if not path.exists():
                self._emit(
                    run,
                    InvestigationRunEventType.WARNING,
                    f"{source.name}: CSV file was not found.",
                    severity=InvestigationRunEventSeverity.WARNING,
                    metadata={"data_source_id": data_source_id, "location": source.location},
                )
                continue
            try:
                self._emit(
                    run,
                    InvestigationRunEventType.INFO,
                    f"Loaded CSV data for {source.name}.",
                    metadata={"data_source_id": data_source_id, "location": source.location},
                )
                return pd.read_csv(path)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()
            except Exception as exc:
                self._emit(
                    run,
                    InvestigationRunEventType.WARNING,
                    f"{source.name}: could not load CSV data.",
                    severity=InvestigationRunEventSeverity.WARNING,
                    metadata={"data_source_id": data_source_id, "error": str(exc)},
                )
                continue
        return None

    def _transition(
        self,
        run: InvestigationRun,
        status: InvestigationRunStatus,
        stage: InvestigationRunStage,
    ) -> InvestigationRun:
        previous_stage = run.current_stage
        if run.started_at is not None and previous_stage != stage:
            self._emit_stage_completed(run, previous_stage)
        run.status = status
        run.current_stage = stage
        if run.started_at is None:
            run.started_at = utc_now()
        updated = self.store.update_investigation_run(run)
        self._emit(
            run,
            InvestigationRunEventType.STAGE_STARTED,
            _stage_label(stage) + " started.",
            stage=stage,
        )
        return updated

    def _emit_stage_completed(self, run: InvestigationRun, stage: InvestigationRunStage) -> None:
        self._emit(
            run,
            InvestigationRunEventType.STAGE_COMPLETED,
            _stage_label(stage) + " completed.",
            stage=stage,
        )

    def _emit(
        self,
        run: InvestigationRun,
        event_type: InvestigationRunEventType,
        message: str,
        stage: InvestigationRunStage | None = None,
        severity: InvestigationRunEventSeverity = InvestigationRunEventSeverity.INFO,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            self.store.add_investigation_run_event(
                InvestigationRunEvent(
                    run_id=run.run_id,
                    investigation_id=run.investigation_id,
                    event_type=event_type,
                    stage=stage or run.current_stage,
                    message=message,
                    severity=severity,
                    metadata=metadata or {},
                )
            )
        except Exception:
            return


def _build_run_context_summary(contexts: list[Any]) -> dict[str, Any]:
    return {
        "data_sources_count": len(contexts),
        "data_sources": [
            {
                "data_source_id": context.data_source_id,
                "name": context.name,
                "status": context.status.value,
                "schema_summary": context.schema_summary,
                "caveats": list(context.caveats),
                "previous_questions": list(context.previous_questions),
                "semantic_column_notes_count": sum(
                    1
                    for column in context.column_summaries
                    if any("Business meaning:" in note or "Display name:" in note for note in column.notes)
                ),
            }
            for context in contexts
        ],
        "caveats": [caveat for context in contexts for caveat in context.caveats],
        "previous_questions": [question for context in contexts for question in context.previous_questions],
    }


def _unique_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _stage_label(stage: InvestigationRunStage) -> str:
    labels = {
        InvestigationRunStage.PREPARING_DATA: "Preparing data",
        InvestigationRunStage.BUILDING_CONTEXT: "Building context",
        InvestigationRunStage.RUNNING_ANALYSIS: "Running analysis",
        InvestigationRunStage.VALIDATING_RESULTS: "Validating results",
        InvestigationRunStage.GENERATING_REPORT: "Generating report",
        InvestigationRunStage.COMPLETED: "Completed",
    }
    return labels.get(stage, stage.value.replace("_", " ").title())


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
