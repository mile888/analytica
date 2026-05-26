from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from source.config import PROJECT_ROOT, RUNTIME_DIR
from source.product.data_sources import DataSourceProfile
from source.product.dataframe_resolver import resolve_dataframe_from_data_source
from source.product.investigation import utc_now
from source.product.execution_planner import AuthoritativeExecutionPlanner
from source.product.question_routing import classify_question_intent, is_non_analytical_intent


DEFAULT_RUNTIME_DIR = RUNTIME_DIR
RUNTIME_TABLE = "rows"
EXECUTION_REQUIRED_INTENTS = {
    "rank_groups",
    "extremum",
    "histogram",
    "seasonality_heatmap",
    "growth",
    "temporal_trend",
    "chart_request",
    "constrained_aggregation",
    "shipping_delay",
    "binning",
}


class ExecutionContextUnavailableError(RuntimeError):
    """Raised when an analytical query requires raw executable rows."""

    def __init__(self, message: str, *, data_source_id: str | None = None, reason: str = "runtime_unavailable") -> None:
        super().__init__(message)
        self.data_source_id = data_source_id
        self.reason = reason

    def to_payload(self) -> dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "data_source_id": self.data_source_id,
            "reason": self.reason,
            "message": str(self),
        }


@dataclass
class DatasetExecutionContext:
    dataset_id: str
    dataset_runtime_reference: str
    schema_metadata: dict[str, Any] = field(default_factory=dict)
    profile_metadata: dict[str, Any] = field(default_factory=dict)
    executable_available: bool = False
    row_count: int = 0
    storage_reference: str = ""
    created_at: str = field(default_factory=lambda: utc_now().isoformat())
    last_loaded_at: str = field(default_factory=lambda: utc_now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_metadata(profile: DataSourceProfile | None) -> dict[str, Any]:
    if profile is None:
        return {}
    return {
        "row_count": int(profile.row_count),
        "column_count": int(profile.column_count),
        "generated_at": profile.generated_at.isoformat(),
    }


def schema_metadata(profile: DataSourceProfile | None, df: pd.DataFrame | None = None) -> dict[str, Any]:
    if profile is not None:
        return {
            "columns": [
                {
                    "name": column.name,
                    "dtype": column.dtype,
                    "nullable": column.nullable,
                    "unique_count": column.unique_count,
                }
                for column in profile.columns
            ],
            "row_count": int(profile.row_count),
            "column_count": int(profile.column_count),
        }
    if isinstance(df, pd.DataFrame):
        return {
            "columns": [{"name": str(column), "dtype": str(dtype)} for column, dtype in df.dtypes.items()],
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
        }
    return {"columns": [], "row_count": 0, "column_count": 0}


def persist_dataset_runtime(
    store: Any,
    data_source_id: str,
    df: pd.DataFrame,
    *,
    runtime_dir: str | Path = DEFAULT_RUNTIME_DIR,
) -> DatasetExecutionContext:
    source = store.get_data_source(data_source_id)
    directory = Path(runtime_dir)
    directory.mkdir(parents=True, exist_ok=True)
    storage_path = directory / f"{data_source_id}.sqlite"
    with sqlite3.connect(storage_path) as conn:
        df.to_sql(RUNTIME_TABLE, conn, if_exists="replace", index=False)
    try:
        profile = store.get_data_source_profile(data_source_id)
    except Exception:
        profile = None
    previous = source.metadata.get("execution_context") if isinstance(source.metadata, dict) else {}
    previous = previous if isinstance(previous, dict) else {}
    created_at = str(previous.get("created_at") or source.created_at.isoformat())
    context = DatasetExecutionContext(
        dataset_id=data_source_id,
        dataset_runtime_reference=f"sqlite:{storage_path}:{RUNTIME_TABLE}",
        schema_metadata=schema_metadata(profile, df),
        profile_metadata=profile_metadata(profile),
        executable_available=True,
        row_count=int(len(df)),
        storage_reference=str(storage_path),
        created_at=created_at,
        last_loaded_at=utc_now().isoformat(),
    )
    source.metadata = dict(source.metadata or {})
    source.metadata.update(context.to_dict())
    source.metadata["execution_context"] = context.to_dict()
    store.update_data_source(source)
    return context


def mark_profile_only_runtime(store: Any, data_source_id: str, *, reason: str = "raw_rows_unavailable") -> DatasetExecutionContext:
    source = store.get_data_source(data_source_id)
    try:
        profile = store.get_data_source_profile(data_source_id)
    except Exception:
        profile = None
    previous = source.metadata.get("execution_context") if isinstance(source.metadata, dict) else {}
    previous = previous if isinstance(previous, dict) else {}
    context = DatasetExecutionContext(
        dataset_id=data_source_id,
        dataset_runtime_reference="",
        schema_metadata=schema_metadata(profile),
        profile_metadata=profile_metadata(profile),
        executable_available=False,
        row_count=int(getattr(profile, "row_count", 0) or 0),
        storage_reference="",
        created_at=str(previous.get("created_at") or source.created_at.isoformat()),
        last_loaded_at=str(previous.get("last_loaded_at") or source.updated_at.isoformat()),
    )
    payload = context.to_dict()
    payload["unavailable_reason"] = reason
    source.metadata = dict(source.metadata or {})
    source.metadata.update(payload)
    source.metadata["execution_context"] = payload
    store.update_data_source(source)
    return context


def resolve_dataset_runtime(store: Any, data_source_id: str) -> tuple[pd.DataFrame, DatasetExecutionContext]:
    source = store.get_data_source(data_source_id)
    metadata = source.metadata if isinstance(source.metadata, dict) else {}
    context_payload = metadata.get("execution_context") if isinstance(metadata.get("execution_context"), dict) else metadata
    storage_reference = str(context_payload.get("storage_reference") or "")
    if storage_reference:
        storage_path = Path(storage_reference).expanduser()
        if not storage_path.is_absolute():
            storage_path = PROJECT_ROOT / storage_path
        if storage_path.exists() and storage_path.is_file():
            try:
                with sqlite3.connect(storage_path) as conn:
                    df = pd.read_sql_query(f'SELECT * FROM "{RUNTIME_TABLE}"', conn)
                if not isinstance(df, pd.DataFrame):
                    raise ValueError("runtime did not load as a dataframe")
                context = _context_from_payload(data_source_id, context_payload, df, storage_path)
                source.metadata = dict(metadata)
                source.metadata.update(context.to_dict())
                source.metadata["execution_context"] = context.to_dict()
                store.update_data_source(source)
                return df, context
            except Exception as exc:
                raise ExecutionContextUnavailableError(
                    "Executable dataset storage exists but could not be loaded.",
                    data_source_id=data_source_id,
                    reason=f"runtime_load_failed:{type(exc).__name__}",
                ) from exc

    df = resolve_dataframe_from_data_source(store, data_source_id)
    if isinstance(df, pd.DataFrame):
        context = persist_dataset_runtime(store, data_source_id, df)
        return df, context

    mark_profile_only_runtime(store, data_source_id)
    raise ExecutionContextUnavailableError(
        "Executable dataset rows are unavailable. The saved schema/profile can be used for metadata reasoning only.",
        data_source_id=data_source_id,
        reason="raw_rows_unavailable",
    )


def execution_context_failure_output(question: str, error: ExecutionContextUnavailableError | None = None) -> dict[str, Any]:
    payload = error.to_payload() if error else {
        "error_type": "ExecutionContextUnavailableError",
        "reason": "raw_rows_unavailable",
        "message": "Executable dataset rows are unavailable.",
    }
    try:
        failed_plan = AuthoritativeExecutionPlanner.plan(question, schema_only_dataframe_from_context(None)).to_payload()
    except Exception:
        failed_plan = {"raw_question": question, "execution_required": True}
    summary = (
        "I can identify the metric and grouping, but this investigation only has schema/profile metadata loaded. "
        "Raw rows are not currently attached, so I cannot compute this result. "
        "Reattach or re-upload the dataset to run the calculation."
    )
    return {
        "query": question,
        "summary": summary,
        "final_answer": summary,
        "key_findings": [],
        "evidence": [],
        "limitations": [payload["message"]],
        "next_steps": ["Reattach or reload the dataset so row-level execution can run."],
        "artifacts": [],
        "result_preview": "",
        "generated_code": "",
        "exec_error": "",
        "trace_metadata": {
            "error_type": "ExecutionContextUnavailableError",
            "execution_context_unavailable": True,
            "execution_required": True,
            "suppress_key_findings": True,
            "last_failed_plan": failed_plan,
            **payload,
        },
        "tool_timeline": [
            {
                "tool": "resolve_dataset_runtime",
                "status": "failed",
                "metadata": payload,
            }
        ],
    }


def execution_required_for_question(
    question: str,
    *,
    df: pd.DataFrame | None = None,
    data_context: dict[str, Any] | None = None,
) -> bool:
    if not str(question or "").strip():
        return False
    if is_non_analytical_intent(classify_question_intent(question)):
        return False
    planning_df = df if isinstance(df, pd.DataFrame) else schema_only_dataframe_from_context(data_context)
    try:
        plan = AuthoritativeExecutionPlanner.plan(question, planning_df)
        if plan.intent in EXECUTION_REQUIRED_INTENTS:
            return True
        if plan.transformation or plan.filters:
            return True
        if plan.intent == "general" and plan.aggregation in {"sum", "mean", "median", "count", "distribution"}:
            return True
    except Exception:
        pass
    normalized = " ".join(str(question or "").casefold().replace("_", " ").replace("-", " ").split())
    return any(
        marker in normalized
        for marker in (
            " by ",
            "group",
            "rank",
            "top",
            "lowest",
            "highest",
            "minimum",
            "maximum",
            "distribution",
            "histogram",
            "compare",
            "comparison",
            "trend",
            "over time",
            "time series",
            "transform",
            "relationship",
            "correlation",
            "anomaly",
            "anomalies",
            "outlier",
            "outliers",
            "unusual",
            "extreme",
            "aggregate",
            "average",
            "mean",
            "median",
            "sum",
            "total",
            "по ",
            "сравн",
            "тренд",
            "динамик",
            "распредел",
            "гистограмм",
            "аномал",
            "выброс",
            "необыч",
        )
    )


def execution_unavailable_in_context(data_context: dict[str, Any] | None) -> bool:
    context = data_context or {}
    failures = context.get("execution_context_failures")
    if isinstance(failures, list) and failures:
        return True
    execution_contexts = [item for item in context.get("execution_contexts") or [] if isinstance(item, dict)]
    if execution_contexts:
        return not any(item.get("executable_available") is True for item in execution_contexts)
    for item in context.get("execution_contexts") or []:
        if isinstance(item, dict) and item.get("executable_available") is False:
            return True
    for source_context in context.get("data_source_usage_contexts") or []:
        if not isinstance(source_context, dict):
            continue
        schema = source_context.get("schema_summary") if isinstance(source_context.get("schema_summary"), dict) else {}
        runtime = schema.get("execution_context") if isinstance(schema.get("execution_context"), dict) else {}
        if runtime.get("executable_available") is False:
            return True
    return False


def schema_only_dataframe_from_context(data_context: dict[str, Any] | None) -> pd.DataFrame | None:
    contexts = (data_context or {}).get("data_source_usage_contexts") or []
    if not isinstance(contexts, list):
        return None
    columns: list[str] = []
    for context in contexts:
        if not isinstance(context, dict):
            continue
        summaries = context.get("column_summaries") or []
        if isinstance(summaries, list):
            for item in summaries:
                if isinstance(item, dict) and str(item.get("name") or "").strip():
                    columns.append(str(item["name"]))
        schema = context.get("schema_summary") if isinstance(context.get("schema_summary"), dict) else {}
        for item in schema.get("columns", []) if isinstance(schema.get("columns"), list) else []:
            if isinstance(item, dict) and str(item.get("name") or "").strip():
                columns.append(str(item["name"]))
    unique = list(dict.fromkeys(columns))
    return pd.DataFrame(columns=unique) if unique else None


def _context_from_payload(
    data_source_id: str,
    payload: dict[str, Any],
    df: pd.DataFrame,
    storage_path: Path,
) -> DatasetExecutionContext:
    return DatasetExecutionContext(
        dataset_id=str(payload.get("dataset_id") or data_source_id),
        dataset_runtime_reference=str(payload.get("dataset_runtime_reference") or f"sqlite:{storage_path}:{RUNTIME_TABLE}"),
        schema_metadata=dict(payload.get("schema_metadata") or schema_metadata(None, df)),
        profile_metadata=dict(payload.get("profile_metadata") or {}),
        executable_available=True,
        row_count=int(len(df)),
        storage_reference=str(storage_path),
        created_at=str(payload.get("created_at") or utc_now().isoformat()),
        last_loaded_at=utc_now().isoformat(),
    )
