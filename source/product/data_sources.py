from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from source.product.investigation import new_id, utc_now


class DataSourceType(StrEnum):
    CSV = "csv"
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    DUCKDB = "duckdb"
    UNKNOWN = "unknown"


class DataSourceStatus(StrEnum):
    ACTIVE = "active"
    STALE = "stale"
    ERROR = "error"
    ARCHIVED = "archived"


class ColumnInferredRole(StrEnum):
    METRIC = "metric"
    DIMENSION = "dimension"
    TIMESTAMP = "timestamp"
    IDENTIFIER = "identifier"
    TEXT = "text"
    UNKNOWN = "unknown"


class ColumnSemanticRole(StrEnum):
    METRIC = "metric"
    DIMENSION = "dimension"
    TIMESTAMP = "timestamp"
    IDENTIFIER = "identifier"
    TEXT = "text"
    TARGET = "target"
    UNKNOWN = "unknown"


@dataclass
class DataSourceColumn:
    name: str
    dtype: str = ""
    nullable: bool = False
    unique_count: int | None = None
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class DataSourceProfile:
    row_count: int = 0
    column_count: int = 0
    columns: list[DataSourceColumn] = field(default_factory=list)
    missing_summary: dict[str, int] = field(default_factory=dict)
    numeric_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    categorical_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    sampled_rows: list[dict[str, Any]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=utc_now)


@dataclass
class ColumnUsageSummary:
    name: str
    dtype: str = ""
    nullable: bool = False
    unique_count: int | None = None
    sample_values: list[Any] = field(default_factory=list)
    inferred_role: ColumnInferredRole | ColumnSemanticRole = ColumnInferredRole.UNKNOWN
    notes: list[str] = field(default_factory=list)


@dataclass
class DataSourceUsageContext:
    data_source_id: str
    name: str
    type: DataSourceType = DataSourceType.UNKNOWN
    status: DataSourceStatus = DataSourceStatus.ACTIVE
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    schema_summary: dict[str, Any] = field(default_factory=dict)
    column_summaries: list[ColumnUsageSummary] = field(default_factory=list)
    sample_rows: list[dict[str, Any]] = field(default_factory=list)
    missing_summary: dict[str, int] = field(default_factory=dict)
    numeric_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    categorical_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    freshness: dict[str, Any] = field(default_factory=dict)
    linked_investigation_ids: list[str] = field(default_factory=list)
    previous_questions: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=utc_now)


@dataclass
class ColumnSemanticNote:
    column_name: str
    display_name: str | None = None
    description: str | None = None
    business_meaning: str | None = None
    semantic_role: ColumnSemanticRole = ColumnSemanticRole.UNKNOWN
    caveats: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass
class DataSourceSemanticNotes:
    data_source_id: str
    source_description: str | None = None
    business_context: str | None = None
    global_caveats: list[str] = field(default_factory=list)
    column_notes: list[ColumnSemanticNote] = field(default_factory=list)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class DataSource:
    name: str
    data_source_type: DataSourceType = DataSourceType.UNKNOWN
    status: DataSourceStatus = DataSourceStatus.ACTIVE
    location: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    linked_investigation_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    data_source_id: str = field(default_factory=lambda: new_id("ds"))
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    @property
    def id(self) -> str:
        return self.data_source_id
