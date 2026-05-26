from __future__ import annotations

import sqlite3

from source.product.investigation import utc_now


V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS investigations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    data_sources_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error_message TEXT,
    raw_output_json TEXT NOT NULL DEFAULT '{}',
    trace_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    content_json TEXT NOT NULL DEFAULT 'null',
    visibility TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    path TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    evidence_artifact_ids_json TEXT NOT NULL DEFAULT '[]',
    evidence_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL,
    status TEXT NOT NULL DEFAULT 'proposed',
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    key_findings_json TEXT NOT NULL DEFAULT '[]',
    evidence_json TEXT NOT NULL DEFAULT '[]',
    limitations_json TEXT NOT NULL DEFAULT '[]',
    next_steps_json TEXT NOT NULL DEFAULT '[]',
    artifact_ids_json TEXT NOT NULL DEFAULT '[]',
    content TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
"""


MIGRATIONS: list[tuple[int, str, str]] = [
    (1, "initial_product_schema", V1_SCHEMA),
    (
        2,
        "shareable_reports",
        """
        CREATE TABLE IF NOT EXISTS shareable_reports (
            id TEXT PRIMARY KEY,
            investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            template TEXT NOT NULL,
            status TEXT NOT NULL,
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            sections_json TEXT NOT NULL DEFAULT '[]',
            source_finding_ids_json TEXT NOT NULL DEFAULT '[]',
            source_artifact_ids_json TEXT NOT NULL DEFAULT '[]',
            include_technical INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        """,
    ),
    (
        3,
        "shareable_report_versioning",
        """
        -- Applied through _add_column_if_missing because SQLite cannot
        -- run ALTER TABLE ADD COLUMN idempotently.
        """,
    ),
    (
        4,
        "shareable_report_review_annotations",
        """
        CREATE TABLE IF NOT EXISTS report_comments (
            id TEXT PRIMARY KEY,
            report_id TEXT NOT NULL REFERENCES shareable_reports(id) ON DELETE CASCADE,
            section_id TEXT NOT NULL,
            text TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            resolved_at TEXT,
            author TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        """,
    ),
    (
        5,
        "report_readiness_and_section_review_status",
        """
        -- Section review status is persisted inside shareable_reports.sections_json.
        -- Readiness results are computed on demand and are not stored yet.
        """,
    ),
    (
        6,
        "final_report_snapshots",
        """
        CREATE TABLE IF NOT EXISTS final_report_snapshots (
            id TEXT PRIMARY KEY,
            report_id TEXT NOT NULL REFERENCES shareable_reports(id) ON DELETE CASCADE,
            investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
            report_version INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            created_by TEXT NOT NULL,
            status TEXT NOT NULL,
            markdown_content TEXT NOT NULL,
            html_content TEXT NOT NULL,
            readiness_snapshot_json TEXT NOT NULL DEFAULT '{}',
            approval_status TEXT NOT NULL,
            approved_at TEXT,
            approved_by TEXT,
            source_report_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        """,
    ),
    (
        7,
        "decision_library_metadata",
        """
        -- Applied through _add_column_if_missing for existing local databases.
        """,
    ),
    (
        8,
        "data_source_registry",
        """
        CREATE TABLE IF NOT EXISTS data_sources (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL,
            location TEXT,
            description TEXT,
            tags_json TEXT NOT NULL DEFAULT '[]',
            linked_investigation_ids_json TEXT NOT NULL DEFAULT '[]',
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS data_source_profiles (
            data_source_id TEXT PRIMARY KEY REFERENCES data_sources(id) ON DELETE CASCADE,
            profile_json TEXT NOT NULL DEFAULT '{}',
            generated_at TEXT NOT NULL
        );
        """,
    ),
    (
        9,
        "data_source_semantic_notes",
        """
        CREATE TABLE IF NOT EXISTS data_source_semantic_notes (
            data_source_id TEXT PRIMARY KEY REFERENCES data_sources(id) ON DELETE CASCADE,
            notes_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        );
        """,
    ),
    (
        10,
        "api_driven_investigation_runs",
        """
        CREATE TABLE IF NOT EXISTS investigation_runs (
            id TEXT PRIMARY KEY,
            investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            status TEXT NOT NULL,
            current_stage TEXT NOT NULL,
            data_source_ids_json TEXT NOT NULL DEFAULT '[]',
            run_context_summary_json TEXT NOT NULL DEFAULT '{}',
            error_message TEXT,
            artifact_ids_json TEXT NOT NULL DEFAULT '[]',
            report_ids_json TEXT NOT NULL DEFAULT '[]',
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        """,
    ),
    (
        11,
        "investigation_run_events",
        """
        CREATE TABLE IF NOT EXISTS investigation_run_events (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES investigation_runs(id) ON DELETE CASCADE,
            investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            stage TEXT,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            severity TEXT NOT NULL
        );
        """,
    ),
    (
        12,
        "investigation_messages",
        """
        CREATE TABLE IF NOT EXISTS investigation_messages (
            id TEXT PRIMARY KEY,
            investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
            run_id TEXT REFERENCES investigation_runs(id) ON DELETE SET NULL,
            role TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        """,
    ),
    (
        13,
        "final_report_txt_content",
        """
        -- Applied through _add_column_if_missing because SQLite cannot
        -- run ALTER TABLE ADD COLUMN idempotently.
        """,
    ),
    (
        14,
        "investigation_memory",
        """
        CREATE TABLE IF NOT EXISTS investigation_memory (
            id TEXT PRIMARY KEY,
            investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
            type TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        """,
    ),
]


def ensure_migration_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )


def get_schema_version(conn: sqlite3.Connection) -> int:
    ensure_migration_table(conn)
    row = conn.execute("SELECT MAX(version) AS version FROM schema_migrations").fetchone()
    if row is None:
        return 0
    value = row["version"] if isinstance(row, sqlite3.Row) else row[0]
    return int(value or 0)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND lower(name) = lower(?)",
        (table,),
    ).fetchone()
    return row is not None


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        raise RuntimeError(f"Cannot add column to missing SQLite table: {table}")
    quoted_table = _quote_identifier(table)
    return {str(row[1]).lower() for row in conn.execute(f"PRAGMA table_info({quoted_table})").fetchall()}


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    if column.lower() not in _column_names(conn, table):
        conn.execute(f"ALTER TABLE {_quote_identifier(table)} ADD COLUMN {definition}")


def _apply_migration(conn: sqlite3.Connection, version: int, sql: str) -> None:
    if version not in {3, 4, 7, 8, 13}:
        conn.executescript(sql)
        return

    if version == 13:
        _add_column_if_missing(
            conn,
            "final_report_snapshots",
            "txt_content",
            "txt_content TEXT NOT NULL DEFAULT ''",
        )
        return

    if version == 3:
        _add_column_if_missing(conn, "shareable_reports", "previous_version_id", "previous_version_id TEXT")
        _add_column_if_missing(
            conn,
            "shareable_reports",
            "is_latest",
            "is_latest INTEGER NOT NULL DEFAULT 1",
        )
        _add_column_if_missing(
            conn,
            "shareable_reports",
            "version_note",
            "version_note TEXT NOT NULL DEFAULT ''",
        )
        return

    conn.executescript(sql)
    if version == 8:
        _add_column_if_missing(
            conn,
            "investigations",
            "linked_data_source_ids_json",
            "linked_data_source_ids_json TEXT NOT NULL DEFAULT '[]'",
        )
        return
    _add_column_if_missing(
        conn,
        "shareable_reports",
        "approval_status",
        "approval_status TEXT NOT NULL DEFAULT 'draft'",
    )
    _add_column_if_missing(
        conn,
        "shareable_reports",
        "reviewer_notes",
        "reviewer_notes TEXT NOT NULL DEFAULT ''",
    )
    _add_column_if_missing(conn, "shareable_reports", "approved_at", "approved_at TEXT")
    _add_column_if_missing(conn, "shareable_reports", "approved_by", "approved_by TEXT")
    if version == 4:
        return

    _add_column_if_missing(
        conn,
        "final_report_snapshots",
        "decision_metadata_json",
        "decision_metadata_json TEXT NOT NULL DEFAULT '{}'",
    )


def apply_migrations(conn: sqlite3.Connection) -> None:
    ensure_migration_table(conn)
    current_version = get_schema_version(conn)
    for version, name, sql in MIGRATIONS:
        if version <= current_version:
            continue
        _apply_migration(conn, version, sql)
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
            (version, name, utc_now().isoformat()),
        )
