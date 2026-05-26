from __future__ import annotations

from pathlib import Path
import sqlite3

from source.product.adapter import agent_output_to_investigation_update
from source.product.investigation import (
    Artifact,
    ArtifactType,
    ArtifactVisibility,
    DecisionReport,
    Finding,
    FindingStatus,
    InvestigationRun,
    InvestigationMemoryItem,
    InvestigationMemoryStatus,
    InvestigationMemoryType,
    InvestigationStatus,
)
from source.product.sqlite_store import SQLiteInvestigationStore
from source.product.store_factory import create_investigation_store

def test_sqlite_create_list_get_persists_after_new_store_instance(tmp_path: Path) -> None:
    db_path = tmp_path / "investigations.sqlite"
    store = SQLiteInvestigationStore(db_path)
    investigation = store.create_investigation("Какие группы отличаются по выбранной метрике?")
    store.update_status(investigation.investigation_id, InvestigationStatus.NEEDS_REVIEW)

    reloaded_store = SQLiteInvestigationStore(db_path)
    loaded = reloaded_store.get_investigation(investigation.investigation_id)
    listed = reloaded_store.list_investigations()

    assert loaded.user_question == investigation.user_question
    assert loaded.status == InvestigationStatus.NEEDS_REVIEW
    assert listed[0].investigation_id == investigation.investigation_id

def test_sqlite_txt_content_migration_is_idempotent_for_partially_updated_db(tmp_path: Path) -> None:
    db_path = tmp_path / "partial.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )
        for version in range(1, 13):
            conn.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                (version, f"migration_{version}", "2026-01-01T00:00:00"),
            )
        conn.execute(
            """
            CREATE TABLE final_report_snapshots (
                id TEXT PRIMARY KEY,
                report_id TEXT NOT NULL,
                investigation_id TEXT NOT NULL,
                report_version INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL,
                status TEXT NOT NULL,
                markdown_content TEXT NOT NULL,
                html_content TEXT NOT NULL,
                txt_content TEXT NOT NULL DEFAULT '',
                readiness_snapshot_json TEXT NOT NULL DEFAULT '{}',
                approval_status TEXT NOT NULL,
                approved_at TEXT,
                approved_by TEXT,
                source_report_json TEXT NOT NULL DEFAULT '{}',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                decision_metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )

    store = SQLiteInvestigationStore(db_path)

    assert store.get_schema_version() == 15
    with sqlite3.connect(db_path) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(final_report_snapshots)").fetchall()]
    assert columns.count("txt_content") == 1
