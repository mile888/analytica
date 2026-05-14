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
    InvestigationStatus,
)
from source.product.sqlite_store import SQLiteInvestigationStore
from source.product.store_factory import create_investigation_store


def test_sqlite_store_creates_db_and_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "investigations.sqlite"

    SQLiteInvestigationStore(db_path)

    assert db_path.exists()


def test_sqlite_create_list_get_persists_after_new_store_instance(tmp_path: Path) -> None:
    db_path = tmp_path / "investigations.sqlite"
    store = SQLiteInvestigationStore(db_path)
    investigation = store.create_investigation("Как режим доставки связан с объемом продаж?")
    store.update_status(investigation.investigation_id, InvestigationStatus.NEEDS_REVIEW)

    reloaded_store = SQLiteInvestigationStore(db_path)
    loaded = reloaded_store.get_investigation(investigation.investigation_id)
    listed = reloaded_store.list_investigations()

    assert loaded.user_question == investigation.user_question
    assert loaded.status == InvestigationStatus.NEEDS_REVIEW
    assert listed[0].investigation_id == investigation.investigation_id


def test_sqlite_artifact_visibility_saved_and_restored(tmp_path: Path) -> None:
    store = SQLiteInvestigationStore(tmp_path / "investigations.sqlite")
    investigation = store.create_investigation("Question")
    run = InvestigationRun(status=InvestigationStatus.NEEDS_REVIEW)
    store.add_run(investigation.investigation_id, run)
    artifact = Artifact(
        artifact_type=ArtifactType.PYTHON_CODE,
        title="Generated Python",
        content="result = df.head()",
        visibility=ArtifactVisibility.TECHNICAL,
        pinned=True,
        run_id=run.run_id,
    )
    store.add_artifact(investigation.investigation_id, artifact)

    reloaded = SQLiteInvestigationStore(tmp_path / "investigations.sqlite").get_investigation(
        investigation.investigation_id
    )

    assert reloaded.artifacts[0].visibility == ArtifactVisibility.TECHNICAL
    assert reloaded.artifacts[0].pinned is True
    assert reloaded.artifacts[0].run_id == run.run_id


def test_adapter_marks_technical_artifacts_correctly() -> None:
    update = agent_output_to_investigation_update(
        {
            "structured_report": {"summary": "Answer", "key_findings": ["Finding"]},
            "generated_code": "result = df.head()",
            "sql_metadata": {"query": "SELECT * FROM data"},
            "tool_timeline": [{"tool": "run", "status": "ok"}],
        }
    )

    by_title = {artifact.title: artifact for artifact in update.artifacts}

    assert by_title["Decision report"].visibility == ArtifactVisibility.USER
    assert by_title["Generated Python"].visibility == ArtifactVisibility.TECHNICAL
    assert by_title["SQL context"].visibility == ArtifactVisibility.TECHNICAL
    assert by_title["Run trace"].visibility == ArtifactVisibility.TECHNICAL


def test_decision_report_structured_fields_are_present(tmp_path: Path) -> None:
    store = SQLiteInvestigationStore(tmp_path / "investigations.sqlite")
    investigation = store.create_investigation("Question")
    run = InvestigationRun(status=InvestigationStatus.NEEDS_REVIEW)
    store.add_run(investigation.investigation_id, run)
    report = DecisionReport(
        question="Question",
        answer="Answer",
        key_findings=["Finding"],
        evidence=["Evidence"],
        limitations=["Limitation"],
        next_steps=["Next step"],
        artifact_ids=["artifact_1"],
        run_id=run.run_id,
    )
    store.set_report(investigation.investigation_id, report)

    loaded = SQLiteInvestigationStore(tmp_path / "investigations.sqlite").get_investigation(
        investigation.investigation_id
    )

    assert loaded.report is not None
    assert loaded.report.question == "Question"
    assert loaded.report.answer == "Answer"
    assert loaded.report.key_findings == ["Finding"]
    assert loaded.report.evidence == ["Evidence"]
    assert loaded.report.artifact_ids == ["artifact_1"]


def test_review_status_updates_persist(tmp_path: Path) -> None:
    store = SQLiteInvestigationStore(tmp_path / "investigations.sqlite")
    investigation = store.create_investigation("Question")
    finding = Finding(text="Finding", title="Finding")
    store.add_finding(investigation.investigation_id, finding)

    store.update_finding_status(investigation.investigation_id, finding.finding_id, FindingStatus.ACCEPTED)
    store.update_status(investigation.investigation_id, InvestigationStatus.VERIFIED)

    loaded = SQLiteInvestigationStore(tmp_path / "investigations.sqlite").get_investigation(
        investigation.investigation_id
    )

    assert loaded.status == InvestigationStatus.VERIFIED
    assert loaded.findings[0].status == FindingStatus.ACCEPTED


def test_store_factory_uses_memory_and_sqlite(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANALYTICA_INVESTIGATION_STORE", "memory")
    memory_store = create_investigation_store()
    assert memory_store.__class__.__name__ == "InvestigationStore"

    monkeypatch.setenv("ANALYTICA_INVESTIGATION_STORE", "sqlite")
    monkeypatch.setenv("ANALYTICA_INVESTIGATION_DB_PATH", str(tmp_path / "factory.sqlite"))
    sqlite_store = create_investigation_store()
    assert isinstance(sqlite_store, SQLiteInvestigationStore)


def test_sqlite_store_creates_schema_migrations(tmp_path: Path) -> None:
    db_path = tmp_path / "investigations.sqlite"
    store = SQLiteInvestigationStore(db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
        ).fetchone()

    assert row is not None
    assert store.get_schema_version() == 11


def test_reopening_old_db_without_migration_table_does_not_fail(tmp_path: Path) -> None:
    db_path = tmp_path / "old.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE investigations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_sources_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )

    store = SQLiteInvestigationStore(db_path)

    assert store.get_schema_version() == 11
    investigation = store.create_investigation("Question")
    assert store.get_investigation(investigation.investigation_id).user_question == "Question"
