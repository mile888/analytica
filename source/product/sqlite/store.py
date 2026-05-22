from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from source.product.data_sources import (
    ColumnSemanticNote,
    DataSource,
    DataSourceProfile,
    DataSourceSemanticNotes,
    DataSourceStatus,
    DataSourceType,
)
from source.product.investigation import (
    Artifact,
    ArtifactType,
    ArtifactVisibility,
    DecisionReport,
    FinalReportSnapshot,
    FinalReportStatus,
    Finding,
    FindingStatus,
    Investigation,
    InvestigationMessage,
    InvestigationMessageRole,
    InvestigationMessageType,
    InvestigationMemoryItem,
    InvestigationMemoryStatus,
    InvestigationMemoryType,
    InvestigationRun,
    InvestigationRunEvent,
    InvestigationRunEventSeverity,
    InvestigationRunEventType,
    InvestigationRunStage,
    InvestigationRunStatus,
    InvestigationStatus,
    ReportApprovalStatus,
    ReportComment,
    ReportCommentStatus,
    ShareableReport,
    ShareableReportStatus,
    ShareableReportTemplate,
    new_id,
    utc_now,
)
from source.product.migrations import apply_migrations, get_schema_version
from source.product.file_storage import delete_uploaded_file_if_safe
from source.product.store import InvestigationStore
from source.product.sqlite.serialization import (
    _decision_metadata_from_dict,
    _decision_metadata_to_dict,
    _dt,
    _json,
    _loads,
    _normalize_tags,
    _parse_dt,
    _profile_from_dict,
    _profile_to_dict,
    _row_value,
    _section_from_dict,
    _section_to_dict,
    _semantic_notes_from_dict,
    _semantic_notes_to_dict,
)


DEFAULT_INVESTIGATION_DB_PATH = ".analytica/investigations.sqlite"


class SQLiteInvestigationStore:
    """SQLite-backed repository with the same public API as InvestigationStore."""

    def __init__(self, db_path: str | Path = DEFAULT_INVESTIGATION_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def create_investigation(self, question: str, title: Optional[str] = None) -> Investigation:
        resolved_title = title or InvestigationStore._title_from_question(question)
        investigation = Investigation(title=resolved_title, user_question=question)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO investigations
                (id, title, question, status, created_at, updated_at, data_sources_json,
                 linked_data_source_ids_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    investigation.investigation_id,
                    investigation.title,
                    investigation.user_question,
                    investigation.status.value,
                    _dt(investigation.created_at),
                    _dt(investigation.updated_at),
                    _json(investigation.data_sources),
                    _json(investigation.linked_data_source_ids),
                    _json(investigation.metadata),
                ),
            )
        return investigation

    def link_data_source_to_investigation(self, investigation_id: str, data_source_id: str) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        data_source = self.get_data_source(data_source_id)
        linked_ids = list(dict.fromkeys([*investigation.linked_data_source_ids, data_source_id]))
        data_sources = list(dict.fromkeys([*investigation.data_sources, data_source_id]))
        source_links = list(dict.fromkeys([*data_source.linked_investigation_ids, investigation_id]))
        data_source.linked_investigation_ids = source_links
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE investigations
                SET linked_data_source_ids_json = ?, data_sources_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (_json(linked_ids), _json(data_sources), _dt(utc_now()), investigation_id),
            )
            self._upsert_data_source(conn, data_source)
        return self.get_investigation(investigation_id)

    def update_investigation_metadata(self, investigation_id: str, metadata: dict) -> Investigation:
        self.get_investigation(investigation_id)
        with self._connect() as conn:
            conn.execute(
                "UPDATE investigations SET metadata_json = ?, updated_at = ? WHERE id = ?",
                (_json(metadata or {}), _dt(utc_now()), investigation_id),
            )
        return self.get_investigation(investigation_id)

    def get_investigation(self, investigation_id: str) -> Investigation:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM investigations WHERE id = ?", (investigation_id,)).fetchone()
            if row is None:
                raise KeyError(f"Investigation not found: {investigation_id}")
            return self._hydrate_investigation(conn, row)

    def list_investigations(self) -> list[Investigation]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM investigations ORDER BY updated_at DESC").fetchall()
            return [self._hydrate_investigation(conn, row) for row in rows]

    def update_status(self, investigation_id: str, status: InvestigationStatus | str) -> Investigation:
        resolved = InvestigationStatus(status)
        self._touch_investigation(investigation_id, status=resolved.value)
        return self.get_investigation(investigation_id)

    def delete_investigation(self, investigation_id: str) -> None:
        self.get_investigation(investigation_id)
        with self._connect() as conn:
            sources = conn.execute("SELECT * FROM data_sources").fetchall()
            for row in sources:
                source = self._data_source_from_row(row)
                if investigation_id in source.linked_investigation_ids:
                    source.linked_investigation_ids = [
                        item for item in source.linked_investigation_ids if item != investigation_id
                    ]
                    source.updated_at = utc_now()
                    self._upsert_data_source(conn, source)
            conn.execute("DELETE FROM investigations WHERE id = ?", (investigation_id,))

    def create_investigation_run(self, run: InvestigationRun) -> InvestigationRun:
        self.get_investigation(run.investigation_id)
        run.status = InvestigationRunStatus(run.status)
        with self._connect() as conn:
            self._upsert_investigation_run(conn, run)
        return self.get_investigation_run(run.run_id)

    def update_investigation_run(self, run: InvestigationRun) -> InvestigationRun:
        self.get_investigation_run(run.run_id)
        if isinstance(run.status, str):
            run.status = InvestigationRunStatus(run.status)
        with self._connect() as conn:
            self._upsert_investigation_run(conn, run)
        return self.get_investigation_run(run.run_id)

    def get_investigation_run(self, run_id: str) -> InvestigationRun:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM investigation_runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"InvestigationRun not found: {run_id}")
            return self._investigation_run_from_row(row)

    def list_investigation_runs(self) -> list[InvestigationRun]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM investigation_runs ORDER BY created_at DESC").fetchall()
            return [self._investigation_run_from_row(row) for row in rows]

    def list_runs_for_investigation(self, investigation_id: str) -> list[InvestigationRun]:
        self.get_investigation(investigation_id)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM investigation_runs WHERE investigation_id = ? ORDER BY created_at DESC",
                (investigation_id,),
            ).fetchall()
            return [self._investigation_run_from_row(row) for row in rows]

    def add_investigation_run_event(self, event: InvestigationRunEvent) -> InvestigationRunEvent:
        self.get_investigation_run(event.run_id)
        self.get_investigation(event.investigation_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO investigation_run_events
                (id, run_id, investigation_id, event_type, stage, message, created_at, metadata_json, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.run_id,
                    event.investigation_id,
                    event.event_type.value,
                    event.stage.value if event.stage else None,
                    event.message,
                    _dt(event.created_at),
                    _json(event.metadata),
                    event.severity.value,
                ),
            )
        return event

    def list_investigation_run_events(self, run_id: str) -> list[InvestigationRunEvent]:
        self.get_investigation_run(run_id)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM investigation_run_events WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
            return [self._investigation_run_event_from_row(row) for row in rows]

    def list_events_for_investigation(
        self,
        investigation_id: str,
        limit: int = 100,
    ) -> list[InvestigationRunEvent]:
        self.get_investigation(investigation_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM investigation_run_events
                WHERE investigation_id = ?
                ORDER BY created_at ASC
                """,
                (investigation_id,),
            ).fetchall()
            return [self._investigation_run_event_from_row(row) for row in rows][-limit:]

    def add_investigation_message(self, message: InvestigationMessage) -> InvestigationMessage:
        self.get_investigation(message.investigation_id)
        if message.run_id:
            self.get_investigation_run(message.run_id)
        message.role = InvestigationMessageRole(message.role)
        message.message_type = InvestigationMessageType(message.message_type)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO investigation_messages
                (id, investigation_id, run_id, role, type, content, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.message_id,
                    message.investigation_id,
                    message.run_id,
                    message.role.value,
                    message.message_type.value,
                    message.content,
                    _dt(message.created_at),
                    _json(message.metadata),
                ),
            )
            self._touch_investigation_conn(conn, message.investigation_id)
        return self.get_investigation_message(message.message_id)

    def list_investigation_messages(
        self,
        investigation_id: str,
        limit: int | None = None,
    ) -> list[InvestigationMessage]:
        self.get_investigation(investigation_id)
        with self._connect() as conn:
            sql = "SELECT * FROM investigation_messages WHERE investigation_id = ? ORDER BY created_at ASC"
            rows = conn.execute(sql, (investigation_id,)).fetchall()
            messages = [self._investigation_message_from_row(row) for row in rows]
            return messages[-limit:] if limit else messages

    def get_investigation_message(self, message_id: str) -> InvestigationMessage:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM investigation_messages WHERE id = ?", (message_id,)).fetchone()
            if row is None:
                raise KeyError(f"InvestigationMessage not found: {message_id}")
            return self._investigation_message_from_row(row)

    def add_investigation_memory_item(self, item: InvestigationMemoryItem) -> InvestigationMemoryItem:
        self.get_investigation(item.investigation_id)
        item.memory_type = InvestigationMemoryType(item.memory_type)
        item.status = InvestigationMemoryStatus(item.status)
        item.updated_at = utc_now()
        with self._connect() as conn:
            self._upsert_investigation_memory_item(conn, item)
            self._touch_investigation_conn(conn, item.investigation_id)
        return self.get_investigation_memory_item(item.memory_id)

    def list_investigation_memory(
        self,
        investigation_id: str,
        memory_type: InvestigationMemoryType | str | None = None,
    ) -> list[InvestigationMemoryItem]:
        self.get_investigation(investigation_id)
        with self._connect() as conn:
            if memory_type:
                resolved = InvestigationMemoryType(memory_type)
                rows = conn.execute(
                    """
                    SELECT * FROM investigation_memory
                    WHERE investigation_id = ? AND type = ?
                    ORDER BY updated_at DESC
                    """,
                    (investigation_id, resolved.value),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM investigation_memory
                    WHERE investigation_id = ?
                    ORDER BY updated_at DESC
                    """,
                    (investigation_id,),
                ).fetchall()
            return [self._investigation_memory_item_from_row(row) for row in rows]

    def get_investigation_memory_item(self, memory_id: str) -> InvestigationMemoryItem:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM investigation_memory WHERE id = ?", (memory_id,)).fetchone()
            if row is None:
                raise KeyError(f"InvestigationMemoryItem not found: {memory_id}")
            return self._investigation_memory_item_from_row(row)

    def update_investigation_memory_item(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        title: str | None = None,
        status: InvestigationMemoryStatus | str | None = None,
        metadata: dict | None = None,
    ) -> InvestigationMemoryItem:
        item = self.get_investigation_memory_item(memory_id)
        if content is not None:
            item.content = content
        if title is not None:
            item.title = title
        if status is not None:
            item.status = InvestigationMemoryStatus(status)
        if metadata is not None:
            item.metadata = metadata
        item.updated_at = utc_now()
        with self._connect() as conn:
            self._upsert_investigation_memory_item(conn, item)
            self._touch_investigation_conn(conn, item.investigation_id)
        return self.get_investigation_memory_item(memory_id)

    def add_run(self, investigation_id: str, run: InvestigationRun) -> Investigation:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (id, investigation_id, status, started_at, finished_at, error_message, raw_output_json, trace_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    investigation_id,
                    run.status.value,
                    _dt(run.started_at),
                    _dt(run.finished_at),
                    run.error,
                    _json(run.output),
                    _json(run.trace),
                    _json(run.metadata),
                ),
            )
            self._touch_investigation_conn(conn, investigation_id)
        return self.get_investigation(investigation_id)

    def add_artifact(self, investigation_id: str, artifact: Artifact) -> Investigation:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifacts
                (id, investigation_id, run_id, type, title, content_json, visibility, created_at, pinned, path, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.artifact_id,
                    investigation_id,
                    artifact.run_id,
                    artifact.artifact_type.value,
                    artifact.title,
                    _json(artifact.content),
                    artifact.visibility.value,
                    _dt(artifact.created_at),
                    1 if artifact.pinned else 0,
                    artifact.path,
                    _json(artifact.metadata),
                ),
            )
            self._touch_investigation_conn(conn, investigation_id)
        return self.get_investigation(investigation_id)

    def add_finding(self, investigation_id: str, finding: Finding) -> Investigation:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO findings
                (id, investigation_id, run_id, title, text, evidence_artifact_ids_json, evidence_json,
                 confidence, status, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    finding.finding_id,
                    investigation_id,
                    finding.run_id,
                    finding.title,
                    finding.text,
                    _json(finding.evidence_artifact_ids),
                    _json(finding.evidence),
                    finding.confidence,
                    finding.status.value,
                    _dt(finding.created_at),
                    _json(finding.metadata),
                ),
            )
            self._touch_investigation_conn(conn, investigation_id)
        return self.get_investigation(investigation_id)

    def update_finding(self, investigation_id: str, finding: Finding) -> Investigation:
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE findings
                SET title = ?, text = ?, evidence_artifact_ids_json = ?, evidence_json = ?,
                    confidence = ?, status = ?, metadata_json = ?
                WHERE id = ? AND investigation_id = ?
                """,
                (
                    finding.title,
                    finding.text,
                    _json(finding.evidence_artifact_ids),
                    _json(finding.evidence),
                    finding.confidence,
                    finding.status.value,
                    _json(finding.metadata),
                    finding.finding_id,
                    investigation_id,
                ),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Finding not found: {finding.finding_id}")
            self._touch_investigation_conn(conn, investigation_id)
        return self.get_investigation(investigation_id)

    def add_finding_evidence_link(
        self,
        investigation_id: str,
        finding_id: str,
        evidence: dict,
    ) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        for finding in investigation.findings:
            if finding.finding_id == finding_id:
                links = list(finding.metadata.get("linked_evidence") or [])
                key = (evidence.get("type"), evidence.get("id"))
                if key not in {(item.get("type"), item.get("id")) for item in links}:
                    links.append(evidence)
                finding.metadata["linked_evidence"] = links
                return self.update_finding(investigation_id, finding)
        raise KeyError(f"Finding not found: {finding_id}")

    def set_report(self, investigation_id: str, report: DecisionReport) -> Investigation:
        report.updated_at = utc_now()
        with self._connect() as conn:
            conn.execute("DELETE FROM reports WHERE investigation_id = ?", (investigation_id,))
            conn.execute(
                """
                INSERT INTO reports
                (id, investigation_id, run_id, question, answer, key_findings_json, evidence_json,
                 limitations_json, next_steps_json, artifact_ids_json, content, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    investigation_id,
                    report.run_id,
                    report.question,
                    report.answer or report.summary,
                    _json(report.key_findings),
                    _json(report.evidence),
                    _json(report.limitations),
                    _json(report.next_steps),
                    _json(report.artifact_ids),
                    report.content,
                    _dt(report.created_at),
                    _json(report.metadata),
                ),
            )
            self._touch_investigation_conn(conn, investigation_id)
        return self.get_investigation(investigation_id)

    def replace_data_sources(self, investigation_id: str, data_sources: list[str]) -> Investigation:
        with self._connect() as conn:
            conn.execute(
                "UPDATE investigations SET data_sources_json = ?, updated_at = ? WHERE id = ?",
                (_json(data_sources), _dt(utc_now()), investigation_id),
            )
        return self.get_investigation(investigation_id)

    def update_finding_status(
        self,
        investigation_id: str,
        finding_id: str,
        status: FindingStatus | str,
    ) -> Investigation:
        resolved = FindingStatus(status)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE findings SET status = ? WHERE id = ? AND investigation_id = ?",
                (resolved.value, finding_id, investigation_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Finding not found: {finding_id}")
            self._touch_investigation_conn(conn, investigation_id)
        return self.get_investigation(investigation_id)

    def set_artifact_pinned(self, investigation_id: str, artifact_id: str, pinned: bool) -> Investigation:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE artifacts SET pinned = ? WHERE id = ? AND investigation_id = ?",
                (1 if pinned else 0, artifact_id, investigation_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Artifact not found: {artifact_id}")
            self._touch_investigation_conn(conn, investigation_id)
        return self.get_investigation(investigation_id)

    def update_artifact_visibility(
        self,
        investigation_id: str,
        artifact_id: str,
        visibility: ArtifactVisibility | str,
    ) -> Investigation:
        resolved = ArtifactVisibility(visibility)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE artifacts SET visibility = ? WHERE id = ? AND investigation_id = ?",
                (resolved.value, artifact_id, investigation_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Artifact not found: {artifact_id}")
            self._touch_investigation_conn(conn, investigation_id)
        return self.get_investigation(investigation_id)

    def update_artifact_metadata(
        self,
        investigation_id: str,
        artifact_id: str,
        metadata: dict,
    ) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        artifact = next((item for item in investigation.artifacts if item.artifact_id == artifact_id), None)
        if artifact is None:
            raise KeyError(f"Artifact not found: {artifact_id}")
        current = artifact.metadata if isinstance(artifact.metadata, dict) else {}
        merged = {**current, **dict(metadata or {})}
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE artifacts SET metadata_json = ? WHERE id = ? AND investigation_id = ?",
                (_json(merged), artifact_id, investigation_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Artifact not found: {artifact_id}")
            self._touch_investigation_conn(conn, investigation_id)
        return self.get_investigation(investigation_id)

    def get_schema_version(self) -> int:
        with self._connect() as conn:
            return get_schema_version(conn)

    def create_shareable_report(self, report: ShareableReport) -> ShareableReport:
        self.get_investigation(report.investigation_id)
        with self._connect() as conn:
            if report.is_latest:
                for version in self._list_report_versions_conn(conn, report.previous_version_id or report.report_id):
                    version.is_latest = False
                    self._upsert_shareable_report(conn, version)
            self._upsert_shareable_report(conn, report)
        return self.get_shareable_report(report.report_id)

    def get_shareable_report(self, report_id: str) -> ShareableReport:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM shareable_reports WHERE id = ?", (report_id,)).fetchone()
            if row is None:
                raise KeyError(f"ShareableReport not found: {report_id}")
            return self._shareable_report_from_row(row)

    def list_shareable_reports(self, investigation_id: str | None = None) -> list[ShareableReport]:
        with self._connect() as conn:
            if investigation_id is None:
                rows = conn.execute("SELECT * FROM shareable_reports ORDER BY updated_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM shareable_reports WHERE investigation_id = ? ORDER BY updated_at DESC",
                    (investigation_id,),
                ).fetchall()
            return [self._shareable_report_from_row(row) for row in rows]

    def update_shareable_report(self, report: ShareableReport) -> ShareableReport:
        self.get_shareable_report(report.report_id)
        report.updated_at = utc_now()
        with self._connect() as conn:
            self._upsert_shareable_report(conn, report)
        return self.get_shareable_report(report.report_id)

    def archive_shareable_report(self, report_id: str) -> ShareableReport:
        report = self.get_shareable_report(report_id)
        report.status = ShareableReportStatus.ARCHIVED
        report.updated_at = utc_now()
        return self.update_shareable_report(report)

    def add_report_comment(self, comment: ReportComment) -> ReportComment:
        self.get_shareable_report(comment.report_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO report_comments
                (id, report_id, section_id, text, status, created_at, updated_at, resolved_at, author, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    comment.comment_id,
                    comment.report_id,
                    comment.section_id,
                    comment.text,
                    comment.status.value,
                    _dt(comment.created_at),
                    _dt(comment.updated_at),
                    _dt(comment.resolved_at),
                    comment.author,
                    _json(comment.metadata),
                ),
            )
        return comment

    def list_report_comments(self, report_id: str, section_id: str | None = None) -> list[ReportComment]:
        self.get_shareable_report(report_id)
        with self._connect() as conn:
            if section_id is None:
                rows = conn.execute(
                    "SELECT * FROM report_comments WHERE report_id = ? ORDER BY created_at ASC",
                    (report_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM report_comments WHERE report_id = ? AND section_id = ? ORDER BY created_at ASC",
                    (report_id, section_id),
                ).fetchall()
            return [self._comment_from_row(row) for row in rows]

    def update_report_comment(self, comment: ReportComment) -> ReportComment:
        comment.updated_at = utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE report_comments
                SET text = ?, status = ?, updated_at = ?, resolved_at = ?, author = ?, metadata_json = ?
                WHERE id = ?
                """,
                (
                    comment.text,
                    comment.status.value,
                    _dt(comment.updated_at),
                    _dt(comment.resolved_at),
                    comment.author,
                    _json(comment.metadata),
                    comment.comment_id,
                ),
            )
            if cur.rowcount == 0:
                raise KeyError(f"ReportComment not found: {comment.comment_id}")
        return comment

    def resolve_report_comment(self, comment_id: str) -> ReportComment:
        with self._connect() as conn:
            resolved_at = utc_now()
            cur = conn.execute(
                "UPDATE report_comments SET status = ?, resolved_at = ?, updated_at = ? WHERE id = ?",
                (ReportCommentStatus.RESOLVED.value, _dt(resolved_at), _dt(resolved_at), comment_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"ReportComment not found: {comment_id}")
            row = conn.execute("SELECT * FROM report_comments WHERE id = ?", (comment_id,)).fetchone()
            return self._comment_from_row(row)

    def delete_report_comment(self, comment_id: str) -> None:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM report_comments WHERE id = ?", (comment_id,))
            if cur.rowcount == 0:
                raise KeyError(f"ReportComment not found: {comment_id}")

    def set_report_approval_status(
        self,
        report_id: str,
        status: ReportApprovalStatus | str,
        reviewer_notes: str | None = None,
        approved_by: str | None = None,
    ) -> ShareableReport:
        report = self.get_shareable_report(report_id)
        report.approval_status = ReportApprovalStatus(status)
        if reviewer_notes is not None:
            report.reviewer_notes = reviewer_notes
        if report.approval_status == ReportApprovalStatus.APPROVED:
            report.approved_at = utc_now()
            report.approved_by = approved_by or report.approved_by
        else:
            report.approved_at = None
            report.approved_by = None
        return self.update_shareable_report(report)

    def list_report_versions(self, report_id: str) -> list[ShareableReport]:
        self.get_shareable_report(report_id)
        with self._connect() as conn:
            return self._list_report_versions_conn(conn, report_id)

    def get_latest_report_version(self, report_id: str) -> ShareableReport:
        versions = self.list_report_versions(report_id)
        for report in versions:
            if report.is_latest:
                return report
        return max(versions, key=lambda item: item.version)

    def restore_report_version(self, version_id: str) -> ShareableReport:
        import copy

        source = self.get_shareable_report(version_id)
        latest = self.get_latest_report_version(version_id)
        restored = copy.deepcopy(source)
        restored.report_id = new_id("share")
        restored.previous_version_id = latest.report_id
        restored.version = max(report.version for report in self.list_report_versions(version_id)) + 1
        restored.is_latest = True
        restored.status = ShareableReportStatus.DRAFT
        restored.version_note = f"Restored from v{source.version}"
        restored.created_at = utc_now()
        restored.updated_at = utc_now()
        latest.is_latest = False
        self.update_shareable_report(latest)
        return self.create_shareable_report(restored)

    def create_final_report_snapshot(self, snapshot: FinalReportSnapshot) -> FinalReportSnapshot:
        self.get_shareable_report(snapshot.report_id)
        with self._connect() as conn:
            self._upsert_final_report_snapshot(conn, snapshot)
        return self.get_final_report_snapshot(snapshot.snapshot_id)

    def get_final_report_snapshot(self, snapshot_id: str) -> FinalReportSnapshot:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM final_report_snapshots WHERE id = ?", (snapshot_id,)).fetchone()
            if row is None:
                raise KeyError(f"FinalReportSnapshot not found: {snapshot_id}")
            return self._final_snapshot_from_row(row)

    def list_final_report_snapshots(
        self,
        report_id: str | None = None,
        investigation_id: str | None = None,
        status: str | None = None,
    ) -> list[FinalReportSnapshot]:
        query = "SELECT * FROM final_report_snapshots"
        params: list[str] = []
        where: list[str] = []
        if report_id is not None:
            where.append("report_id = ?")
            params.append(report_id)
        if investigation_id is not None:
            where.append("investigation_id = ?")
            params.append(investigation_id)
        if status is not None:
            where.append("status = ?")
            params.append(status)
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._final_snapshot_from_row(row) for row in rows]

    def update_final_report_snapshot(self, snapshot: FinalReportSnapshot) -> FinalReportSnapshot:
        self.get_final_report_snapshot(snapshot.snapshot_id)
        with self._connect() as conn:
            self._upsert_final_report_snapshot(conn, snapshot)
        return self.get_final_report_snapshot(snapshot.snapshot_id)

    def create_data_source(self, data_source: DataSource) -> DataSource:
        data_source.tags = _normalize_tags(data_source.tags)
        with self._connect() as conn:
            self._upsert_data_source(conn, data_source)
        return self.get_data_source(data_source.data_source_id)

    def get_data_source(self, data_source_id: str) -> DataSource:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM data_sources WHERE id = ?", (data_source_id,)).fetchone()
            if row is None:
                raise KeyError(f"DataSource not found: {data_source_id}")
            return self._data_source_from_row(row)

    def list_data_sources(self, status: str | None = None) -> list[DataSource]:
        with self._connect() as conn:
            if status is None:
                rows = conn.execute("SELECT * FROM data_sources ORDER BY updated_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM data_sources WHERE status = ? ORDER BY updated_at DESC",
                    (status,),
                ).fetchall()
            return [self._data_source_from_row(row) for row in rows]

    def update_data_source(self, data_source: DataSource) -> DataSource:
        self.get_data_source(data_source.data_source_id)
        data_source.tags = _normalize_tags(data_source.tags)
        data_source.updated_at = utc_now()
        with self._connect() as conn:
            self._upsert_data_source(conn, data_source)
        return self.get_data_source(data_source.data_source_id)

    def archive_data_source(self, data_source_id: str) -> DataSource:
        data_source = self.get_data_source(data_source_id)
        data_source.status = DataSourceStatus.ARCHIVED
        return self.update_data_source(data_source)

    def delete_data_source(self, data_source_id: str, delete_file: bool = True) -> bool:
        data_source = self.get_data_source(data_source_id)
        file_deleted = False
        if delete_file and data_source.data_source_type == DataSourceType.CSV:
            file_deleted = delete_uploaded_file_if_safe(data_source.location)
        with self._connect() as conn:
            investigations = conn.execute("SELECT * FROM investigations").fetchall()
            for row in investigations:
                linked_ids = [
                    item for item in _loads(row["linked_data_source_ids_json"], []) if item != data_source_id
                ]
                data_sources = [
                    item for item in _loads(row["data_sources_json"], []) if item != data_source_id
                ]
                if len(linked_ids) != len(_loads(row["linked_data_source_ids_json"], [])) or len(data_sources) != len(_loads(row["data_sources_json"], [])):
                    conn.execute(
                        """
                        UPDATE investigations
                        SET linked_data_source_ids_json = ?, data_sources_json = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (_json(linked_ids), _json(data_sources), _dt(utc_now()), row["id"]),
                    )
            conn.execute("DELETE FROM data_sources WHERE id = ?", (data_source_id,))
        return file_deleted

    def save_data_source_profile(self, data_source_id: str, profile: DataSourceProfile) -> DataSourceProfile:
        source = self.get_data_source(data_source_id)
        runtime = source.metadata.get("execution_context") if isinstance(source.metadata, dict) else {}
        if not isinstance(runtime, dict) or "executable_available" not in runtime:
            source.metadata = dict(source.metadata or {})
            source.metadata.setdefault("dataset_id", data_source_id)
            source.metadata.setdefault("dataset_runtime_reference", "")
            source.metadata.setdefault("executable_available", False)
            source.metadata.setdefault("row_count", int(profile.row_count))
            source.metadata.setdefault("storage_reference", "")
            source.metadata.setdefault("created_at", source.created_at.isoformat())
            source.metadata.setdefault("last_loaded_at", source.updated_at.isoformat())
            source.metadata.setdefault(
                "execution_context",
                {
                    "dataset_id": data_source_id,
                    "dataset_runtime_reference": "",
                    "schema_metadata": {
                        "columns": [{"name": column.name, "dtype": column.dtype} for column in profile.columns],
                        "row_count": int(profile.row_count),
                        "column_count": int(profile.column_count),
                    },
                    "profile_metadata": {
                        "row_count": int(profile.row_count),
                        "column_count": int(profile.column_count),
                        "generated_at": profile.generated_at.isoformat(),
                    },
                    "executable_available": False,
                    "row_count": int(profile.row_count),
                    "storage_reference": "",
                    "created_at": source.created_at.isoformat(),
                    "last_loaded_at": source.updated_at.isoformat(),
                    "unavailable_reason": "profile_only",
                },
            )
        with self._connect() as conn:
            self._upsert_data_source(conn, source)
            conn.execute(
                """
                INSERT OR REPLACE INTO data_source_profiles
                (data_source_id, profile_json, generated_at)
                VALUES (?, ?, ?)
                """,
                (data_source_id, _json(_profile_to_dict(profile)), _dt(profile.generated_at)),
            )
        return profile

    def get_data_source_profile(self, data_source_id: str) -> DataSourceProfile:
        self.get_data_source(data_source_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM data_source_profiles WHERE data_source_id = ?",
                (data_source_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"DataSourceProfile not found: {data_source_id}")
            return _profile_from_dict(_loads(row["profile_json"], {}))

    def get_data_source_semantic_notes(self, data_source_id: str) -> DataSourceSemanticNotes:
        self.get_data_source(data_source_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM data_source_semantic_notes WHERE data_source_id = ?",
                (data_source_id,),
            ).fetchone()
            if row is None:
                return DataSourceSemanticNotes(data_source_id=data_source_id)
            notes = _semantic_notes_from_dict(_loads(row["notes_json"], {}), data_source_id=data_source_id)
            notes.updated_at = _parse_dt(row["updated_at"])
            return notes

    def save_data_source_semantic_notes(self, notes: DataSourceSemanticNotes) -> DataSourceSemanticNotes:
        self.get_data_source(notes.data_source_id)
        notes.updated_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO data_source_semantic_notes
                (data_source_id, notes_json, updated_at)
                VALUES (?, ?, ?)
                """,
                (notes.data_source_id, _json(_semantic_notes_to_dict(notes)), _dt(notes.updated_at)),
            )
        return self.get_data_source_semantic_notes(notes.data_source_id)

    def update_column_semantic_note(
        self,
        data_source_id: str,
        column_name: str,
        note: ColumnSemanticNote,
    ) -> DataSourceSemanticNotes:
        notes = self.get_data_source_semantic_notes(data_source_id)
        note.column_name = column_name
        notes.column_notes = [item for item in notes.column_notes if item.column_name != column_name]
        notes.column_notes.append(note)
        return self.save_data_source_semantic_notes(notes)

    def delete_column_semantic_note(self, data_source_id: str, column_name: str) -> DataSourceSemanticNotes:
        notes = self.get_data_source_semantic_notes(data_source_id)
        notes.column_notes = [item for item in notes.column_notes if item.column_name != column_name]
        return self.save_data_source_semantic_notes(notes)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            apply_migrations(conn)

    @staticmethod
    def _upsert_shareable_report(conn: sqlite3.Connection, report: ShareableReport) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO shareable_reports
            (id, investigation_id, title, template, status, version, created_at, updated_at,
             sections_json, source_finding_ids_json, source_artifact_ids_json, include_technical, metadata_json,
             previous_version_id, is_latest, version_note, approval_status, reviewer_notes, approved_at, approved_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.report_id,
                report.investigation_id,
                report.title,
                report.template.value,
                report.status.value,
                report.version,
                _dt(report.created_at),
                _dt(report.updated_at),
                _json([_section_to_dict(section) for section in report.sections]),
                _json(report.source_finding_ids),
                _json(report.source_artifact_ids),
                1 if report.include_technical else 0,
                _json(report.metadata),
                report.previous_version_id,
                1 if report.is_latest else 0,
                report.version_note,
                report.approval_status.value,
                report.reviewer_notes,
                _dt(report.approved_at),
                report.approved_by,
            ),
        )

    @staticmethod
    def _upsert_final_report_snapshot(conn: sqlite3.Connection, snapshot: FinalReportSnapshot) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO final_report_snapshots
            (id, report_id, investigation_id, report_version, title, created_at, created_by, status,
             markdown_content, html_content, txt_content, readiness_snapshot_json, approval_status, approved_at,
             approved_by, source_report_json, metadata_json, decision_metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.snapshot_id,
                snapshot.report_id,
                snapshot.investigation_id,
                snapshot.report_version,
                snapshot.title,
                _dt(snapshot.created_at),
                snapshot.created_by,
                snapshot.status.value,
                snapshot.markdown_content,
                snapshot.html_content,
                snapshot.txt_content,
                _json(snapshot.readiness_snapshot),
                snapshot.approval_status.value,
                _dt(snapshot.approved_at),
                snapshot.approved_by,
                _json(snapshot.source_report_json),
                _json(snapshot.metadata),
                _json(_decision_metadata_to_dict(snapshot.decision_metadata)),
            ),
        )

    @staticmethod
    def _upsert_data_source(conn: sqlite3.Connection, data_source: DataSource) -> None:
        conn.execute(
            """
            INSERT INTO data_sources
            (id, name, type, created_at, updated_at, status, location, description,
             tags_json, linked_investigation_ids_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                type = excluded.type,
                updated_at = excluded.updated_at,
                status = excluded.status,
                location = excluded.location,
                description = excluded.description,
                tags_json = excluded.tags_json,
                linked_investigation_ids_json = excluded.linked_investigation_ids_json,
                metadata_json = excluded.metadata_json
            """,
            (
                data_source.data_source_id,
                data_source.name,
                data_source.data_source_type.value,
                _dt(data_source.created_at),
                _dt(data_source.updated_at),
                data_source.status.value,
                data_source.location,
                data_source.description,
                _json(data_source.tags),
                _json(data_source.linked_investigation_ids),
                _json(data_source.metadata),
            ),
        )

    @staticmethod
    def _upsert_investigation_memory_item(conn: sqlite3.Connection, item: InvestigationMemoryItem) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO investigation_memory
            (id, investigation_id, type, title, content, status, created_at, updated_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.memory_id,
                item.investigation_id,
                item.memory_type.value,
                item.title,
                item.content,
                item.status.value,
                _dt(item.created_at),
                _dt(item.updated_at),
                _json(item.metadata),
            ),
        )

    @staticmethod
    def _upsert_investigation_run(conn: sqlite3.Connection, run: InvestigationRun) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO investigation_runs
            (id, investigation_id, created_at, started_at, completed_at, status, current_stage,
             data_source_ids_json, run_context_summary_json, error_message,
             artifact_ids_json, report_ids_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.investigation_id,
                _dt(run.created_at),
                _dt(run.started_at),
                _dt(run.completed_at),
                run.status.value,
                run.current_stage.value,
                _json(run.data_source_ids),
                _json(run.run_context_summary),
                run.error_message or run.error,
                _json(run.artifact_ids),
                _json(run.report_ids),
                _json(run.metadata),
            ),
        )

    def _list_report_versions_conn(self, conn: sqlite3.Connection, report_id: str) -> list[ShareableReport]:
        rows = conn.execute("SELECT * FROM shareable_reports").fetchall()
        reports = [self._shareable_report_from_row(row) for row in rows]
        by_id = {report.report_id: report for report in reports}
        if report_id not in by_id:
            return []
        edges: dict[str, set[str]] = {report.report_id: set() for report in reports}
        for report in reports:
            if report.previous_version_id and report.previous_version_id in edges:
                edges[report.report_id].add(report.previous_version_id)
                edges[report.previous_version_id].add(report.report_id)
        seen = {report_id}
        stack = [report_id]
        while stack:
            current = stack.pop()
            for nxt in edges.get(current, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return sorted([by_id[item] for item in seen], key=lambda item: item.version)

    def _hydrate_investigation(self, conn: sqlite3.Connection, row: sqlite3.Row) -> Investigation:
        investigation = Investigation(
            title=row["title"],
            user_question=row["question"],
            status=InvestigationStatus(row["status"]),
            data_sources=_loads(row["data_sources_json"], []),
            linked_data_source_ids=_loads(_row_value(row, "linked_data_source_ids_json"), []),
            investigation_id=row["id"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
            metadata=_loads(row["metadata_json"], {}),
        )
        run_rows = conn.execute(
            "SELECT * FROM runs WHERE investigation_id = ? ORDER BY started_at ASC",
            (investigation.investigation_id,),
        ).fetchall()
        investigation.runs = [self._run_from_row(item) for item in run_rows]
        investigation.trace = [event for run in investigation.runs for event in run.trace]

        artifact_rows = conn.execute(
            "SELECT * FROM artifacts WHERE investigation_id = ? ORDER BY pinned DESC, created_at ASC",
            (investigation.investigation_id,),
        ).fetchall()
        investigation.artifacts = [self._artifact_from_row(item) for item in artifact_rows]

        finding_rows = conn.execute(
            "SELECT * FROM findings WHERE investigation_id = ? ORDER BY created_at ASC",
            (investigation.investigation_id,),
        ).fetchall()
        investigation.findings = [self._finding_from_row(item) for item in finding_rows]

        report_row = conn.execute(
            "SELECT * FROM reports WHERE investigation_id = ? ORDER BY created_at DESC LIMIT 1",
            (investigation.investigation_id,),
        ).fetchone()
        investigation.report = self._report_from_row(report_row) if report_row else None
        return investigation

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> InvestigationRun:
        return InvestigationRun(
            status=InvestigationStatus(row["status"]),
            trace=_loads(row["trace_json"], []),
            output=_loads(row["raw_output_json"], {}),
            error=row["error_message"],
            run_id=row["id"],
            started_at=_parse_dt(row["started_at"]),
            finished_at=_parse_dt(row["finished_at"]) if row["finished_at"] else None,
            metadata=_loads(row["metadata_json"], {}),
        )

    @staticmethod
    def _investigation_run_from_row(row: sqlite3.Row) -> InvestigationRun:
        return InvestigationRun(
            run_id=row["id"],
            investigation_id=row["investigation_id"],
            created_at=_parse_dt(row["created_at"]),
            started_at=_parse_dt(row["started_at"]) if row["started_at"] else None,
            completed_at=_parse_dt(row["completed_at"]) if row["completed_at"] else None,
            finished_at=_parse_dt(row["completed_at"]) if row["completed_at"] else None,
            status=InvestigationRunStatus(row["status"]),
            current_stage=InvestigationRunStage(row["current_stage"]),
            data_source_ids=_loads(row["data_source_ids_json"], []),
            run_context_summary=_loads(row["run_context_summary_json"], {}),
            error_message=row["error_message"],
            error=row["error_message"],
            artifact_ids=_loads(row["artifact_ids_json"], []),
            report_ids=_loads(row["report_ids_json"], []),
            metadata=_loads(row["metadata_json"], {}),
        )

    @staticmethod
    def _investigation_run_event_from_row(row: sqlite3.Row) -> InvestigationRunEvent:
        return InvestigationRunEvent(
            event_id=row["id"],
            run_id=row["run_id"],
            investigation_id=row["investigation_id"],
            event_type=InvestigationRunEventType(row["event_type"]),
            stage=InvestigationRunStage(row["stage"]) if row["stage"] else None,
            message=row["message"],
            created_at=_parse_dt(row["created_at"]),
            metadata=_loads(row["metadata_json"], {}),
            severity=InvestigationRunEventSeverity(row["severity"]),
        )

    @staticmethod
    def _investigation_message_from_row(row: sqlite3.Row) -> InvestigationMessage:
        return InvestigationMessage(
            message_id=row["id"],
            investigation_id=row["investigation_id"],
            run_id=row["run_id"],
            role=InvestigationMessageRole(row["role"]),
            message_type=InvestigationMessageType(row["type"]),
            content=row["content"],
            created_at=_parse_dt(row["created_at"]),
            metadata=_loads(row["metadata_json"], {}),
        )

    @staticmethod
    def _investigation_memory_item_from_row(row: sqlite3.Row) -> InvestigationMemoryItem:
        return InvestigationMemoryItem(
            memory_id=row["id"],
            investigation_id=row["investigation_id"],
            memory_type=InvestigationMemoryType(row["type"]),
            title=row["title"],
            content=row["content"],
            status=InvestigationMemoryStatus(row["status"]),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
            metadata=_loads(row["metadata_json"], {}),
        )

    @staticmethod
    def _artifact_from_row(row: sqlite3.Row) -> Artifact:
        return Artifact(
            artifact_type=ArtifactType(row["type"]),
            title=row["title"],
            content=_loads(row["content_json"], None),
            path=row["path"],
            metadata=_loads(row["metadata_json"], {}),
            visibility=ArtifactVisibility(row["visibility"]),
            pinned=bool(row["pinned"]),
            run_id=row["run_id"],
            artifact_id=row["id"],
            created_at=_parse_dt(row["created_at"]),
        )

    @staticmethod
    def _finding_from_row(row: sqlite3.Row) -> Finding:
        return Finding(
            title=row["title"],
            text=row["text"],
            evidence=_loads(row["evidence_json"], []),
            evidence_artifact_ids=_loads(row["evidence_artifact_ids_json"], []),
            confidence=row["confidence"],
            status=FindingStatus(row["status"]),
            metadata=_loads(row["metadata_json"], {}),
            run_id=row["run_id"],
            finding_id=row["id"],
            created_at=_parse_dt(row["created_at"]),
        )

    @staticmethod
    def _report_from_row(row: sqlite3.Row) -> DecisionReport:
        return DecisionReport(
            question=row["question"],
            answer=row["answer"],
            summary=row["answer"],
            key_findings=_loads(row["key_findings_json"], []),
            evidence=_loads(row["evidence_json"], []),
            limitations=_loads(row["limitations_json"], []),
            next_steps=_loads(row["next_steps_json"], []),
            artifact_ids=_loads(row["artifact_ids_json"], []),
            content=row["content"],
            metadata=_loads(row["metadata_json"], {}),
            run_id=row["run_id"],
            report_id=row["id"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["created_at"]),
        )

    @staticmethod
    def _shareable_report_from_row(row: sqlite3.Row) -> ShareableReport:
        return ShareableReport(
            report_id=row["id"],
            investigation_id=row["investigation_id"],
            title=row["title"],
            template=ShareableReportTemplate(row["template"]),
            status=ShareableReportStatus(row["status"]),
            version=int(row["version"]),
            previous_version_id=_row_value(row, "previous_version_id"),
            is_latest=bool(_row_value(row, "is_latest", 1)),
            version_note=str(_row_value(row, "version_note", "") or ""),
            approval_status=ReportApprovalStatus(_row_value(row, "approval_status", "draft") or "draft"),
            reviewer_notes=str(_row_value(row, "reviewer_notes", "") or ""),
            approved_at=_parse_dt(_row_value(row, "approved_at")) if _row_value(row, "approved_at") else None,
            approved_by=_row_value(row, "approved_by"),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
            sections=[_section_from_dict(item) for item in _loads(row["sections_json"], [])],
            source_finding_ids=_loads(row["source_finding_ids_json"], []),
            source_artifact_ids=_loads(row["source_artifact_ids_json"], []),
            include_technical=bool(row["include_technical"]),
            metadata=_loads(row["metadata_json"], {}),
        )

    @staticmethod
    def _comment_from_row(row: sqlite3.Row) -> ReportComment:
        return ReportComment(
            comment_id=row["id"],
            report_id=row["report_id"],
            section_id=row["section_id"],
            text=row["text"],
            status=ReportCommentStatus(row["status"]),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
            resolved_at=_parse_dt(row["resolved_at"]) if row["resolved_at"] else None,
            author=row["author"],
            metadata=_loads(row["metadata_json"], {}),
        )

    @staticmethod
    def _final_snapshot_from_row(row: sqlite3.Row) -> FinalReportSnapshot:
        return FinalReportSnapshot(
            snapshot_id=row["id"],
            report_id=row["report_id"],
            investigation_id=row["investigation_id"],
            report_version=int(row["report_version"]),
            title=row["title"],
            created_at=_parse_dt(row["created_at"]),
            created_by=row["created_by"],
            status=FinalReportStatus(row["status"]),
            markdown_content=row["markdown_content"],
            html_content=row["html_content"],
            txt_content=_row_value(row, "txt_content", ""),
            readiness_snapshot=_loads(row["readiness_snapshot_json"], {}),
            approval_status=ReportApprovalStatus(row["approval_status"]),
            approved_at=_parse_dt(row["approved_at"]) if row["approved_at"] else None,
            approved_by=row["approved_by"],
            source_report_json=_loads(row["source_report_json"], {}),
            decision_metadata=_decision_metadata_from_dict(_loads(_row_value(row, "decision_metadata_json"), {})),
            metadata=_loads(row["metadata_json"], {}),
        )

    @staticmethod
    def _data_source_from_row(row: sqlite3.Row) -> DataSource:
        return DataSource(
            data_source_id=row["id"],
            name=row["name"],
            data_source_type=DataSourceType(row["type"]),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
            status=DataSourceStatus(row["status"]),
            location=row["location"],
            description=row["description"],
            tags=_loads(row["tags_json"], []),
            linked_investigation_ids=_loads(row["linked_investigation_ids_json"], []),
            metadata=_loads(row["metadata_json"], {}),
        )

    def _touch_investigation(self, investigation_id: str, status: str | None = None) -> None:
        with self._connect() as conn:
            self._touch_investigation_conn(conn, investigation_id, status=status)

    @staticmethod
    def _touch_investigation_conn(
        conn: sqlite3.Connection,
        investigation_id: str,
        status: str | None = None,
    ) -> None:
        if status is None:
            conn.execute("UPDATE investigations SET updated_at = ? WHERE id = ?", (_dt(utc_now()), investigation_id))
        else:
            conn.execute(
                "UPDATE investigations SET status = ?, updated_at = ? WHERE id = ?",
                (status, _dt(utc_now()), investigation_id),
            )
