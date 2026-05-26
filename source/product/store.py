from __future__ import annotations

from typing import Optional

from source.api.session import get_current_session_id
from source.product.investigation import (
    Artifact,
    ArtifactVisibility,
    DecisionReport,
    FinalReportSnapshot,
    Finding,
    FindingStatus,
    Investigation,
    InvestigationMessage,
    InvestigationMemoryItem,
    InvestigationMemoryStatus,
    InvestigationMemoryType,
    InvestigationMessageRole,
    InvestigationMessageType,
    InvestigationRun,
    InvestigationRunEvent,
    InvestigationRunStatus,
    InvestigationStatus,
    ReportApprovalStatus,
    ReportComment,
    ReportCommentStatus,
    ShareableReport,
    ShareableReportStatus,
    new_id,
    utc_now,
)
from source.product.data_sources import (
    ColumnSemanticNote,
    DataSource,
    DataSourceProfile,
    DataSourceSemanticNotes,
    DataSourceStatus,
)
from source.product.file_storage import delete_uploaded_file_if_safe


class InvestigationStore:
    """Small replaceable repository for product investigations.

    The first implementation is intentionally in-memory. The public methods are
    shaped like a repository so a later SQLite/Postgres store can keep the same
    service contract.
    """

    def __init__(self) -> None:
        self._investigations: dict[str, Investigation] = {}
        self._shareable_reports: dict[str, ShareableReport] = {}
        self._report_comments: dict[str, ReportComment] = {}
        self._final_report_snapshots: dict[str, FinalReportSnapshot] = {}
        self._investigation_runs: dict[str, InvestigationRun] = {}
        self._investigation_run_events: dict[str, InvestigationRunEvent] = {}
        self._investigation_messages: dict[str, InvestigationMessage] = {}
        self._investigation_memory: dict[str, InvestigationMemoryItem] = {}
        self._data_sources: dict[str, DataSource] = {}
        self._data_source_profiles: dict[str, DataSourceProfile] = {}
        self._data_source_semantic_notes: dict[str, DataSourceSemanticNotes] = {}
        self._owners: dict[str, dict[str, str | None]] = {
            "investigations": {},
            "shareable_reports": {},
            "report_comments": {},
            "final_report_snapshots": {},
            "investigation_runs": {},
            "investigation_run_events": {},
            "investigation_messages": {},
            "investigation_memory": {},
            "data_sources": {},
            "data_source_profiles": {},
            "data_source_semantic_notes": {},
            "runs": {},
            "artifacts": {},
            "findings": {},
            "reports": {},
        }

    def create_investigation(self, question: str, title: Optional[str] = None) -> Investigation:
        resolved_title = title or self._title_from_question(question)
        investigation = Investigation(title=resolved_title, user_question=question)
        self._investigations[investigation.investigation_id] = investigation
        self._set_owner("investigations", investigation.investigation_id)
        return investigation

    def link_data_source_to_investigation(self, investigation_id: str, data_source_id: str) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        data_source = self.get_data_source(data_source_id)
        if data_source_id not in investigation.linked_data_source_ids:
            investigation.linked_data_source_ids.append(data_source_id)
        if data_source_id not in investigation.data_sources:
            investigation.data_sources.append(data_source_id)
        if investigation_id not in data_source.linked_investigation_ids:
            data_source.linked_investigation_ids.append(investigation_id)
            data_source.updated_at = utc_now()
        self._touch(investigation)
        return investigation

    def get_investigation(self, investigation_id: str) -> Investigation:
        self._require_owner("investigations", investigation_id)
        try:
            return self._investigations[investigation_id]
        except KeyError as exc:
            raise KeyError(f"Investigation not found: {investigation_id}") from exc

    def list_investigations(self) -> list[Investigation]:
        return sorted(
            [
                item
                for item in self._investigations.values()
                if self._matches_owner("investigations", item.investigation_id)
            ],
            key=lambda item: item.updated_at,
            reverse=True,
        )

    def update_status(self, investigation_id: str, status: InvestigationStatus | str) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        investigation.status = InvestigationStatus(status)
        self._touch(investigation)
        return investigation

    def update_investigation_metadata(self, investigation_id: str, metadata: dict) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        investigation.metadata = dict(metadata or {})
        self._touch(investigation)
        return investigation

    def delete_investigation(self, investigation_id: str) -> None:
        self.get_investigation(investigation_id)
        report_ids = {
            report.report_id
            for report in self._shareable_reports.values()
            if report.investigation_id == investigation_id
        }
        run_ids = {
            run.run_id
            for run in self._investigation_runs.values()
            if run.investigation_id == investigation_id
        }
        for source in self._data_sources.values():
            if investigation_id in source.linked_investigation_ids:
                source.linked_investigation_ids = [
                    item for item in source.linked_investigation_ids if item != investigation_id
                ]
                source.updated_at = utc_now()
        self._report_comments = {
            key: comment for key, comment in self._report_comments.items() if comment.report_id not in report_ids
        }
        self._final_report_snapshots = {
            key: snapshot
            for key, snapshot in self._final_report_snapshots.items()
            if snapshot.investigation_id != investigation_id and snapshot.report_id not in report_ids
        }
        self._shareable_reports = {
            key: report for key, report in self._shareable_reports.items() if report.investigation_id != investigation_id
        }
        self._investigation_run_events = {
            key: event for key, event in self._investigation_run_events.items() if event.investigation_id != investigation_id
        }
        self._investigation_runs = {
            key: run for key, run in self._investigation_runs.items() if run.investigation_id != investigation_id
        }
        self._investigation_messages = {
            key: message
            for key, message in self._investigation_messages.items()
            if message.investigation_id != investigation_id and message.run_id not in run_ids
        }
        self._investigation_memory = {
            key: item for key, item in self._investigation_memory.items() if item.investigation_id != investigation_id
        }
        del self._investigations[investigation_id]

    def create_investigation_run(self, run: InvestigationRun) -> InvestigationRun:
        self.get_investigation(run.investigation_id)
        run.status = InvestigationRunStatus(run.status)
        self._investigation_runs[run.run_id] = run
        self._set_owner("investigation_runs", run.run_id)
        return run

    def update_investigation_run(self, run: InvestigationRun) -> InvestigationRun:
        self.get_investigation_run(run.run_id)
        if isinstance(run.status, str):
            run.status = InvestigationRunStatus(run.status)
        self._investigation_runs[run.run_id] = run
        return run

    def get_investigation_run(self, run_id: str) -> InvestigationRun:
        self._require_owner("investigation_runs", run_id)
        try:
            return self._investigation_runs[run_id]
        except KeyError as exc:
            raise KeyError(f"InvestigationRun not found: {run_id}") from exc

    def list_investigation_runs(self) -> list[InvestigationRun]:
        return sorted(
            [
                item
                for item in self._investigation_runs.values()
                if self._matches_owner("investigation_runs", item.run_id)
            ],
            key=lambda item: item.created_at,
            reverse=True,
        )

    def list_runs_for_investigation(self, investigation_id: str) -> list[InvestigationRun]:
        self.get_investigation(investigation_id)
        return [
            run
            for run in self.list_investigation_runs()
            if run.investigation_id == investigation_id
        ]

    def add_investigation_run_event(self, event: InvestigationRunEvent) -> InvestigationRunEvent:
        self.get_investigation_run(event.run_id)
        self.get_investigation(event.investigation_id)
        self._investigation_run_events[event.event_id] = event
        self._set_owner("investigation_run_events", event.event_id)
        return event

    def list_investigation_run_events(self, run_id: str) -> list[InvestigationRunEvent]:
        self.get_investigation_run(run_id)
        return sorted(
            [event for event in self._investigation_run_events.values() if event.run_id == run_id],
            key=lambda item: item.created_at,
        )

    def list_events_for_investigation(
        self,
        investigation_id: str,
        limit: int = 100,
    ) -> list[InvestigationRunEvent]:
        self.get_investigation(investigation_id)
        events = sorted(
            [
                event
                for event in self._investigation_run_events.values()
                if event.investigation_id == investigation_id
            ],
            key=lambda item: item.created_at,
        )
        return events[-limit:]

    def add_investigation_message(self, message: InvestigationMessage) -> InvestigationMessage:
        self.get_investigation(message.investigation_id)
        if message.run_id:
            self.get_investigation_run(message.run_id)
        message.role = InvestigationMessageRole(message.role)
        message.message_type = InvestigationMessageType(message.message_type)
        self._investigation_messages[message.message_id] = message
        self._set_owner("investigation_messages", message.message_id)
        self._touch(self.get_investigation(message.investigation_id))
        return message

    def list_investigation_messages(
        self,
        investigation_id: str,
        limit: int | None = None,
    ) -> list[InvestigationMessage]:
        self.get_investigation(investigation_id)
        messages = sorted(
            [
                message
                for message in self._investigation_messages.values()
                if message.investigation_id == investigation_id
            ],
            key=lambda item: item.created_at,
        )
        return messages[-limit:] if limit else messages

    def get_investigation_message(self, message_id: str) -> InvestigationMessage:
        self._require_owner("investigation_messages", message_id)
        try:
            return self._investigation_messages[message_id]
        except KeyError as exc:
            raise KeyError(f"InvestigationMessage not found: {message_id}") from exc

    def add_investigation_memory_item(self, item: InvestigationMemoryItem) -> InvestigationMemoryItem:
        self.get_investigation(item.investigation_id)
        item.memory_type = InvestigationMemoryType(item.memory_type)
        item.status = InvestigationMemoryStatus(item.status)
        item.updated_at = utc_now()
        self._investigation_memory[item.memory_id] = item
        self._set_owner("investigation_memory", item.memory_id)
        self._touch(self.get_investigation(item.investigation_id))
        return item

    def list_investigation_memory(
        self,
        investigation_id: str,
        memory_type: InvestigationMemoryType | str | None = None,
    ) -> list[InvestigationMemoryItem]:
        self.get_investigation(investigation_id)
        resolved_type = InvestigationMemoryType(memory_type) if memory_type else None
        items = [item for item in self._investigation_memory.values() if item.investigation_id == investigation_id]
        if resolved_type:
            items = [item for item in items if item.memory_type == resolved_type]
        return sorted(items, key=lambda item: item.updated_at, reverse=True)

    def get_investigation_memory_item(self, memory_id: str) -> InvestigationMemoryItem:
        self._require_owner("investigation_memory", memory_id)
        try:
            return self._investigation_memory[memory_id]
        except KeyError as exc:
            raise KeyError(f"InvestigationMemoryItem not found: {memory_id}") from exc

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
        self._investigation_memory[memory_id] = item
        self._touch(self.get_investigation(item.investigation_id))
        return item

    def update_finding_status(
        self,
        investigation_id: str,
        finding_id: str,
        status: FindingStatus | str,
    ) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        resolved = FindingStatus(status)
        for finding in investigation.findings:
            if finding.finding_id == finding_id:
                finding.status = resolved
                self._touch(investigation)
                return investigation
        raise KeyError(f"Finding not found: {finding_id}")

    def set_artifact_pinned(self, investigation_id: str, artifact_id: str, pinned: bool) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        for artifact in investigation.artifacts:
            if artifact.artifact_id == artifact_id:
                artifact.pinned = bool(pinned)
                self._touch(investigation)
                return investigation
        raise KeyError(f"Artifact not found: {artifact_id}")

    def update_artifact_visibility(
        self,
        investigation_id: str,
        artifact_id: str,
        visibility: ArtifactVisibility | str,
    ) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        resolved = ArtifactVisibility(visibility)
        for artifact in investigation.artifacts:
            if artifact.artifact_id == artifact_id:
                artifact.visibility = resolved
                self._touch(investigation)
                return investigation
        raise KeyError(f"Artifact not found: {artifact_id}")

    def update_artifact_metadata(
        self,
        investigation_id: str,
        artifact_id: str,
        metadata: dict,
    ) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        for artifact in investigation.artifacts:
            if artifact.artifact_id == artifact_id:
                current = artifact.metadata if isinstance(artifact.metadata, dict) else {}
                artifact.metadata = {**current, **dict(metadata or {})}
                self._touch(investigation)
                return investigation
        raise KeyError(f"Artifact not found: {artifact_id}")

    def create_shareable_report(self, report: ShareableReport) -> ShareableReport:
        self.get_investigation(report.investigation_id)
        if report.is_latest:
            anchor_id = report.previous_version_id or report.report_id
            if anchor_id in self._shareable_reports:
                for existing in self.list_report_versions(anchor_id):
                    existing.is_latest = False
        self._shareable_reports[report.report_id] = report
        self._set_owner("shareable_reports", report.report_id)
        return report

    def get_shareable_report(self, report_id: str) -> ShareableReport:
        self._require_owner("shareable_reports", report_id)
        try:
            return self._shareable_reports[report_id]
        except KeyError as exc:
            raise KeyError(f"ShareableReport not found: {report_id}") from exc

    def list_shareable_reports(self, investigation_id: str | None = None) -> list[ShareableReport]:
        reports = list(self._shareable_reports.values())
        reports = [
            report
            for report in reports
            if self._matches_owner("shareable_reports", report.report_id)
        ]
        if investigation_id is not None:
            reports = [report for report in reports if report.investigation_id == investigation_id]
        return sorted(reports, key=lambda item: item.updated_at, reverse=True)

    def update_shareable_report(self, report: ShareableReport) -> ShareableReport:
        self.get_shareable_report(report.report_id)
        report.updated_at = utc_now()
        self._shareable_reports[report.report_id] = report
        return report

    def archive_shareable_report(self, report_id: str) -> ShareableReport:
        report = self.get_shareable_report(report_id)
        report.status = ShareableReportStatus.ARCHIVED
        report.updated_at = utc_now()
        self._shareable_reports[report_id] = report
        return report

    def add_report_comment(self, comment: ReportComment) -> ReportComment:
        self.get_shareable_report(comment.report_id)
        self._report_comments[comment.comment_id] = comment
        self._set_owner("report_comments", comment.comment_id)
        return comment

    def list_report_comments(self, report_id: str, section_id: str | None = None) -> list[ReportComment]:
        self.get_shareable_report(report_id)
        comments = [comment for comment in self._report_comments.values() if comment.report_id == report_id]
        if section_id is not None:
            comments = [comment for comment in comments if comment.section_id == section_id]
        return sorted(comments, key=lambda item: item.created_at)

    def update_report_comment(self, comment: ReportComment) -> ReportComment:
        self._require_owner("report_comments", comment.comment_id)
        if comment.comment_id not in self._report_comments:
            raise KeyError(f"ReportComment not found: {comment.comment_id}")
        comment.updated_at = utc_now()
        self._report_comments[comment.comment_id] = comment
        return comment

    def resolve_report_comment(self, comment_id: str) -> ReportComment:
        self._require_owner("report_comments", comment_id)
        try:
            comment = self._report_comments[comment_id]
        except KeyError as exc:
            raise KeyError(f"ReportComment not found: {comment_id}") from exc
        comment.status = ReportCommentStatus.RESOLVED
        comment.resolved_at = utc_now()
        comment.updated_at = comment.resolved_at
        self._report_comments[comment_id] = comment
        return comment

    def delete_report_comment(self, comment_id: str) -> None:
        self._require_owner("report_comments", comment_id)
        if comment_id not in self._report_comments:
            raise KeyError(f"ReportComment not found: {comment_id}")
        del self._report_comments[comment_id]

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
        ids = self._version_component_ids(report_id)
        reports = [self._shareable_reports[item] for item in ids]
        return sorted(reports, key=lambda item: item.version)

    def get_latest_report_version(self, report_id: str) -> ShareableReport:
        versions = self.list_report_versions(report_id)
        for report in versions:
            if report.is_latest:
                return report
        return max(versions, key=lambda item: item.version)

    def restore_report_version(self, version_id: str) -> ShareableReport:
        source = self.get_shareable_report(version_id)
        latest = self.get_latest_report_version(version_id)
        import copy

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
        self._final_report_snapshots[snapshot.snapshot_id] = snapshot
        self._set_owner("final_report_snapshots", snapshot.snapshot_id)
        return snapshot

    def get_final_report_snapshot(self, snapshot_id: str) -> FinalReportSnapshot:
        self._require_owner("final_report_snapshots", snapshot_id)
        try:
            return self._final_report_snapshots[snapshot_id]
        except KeyError as exc:
            raise KeyError(f"FinalReportSnapshot not found: {snapshot_id}") from exc

    def list_final_report_snapshots(
        self,
        report_id: str | None = None,
        investigation_id: str | None = None,
        status: str | None = None,
    ) -> list[FinalReportSnapshot]:
        snapshots = list(self._final_report_snapshots.values())
        snapshots = [
            snapshot
            for snapshot in snapshots
            if self._matches_owner("final_report_snapshots", snapshot.snapshot_id)
        ]
        if report_id is not None:
            snapshots = [snapshot for snapshot in snapshots if snapshot.report_id == report_id]
        if investigation_id is not None:
            snapshots = [snapshot for snapshot in snapshots if snapshot.investigation_id == investigation_id]
        if status is not None:
            snapshots = [snapshot for snapshot in snapshots if snapshot.status.value == status]
        return sorted(snapshots, key=lambda item: item.created_at, reverse=True)

    def update_final_report_snapshot(self, snapshot: FinalReportSnapshot) -> FinalReportSnapshot:
        self.get_final_report_snapshot(snapshot.snapshot_id)
        self._final_report_snapshots[snapshot.snapshot_id] = snapshot
        return snapshot

    def create_data_source(self, data_source: DataSource) -> DataSource:
        data_source.tags = _normalize_tags(data_source.tags)
        self._data_sources[data_source.data_source_id] = data_source
        self._set_owner("data_sources", data_source.data_source_id)
        return data_source

    def get_data_source(self, data_source_id: str) -> DataSource:
        self._require_owner("data_sources", data_source_id)
        try:
            return self._data_sources[data_source_id]
        except KeyError as exc:
            raise KeyError(f"DataSource not found: {data_source_id}") from exc

    def list_data_sources(self, status: str | None = None) -> list[DataSource]:
        sources = list(self._data_sources.values())
        sources = [
            source
            for source in sources
            if self._matches_owner("data_sources", source.data_source_id)
        ]
        if status is not None:
            sources = [source for source in sources if source.status.value == status]
        return sorted(sources, key=lambda item: item.updated_at, reverse=True)

    def update_data_source(self, data_source: DataSource) -> DataSource:
        self.get_data_source(data_source.data_source_id)
        data_source.tags = _normalize_tags(data_source.tags)
        data_source.updated_at = utc_now()
        self._data_sources[data_source.data_source_id] = data_source
        return data_source

    def archive_data_source(self, data_source_id: str) -> DataSource:
        data_source = self.get_data_source(data_source_id)
        data_source.status = DataSourceStatus.ARCHIVED
        return self.update_data_source(data_source)

    def delete_data_source(self, data_source_id: str, delete_file: bool = True) -> bool:
        data_source = self.get_data_source(data_source_id)
        file_deleted = False
        if delete_file and data_source.data_source_type == "csv":
            file_deleted = delete_uploaded_file_if_safe(data_source.location)
        for investigation in self._investigations.values():
            investigation.linked_data_source_ids = [
                item for item in investigation.linked_data_source_ids if item != data_source_id
            ]
            investigation.data_sources = [
                item for item in investigation.data_sources if item != data_source_id
            ]
            self._touch(investigation)
        self._data_source_profiles.pop(data_source_id, None)
        self._data_source_semantic_notes.pop(data_source_id, None)
        del self._data_sources[data_source_id]
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
        self._data_source_profiles[data_source_id] = profile
        self._set_owner("data_source_profiles", data_source_id)
        return profile

    def get_data_source_profile(self, data_source_id: str) -> DataSourceProfile:
        self.get_data_source(data_source_id)
        self._require_owner("data_source_profiles", data_source_id)
        try:
            return self._data_source_profiles[data_source_id]
        except KeyError as exc:
            raise KeyError(f"DataSourceProfile not found: {data_source_id}") from exc

    def get_data_source_semantic_notes(self, data_source_id: str) -> DataSourceSemanticNotes:
        self.get_data_source(data_source_id)
        return self._data_source_semantic_notes.get(
            data_source_id,
            DataSourceSemanticNotes(data_source_id=data_source_id),
        )

    def save_data_source_semantic_notes(self, notes: DataSourceSemanticNotes) -> DataSourceSemanticNotes:
        self.get_data_source(notes.data_source_id)
        notes.column_notes = [_normalize_column_note(note) for note in notes.column_notes if note.column_name]
        notes.global_caveats = _clean_list(notes.global_caveats)
        notes.updated_at = utc_now()
        self._data_source_semantic_notes[notes.data_source_id] = notes
        self._set_owner("data_source_semantic_notes", notes.data_source_id)
        return notes

    def update_column_semantic_note(
        self,
        data_source_id: str,
        column_name: str,
        note: ColumnSemanticNote,
    ) -> DataSourceSemanticNotes:
        notes = self.get_data_source_semantic_notes(data_source_id)
        note.column_name = column_name
        normalized = _normalize_column_note(note)
        notes.column_notes = [item for item in notes.column_notes if item.column_name != column_name]
        notes.column_notes.append(normalized)
        return self.save_data_source_semantic_notes(notes)

    def delete_column_semantic_note(self, data_source_id: str, column_name: str) -> DataSourceSemanticNotes:
        notes = self.get_data_source_semantic_notes(data_source_id)
        notes.column_notes = [item for item in notes.column_notes if item.column_name != column_name]
        return self.save_data_source_semantic_notes(notes)

    def _version_component_ids(self, report_id: str) -> set[str]:
        edges: dict[str, set[str]] = {key: set() for key in self._shareable_reports}
        for report in self._shareable_reports.values():
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
        return seen

    def add_run(self, investigation_id: str, run: InvestigationRun) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        investigation.runs.append(run)
        self._set_owner("runs", run.run_id)
        if run.trace:
            investigation.trace.extend(run.trace)
        self._touch(investigation)
        return investigation

    def add_artifact(self, investigation_id: str, artifact: Artifact) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        investigation.artifacts.append(artifact)
        self._set_owner("artifacts", artifact.artifact_id)
        self._touch(investigation)
        return investigation

    def add_finding(self, investigation_id: str, finding: Finding) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        investigation.findings.append(finding)
        self._set_owner("findings", finding.finding_id)
        self._touch(investigation)
        return investigation

    def update_finding(self, investigation_id: str, finding: Finding) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        for idx, existing in enumerate(investigation.findings):
            if existing.finding_id == finding.finding_id:
                investigation.findings[idx] = finding
                self._touch(investigation)
                return investigation
        raise KeyError(f"Finding not found: {finding.finding_id}")

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
                self._touch(investigation)
                return investigation
        raise KeyError(f"Finding not found: {finding_id}")

    def set_report(self, investigation_id: str, report: DecisionReport) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        report.updated_at = utc_now()
        investigation.report = report
        self._set_owner("reports", report.report_id)
        self._touch(investigation)
        return investigation

    def replace_data_sources(self, investigation_id: str, data_sources: list[str]) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        investigation.data_sources = list(data_sources)
        self._touch(investigation)
        return investigation

    @staticmethod
    def _title_from_question(question: str) -> str:
        cleaned = " ".join((question or "").strip().split())
        if not cleaned:
            return "Untitled investigation"
        return cleaned[:80]

    @staticmethod
    def _touch(investigation: Investigation) -> None:
        investigation.updated_at = utc_now()

    @staticmethod
    def _owner_session_id() -> str | None:
        return get_current_session_id()

    def _set_owner(self, collection: str, item_id: str, owner_session_id: str | None = None) -> None:
        self._owners.setdefault(collection, {})[item_id] = owner_session_id if owner_session_id is not None else self._owner_session_id()

    def _matches_owner(self, collection: str, item_id: str) -> bool:
        owner_session_id = self._owner_session_id()
        if owner_session_id is None:
            return True
        return self._owners.get(collection, {}).get(item_id) == owner_session_id

    def _require_owner(self, collection: str, item_id: str) -> None:
        if not self._matches_owner(collection, item_id):
            raise KeyError(f"{collection} not found: {item_id}")


def _normalize_tags(tags: list[str]) -> list[str]:
    normalized: list[str] = []
    for tag in tags:
        clean = " ".join(str(tag or "").strip().lower().split())
        if clean and clean not in normalized:
            normalized.append(clean)
    return normalized


def _clean_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        item = " ".join(str(value or "").strip().split())
        if item:
            cleaned.append(item)
    return cleaned


def _normalize_column_note(note: ColumnSemanticNote) -> ColumnSemanticNote:
    note.column_name = str(note.column_name or "").strip()
    note.display_name = str(note.display_name).strip() if note.display_name else None
    note.description = str(note.description).strip() if note.description else None
    note.business_meaning = str(note.business_meaning).strip() if note.business_meaning else None
    note.caveats = _clean_list(note.caveats)
    note.examples = _clean_list(note.examples)
    return note
