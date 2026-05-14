from __future__ import annotations

from typing import Optional

from source.product.investigation import (
    Artifact,
    ArtifactVisibility,
    DecisionReport,
    FinalReportSnapshot,
    Finding,
    FindingStatus,
    Investigation,
    InvestigationRun,
    InvestigationRunEvent,
    InvestigationRunStage,
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
        self._data_sources: dict[str, DataSource] = {}
        self._data_source_profiles: dict[str, DataSourceProfile] = {}
        self._data_source_semantic_notes: dict[str, DataSourceSemanticNotes] = {}

    def create_investigation(self, question: str, title: Optional[str] = None) -> Investigation:
        resolved_title = title or self._title_from_question(question)
        investigation = Investigation(title=resolved_title, user_question=question)
        self._investigations[investigation.investigation_id] = investigation
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
        try:
            return self._investigations[investigation_id]
        except KeyError as exc:
            raise KeyError(f"Investigation not found: {investigation_id}") from exc

    def list_investigations(self) -> list[Investigation]:
        return sorted(self._investigations.values(), key=lambda item: item.updated_at, reverse=True)

    def update_status(self, investigation_id: str, status: InvestigationStatus | str) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        investigation.status = InvestigationStatus(status)
        self._touch(investigation)
        return investigation

    def create_investigation_run(self, run: InvestigationRun) -> InvestigationRun:
        self.get_investigation(run.investigation_id)
        run.status = InvestigationRunStatus(run.status)
        self._investigation_runs[run.run_id] = run
        return run

    def update_investigation_run(self, run: InvestigationRun) -> InvestigationRun:
        self.get_investigation_run(run.run_id)
        if isinstance(run.status, str):
            run.status = InvestigationRunStatus(run.status)
        self._investigation_runs[run.run_id] = run
        return run

    def get_investigation_run(self, run_id: str) -> InvestigationRun:
        try:
            return self._investigation_runs[run_id]
        except KeyError as exc:
            raise KeyError(f"InvestigationRun not found: {run_id}") from exc

    def list_investigation_runs(self) -> list[InvestigationRun]:
        return sorted(self._investigation_runs.values(), key=lambda item: item.created_at, reverse=True)

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

    def create_shareable_report(self, report: ShareableReport) -> ShareableReport:
        self.get_investigation(report.investigation_id)
        if report.is_latest:
            anchor_id = report.previous_version_id or report.report_id
            if anchor_id in self._shareable_reports:
                for existing in self.list_report_versions(anchor_id):
                    existing.is_latest = False
        self._shareable_reports[report.report_id] = report
        return report

    def get_shareable_report(self, report_id: str) -> ShareableReport:
        try:
            return self._shareable_reports[report_id]
        except KeyError as exc:
            raise KeyError(f"ShareableReport not found: {report_id}") from exc

    def list_shareable_reports(self, investigation_id: str | None = None) -> list[ShareableReport]:
        reports = list(self._shareable_reports.values())
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
        return comment

    def list_report_comments(self, report_id: str, section_id: str | None = None) -> list[ReportComment]:
        self.get_shareable_report(report_id)
        comments = [comment for comment in self._report_comments.values() if comment.report_id == report_id]
        if section_id is not None:
            comments = [comment for comment in comments if comment.section_id == section_id]
        return sorted(comments, key=lambda item: item.created_at)

    def update_report_comment(self, comment: ReportComment) -> ReportComment:
        if comment.comment_id not in self._report_comments:
            raise KeyError(f"ReportComment not found: {comment.comment_id}")
        comment.updated_at = utc_now()
        self._report_comments[comment.comment_id] = comment
        return comment

    def resolve_report_comment(self, comment_id: str) -> ReportComment:
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
        return snapshot

    def get_final_report_snapshot(self, snapshot_id: str) -> FinalReportSnapshot:
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
        return data_source

    def get_data_source(self, data_source_id: str) -> DataSource:
        try:
            return self._data_sources[data_source_id]
        except KeyError as exc:
            raise KeyError(f"DataSource not found: {data_source_id}") from exc

    def list_data_sources(self, status: str | None = None) -> list[DataSource]:
        sources = list(self._data_sources.values())
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

    def save_data_source_profile(self, data_source_id: str, profile: DataSourceProfile) -> DataSourceProfile:
        self.get_data_source(data_source_id)
        self._data_source_profiles[data_source_id] = profile
        return profile

    def get_data_source_profile(self, data_source_id: str) -> DataSourceProfile:
        self.get_data_source(data_source_id)
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
        if run.trace:
            investigation.trace.extend(run.trace)
        self._touch(investigation)
        return investigation

    def add_artifact(self, investigation_id: str, artifact: Artifact) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        investigation.artifacts.append(artifact)
        self._touch(investigation)
        return investigation

    def add_finding(self, investigation_id: str, finding: Finding) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        investigation.findings.append(finding)
        self._touch(investigation)
        return investigation

    def set_report(self, investigation_id: str, report: DecisionReport) -> Investigation:
        investigation = self.get_investigation(investigation_id)
        report.updated_at = utc_now()
        investigation.report = report
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
