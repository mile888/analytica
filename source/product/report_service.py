from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from source.product.exporter import (
    export_shareable_report_html,
    export_shareable_report_markdown,
    export_shareable_report_txt,
)
from source.product.investigation import (
    DecisionMetadata,
    DecisionStatus,
    FinalReportSnapshot,
    FinalReportStatus,
    ReportApprovalStatus,
    ReportComment,
    ReportSection,
    ShareableReport,
    ShareableReportStatus,
    SectionReviewStatus,
    new_id,
    utc_now,
)
from source.product.readiness import evaluate_report_readiness


class ReportEditingService:
    def __init__(self, store) -> None:
        self.store = store

    def update_section_title(self, report_id: str, section_id: str, title: str, user: str = "user") -> ShareableReport:
        report = self._editable_snapshot(report_id, f"Updated section title: {section_id}")
        section = self._find_section(report, section_id)
        self._record_edit(section, "title", section.title, title, user)
        section.title = title
        return self.store.create_shareable_report(report)

    def update_section_content(self, report_id: str, section_id: str, content: str, user: str = "user") -> ShareableReport:
        report = self._editable_snapshot(report_id, f"Updated section content: {section_id}")
        section = self._find_section(report, section_id)
        self._record_edit(section, "content", section.content, content, user)
        section.content = content
        return self.store.create_shareable_report(report)

    def update_section(
        self,
        report_id: str,
        section_id: str,
        title: str | None = None,
        content: str | None = None,
        user: str = "user",
    ) -> ShareableReport:
        report = self._editable_snapshot(report_id, f"Updated section: {section_id}")
        section = self._find_section(report, section_id)
        if title is not None:
            self._record_edit(section, "title", section.title, title, user)
            section.title = title
        if content is not None:
            self._record_edit(section, "content", section.content, content, user)
            section.content = content
        return self.store.create_shareable_report(report)

    def reorder_sections(self, report_id: str, section_ids: Sequence[str], user: str = "user") -> ShareableReport:
        report = self._editable_snapshot(report_id, "Reordered sections")
        by_id = {section.section_id: section for section in report.sections}
        ordered = [by_id[section_id] for section_id in section_ids if section_id in by_id]
        ordered.extend(section for section in report.sections if section.section_id not in set(section_ids))
        for idx, section in enumerate(ordered, start=1):
            old_order = section.order
            section.order = idx * 10
            self._record_edit(section, "order", old_order, section.order, user)
        report.sections = ordered
        return self.store.create_shareable_report(report)

    def add_section(
        self,
        report_id: str,
        title: str,
        content: str = "",
        order: int | None = None,
        user: str = "user",
    ) -> ShareableReport:
        report = self._editable_snapshot(report_id, f"Added section: {title}")
        resolved_order = order if order is not None else ((max((section.order for section in report.sections), default=0)) + 10)
        section = ReportSection(
            title=title,
            content=content,
            order=resolved_order,
            edited_by_user=True,
            created_by=user,
            edit_history=[
                {
                    "action": "created",
                    "user": user,
                    "timestamp": utc_now().isoformat(),
                }
            ],
        )
        report.sections.append(section)
        report.sections.sort(key=lambda item: (item.order, item.title))
        return self.store.create_shareable_report(report)

    def remove_section(self, report_id: str, section_id: str, user: str = "user") -> ShareableReport:
        report = self._editable_snapshot(report_id, f"Removed section: {section_id}")
        self._find_section(report, section_id)
        report.sections = [section for section in report.sections if section.section_id != section_id]
        report.metadata.setdefault("edit_history", []).append(
            {"action": "removed_section", "section_id": section_id, "user": user, "timestamp": utc_now().isoformat()}
        )
        return self.store.create_shareable_report(report)

    def duplicate_section(self, report_id: str, section_id: str, user: str = "user") -> ShareableReport:
        report = self._editable_snapshot(report_id, f"Duplicated section: {section_id}")
        section = self._find_section(report, section_id)
        duplicate = copy.deepcopy(section)
        duplicate.section_id = new_id("section")
        duplicate.title = f"{section.title} copy"
        duplicate.order = section.order + 1
        duplicate.version = 1
        duplicate.edited_by_user = True
        duplicate.created_by = user
        duplicate.updated_at = utc_now()
        duplicate.edit_history = [
            {
                "action": "duplicated",
                "source_section_id": section.section_id,
                "user": user,
                "timestamp": duplicate.updated_at.isoformat(),
            }
        ]
        report.sections.append(duplicate)
        report.sections.sort(key=lambda item: (item.order, item.title))
        for idx, item in enumerate(report.sections, start=1):
            item.order = idx * 10
        return self.store.create_shareable_report(report)

    def add_comment(
        self,
        report_id: str,
        section_id: str,
        text: str,
        author: str = "reviewer",
    ) -> ReportComment:
        self._find_section(self.store.get_shareable_report(report_id), section_id)
        comment = ReportComment(report_id=report_id, section_id=section_id, text=text, author=author)
        return self.store.add_report_comment(comment)

    def resolve_comment(self, comment_id: str) -> ReportComment:
        return self.store.resolve_report_comment(comment_id)

    def set_section_review_status(
        self,
        report_id: str,
        section_id: str,
        status: SectionReviewStatus | str,
    ) -> ShareableReport:
        resolved = SectionReviewStatus(status)
        report = self._editable_snapshot(report_id, f"Set section review status: {section_id} -> {resolved.value}")
        section = self._find_section(report, section_id)
        self._record_edit(section, "review_status", section.review_status.value, resolved.value, "reviewer")
        section.review_status = resolved
        return self.store.create_shareable_report(report)

    def approve_section(self, report_id: str, section_id: str) -> ShareableReport:
        return self.set_section_review_status(report_id, section_id, SectionReviewStatus.APPROVED)

    def request_section_changes(self, report_id: str, section_id: str) -> ShareableReport:
        return self.set_section_review_status(report_id, section_id, SectionReviewStatus.CHANGES_REQUESTED)

    def send_to_review(self, report_id: str) -> ShareableReport:
        return self.store.set_report_approval_status(report_id, ReportApprovalStatus.IN_REVIEW)

    def request_changes(self, report_id: str, notes: str) -> ShareableReport:
        return self.store.set_report_approval_status(
            report_id,
            ReportApprovalStatus.CHANGES_REQUESTED,
            reviewer_notes=notes,
        )

    def approve_report(self, report_id: str, approved_by: str = "reviewer", force: bool = False) -> ShareableReport:
        report = self.store.get_shareable_report(report_id)
        comments = self.store.list_report_comments(report_id)
        try:
            investigation = self.store.get_investigation(report.investigation_id)
        except KeyError:
            investigation = None
        readiness = evaluate_report_readiness(report, investigation=investigation, comments=comments)
        if not readiness.is_ready and not force:
            failed = [check.message for check in readiness.checks if check.blocking and check.status.value == "failed"]
            raise ValueError("Cannot approve report: " + "; ".join(failed))
        return self.store.set_report_approval_status(
            report_id,
            ReportApprovalStatus.APPROVED,
            approved_by=approved_by,
        )

    def create_final_report_snapshot(
        self,
        report_id: str,
        created_by: str = "user",
        force: bool = False,
    ) -> FinalReportSnapshot:
        report = self.store.get_shareable_report(report_id)
        comments = self.store.list_report_comments(report_id)
        try:
            investigation = self.store.get_investigation(report.investigation_id)
        except KeyError:
            investigation = None
        readiness = evaluate_report_readiness(report, investigation=investigation, comments=comments)
        if report.approval_status != ReportApprovalStatus.APPROVED and not force:
            raise ValueError("Cannot create final snapshot: report is not approved.")
        if not readiness.is_ready and not force:
            failed = [check.message for check in readiness.checks if check.blocking and check.status.value == "failed"]
            raise ValueError("Cannot create final snapshot: " + "; ".join(failed))

        snapshot = FinalReportSnapshot(
            report_id=report.report_id,
            investigation_id=report.investigation_id,
            report_version=report.version,
            title=report.title,
            created_by=created_by,
            markdown_content=export_shareable_report_markdown(report),
            html_content=export_shareable_report_html(report),
            txt_content=export_shareable_report_txt(report),
            readiness_snapshot=_jsonable(readiness),
            approval_status=report.approval_status,
            approved_at=report.approved_at,
            approved_by=report.approved_by,
            source_report_json=_jsonable(report),
        )
        return self.store.create_final_report_snapshot(snapshot)

    def get_final_report_snapshot(self, snapshot_id: str) -> FinalReportSnapshot:
        return self.store.get_final_report_snapshot(snapshot_id)

    def list_final_report_snapshots(
        self,
        report_id: str | None = None,
        investigation_id: str | None = None,
    ) -> list[FinalReportSnapshot]:
        return self.store.list_final_report_snapshots(report_id=report_id, investigation_id=investigation_id)

    def revoke_final_report_snapshot(self, snapshot_id: str, reason: str | None = None) -> FinalReportSnapshot:
        snapshot = self.store.get_final_report_snapshot(snapshot_id)
        snapshot.status = FinalReportStatus.REVOKED
        if reason:
            snapshot.metadata["revocation_reason"] = reason
        snapshot.metadata["revoked_at"] = utc_now().isoformat()
        return self.store.update_final_report_snapshot(snapshot)

    def update_final_report_metadata(
        self,
        snapshot_id: str,
        decision_metadata: DecisionMetadata | dict[str, Any],
    ) -> FinalReportSnapshot:
        snapshot = self.store.get_final_report_snapshot(snapshot_id)
        snapshot.decision_metadata = _coerce_decision_metadata(decision_metadata)
        return self.store.update_final_report_snapshot(snapshot)

    def add_final_report_tags(self, snapshot_id: str, tags: Sequence[str]) -> FinalReportSnapshot:
        snapshot = self.store.get_final_report_snapshot(snapshot_id)
        existing = _normalize_tags(snapshot.decision_metadata.tags)
        for tag in _normalize_tags(tags):
            if tag not in existing:
                existing.append(tag)
        snapshot.decision_metadata.tags = existing
        return self.store.update_final_report_snapshot(snapshot)

    def remove_final_report_tag(self, snapshot_id: str, tag: str) -> FinalReportSnapshot:
        snapshot = self.store.get_final_report_snapshot(snapshot_id)
        normalized = _normalize_tag(tag)
        snapshot.decision_metadata.tags = [
            existing for existing in _normalize_tags(snapshot.decision_metadata.tags) if existing != normalized
        ]
        return self.store.update_final_report_snapshot(snapshot)

    def set_decision_status(self, snapshot_id: str, status: DecisionStatus | str) -> FinalReportSnapshot:
        snapshot = self.store.get_final_report_snapshot(snapshot_id)
        snapshot.decision_metadata.decision_status = DecisionStatus(status)
        return self.store.update_final_report_snapshot(snapshot)

    def _editable_snapshot(self, report_id: str, note: str) -> ShareableReport:
        base = self.store.get_latest_report_version(report_id)
        report = copy.deepcopy(base)
        report.report_id = new_id("share")
        report.previous_version_id = base.report_id
        report.version = max(item.version for item in self.store.list_report_versions(report_id)) + 1
        report.is_latest = True
        report.status = ShareableReportStatus.DRAFT
        report.version_note = note
        report.created_at = utc_now()
        report.updated_at = report.created_at
        base.is_latest = False
        self.store.update_shareable_report(base)
        return report

    @staticmethod
    def _find_section(report: ShareableReport, section_id: str) -> ReportSection:
        for section in report.sections:
            if section.section_id == section_id:
                return section
        raise KeyError(f"ReportSection not found: {section_id}")

    @staticmethod
    def _record_edit(section: ReportSection, field: str, before: Any, after: Any, user: str) -> None:
        section.edited_by_user = True
        section.version += 1
        if field in {"title", "content"}:
            section.review_status = SectionReviewStatus.DRAFT
        section.updated_at = utc_now()
        section.edit_history.append(
            {
                "field": field,
                "before": before,
                "after": after,
                "user": user,
                "timestamp": section.updated_at.isoformat(),
            }
        )


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


def _coerce_decision_metadata(value: DecisionMetadata | dict[str, Any]) -> DecisionMetadata:
    if isinstance(value, DecisionMetadata):
        value.tags = _normalize_tags(value.tags)
        return value
    return DecisionMetadata(
        tags=_normalize_tags(value.get("tags") or []),
        owner=value.get("owner") or None,
        audience=value.get("audience") or None,
        business_area=value.get("business_area") or None,
        decision_date=value.get("decision_date") or None,
        decision_status=DecisionStatus(value.get("decision_status") or "unknown"),
        short_description=value.get("short_description") or None,
    )


def _normalize_tags(tags: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for tag in tags:
        clean = _normalize_tag(tag)
        if clean and clean not in normalized:
            normalized.append(clean)
    return normalized


def _normalize_tag(tag: str) -> str:
    return " ".join(str(tag or "").strip().lower().split())
