from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class InvestigationStatus(StrEnum):
    DRAFT = "draft"
    RUNNING = "running"
    NEEDS_REVIEW = "needs_review"
    NEEDS_MORE_ANALYSIS = "needs_more_analysis"
    VERIFIED = "verified"
    FAILED = "failed"
    ARCHIVED = "archived"


class ArtifactType(StrEnum):
    TABLE = "table"
    CHART = "chart"
    SQL = "sql"
    PYTHON_CODE = "python_code"
    TEXT = "text"
    REPORT = "report"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


class ArtifactVisibility(StrEnum):
    USER = "user"
    TECHNICAL = "technical"
    HIDDEN = "hidden"


class FindingStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ShareableReportStatus(StrEnum):
    DRAFT = "draft"
    READY = "ready"
    ARCHIVED = "archived"


class ShareableReportTemplate(StrEnum):
    EXECUTIVE_SUMMARY = "executive_summary"
    PRODUCT_DECISION_MEMO = "product_decision_memo"
    TECHNICAL_APPENDIX = "technical_appendix"


class ReportCommentStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"


class ReportApprovalStatus(StrEnum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    CHANGES_REQUESTED = "changes_requested"
    APPROVED = "approved"


class SectionReviewStatus(StrEnum):
    DRAFT = "draft"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"


class ReadinessCheckStatus(StrEnum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class ReadinessSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FinalReportStatus(StrEnum):
    FINAL = "final"
    REVOKED = "revoked"


class DecisionStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    UNKNOWN = "unknown"


class InvestigationRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InvestigationRunStage(StrEnum):
    PREPARING_DATA = "preparing_data"
    BUILDING_CONTEXT = "building_context"
    RUNNING_ANALYSIS = "running_analysis"
    VALIDATING_RESULTS = "validating_results"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"


class InvestigationRunEventType(StrEnum):
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    ARTIFACT_CREATED = "artifact_created"
    REPORT_CREATED = "report_created"
    VALIDATION_CHECK = "validation_check"


class InvestigationRunEventSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Artifact:
    artifact_type: ArtifactType = ArtifactType.UNKNOWN
    title: str = ""
    content: Any = None
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    visibility: ArtifactVisibility = ArtifactVisibility.USER
    pinned: bool = False
    run_id: str | None = None
    artifact_id: str = field(default_factory=lambda: new_id("artifact"))
    created_at: datetime = field(default_factory=utc_now)

    @property
    def id(self) -> str:
        return self.artifact_id


@dataclass
class Finding:
    text: str
    title: str = ""
    evidence: list[str] = field(default_factory=list)
    evidence_artifact_ids: list[str] = field(default_factory=list)
    confidence: float | None = None
    status: FindingStatus = FindingStatus.PROPOSED
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    finding_id: str = field(default_factory=lambda: new_id("finding"))
    created_at: datetime = field(default_factory=utc_now)

    @property
    def id(self) -> str:
        return self.finding_id


@dataclass
class DecisionReport:
    summary: str = ""
    question: str = ""
    answer: str = ""
    key_findings: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    report_id: str = field(default_factory=lambda: new_id("report"))
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    @property
    def id(self) -> str:
        return self.report_id

    def __post_init__(self) -> None:
        if not self.answer and self.summary:
            self.answer = self.summary
        if not self.summary and self.answer:
            self.summary = self.answer


@dataclass
class InvestigationRun:
    status: InvestigationStatus | InvestigationRunStatus = InvestigationStatus.RUNNING
    trace: list[dict[str, Any]] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    investigation_id: str = ""
    current_stage: InvestigationRunStage = InvestigationRunStage.PREPARING_DATA
    data_source_ids: list[str] = field(default_factory=list)
    run_context_summary: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    artifact_ids: list[str] = field(default_factory=list)
    report_ids: list[str] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: new_id("run"))
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.run_id

    def __post_init__(self) -> None:
        if self.error_message and not self.error:
            self.error = self.error_message
        if self.error and not self.error_message:
            self.error_message = self.error
        if self.completed_at and not self.finished_at:
            self.finished_at = self.completed_at
        if self.finished_at and not self.completed_at:
            self.completed_at = self.finished_at


@dataclass
class InvestigationRunEvent:
    run_id: str
    investigation_id: str
    event_type: InvestigationRunEventType = InvestigationRunEventType.INFO
    stage: InvestigationRunStage | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    severity: InvestigationRunEventSeverity = InvestigationRunEventSeverity.INFO
    event_id: str = field(default_factory=lambda: new_id("event"))
    created_at: datetime = field(default_factory=utc_now)

    @property
    def id(self) -> str:
        return self.event_id


@dataclass
class Investigation:
    title: str
    user_question: str
    status: InvestigationStatus = InvestigationStatus.DRAFT
    data_sources: list[str] = field(default_factory=list)
    linked_data_source_ids: list[str] = field(default_factory=list)
    runs: list[InvestigationRun] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    report: DecisionReport | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    investigation_id: str = field(default_factory=lambda: new_id("inv"))
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.investigation_id


@dataclass
class ReportSection:
    title: str
    content: str = ""
    order: int = 0
    artifact_ids: list[str] = field(default_factory=list)
    edited_by_user: bool = False
    created_by: str = "ai"
    version: int = 1
    review_status: SectionReviewStatus = SectionReviewStatus.DRAFT
    updated_at: datetime = field(default_factory=utc_now)
    edit_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    section_id: str = field(default_factory=lambda: new_id("section"))

    @property
    def id(self) -> str:
        return self.section_id


@dataclass
class ShareableReport:
    investigation_id: str
    title: str
    template: ShareableReportTemplate = ShareableReportTemplate.EXECUTIVE_SUMMARY
    status: ShareableReportStatus = ShareableReportStatus.DRAFT
    version: int = 1
    previous_version_id: str | None = None
    is_latest: bool = True
    version_note: str = ""
    approval_status: ReportApprovalStatus = ReportApprovalStatus.DRAFT
    reviewer_notes: str = ""
    approved_at: datetime | None = None
    approved_by: str | None = None
    sections: list[ReportSection] = field(default_factory=list)
    source_finding_ids: list[str] = field(default_factory=list)
    source_artifact_ids: list[str] = field(default_factory=list)
    include_technical: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: new_id("share"))
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    @property
    def id(self) -> str:
        return self.report_id


@dataclass
class ReportComment:
    report_id: str
    section_id: str
    text: str
    status: ReportCommentStatus = ReportCommentStatus.OPEN
    author: str = "reviewer"
    metadata: dict[str, Any] = field(default_factory=dict)
    comment_id: str = field(default_factory=lambda: new_id("comment"))
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    resolved_at: datetime | None = None

    @property
    def id(self) -> str:
        return self.comment_id


@dataclass
class ReadinessCheck:
    name: str
    status: ReadinessCheckStatus
    severity: ReadinessSeverity
    message: str
    blocking: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    check_id: str = field(default_factory=lambda: new_id("check"))

    @property
    def id(self) -> str:
        return self.check_id


@dataclass
class ReportReadinessResult:
    report_id: str
    is_ready: bool
    blocking_count: int
    warning_count: int
    checks: list[ReadinessCheck] = field(default_factory=list)
    generated_at: datetime = field(default_factory=utc_now)


@dataclass
class DecisionMetadata:
    tags: list[str] = field(default_factory=list)
    owner: str | None = None
    audience: str | None = None
    business_area: str | None = None
    decision_date: str | None = None
    decision_status: DecisionStatus = DecisionStatus.UNKNOWN
    short_description: str | None = None


@dataclass
class FinalReportSnapshot:
    report_id: str
    investigation_id: str
    report_version: int
    title: str
    markdown_content: str
    html_content: str
    readiness_snapshot: dict[str, Any]
    approval_status: ReportApprovalStatus
    source_report_json: dict[str, Any]
    created_by: str = "user"
    status: FinalReportStatus = FinalReportStatus.FINAL
    approved_at: datetime | None = None
    approved_by: str | None = None
    decision_metadata: DecisionMetadata = field(default_factory=DecisionMetadata)
    metadata: dict[str, Any] = field(default_factory=dict)
    snapshot_id: str = field(default_factory=lambda: new_id("final"))
    created_at: datetime = field(default_factory=utc_now)

    @property
    def id(self) -> str:
        return self.snapshot_id
