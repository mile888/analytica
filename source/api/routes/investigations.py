from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from source.api.deps import get_store
from source.api.serialization import to_jsonable
from source.product.event_stream import build_event_stream_response
from source.product.cross_investigation import extract_cross_investigation_patterns, extract_reusable_hypotheses
from source.product.investigation import (
    ArtifactType,
    InvestigationMessage,
    InvestigationMessageRole,
    InvestigationMessageType,
    InvestigationMemoryItem,
    InvestigationMemoryStatus,
    InvestigationMemoryType,
)
from source.product.question_suggestions import (
    build_cross_investigation_question_suggestions,
    build_data_aware_question_suggestions,
    build_thread_aware_question_suggestions,
)
from source.product.branch_workspace import activate_branch, branch_dtos_for_investigation
from source.product.report_builder import build_shareable_report
from source.product.run_service import InvestigationRunService
from source.product.semantic_layer import build_investigation_thread_state
from source.product.organizational_workflows import (
    build_organizational_playbooks,
    evaluate_workflow_progress,
    recommend_organizational_workflow,
    report_standard_for_template,
    review_investigation,
    validation_expectations_for_investigation,
)


router = APIRouter(prefix="/investigations", tags=["investigations"])
runs_router = APIRouter(prefix="/investigation-runs", tags=["investigation-runs"])


class InvestigationCreate(BaseModel):
    question: str
    title: str | None = None
    data_source_ids: list[str] = Field(default_factory=list)


class InvestigationRunRequest(BaseModel):
    data_source_ids: list[str] = Field(default_factory=list)
    force_refresh_context: bool = False
    message_id: str | None = None
    analysis_mode: str = "exploration"


class InvestigationMessageCreate(BaseModel):
    content: str
    type: str = "follow_up"
    role: str = "user"
    metadata: dict = Field(default_factory=dict)


class InvestigationMemoryCreate(BaseModel):
    type: str
    content: str
    title: str = ""
    status: str = "active"
    metadata: dict = Field(default_factory=dict)


class InvestigationMemoryUpdate(BaseModel):
    content: str | None = None
    title: str | None = None
    status: str | None = None
    metadata: dict | None = None


class PromoteFindingPayload(BaseModel):
    type: str


class EvidenceLinkPayload(BaseModel):
    type: str
    id: str
    label: str | None = None
    href: str | None = None


class ShareableReportCreate(BaseModel):
    template: str = "executive_summary"
    include_technical: bool = False


class ArtifactReportSelectionPayload(BaseModel):
    selected: bool = True


@router.get("")
def list_investigations():
    return [to_jsonable(item) for item in get_store().list_investigations()]


@router.post("")
def create_investigation(payload: InvestigationCreate):
    store = get_store()
    investigation = store.create_investigation(payload.question, title=payload.title)
    for data_source_id in payload.data_source_ids:
        store.link_data_source_to_investigation(investigation.investigation_id, data_source_id)
    return to_jsonable(store.get_investigation(investigation.investigation_id))


@router.get("/{investigation_id}")
def get_investigation(investigation_id: str):
    try:
        return to_jsonable(get_store().get_investigation(investigation_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{investigation_id}/branches")
def list_investigation_branches(investigation_id: str):
    try:
        return branch_dtos_for_investigation(get_store().get_investigation(investigation_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{investigation_id}/branches/{branch_id}/activate")
def activate_investigation_branch(investigation_id: str, branch_id: str):
    try:
        store = get_store()
        investigation = store.get_investigation(investigation_id)
        metadata = activate_branch(investigation, branch_id)
        updater = getattr(store, "update_investigation_metadata", None)
        if callable(updater):
            updater(investigation_id, metadata)
        return {"active_branch_id": branch_id, "branches": branch_dtos_for_investigation(store.get_investigation(investigation_id))}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{investigation_id}")
def delete_investigation(investigation_id: str):
    try:
        get_store().delete_investigation(investigation_id)
        return {"deleted": True, "id": investigation_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{investigation_id}/run")
def run_investigation(investigation_id: str, payload: InvestigationRunRequest | None = None):
    body = payload or InvestigationRunRequest()
    try:
        service = InvestigationRunService(get_store())
        return to_jsonable(
            service.run_investigation(
                investigation_id,
                data_source_ids=body.data_source_ids,
                force_refresh_context=body.force_refresh_context,
                message_id=body.message_id,
                analysis_mode=body.analysis_mode,
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{investigation_id}/messages")
def add_investigation_message(investigation_id: str, payload: InvestigationMessageCreate):
    try:
        message = InvestigationMessage(
            investigation_id=investigation_id,
            role=InvestigationMessageRole(payload.role),
            message_type=InvestigationMessageType(payload.type),
            content=payload.content,
            metadata=payload.metadata,
        )
        return to_jsonable(get_store().add_investigation_message(message))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{investigation_id}/messages")
def list_investigation_messages(investigation_id: str, limit: int | None = None):
    try:
        return [to_jsonable(item) for item in get_store().list_investigation_messages(investigation_id, limit=limit)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{investigation_id}/memory")
def list_investigation_memory(investigation_id: str, type: str | None = None):
    try:
        return [to_jsonable(item) for item in get_store().list_investigation_memory(investigation_id, memory_type=type)]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{investigation_id}/memory")
def add_investigation_memory(investigation_id: str, payload: InvestigationMemoryCreate):
    try:
        item = InvestigationMemoryItem(
            investigation_id=investigation_id,
            memory_type=InvestigationMemoryType(payload.type),
            content=payload.content,
            title=payload.title,
            status=InvestigationMemoryStatus(payload.status),
            metadata=payload.metadata,
        )
        return to_jsonable(get_store().add_investigation_memory_item(item))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/{investigation_id}/memory/{memory_id}")
def update_investigation_memory(investigation_id: str, memory_id: str, payload: InvestigationMemoryUpdate):
    try:
        store = get_store()
        item = store.get_investigation_memory_item(memory_id)
        if item.investigation_id != investigation_id:
            raise KeyError(f"InvestigationMemoryItem not found for investigation {investigation_id}: {memory_id}")
        return to_jsonable(
            store.update_investigation_memory_item(
                memory_id,
                content=payload.content,
                title=payload.title,
                status=payload.status,
                metadata=payload.metadata,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{investigation_id}/suggested-questions")
def get_investigation_suggested_questions(investigation_id: str, limit: int = 8):
    store = get_store()
    try:
        investigation = store.get_investigation(investigation_id)
        suggestions: list[str] = []
        for data_source_id in investigation.linked_data_source_ids:
            profile = None
            context = None
            notes = None
            with suppress(KeyError):
                profile = store.get_data_source_profile(data_source_id)
            with suppress(KeyError):
                from source.product.data_context import build_data_source_usage_context

                context = build_data_source_usage_context(store, data_source_id)
            with suppress(KeyError):
                notes = store.get_data_source_semantic_notes(data_source_id)
            suggestions.extend(
                build_data_aware_question_suggestions(
                    profile=profile,
                    usage_context=context,
                    semantic_notes=notes,
                    limit=limit,
                )
            )
        thread_state = build_investigation_thread_state(
            {
                "initial_question": investigation.user_question,
                "latest_chart_context": _latest_chart_context(investigation.artifacts),
                "latest_findings": [
                    item.text
                    for item in investigation.findings[-5:]
                    if getattr(item, "status", None) != "rejected"
                ],
                "artifact_titles": [
                    artifact.title
                    for artifact in investigation.artifacts[-8:]
                    if getattr(artifact, "visibility", None) != "hidden"
                ],
            }
        )
        thread_suggestions = build_thread_aware_question_suggestions(thread_state, limit=limit)
        historical_investigations = store.list_investigations()
        patterns = extract_cross_investigation_patterns(
            historical_investigations,
            target_investigation_id=investigation_id,
        )
        reusable_hypotheses = extract_reusable_hypotheses(historical_investigations)
        cross_suggestions = build_cross_investigation_question_suggestions(
            patterns,
            reusable_hypotheses,
            limit=limit,
        )
        suggestions = [*thread_suggestions, *cross_suggestions, *suggestions]
        if investigation.findings:
            suggestions.append("Which current findings need stronger evidence?")
        deduped = list(dict.fromkeys(suggestions))
        return {"suggestions": deduped[: max(1, min(limit, 20))]}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{investigation_id}/workflow-guidance")
def get_investigation_workflow_guidance(investigation_id: str):
    store = get_store()
    try:
        investigation = store.get_investigation(investigation_id)
        patterns = extract_cross_investigation_patterns(
            store.list_investigations(),
            target_investigation_id=investigation_id,
        )
        workflow = recommend_organizational_workflow(investigation, patterns)
        progress = evaluate_workflow_progress(investigation, workflow)
        review = review_investigation(investigation, patterns)
        expectations = validation_expectations_for_investigation(investigation, patterns)
        playbooks = build_organizational_playbooks(patterns)
        standard = report_standard_for_template("executive_summary")
        return to_jsonable(
            {
                "workflow": asdict(workflow),
                "progress": [asdict(item) for item in progress],
                "review": asdict(review),
                "validation_expectations": [asdict(item) for item in expectations],
                "playbooks": [asdict(item) for item in playbooks],
                "report_standard": asdict(standard),
            }
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _latest_chart_context(artifacts: list[object]) -> dict[str, object]:
    for artifact in reversed(artifacts or []):
        if getattr(artifact, "artifact_type", None) != ArtifactType.CHART:
            continue
        content = getattr(artifact, "content", None)
        metadata = getattr(artifact, "metadata", None)
        content = content if isinstance(content, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        return {
            "title": getattr(artifact, "title", ""),
            "chart_type": content.get("chart_type") or metadata.get("chart_type"),
            "metric": content.get("metric") or metadata.get("metric") or content.get("y"),
            "dimension": metadata.get("dimension") or content.get("dimension") or content.get("x"),
            "time_axis": content.get("timestamp") or metadata.get("timestamp"),
        }
    return {}


@router.get("/{investigation_id}/runs")
def list_runs_for_investigation(investigation_id: str):
    try:
        return [to_jsonable(item) for item in get_store().list_runs_for_investigation(investigation_id)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{investigation_id}/findings/{finding_id}/promote-memory")
def promote_finding_to_memory(investigation_id: str, finding_id: str, payload: PromoteFindingPayload):
    try:
        memory_type = InvestigationMemoryType(payload.type)
        if memory_type not in {
            InvestigationMemoryType.ASSUMPTION,
            InvestigationMemoryType.RISK,
            InvestigationMemoryType.DECISION,
        }:
            raise ValueError("Findings can be promoted only to assumption, risk, or decision.")
        investigation = get_store().get_investigation(investigation_id)
        finding = next((item for item in investigation.findings if item.finding_id == finding_id), None)
        if finding is None:
            raise KeyError(f"Finding not found: {finding_id}")
        memory = InvestigationMemoryItem(
            investigation_id=investigation_id,
            memory_type=memory_type,
            title=finding.title or "Promoted finding",
            content=finding.text,
            metadata={
                "source_type": "finding",
                "source_id": finding.finding_id,
                "promoted_from": "finding",
                "promoted_at": finding.created_at.isoformat(),
            },
        )
        return to_jsonable(get_store().add_investigation_memory_item(memory))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{investigation_id}/findings/{finding_id}/evidence")
def link_finding_evidence(investigation_id: str, finding_id: str, payload: EvidenceLinkPayload):
    try:
        evidence = {
            "type": payload.type,
            "id": payload.id,
            "label": payload.label or payload.id,
            "href": payload.href,
        }
        investigation = get_store().add_finding_evidence_link(investigation_id, finding_id, evidence)
        finding = next((item for item in investigation.findings if item.finding_id == finding_id), None)
        if finding is None:
            raise KeyError(f"Finding not found: {finding_id}")
        return to_jsonable(finding)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{investigation_id}/artifacts/{artifact_id}/report-selection")
def update_artifact_report_selection(
    investigation_id: str,
    artifact_id: str,
    payload: ArtifactReportSelectionPayload | None = None,
):
    body = payload or ArtifactReportSelectionPayload()
    try:
        store = get_store()
        investigation = store.update_artifact_metadata(
            investigation_id,
            artifact_id,
            {"selected_for_report": bool(body.selected)},
        )
        artifact = next((item for item in investigation.artifacts if item.artifact_id == artifact_id), None)
        if artifact is None:
            raise KeyError(f"Artifact not found: {artifact_id}")
        return to_jsonable(artifact)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{investigation_id}/reports")
def list_reports_for_investigation(investigation_id: str):
    try:
        store = get_store()
        store.get_investigation(investigation_id)
        return [to_jsonable(item) for item in store.list_shareable_reports(investigation_id=investigation_id)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{investigation_id}/reports")
def create_report_for_investigation(investigation_id: str, payload: ShareableReportCreate | None = None):
    try:
        store = get_store()
        investigation = store.get_investigation(investigation_id)
        body = payload or ShareableReportCreate()
        patterns = extract_cross_investigation_patterns(
            store.list_investigations(),
            target_investigation_id=investigation_id,
        )
        report = build_shareable_report(
            investigation,
            template=body.template,
            include_technical=body.include_technical,
            analytical_patterns=patterns,
            organizational_review=review_investigation(investigation, patterns),
            report_standard=report_standard_for_template(body.template),
        )
        previous_versions = [
            item
            for item in store.list_shareable_reports(investigation_id=investigation_id)
            if item.template.value == report.template.value and item.is_latest
        ]
        if previous_versions:
            latest = sorted(previous_versions, key=lambda item: item.updated_at.isoformat())[-1]
            report.previous_version_id = latest.report_id
            report.version = latest.version + 1
        return to_jsonable(store.create_shareable_report(report))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{investigation_id}/events")
def list_events_for_investigation(investigation_id: str, after: str | None = None, limit: int = 100):
    try:
        events = get_store().list_events_for_investigation(investigation_id, limit=5000)
        return to_jsonable(build_event_stream_response(events, cursor=after, limit=limit))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@runs_router.get("/{run_id}")
def get_investigation_run(run_id: str):
    try:
        return to_jsonable(get_store().get_investigation_run(run_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@runs_router.get("/{run_id}/events")
def list_investigation_run_events(run_id: str, after: str | None = None, limit: int = 100):
    try:
        events = get_store().list_investigation_run_events(run_id)
        return to_jsonable(build_event_stream_response(events, cursor=after, limit=limit))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
