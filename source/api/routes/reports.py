from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from source.api.deps import get_store
from source.api.serialization import to_jsonable
from source.product.readiness import evaluate_report_readiness
from source.product.report_service import ReportEditingService


router = APIRouter(prefix="/reports", tags=["reports"])


class CommentCreate(BaseModel):
    section_id: str
    text: str
    author: str = "reviewer"


class ApprovalRequest(BaseModel):
    approved_by: str = "reviewer"
    force: bool = False


class FinalizeRequest(BaseModel):
    created_by: str = "user"
    force: bool = False


class RequestChangesPayload(BaseModel):
    notes: str


class RevokeSnapshotPayload(BaseModel):
    reason: str | None = None


@router.get("/{report_id}")
def get_report(report_id: str):
    try:
        return to_jsonable(get_store().get_shareable_report(report_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{report_id}/versions")
def get_report_versions(report_id: str):
    try:
        return [to_jsonable(item) for item in get_store().list_report_versions(report_id)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{report_id}/finalize")
def finalize_report(report_id: str, payload: FinalizeRequest | None = None):
    try:
        body = payload or FinalizeRequest()
        return to_jsonable(
            ReportEditingService(get_store()).create_final_report_snapshot(
                report_id,
                created_by=body.created_by,
                force=body.force,
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{report_id}/final-snapshots")
def get_report_final_snapshots(report_id: str):
    try:
        get_store().get_shareable_report(report_id)
        return [to_jsonable(item) for item in get_store().list_final_report_snapshots(report_id=report_id)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{report_id}/readiness")
def get_report_readiness(report_id: str):
    store = get_store()
    try:
        report = store.get_shareable_report(report_id)
        try:
            investigation = store.get_investigation(report.investigation_id)
        except KeyError:
            investigation = None
        comments = store.list_report_comments(report_id)
        return to_jsonable(evaluate_report_readiness(report, investigation=investigation, comments=comments))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/final-snapshots/{snapshot_id}")
def get_final_snapshot(snapshot_id: str):
    try:
        return to_jsonable(get_store().get_final_report_snapshot(snapshot_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/final-snapshots/{snapshot_id}/revoke")
def revoke_final_snapshot(snapshot_id: str, payload: RevokeSnapshotPayload | None = None):
    try:
        body = payload or RevokeSnapshotPayload()
        return to_jsonable(ReportEditingService(get_store()).revoke_final_report_snapshot(snapshot_id, body.reason))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{report_id}/comments")
def get_report_comments(report_id: str, section_id: str | None = None):
    try:
        return [to_jsonable(item) for item in get_store().list_report_comments(report_id, section_id=section_id)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{report_id}/comments")
def create_report_comment(report_id: str, payload: CommentCreate):
    try:
        service = ReportEditingService(get_store())
        return to_jsonable(
            service.add_comment(
                report_id,
                section_id=payload.section_id,
                text=payload.text,
                author=payload.author,
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{report_id}/approve")
def approve_report(report_id: str, payload: ApprovalRequest | None = None):
    try:
        body = payload or ApprovalRequest()
        return to_jsonable(
            ReportEditingService(get_store()).approve_report(
                report_id,
                approved_by=body.approved_by,
                force=body.force,
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{report_id}/request-changes")
def request_report_changes(report_id: str, payload: RequestChangesPayload):
    try:
        return to_jsonable(ReportEditingService(get_store()).request_changes(report_id, notes=payload.notes))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{report_id}/sections/{section_id}/approve")
def approve_report_section(report_id: str, section_id: str):
    try:
        return to_jsonable(ReportEditingService(get_store()).approve_section(report_id, section_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{report_id}/sections/{section_id}/request-changes")
def request_report_section_changes(report_id: str, section_id: str):
    try:
        return to_jsonable(ReportEditingService(get_store()).request_section_changes(report_id, section_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
