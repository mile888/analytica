from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from urllib.parse import quote
import re
import unicodedata

from source.api.deps import get_store
from source.api.serialization import to_jsonable
from source.product.investigation import InvestigationMemoryItem, InvestigationMemoryType
from source.product.readiness import evaluate_report_readiness
from source.product.exporter import export_shareable_report_pdf, export_shareable_report_txt
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


class SectionUpdatePayload(BaseModel):
    title: str | None = None
    content: str | None = None


class SectionCreatePayload(BaseModel):
    title: str
    content: str = ""
    order: int | None = None


class SectionReorderPayload(BaseModel):
    section_ids: list[str]


class PromoteCommentPayload(BaseModel):
    investigation_id: str | None = None


class PromoteSectionPayload(BaseModel):
    investigation_id: str | None = None


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


@router.get("/{report_id}/download/txt")
def download_report_txt(report_id: str):
    try:
        report = get_store().get_shareable_report(report_id)
        return Response(content=export_shareable_report_txt(report), media_type="text/plain")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{report_id}/download/pdf")
def download_report_pdf(report_id: str):
    store = get_store()
    try:
        report = store.get_shareable_report(report_id)
        try:
            investigation = store.get_investigation(report.investigation_id)
            artifacts = investigation.artifacts
        except KeyError:
            artifacts = []
        ascii_filename = _ascii_pdf_filename(report.title)
        utf8_filename = quote(f"{report.title}.pdf")
        return Response(
            content=export_shareable_report_pdf(report, artifacts),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{utf8_filename}"},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _ascii_pdf_filename(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title or "")
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_name).strip("-._")
    return f"{ascii_name or 'analytica-report'}.pdf"


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


@router.post("/{report_id}/comments/{comment_id}/resolve")
def resolve_report_comment(report_id: str, comment_id: str):
    try:
        store = get_store()
        if comment_id not in {comment.comment_id for comment in store.list_report_comments(report_id)}:
            raise KeyError(f"ReportComment not found for report {report_id}: {comment_id}")
        return to_jsonable(ReportEditingService(store).resolve_comment(comment_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{report_id}/comments/{comment_id}")
def delete_report_comment(report_id: str, comment_id: str):
    try:
        store = get_store()
        if comment_id not in {comment.comment_id for comment in store.list_report_comments(report_id)}:
            raise KeyError(f"ReportComment not found for report {report_id}: {comment_id}")
        store.delete_report_comment(comment_id)
        return {"deleted": True, "comment_id": comment_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{report_id}/comments/{comment_id}/promote-memory")
def promote_comment_to_memory(report_id: str, comment_id: str, payload: PromoteCommentPayload | None = None):
    try:
        store = get_store()
        report = store.get_shareable_report(report_id)
        comments = store.list_report_comments(report_id)
        comment = next((item for item in comments if item.comment_id == comment_id), None)
        if comment is None:
            raise KeyError(f"ReportComment not found: {comment_id}")
        memory = InvestigationMemoryItem(
            investigation_id=(payload.investigation_id if payload and payload.investigation_id else report.investigation_id),
            memory_type=InvestigationMemoryType.OPEN_QUESTION,
            title="Promoted review comment",
            content=comment.text,
            metadata={
                "source_type": "report_comment",
                "source_id": comment.comment_id,
                "promoted_from": "report_comment",
                "report_id": report_id,
                "section_id": comment.section_id,
                "promoted_at": comment.created_at.isoformat(),
            },
        )
        return to_jsonable(store.add_investigation_memory_item(memory))
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


@router.post("/{report_id}/sections/{section_id}/promote-memory")
def promote_section_to_memory(report_id: str, section_id: str, payload: PromoteSectionPayload | None = None):
    try:
        store = get_store()
        report = store.get_shareable_report(report_id)
        section = next((item for item in report.sections if item.section_id == section_id), None)
        if section is None:
            raise KeyError(f"ReportSection not found: {section_id}")
        memory = InvestigationMemoryItem(
            investigation_id=(payload.investigation_id if payload and payload.investigation_id else report.investigation_id),
            memory_type=InvestigationMemoryType.DECISION,
            title=section.title,
            content=section.content or section.title,
            metadata={
                "source_type": "report_section",
                "source_id": section.section_id,
                "promoted_from": "report_section",
                "report_id": report_id,
                "promoted_at": section.updated_at.isoformat(),
            },
        )
        return to_jsonable(store.add_investigation_memory_item(memory))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/{report_id}/sections/{section_id}")
def update_report_section(report_id: str, section_id: str, payload: SectionUpdatePayload):
    try:
        if payload.title is None and payload.content is None:
            return to_jsonable(get_store().get_shareable_report(report_id))
        return to_jsonable(
            ReportEditingService(get_store()).update_section(
                report_id,
                section_id,
                title=payload.title,
                content=payload.content,
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{report_id}/sections")
def add_report_section(report_id: str, payload: SectionCreatePayload):
    try:
        return to_jsonable(
            ReportEditingService(get_store()).add_section(
                report_id,
                title=payload.title,
                content=payload.content,
                order=payload.order,
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{report_id}/sections/{section_id}")
def delete_report_section(report_id: str, section_id: str):
    try:
        return to_jsonable(ReportEditingService(get_store()).remove_section(report_id, section_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{report_id}/sections/{section_id}/duplicate")
def duplicate_report_section(report_id: str, section_id: str):
    try:
        return to_jsonable(ReportEditingService(get_store()).duplicate_section(report_id, section_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{report_id}/sections/reorder")
def reorder_report_sections(report_id: str, payload: SectionReorderPayload):
    try:
        return to_jsonable(ReportEditingService(get_store()).reorder_sections(report_id, payload.section_ids))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
