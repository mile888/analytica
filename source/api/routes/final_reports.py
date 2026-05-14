from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from source.api.deps import get_store
from source.api.serialization import to_jsonable
from source.product.final_report_registry import list_published_reports
from source.product.report_service import ReportEditingService


router = APIRouter(prefix="/final-reports", tags=["final-reports"])


class DecisionMetadataPayload(BaseModel):
    tags: list[str] = []
    owner: str | None = None
    audience: str | None = None
    business_area: str | None = None
    decision_date: str | None = None
    decision_status: str = "unknown"
    short_description: str | None = None


class TagsPayload(BaseModel):
    tags: list[str]


class DecisionStatusPayload(BaseModel):
    status: str


@router.get("")
def get_final_reports(
    status: str | None = None,
    investigation_id: str | None = None,
    report_id: str | None = None,
    search: str | None = None,
    decision_status: str | None = None,
    business_area: str | None = None,
    tag: str | None = None,
    limit: int = 100,
):
    return to_jsonable(
        list_published_reports(
            get_store(),
            status=status,
            investigation_id=investigation_id,
            report_id=report_id,
            search=search,
            decision_status=decision_status,
            business_area=business_area,
            tag=tag,
            limit=limit,
        )
    )


@router.get("/{snapshot_id}")
def get_final_report(snapshot_id: str):
    try:
        return to_jsonable(get_store().get_final_report_snapshot(snapshot_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{snapshot_id}/download/markdown")
def download_final_markdown(snapshot_id: str):
    try:
        snapshot = get_store().get_final_report_snapshot(snapshot_id)
        return Response(content=snapshot.markdown_content, media_type="text/markdown")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{snapshot_id}/download/html")
def download_final_html(snapshot_id: str):
    try:
        snapshot = get_store().get_final_report_snapshot(snapshot_id)
        return Response(content=snapshot.html_content, media_type="text/html")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{snapshot_id}/revoke")
def revoke_final_report(snapshot_id: str):
    try:
        return to_jsonable(ReportEditingService(get_store()).revoke_final_report_snapshot(snapshot_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/{snapshot_id}/metadata")
def update_final_report_metadata(snapshot_id: str, payload: DecisionMetadataPayload):
    try:
        data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        return to_jsonable(
            ReportEditingService(get_store()).update_final_report_metadata(snapshot_id, data)
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{snapshot_id}/tags")
def add_final_report_tags(snapshot_id: str, payload: TagsPayload):
    try:
        return to_jsonable(ReportEditingService(get_store()).add_final_report_tags(snapshot_id, payload.tags))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{snapshot_id}/tags/{tag}")
def remove_final_report_tag(snapshot_id: str, tag: str):
    try:
        return to_jsonable(ReportEditingService(get_store()).remove_final_report_tag(snapshot_id, tag))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{snapshot_id}/decision-status")
def set_final_report_decision_status(snapshot_id: str, payload: DecisionStatusPayload):
    try:
        return to_jsonable(ReportEditingService(get_store()).set_decision_status(snapshot_id, payload.status))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
