from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from source.api.deps import get_store
from source.api.serialization import to_jsonable
from source.product.event_stream import build_event_stream_response
from source.product.run_service import InvestigationRunService


router = APIRouter(prefix="/investigations", tags=["investigations"])
runs_router = APIRouter(prefix="/investigation-runs", tags=["investigation-runs"])


class InvestigationCreate(BaseModel):
    question: str
    title: str | None = None
    data_source_ids: list[str] = Field(default_factory=list)


class InvestigationRunRequest(BaseModel):
    data_source_ids: list[str] = Field(default_factory=list)
    force_refresh_context: bool = False


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
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{investigation_id}/runs")
def list_runs_for_investigation(investigation_id: str):
    try:
        return [to_jsonable(item) for item in get_store().list_runs_for_investigation(investigation_id)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
