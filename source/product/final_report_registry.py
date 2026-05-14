from __future__ import annotations

from typing import Any

from source.product.investigation import FinalReportSnapshot, Investigation, ShareableReport


def list_published_reports(
    store,
    status: str | None = None,
    investigation_id: str | None = None,
    report_id: str | None = None,
    search: str | None = None,
    decision_status: str | None = None,
    business_area: str | None = None,
    tag: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    snapshots = store.list_final_report_snapshots(
        report_id=report_id,
        investigation_id=investigation_id,
        status=status,
    )
    summaries: list[dict[str, Any]] = []
    query = (search or "").strip().lower()
    for snapshot in snapshots:
        investigation = _safe_get_investigation(store, snapshot.investigation_id)
        report = _safe_get_report(store, snapshot.report_id)
        summary = get_published_report_summary(snapshot, investigation=investigation, report=report)
        if decision_status and summary["decision_status"] != decision_status:
            continue
        if business_area and (summary["business_area"] or "").lower() != business_area.strip().lower():
            continue
        if tag and _normalize(tag) not in {_normalize(item) for item in summary["tags"]}:
            continue
        if query:
            haystack = " ".join(
                [
                    str(summary.get("title") or ""),
                    str(summary.get("investigation_title") or ""),
                    str(summary.get("snapshot_id") or ""),
                    " ".join(summary.get("tags") or []),
                    str(summary.get("owner") or ""),
                    str(summary.get("audience") or ""),
                    str(summary.get("business_area") or ""),
                    str(summary.get("short_description") or ""),
                    str(summary.get("decision_status") or ""),
                ]
            ).lower()
            if query not in haystack:
                continue
        summaries.append(summary)
        if len(summaries) >= limit:
            break
    return summaries


def get_published_report_summary(
    snapshot: FinalReportSnapshot,
    investigation: Investigation | None = None,
    report: ShareableReport | None = None,
) -> dict[str, Any]:
    readiness = snapshot.readiness_snapshot or {}
    decision = snapshot.decision_metadata
    return {
        "snapshot_id": snapshot.snapshot_id,
        "title": snapshot.title,
        "status": snapshot.status.value,
        "created_at": snapshot.created_at.isoformat(),
        "created_by": snapshot.created_by,
        "investigation_id": snapshot.investigation_id,
        "investigation_title": investigation.title if investigation else None,
        "report_id": snapshot.report_id,
        "report_version": snapshot.report_version,
        "approval_status": snapshot.approval_status.value,
        "approved_at": snapshot.approved_at.isoformat() if snapshot.approved_at else None,
        "approved_by": snapshot.approved_by,
        "readiness_is_ready": bool(readiness.get("is_ready", False)),
        "readiness_blocking_count": int(readiness.get("blocking_count") or 0),
        "readiness_warning_count": int(readiness.get("warning_count") or 0),
        "download_available": bool(snapshot.markdown_content or snapshot.html_content),
        "source_report_is_latest": report.is_latest if report else None,
        "tags": decision.tags,
        "owner": decision.owner,
        "audience": decision.audience,
        "business_area": decision.business_area,
        "decision_date": decision.decision_date,
        "decision_status": decision.decision_status.value,
        "short_description": decision.short_description,
    }


def _safe_get_investigation(store, investigation_id: str) -> Investigation | None:
    try:
        return store.get_investigation(investigation_id)
    except KeyError:
        return None


def _safe_get_report(store, report_id: str) -> ShareableReport | None:
    try:
        return store.get_shareable_report(report_id)
    except KeyError:
        return None


def _normalize(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())
