from __future__ import annotations

from datetime import datetime
from typing import Any


MAX_EVENT_STREAM_LIMIT = 500
DEFAULT_EVENT_STREAM_LIMIT = 100


def encode_event_cursor(event: Any) -> str:
    return f"{event.created_at.isoformat()}|{event.event_id}"


def decode_event_cursor(cursor: str) -> tuple[str, str]:
    if not cursor or "|" not in cursor:
        raise ValueError("Invalid event cursor.")
    created_at, event_id = cursor.rsplit("|", 1)
    datetime.fromisoformat(created_at)
    if not event_id:
        raise ValueError("Invalid event cursor.")
    return created_at, event_id


def filter_events_after(events: list[Any], cursor: str | None) -> list[Any]:
    ordered = sorted(events, key=lambda event: (event.created_at.isoformat(), event.event_id))
    if not cursor:
        return ordered
    try:
        cursor_created_at, cursor_event_id = decode_event_cursor(cursor)
    except ValueError:
        return ordered
    return [
        event
        for event in ordered
        if (event.created_at.isoformat(), event.event_id) > (cursor_created_at, cursor_event_id)
    ]


def build_event_stream_response(
    events: list[Any],
    cursor: str | None,
    limit: int = DEFAULT_EVENT_STREAM_LIMIT,
) -> dict[str, Any]:
    resolved_limit = max(1, min(int(limit or DEFAULT_EVENT_STREAM_LIMIT), MAX_EVENT_STREAM_LIMIT))
    filtered = filter_events_after(events, cursor)
    page = filtered[:resolved_limit]
    return {
        "events": page,
        "next_cursor": encode_event_cursor(page[-1]) if page else cursor,
        "has_more": len(filtered) > resolved_limit,
    }
