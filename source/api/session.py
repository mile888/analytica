from __future__ import annotations

from contextvars import ContextVar
from uuid import UUID, uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


SESSION_COOKIE_NAME = "analytica_session_id"

_current_session_id: ContextVar[str | None] = ContextVar("analytica_session_id", default=None)


def get_current_session_id() -> str | None:
    return _current_session_id.get()


def _valid_session_id(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(UUID(value))
    except (TypeError, ValueError):
        return None


class SessionCookieMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        incoming_session_id = _valid_session_id(request.cookies.get(SESSION_COOKIE_NAME))
        session_id = incoming_session_id or str(uuid4())
        token = _current_session_id.set(session_id)
        request.state.current_session_id = session_id
        try:
            response = await call_next(request)
        finally:
            _current_session_id.reset(token)
        response.headers["Cache-Control"] = "no-store"
        response.set_cookie(
            SESSION_COOKIE_NAME,
            session_id,
            httponly=True,
            secure=False,
            samesite="lax",
            path="/",
        )
        return response
