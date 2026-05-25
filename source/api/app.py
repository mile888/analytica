from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from source.config import CORS_ORIGINS
from source.api.routes.data_sources import router as data_sources_router
from source.api.routes.final_reports import router as final_reports_router
from source.api.routes.investigations import router as investigations_router, runs_router
from source.api.routes.reports import router as reports_router

logger = logging.getLogger("analytica.api")

app = FastAPI(title="Analytica Product API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ORIGINS),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(data_sources_router)
app.include_router(final_reports_router)
app.include_router(investigations_router)
app.include_router(runs_router)
app.include_router(reports_router)


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch any unhandled exception and return structured JSON instead of raw 500."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": f"An unexpected error occurred: {type(exc).__name__}",
            "detail": str(exc)[:500],
        },
    )


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    return {"status": "ok"}
