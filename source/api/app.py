from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from source.config import CORS_ORIGINS
from source.api.routes.data_sources import router as data_sources_router
from source.api.routes.final_reports import router as final_reports_router
from source.api.routes.investigations import router as investigations_router, runs_router
from source.api.routes.reports import router as reports_router


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


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    return {"status": "ok"}
