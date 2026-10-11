"""
GET /api/health

Simple liveness endpoint used to verify the frontend <-> backend
connection during Milestone 1. Expand later with subsystem checks
(model registry loaded? GDAL available?) but keep it fast and boring —
this gets polled by the UI status dot.
"""
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="satquery-ai-backend",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
