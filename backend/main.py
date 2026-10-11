"""
SatQuery AI — FastAPI backend entrypoint.

Phase 1-2: app bootstrap, CORS, /api/health.
Phase 3: /api/upload — GeoTIFF/TIFF input validation (P0, PS requirement).
Phase 4: /api/query — query classification + input compatibility +
         tool selection + execution trace (P0, PS requirement).
Phase 5: /api/query now runs a real VQA specialist model
         (models/vqa.py, Salesforce/blip-vqa-base + optional
         BigEarthNet LoRA adapter) for the vqa task. Other tasks
         remain routing-only until Phases 6-8.

Add feature routers here only once they exist and are imported
explicitly — keep this file a thin composition root.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import health, query, upload

app = FastAPI(
    title="SatQuery AI Backend",
    description="Agentic remote-sensing vision-language assistant — orchestration API.",
    version="0.1.0",
)

# Local dev origins only. Tighten before any real deployment.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(query.router, prefix="/api")


@app.get("/")
def root():
    return {"service": "satquery-ai-backend", "status": "running", "docs": "/docs"}
