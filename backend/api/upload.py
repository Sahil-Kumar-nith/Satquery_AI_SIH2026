"""
POST /api/upload
GET  /api/upload/{image_id}

Phase 3 (P0). Satisfies PS requirements:
- "Input upload and compatibility checking"
- GeoTIFF/TIFF input support

Accepts one image file plus its modality (optical/sar) and role
(single/before/after), validates it with Rasterio, stores it on disk,
and returns structured metadata. The returned image_id is what the
agent (Phase 4+) will reference when routing a query and selecting
inputs for a specialist model.

In-memory registry only — fine for a single-process hackathon demo.
Restarting the backend clears uploaded image state; re-upload if needed.
Do not treat this as a real datastore.
"""
from __future__ import annotations

import uuid
from enum import Enum
from pathlib import Path
from typing import Optional

import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from preprocessing.geotiff import InvalidImageError, validate_and_extract
from preprocessing.correction import run_preprocessing_checks

router = APIRouter()

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB — generous for demo scenes


class Modality(str, Enum):
    optical = "optical"
    sar = "sar"


class ImageRole(str, Enum):
    single = "single"
    before = "before"
    after = "after"


class UploadResponse(BaseModel):
    image_id: str
    filename: str
    modality: Modality
    role: ImageRole
    format: str
    driver: str
    width: int
    height: int
    band_count: int
    dtype: str
    crs: Optional[str] = None
    bounds: Optional[tuple] = None
    is_geospatial: bool
    size_bytes: int
    # Phase 13: automated data-integrity / georeferencing / spectral-band /
    # atmospheric-correction diagnostics — see preprocessing/correction.py
    # for exactly what is and isn't performed (honesty contract in there).
    preprocessing_report: Optional[dict] = None


# image_id -> UploadResponse. Phase 4 (query router) looks images up here.
IMAGE_REGISTRY: dict[str, UploadResponse] = {}

# image_id -> on-disk path of the stored file. Kept separate from
# UploadResponse (an API response model) because the filesystem path is
# an internal implementation detail, not something we return to the
# client. Phase 5+ specialist models (models/vqa.py etc.) read from here.
IMAGE_PATHS: dict[str, Path] = {}


@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    modality: Modality = Form(...),
    role: ImageRole = Form(...),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=422, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    image_id = f"img_{uuid.uuid4().hex[:12]}"
    dest_path = UPLOAD_DIR / f"{image_id}{suffix}"

    size_bytes = 0
    try:
        with dest_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size_bytes += len(chunk)
                if size_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File exceeds 200MB limit.")
                out.write(chunk)
    except HTTPException:
        dest_path.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 — surface any I/O failure as 500
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    try:
        meta = validate_and_extract(dest_path)
    except InvalidImageError as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Phase 13: automated integrity / georeferencing / spectral-band /
    # atmospheric-correction diagnostics. Failure here should never block
    # the upload itself (this is diagnostic, not gatekeeping) — degrade to
    # a reported-but-empty report rather than a 500.
    try:
        report = run_preprocessing_checks(dest_path)
        preprocessing_report = {
            "integrity_ok": report.integrity_ok,
            "integrity_note": report.integrity_note,
            "geometric": report.geometric.__dict__,
            "spectral": report.spectral.__dict__,
            "atmospheric": report.atmospheric.__dict__,
        }
    except Exception as exc:  # noqa: BLE001 — diagnostics must never break upload
        preprocessing_report = {
            "integrity_ok": False,
            "integrity_note": f"Preprocessing diagnostics failed to run: {exc}",
            "geometric": None,
            "spectral": None,
            "atmospheric": None,
        }

    response = UploadResponse(
        image_id=image_id,
        filename=file.filename,
        modality=modality,
        role=role,
        format=meta.format,
        driver=meta.driver,
        width=meta.width,
        height=meta.height,
        band_count=meta.band_count,
        dtype=meta.dtype,
        crs=meta.crs,
        bounds=meta.bounds,
        is_geospatial=meta.is_geospatial,
        size_bytes=size_bytes,
        preprocessing_report=preprocessing_report,
    )
    IMAGE_REGISTRY[image_id] = response
    IMAGE_PATHS[image_id] = dest_path
    return response


@router.get("/upload/{image_id}", response_model=UploadResponse)
def get_upload(image_id: str) -> UploadResponse:
    meta = IMAGE_REGISTRY.get(image_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Unknown image_id.")
    return meta


# Simple in-process cache: rendering a GeoTIFF to RGB on every request is
# wasteful when the frontend re-requests the same preview across renders
# (e.g. the before/after slider mounting twice). Cleared implicitly on
# backend restart along with the rest of the in-memory registry.
_PREVIEW_CACHE: dict[str, bytes] = {}


@router.get("/upload/{image_id}/preview")
def get_upload_preview(image_id: str):
    """Phase 10 (P0 — UI). Renders whatever raster was uploaded (GeoTIFF,
    single-band SAR, multi-band optical, PNG/JPEG) into a displayable PNG
    using the same percentile-stretch pipeline the specialist models use
    (preprocessing/raster_to_rgb.py), so what judges see in the Satellite
    View panel is honestly the same pixels the models reasoned over — not
    a separately-styled "demo" image.

    This does NOT return calibrated reflectance/backscatter — it's a
    display convenience only, same disclosure as raster_to_rgb.py.
    """
    if image_id in _PREVIEW_CACHE:
        return Response(content=_PREVIEW_CACHE[image_id], media_type="image/png")

    meta = IMAGE_REGISTRY.get(image_id)
    path = IMAGE_PATHS.get(image_id)
    if meta is None or path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Unknown image_id or source file missing from disk.")

    from preprocessing.raster_to_rgb import load_as_rgb  # deferred: heavy import (rasterio) only when needed

    try:
        img = load_as_rgb(path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not render preview: {exc}") from exc

    # Cap displayed resolution — full-res GeoTIFFs can be huge and the
    # browser only needs enough pixels to fill the Satellite View panel.
    max_dim = 1600
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    _PREVIEW_CACHE[image_id] = data
    return Response(content=data, media_type="image/png")
