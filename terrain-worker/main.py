"""
main.py — DepthWizard ML microservice.

FastAPI service that:
  1. Accepts an uploaded RGB image (PNG/JPG or GeoTIFF)
  2. Runs Depth Anything V2 to get relative depth
  3. If georeferenced + GCPs/SRTM provided -> calibrates to absolute meters
     Otherwise -> returns normalized relative depth (rDSM)
  4. Returns depth as base64 PNG (for quick preview) + base64 float32 raw
     array (for the Three.js terrain mesh, which needs real height values)

Run (SatQuery integration — port 8010, kept off the main backend's 8000
so a heavy depth model can never block/crash the PS-mandated agent):
    uvicorn main:app --host 0.0.0.0 --port 8010 --reload
"""

import base64
import io
import json
import logging
import time

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from calibration import GCP, calibrate_with_gcps, calibrate_with_srtm, normalize_relative
from model import get_estimator
from geo_export import export_dsm_to_geotiff
from validation import validate as validate_dsm

# 1. IMPORT SPECTRAL ROUTER
import spectral
import object_finder 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("depthwizard.main")

app = FastAPI(title="DepthWizard ML Worker", version="0.3.0")

# 3. ADD MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"], # Added "*" to prevent CORS issues during local dev
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. INCLUDE ROUTERS (Now that 'app' actually exists!)
app.include_router(spectral.router)
app.include_router(object_finder.router)

GEO_EXTENSIONS = {".tif", ".tiff"}
MAX_UPLOAD_MB = 25


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/warmup")
def warmup():
    """Force the model to load now instead of on first real request."""
    t0 = time.time()
    get_estimator()
    return {"loaded": True, "seconds": round(time.time() - t0, 2)}


def _is_geotiff(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in GEO_EXTENSIONS)


def _array_to_png_b64(arr: np.ndarray) -> str:
    """8-bit normalized PNG for quick client-side preview/thumbnailing."""
    a_min, a_max = float(arr.min()), float(arr.max())
    span = max(a_max - a_min, 1e-6)
    norm = ((arr - a_min) / span * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(norm).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _array_to_raw_b64(arr: np.ndarray) -> str:
    """Raw float32 buffer, base64-encoded, for the R3F heightmap (full precision)."""
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("utf-8")


@app.post("/process-depth")
async def process_depth(
    file: UploadFile = File(...),
    gcps: str = Form(default="[]"),
    export_geotiff: bool = Form(default=False),
):
    if file.size and file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB}MB limit")

    raw_bytes = await file.read()
    is_geo = _is_geotiff(file.filename or "")

    # --- Load image for the depth model (needs a plain RGB PIL image either way) ---
    try:
        if is_geo:
            import rasterio  # local import: optional dependency, only needed for GeoTIFF path

            with rasterio.open(io.BytesIO(raw_bytes)) as src:
                bands = src.read()  # (bands, H, W)
                crs = src.crs.to_string() if src.crs else None
                bounds = list(src.bounds) if src.crs else None
                transform = src.transform
            rgb = bands[:3].transpose(1, 2, 0) if bands.shape[0] >= 3 else np.repeat(bands[0][..., None], 3, axis=2)
            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        else:
            crs, bounds, transform = None, None, None
            image = Image.open(io.BytesIO(raw_bytes))
    except Exception as e:
        logger.exception("Failed to decode uploaded image")
        raise HTTPException(400, f"Could not read image: {e}")

    # --- Run monocular depth estimation ---
    estimator = get_estimator()
    result = estimator.predict(image)
    relative_depth = result.depth

    # --- Scale calibration ---
    parsed_gcps = json.loads(gcps) if gcps else []

    if is_geo and len(parsed_gcps) >= 2:
        gcp_objs = [GCP(**g) for g in parsed_gcps]
        calib = calibrate_with_gcps(relative_depth, gcp_objs)
    elif is_geo:
        logger.warning("GeoTIFF received with no GCPs — falling back to normalized relative depth.")
        calib = normalize_relative(relative_depth)
    else:
        calib = normalize_relative(relative_depth)

    response = {
        "mode": "absolute" if calib.method in ("gcp", "srtm") else "relative",
        "calibration_method": calib.method,
        "calibration": {
            "scale": calib.scale,
            "offset": calib.offset,
            "r_squared": calib.r_squared,
            "n_points": calib.n_points,
        },
        "width": result.width,
        "height": result.height,
        "inference_ms": round(result.inference_ms, 1),
        "depth_preview_png_b64": _array_to_png_b64(calib.absolute_height),
        "depth_raw_f32_b64": _array_to_raw_b64(calib.absolute_height),
        "geo": {
            "crs": crs,
            "bounds": bounds,
        } if is_geo else None,
    }
    
    if export_geotiff:
        geotiff_bytes = export_dsm_to_geotiff(
            calib.absolute_height,
            crs_string=crs if is_geo else None,
            bounds=bounds if is_geo else None,
            is_absolute=(calib.method in ("gcp", "srtm")),
        )
        response["dsm_geotiff_b64"] = base64.b64encode(geotiff_bytes).decode("utf-8")
    else:
        response["dsm_geotiff_b64"] = None

    return response


@app.post("/validate")
async def validate_endpoint(
    predicted_raw_f32_b64: str = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    reference_file: UploadFile = File(...),
    crs: str = Form(default=""),
    bounds: str = Form(default=""),  # JSON string: "[left,bottom,right,top]"
):
    """
    Compares an already-computed predicted DSM against a reference raster
    (e.g. a LiDAR-derived DSM) and returns RMSE/MAE/correlation.
    """
    predicted_bytes = base64.b64decode(predicted_raw_f32_b64)
    predicted = np.frombuffer(predicted_bytes, dtype=np.float32).reshape(height, width)

    reference_bytes = await reference_file.read()

    target_bounds = json.loads(bounds) if bounds else None
    target_crs = crs if crs else None

    try:
        result = validate_dsm(
            predicted,
            io.BytesIO(reference_bytes),
            target_crs_string=target_crs,
            target_bounds=target_bounds,
        )
    except Exception as e:
        logger.exception("Validation failed")
        raise HTTPException(400, f"Validation failed: {e}")

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)