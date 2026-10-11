"""
calibration.py — converts scale-agnostic relative depth into absolute
(metric) elevation for georeferenced inputs.

Two calibration paths, matching the problem statement:
  1. Ground Control Points (GCPs): a handful of (pixel_x, pixel_y, known_height_m)
     pairs supplied by the user/dataset. Cheapest, no external data needed.
  2. Low-resolution reference DEM (e.g. SRTM 30m): if the caller has a local
     SRTM GeoTIFF covering the same footprint, we resample it onto the image
     grid and regress relative-depth -> SRTM elevation at sparse sample points.

Both reduce to fitting depth_relative -> height_absolute as an affine map
(height = a * depth + b), which is the standard scale+shift ambiguity of
monocular relative-depth predictions. For production you'd want a more
robust fit (RANSAC / per-landcover-class regression) — this is the Day-2
scaffold; swap `_fit_affine` for something fancier later without touching
the API surface.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("depthwizard.calibration")


@dataclass
class GCP:
    pixel_x: int
    pixel_y: int
    height_m: float


@dataclass
class CalibrationResult:
    absolute_height: np.ndarray   # float32 HxW, meters
    method: str                   # "gcp" | "srtm" | "uncalibrated"
    scale: float
    offset: float
    r_squared: Optional[float]
    n_points: int


def _fit_affine(depth_samples: np.ndarray, height_samples: np.ndarray):
    """Least-squares fit of height = a*depth + b. Returns (a, b, r_squared)."""
    a, b = np.polyfit(depth_samples, height_samples, deg=1)
    pred = a * depth_samples + b
    ss_res = np.sum((height_samples - pred) ** 2)
    ss_tot = np.sum((height_samples - np.mean(height_samples)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(b), float(r_squared)


def calibrate_with_gcps(relative_depth: np.ndarray, gcps: list[GCP]) -> CalibrationResult:
    if len(gcps) < 2:
        raise ValueError("Need at least 2 GCPs to fit scale + offset (3+ recommended).")

    depth_samples = np.array([relative_depth[g.pixel_y, g.pixel_x] for g in gcps])
    height_samples = np.array([g.height_m for g in gcps])

    a, b, r2 = _fit_affine(depth_samples, height_samples)
    absolute = (a * relative_depth + b).astype(np.float32)

    logger.info(f"GCP calibration: a={a:.4f} b={b:.4f} r2={r2:.3f} n={len(gcps)}")
    return CalibrationResult(absolute, "gcp", a, b, r2, len(gcps))


def calibrate_with_srtm(
    relative_depth: np.ndarray,
    srtm_resampled: np.ndarray,
    sample_stride: int = 25,
) -> CalibrationResult:
    """
    srtm_resampled: SRTM elevation already reprojected/resampled onto the
    SAME grid as relative_depth (same H,W). Do the reprojection in the
    route handler with rasterio before calling this.
    """
    if srtm_resampled.shape != relative_depth.shape:
        raise ValueError("srtm_resampled must match relative_depth shape — reproject first.")

    ys = np.arange(0, relative_depth.shape[0], sample_stride)
    xs = np.arange(0, relative_depth.shape[1], sample_stride)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")

    depth_samples = relative_depth[grid_y, grid_x].ravel()
    height_samples = srtm_resampled[grid_y, grid_x].ravel()

    valid = np.isfinite(height_samples) & np.isfinite(depth_samples)
    depth_samples, height_samples = depth_samples[valid], height_samples[valid]

    a, b, r2 = _fit_affine(depth_samples, height_samples)
    absolute = (a * relative_depth + b).astype(np.float32)

    logger.info(f"SRTM calibration: a={a:.4f} b={b:.4f} r2={r2:.3f} n={len(depth_samples)}")
    return CalibrationResult(absolute, "srtm", a, b, r2, len(depth_samples))


def normalize_relative(relative_depth: np.ndarray) -> CalibrationResult:
    """
    Fallback for non-georeferenced imagery: no absolute scale available.
    Min-max normalize to [0, 1] purely for visualization — NOT metric.
    """
    d_min, d_max = float(relative_depth.min()), float(relative_depth.max())
    span = max(d_max - d_min, 1e-6)
    normalized = ((relative_depth - d_min) / span).astype(np.float32)

    return CalibrationResult(normalized, "uncalibrated", 1.0 / span, -d_min / span, None, 0)