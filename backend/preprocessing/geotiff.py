"""
GeoTIFF/TIFF validation and metadata extraction using Rasterio.

Phase 3 (P0 — PS requirement: "Input upload and compatibility checking",
GeoTIFF/TIFF input support).

Kept deliberately simple: open with rasterio, pull out what the agent
and evidence engine will need later (dimensions, band count, CRS,
bounds, nodata). No reprojection, no resampling — that is out of scope
for upload-time validation and belongs in Phase 4+ preprocessing steps
that run per-task (e.g. co-registration checks for optical+SAR).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rasterio
from rasterio.errors import RasterioIOError


class InvalidImageError(ValueError):
    """Raised when a file cannot be validated as a usable raster."""


@dataclass
class RasterMetadata:
    format: str
    driver: str
    width: int
    height: int
    band_count: int
    dtype: str
    crs: Optional[str]
    bounds: Optional[tuple]
    is_geospatial: bool
    nodata: Optional[float]


# PS: "Supported formats: GeoTIFF or TIFF for geospatial imagery. PNG and
# JPEG inputs may be accepted only for the prescribed public benchmark
# datasets." We accept both but flag is_geospatial accordingly — PNG/JPEG
# will not carry a CRS/transform and downstream geospatial features
# (map overlay, bounds) must degrade gracefully for those.
GEOSPATIAL_EXTENSIONS = {".tif", ".tiff"}
BENCHMARK_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def validate_and_extract(path: Path) -> RasterMetadata:
    """
    Open `path` with Rasterio and extract metadata.

    Raises InvalidImageError on anything Rasterio can't open, or on
    basic sanity-check failures (zero-size raster, no bands).
    """
    ext = path.suffix.lower()
    if ext not in GEOSPATIAL_EXTENSIONS | BENCHMARK_EXTENSIONS:
        raise InvalidImageError(
            f"Unsupported file extension '{ext}'. Expected GeoTIFF/TIFF "
            f"({', '.join(sorted(GEOSPATIAL_EXTENSIONS))}) or a benchmark "
            f"image ({', '.join(sorted(BENCHMARK_EXTENSIONS))})."
        )

    try:
        with rasterio.open(path) as src:
            if src.width == 0 or src.height == 0:
                raise InvalidImageError("Raster has zero width or height.")
            if src.count == 0:
                raise InvalidImageError("Raster has no bands.")

            crs = src.crs.to_string() if src.crs else None
            bounds = tuple(src.bounds) if src.crs else None

            return RasterMetadata(
                format="GeoTIFF/TIFF" if ext in GEOSPATIAL_EXTENSIONS else "benchmark-image",
                driver=src.driver,
                width=src.width,
                height=src.height,
                band_count=src.count,
                dtype=str(src.dtypes[0]) if src.dtypes else "unknown",
                crs=crs,
                bounds=bounds,
                is_geospatial=crs is not None,
                nodata=src.nodata,
            )
    except RasterioIOError as exc:
        raise InvalidImageError(f"Could not open file as a raster: {exc}") from exc
