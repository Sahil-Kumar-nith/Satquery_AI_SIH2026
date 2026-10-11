"""
geo_export.py — writes a computed DSM/rDSM to GeoTIFF, satisfying the
"output a high-fidelity DSM in a standard geospatial format" requirement.

Two cases:
  - Georeferenced input (GeoTIFF in -> CRS/transform known): output carries
    the same CRS and transform, so it's a real, GIS-loadable absolute DSM.
  - Non-georeferenced input (PNG/JPG in -> no CRS): output is still a valid
    single-band GeoTIFF (so downstream GIS tools can open it), but tagged
    with an identity transform and no CRS, and a metadata flag marking it
    as relative/unitless. This is honest about what the data is — we don't
    fabricate a coordinate system that doesn't exist.
"""

import io
import logging

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS

logger = logging.getLogger("depthwizard.geo_export")


def export_dsm_to_geotiff(
    height_array: np.ndarray,
    crs_string: str | None = None,
    bounds: list[float] | None = None,
    is_absolute: bool = False,
) -> bytes:
    """
    Serializes a height array to an in-memory GeoTIFF and returns raw bytes.

    height_array: float32 HxW array of elevation (meters) or relative depth.
    crs_string: e.g. "EPSG:32643" — from the source GeoTIFF, if any.
    bounds: [left, bottom, right, top] in the source CRS, if georeferenced.
    is_absolute: whether height_array is metric (True) or relative (False).
    """
    h, w = height_array.shape

    if crs_string and bounds:
        crs = CRS.from_string(crs_string)
        left, bottom, right, top = bounds
        transform = rasterio.transform.from_bounds(left, bottom, right, top, w, h)
    else:
        # No georeferencing available — identity transform, no CRS.
        # Still produces a spec-valid single-band GeoTIFF.
        crs = None
        transform = Affine.identity()

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": w,
        "height": h,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
    }

    buf = io.BytesIO()
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(height_array.astype(np.float32), 1)
            dst.update_tags(
                DEPTHWIZARD_MODE="absolute_meters" if is_absolute else "relative_unitless",
                DEPTHWIZARD_SOURCE="DepthAnythingV2 + DepthWizard calibration",
            )
        buf.write(memfile.read())

    buf.seek(0)
    logger.info(f"Exported GeoTIFF: {w}x{h}, crs={crs}, absolute={is_absolute}")
    return buf.getvalue()