"""
Raster -> PIL RGB conversion.

Phase 5. Every specialist vision model we integrate (VQA now, captioning/
grounding/change/fusion later) needs a plain RGB image, but uploads can be
1-band SAR, multi-band optical GeoTIFF, or 3-band PNG/JPEG benchmark
images. This is the single shared place that turns "whatever Rasterio can
open" into "3-channel uint8 PIL.Image" so every models/*.py file stays
model-only and doesn't reimplement band handling.

Band selection is intentionally simple (first 3 bands, or band 1 repeated
for single-band SAR/panchromatic). This is a reasonable prototype default,
not a radiometric-correctness claim — do not present the visualization as
calibrated reflectance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


def _to_uint8(band: np.ndarray) -> np.ndarray:
    """Percentile-stretch a single band to 0-255 uint8 for display/model input."""
    band = band.astype("float32")
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band, dtype="uint8")
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max() or 1.0)
    stretched = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
    return (stretched * 255).astype("uint8")


# Public alias — Phase 7/8 need single-band percentile stretching directly
# (grayscale diffing, SAR backscatter) without going through load_as_rgb's
# 3-channel stacking. Kept as a thin re-export so there's one implementation.
to_uint8 = _to_uint8


def load_as_rgb(path: Path) -> Image.Image:
    """Open any raster Phase 3 accepted and return a 3-channel PIL RGB image."""
    with rasterio.open(path) as src:
        count = src.count
        if count >= 3:
            r, g, b = (src.read(i) for i in (1, 2, 3))
        elif count == 1:
            band = src.read(1)
            r = g = b = band
        else:  # 2-band, uncommon — reuse band 1 for the missing channel
            r = g = src.read(1)
            b = src.read(2)

        rgb = np.dstack([_to_uint8(r), _to_uint8(g), _to_uint8(b)])
        return Image.fromarray(rgb, mode="RGB")
