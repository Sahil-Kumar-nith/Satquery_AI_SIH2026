"""
Cloud coverage estimator for optical imagery.

Phase 9 (P0 — hero feature enabler). Satisfies the master brief's core
innovation: "The system should determine whether available evidence is
sufficient for a query."

MODEL CHOICE — brightness/low-saturation heuristic, NOT a trained cloud
segmentation model (e.g. s2cloudless) and NOT the Sentinel-2 QA60 cloud
mask band. This pipeline accepts arbitrary GeoTIFF/TIFF/PNG/JPEG uploads
and does not guarantee a QA band is present or which band index it would
be — so no "we read the official cloud mask" claim is made.

Heuristic: a pixel is "cloud-like" if it is BOTH bright (after a 2-98
percentile stretch, same normalization used everywhere else in this
codebase) AND low-saturation (small spread between its brightest and
darkest channel — i.e. close to white/grey rather than vividly colored
land cover). This is the same rule of thumb a person uses to eyeball
cloud cover in a thumbnail.

Known false positives (disclose these, don't hide them): bright sand,
snow, salt flats, and pale rooftops will register as "cloud" by this
heuristic. It is a prototype signal, not a validated cloud mask — never
present its output as ground truth to a judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

_BRIGHTNESS_THRESHOLD = 190     # 0-255 scale, after percentile stretch
_SATURATION_SPREAD_THRESHOLD = 30  # max-min channel diff, 0-255 scale


@dataclass
class CloudEstimate:
    cloud_pct: float
    method: str
    note: str


def _percentile_stretch(a: np.ndarray) -> np.ndarray:
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a, dtype="float32")
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        return np.zeros_like(a, dtype="float32")
    return np.clip((a - lo) / (hi - lo), 0, 1) * 255


def estimate_cloud_coverage(optical_path: Path) -> CloudEstimate:
    with rasterio.open(optical_path) as src:
        count = src.count
        if count >= 3:
            r, g, b = (src.read(i).astype("float32") for i in (1, 2, 3))
        else:
            band = src.read(1).astype("float32")
            r = g = b = band

    brightness = _percentile_stretch((r + g + b) / 3.0)
    spread = _percentile_stretch(np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b]))

    cloud_mask = (brightness > _BRIGHTNESS_THRESHOLD) & (spread < _SATURATION_SPREAD_THRESHOLD)
    cloud_pct = float(cloud_mask.mean() * 100)

    note = (
        f"{cloud_pct:.1f}% of pixels are bright and low-color-variation (a 'looks white/grey' "
        "heuristic), not a validated cloud mask such as Sentinel-2 QA60 or s2cloudless. Can "
        "misfire on snow, sand, salt flats, or pale rooftops — treat as a prototype signal, "
        "not ground truth."
    )
    return CloudEstimate(
        cloud_pct=round(cloud_pct, 1),
        method="brightness_saturation_heuristic (deterministic, not a trained cloud model)",
        note=note,
    )
