"""
Optical + SAR cross-modal fusion specialist "model".

Phase 8 (P0). Satisfies PS requirements:
- "Cross-modal pair analysis: The system must extract complementary
   information from a co-registered optical/multispectral and SAR image
   pair."
- This is also the backend half of the master brief's PRIMARY HERO
   FEATURE (cloud obstruction -> evidence insufficiency -> SAR
   activation -> multimodal evidence fusion). The Evidence Sufficiency
   Engine that decides WHEN to activate this is Phase 9 scope (P1/P2 per
   brief §12) — this phase only builds the fusion analysis itself so it
   exists to be activated.

MODEL CHOICE — deterministic per-pixel thresholding + fusion, NOT a
trained multimodal fusion network.
- Same constraint as Phase 7: no pip-installable, genuinely pretrained
  optical+SAR fusion model exists that meets "prefer existing pretrained
  models" + 4-day integration. The master brief explicitly allows this
  ("Optical features + SAR features -> simple fusion/classifier ->
  prediction" — Day 3 guidance).
- Optical channel: brightness-based water/built-up heuristic (NOT NDWI —
  that needs a NIR band this pipeline doesn't guarantee is present/known;
  using a labeled "brightness heuristic", not claiming spectral-index
  validity).
- SAR channel: backscatter-intensity threshold (low backscatter ~ open
  water / flooded surfaces; high backscatter ~ built-up/rough surfaces).
  This is the standard qualitative SAR interpretation heuristic, not a
  calibrated sigma-naught classifier — no radiometric calibration is
  performed on the uploaded raster.
- Fusion rule: simple agreement logic per pixel (BOTH indicate water ->
  high-confidence water; only one indicates it -> low-confidence;
  neither -> not flagged). This directly demonstrates "extract
  complementary information" without inventing a black-box model.
- Every number in the output is computed from the actual uploaded pixels
  at request time — nothing here is a canned/demo constant.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio

from preprocessing.raster_to_rgb import to_uint8

_WATER_DARK_PCT = 25      # optical: bottom 25th percentile brightness -> "dark/possible water"
_SAR_LOW_BACKSCATTER_PCT = 25  # SAR: bottom 25th percentile intensity -> "low backscatter/possible water or flood"
_BUILTUP_BRIGHT_PCT = 85       # optical: top 15% brightness -> "possible built-up/bare surface"
_SAR_HIGH_BACKSCATTER_PCT = 85  # SAR: top 15% intensity -> "possible built-up/rough surface"


@dataclass
class FusionRegionStats:
    label: str
    optical_only_pct: float
    sar_only_pct: float
    agreement_pct: float  # both modalities agree -> high-confidence
    confidence_level: Literal["high", "medium", "low"]


@dataclass
class OpticalSarResult:
    target: str  # "water" | "built_up"
    stats: FusionRegionStats
    answer: str
    width: int
    height: int
    method: str


def _load_gray(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        band = src.read(1).astype("float32")
        if src.count >= 3:
            g, b = src.read(2).astype("float32"), src.read(3).astype("float32")
            band = (band + g + b) / 3.0
    return to_uint8(band).astype("float32")


def _resize_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    if a.shape == b.shape:
        return a, b, False
    from PIL import Image
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a2 = np.array(Image.fromarray(a.astype("uint8")).resize((w, h))).astype("float32")
    b2 = np.array(Image.fromarray(b.astype("uint8")).resize((w, h))).astype("float32")
    return a2, b2, True


def _target_from_query(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ("water", "flood", "flooded", "river", "lake")):
        return "water"
    return "built_up"  # default — matches PS representative query "identify built-up and water-covered regions"


def run_optical_sar(optical_path: Path, sar_path: Path, query: str) -> OpticalSarResult:
    optical = _load_gray(optical_path)
    sar = _load_gray(sar_path)
    optical, sar, was_resized = _resize_to_match(optical, sar)

    target = _target_from_query(query)

    if target == "water":
        opt_thresh = np.percentile(optical, _WATER_DARK_PCT)
        sar_thresh = np.percentile(sar, _SAR_LOW_BACKSCATTER_PCT)
        opt_mask = optical <= opt_thresh   # dark in optical
        sar_mask = sar <= sar_thresh       # low backscatter in SAR
        label = "water / flooded surface"
    else:
        opt_thresh = np.percentile(optical, _BUILTUP_BRIGHT_PCT)
        sar_thresh = np.percentile(sar, _SAR_HIGH_BACKSCATTER_PCT)
        opt_mask = optical >= opt_thresh   # bright in optical
        sar_mask = sar >= sar_thresh       # high backscatter in SAR
        label = "built-up / bare surface"

    agreement = opt_mask & sar_mask
    optical_only = opt_mask & ~sar_mask
    sar_only = sar_mask & ~opt_mask

    total = optical.size
    agreement_pct = float(agreement.sum() / total * 100)
    optical_only_pct = float(optical_only.sum() / total * 100)
    sar_only_pct = float(sar_only.sum() / total * 100)

    if agreement_pct >= 3.0:
        level: Literal["high", "medium", "low"] = "high"
    elif agreement_pct >= 0.5:
        level = "medium"
    else:
        level = "low"

    stats = FusionRegionStats(
        label=label,
        optical_only_pct=round(optical_only_pct, 2),
        sar_only_pct=round(sar_only_pct, 2),
        agreement_pct=round(agreement_pct, 2),
        confidence_level=level,
    )

    answer = (
        f"Both optical and SAR evidence agree on {label} in {stats.agreement_pct}% of the "
        f"scene ({level.upper()} confidence). Optical alone flagged an additional "
        f"{stats.optical_only_pct}% (possible optical-only signal — e.g. shadow or cloud edge "
        f"for water, bare soil for built-up); SAR alone flagged {stats.sar_only_pct}% (possible "
        f"SAR-only signal — e.g. surface roughness unrelated to the target, or optical cloud "
        f"obstruction hiding a real feature)."
    )
    if was_resized:
        answer += (
            " NOTE: optical and SAR images had different pixel dimensions and were resized to "
            "match — this is NOT true geometric co-registration; treat spatial agreement as "
            "approximate, not pixel-exact."
        )

    return OpticalSarResult(
        target=target,
        stats=stats,
        answer=answer,
        width=int(optical.shape[1]),
        height=int(optical.shape[0]),
        method="percentile_threshold_agreement (deterministic, not a trained fusion model)",
    )
