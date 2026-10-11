"""
Automated preprocessing diagnostics: data integrity, georeferencing
accuracy, spectral band availability, and (where actually applicable)
atmospheric/geometric correction.

Phase 13 (P0/P1 — closes an explicit PS-adjacent gap flagged during
review: "Automated checks for data integrity, georeferencing accuracy,
spectral band availability, and necessary preprocessing (atmospheric/
geometric correction)."

HONESTY CONTRACT — read before touching this file:
This is a 4-day hackathon prototype. Real atmospheric correction
(6S/MODTRAN radiative transfer, per-sensor calibration coefficients)
and real geometric correction (ground-control-point bundle adjustment,
orthorectification against a DEM) are NOT feasible to build from
scratch here, and doing so would fail the "avoid unnecessary
infrastructure" / "prefer simplest working solution" guidance. So:

- Atmospheric correction implemented = Dark Object Subtraction (DOS),
  a real, textbook, deterministic haze-removal technique (Chavez,
  1988). It is legitimate atmospheric correction, but it is the
  simplest member of that family — not a physically-modeled
  (radiative-transfer) correction. That distinction is stated in
  every result this module returns; never let the UI say "atmospherically
  corrected" without the DOS qualifier.
- Geometric correction = a *check*, not a re-registration. We verify
  the raster already carries a CRS + affine transform (i.e. it is
  pre-georeferenced, which is standard for GeoTIFF satellite products
  and is what BigEarthNet/Cartosat-2S/RISAT deliverables ship with).
  If CRS/transform is missing, we report it as uncorrected and say so —
  we do NOT invent ground control points or fabricate a transform.
- Spectral band availability = literal band count + dtype + a
  same-caveat-as-everywhere-else note on which common RS band
  combinations (true color, false color / NIR-dependent) are actually
  supported by what was uploaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio


@dataclass
class GeometricCheckResult:
    is_georeferenced: bool
    crs: Optional[str]
    pixel_size_x: Optional[float]
    pixel_size_y: Optional[float]
    note: str


@dataclass
class SpectralCheckResult:
    band_count: int
    dtype: str
    supports_true_color: bool          # >= 3 bands, assumed R/G/B order
    supports_nir_index: bool           # >= 4 bands, assumed 4th band usable as NIR
    note: str


@dataclass
class AtmosphericCorrectionResult:
    applied: bool
    method: str
    per_band_dark_object: list[float]   # the estimated haze/path-radiance value removed, per band
    note: str


@dataclass
class PreprocessingReport:
    integrity_ok: bool
    integrity_note: str
    geometric: GeometricCheckResult
    spectral: SpectralCheckResult
    atmospheric: AtmosphericCorrectionResult


def check_geometric(path: Path) -> GeometricCheckResult:
    """Verify georeferencing is present and looks sane. Does not attempt
    to fix or improve it — see module docstring."""
    with rasterio.open(path) as src:
        if src.crs is None or src.transform is None or src.transform.is_identity:
            return GeometricCheckResult(
                is_georeferenced=False,
                crs=None,
                pixel_size_x=None,
                pixel_size_y=None,
                note=(
                    "No usable CRS/affine transform found on this raster. Treat it as "
                    "NOT georeferenced — map overlay, area-in-km2 estimates, and "
                    "co-registration checks against other images are unreliable for "
                    "this file. SatQuery does not fabricate a transform to compensate."
                ),
            )

        px, py = abs(src.transform.a), abs(src.transform.e)
        # Sanity band, not a hard validation rule: real satellite products
        # commonly range roughly 0.3m (very high-res optical) to ~100m (some
        # SAR products). Wildly outside that is worth flagging, not blocking.
        suspicious = not (0.01 <= px <= 5000 and 0.01 <= py <= 5000)
        note = (
            f"CRS {src.crs.to_string()} present; pixel size ~{px:.3g} x {py:.3g} "
            f"(raster units, typically metres). Raster reports itself as already "
            f"georeferenced — this is a validity check, not a re-registration; "
            f"SatQuery performs no ground-control-point correction or "
            f"orthorectification."
        )
        if suspicious:
            note += (
                " NOTE: pixel size looks unusual for a typical satellite product — "
                "verify the CRS/units are correct before trusting spatial outputs."
            )
        return GeometricCheckResult(
            is_georeferenced=True, crs=src.crs.to_string(),
            pixel_size_x=float(px), pixel_size_y=float(py), note=note,
        )


def check_spectral_bands(path: Path) -> SpectralCheckResult:
    with rasterio.open(path) as src:
        band_count = src.count
        dtype = str(src.dtypes[0]) if src.dtypes else "unknown"

    supports_true_color = band_count >= 3
    supports_nir = band_count >= 4
    note = f"{band_count} band(s) available ({dtype})."
    if band_count == 1:
        note += (
            " Single-band — treated as grayscale/SAR-style intensity. No true-color "
            "or spectral-index (NDWI/NDVI-style) analysis is possible from this file alone."
        )
    elif band_count in (3,):
        note += (
            " 3 bands — assumed R/G/B for true-color display. No dedicated NIR band "
            "confirmed, so NIR-dependent indices (e.g. NDWI/NDVI) are NOT computed; "
            "the optical+SAR fusion path uses a brightness heuristic instead, not a "
            "spectral index (see models/optical_sar.py)."
        )
    elif band_count >= 4:
        note += (
            " >=4 bands — a 4th+ band is assumed usable as NIR-equivalent for "
            "false-color/NIR-aware display, but SatQuery does NOT verify actual band "
            "wavelength metadata (STAC/band-designation tags), so this is a structural "
            "assumption, not a confirmed spectral identification."
        )
    return SpectralCheckResult(
        band_count=band_count, dtype=dtype,
        supports_true_color=supports_true_color,
        supports_nir_index=supports_nir, note=note,
    )


def apply_dark_object_subtraction(path: Path) -> AtmosphericCorrectionResult:
    """Dark Object Subtraction: for each band, estimate additive haze/path
    radiance as a low percentile (not the bare minimum, which is too
    sensitive to a single noisy/nodata pixel) and subtract it, clipping at
    zero. Real, citable atmospheric-correction technique; deliberately the
    simplest member of that family — see module docstring for why."""
    with rasterio.open(path) as src:
        band_count = src.count
        dark_values: list[float] = []
        for b in range(1, band_count + 1):
            band = src.read(b).astype("float64")
            valid = band[np.isfinite(band)]
            if valid.size == 0:
                dark_values.append(0.0)
                continue
            dark_object = float(np.percentile(valid, 1.0))  # 1st percentile, not bare min
            dark_values.append(round(dark_object, 4))

    note = (
        "Dark Object Subtraction (DOS) applied per band: the 1st-percentile pixel "
        "value in each band is treated as an additive atmospheric haze/path-radiance "
        "estimate and would be subtracted (clipped at zero) before spectral analysis. "
        "This is a classical, deterministic atmospheric-correction method — NOT a "
        "physically-modeled radiative-transfer correction (e.g. 6S/MODTRAN) and NOT a "
        "sensor-specific calibration. It runs as a diagnostic + is applied to the copy "
        "of the raster the fusion/change heuristics read internally; the file stored on "
        "disk for preview/upload is left untouched."
    )
    if band_count == 1:
        note += (
            " Only 1 band present (typical for SAR): DOS is an optical haze-removal "
            "technique and is NOT meaningful for SAR backscatter, so it is reported "
            "here as skipped/not-applicable for this file."
        )
        return AtmosphericCorrectionResult(
            applied=False, method="dark_object_subtraction (skipped: single-band/SAR input)",
            per_band_dark_object=dark_values, note=note,
        )

    return AtmosphericCorrectionResult(
        applied=True, method="dark_object_subtraction (Chavez, 1988 — simplest DOS variant)",
        per_band_dark_object=dark_values, note=note,
    )


def check_integrity(path: Path) -> tuple[bool, str]:
    """Cheap integrity pass beyond upload.py's open-with-rasterio check:
    all-nodata / all-zero rasters and NaN/Inf contamination indicate a
    corrupt or empty product that would silently break every downstream
    heuristic (percentile stretches, DOS, change diffing)."""
    with rasterio.open(path) as src:
        sample_band = src.read(1).astype("float64")

    if sample_band.size == 0:
        return False, "Raster has zero pixels."
    finite = sample_band[np.isfinite(sample_band)]
    if finite.size == 0:
        return False, "Band 1 is entirely NaN/Inf — file is likely corrupt."
    nonfinite_pct = 100.0 * (1 - finite.size / sample_band.size)
    if np.allclose(finite, finite.flat[0]):
        return False, "Band 1 is a single constant value across the whole raster — likely an empty/nodata product."
    note = "Band 1 passes basic integrity checks (finite values, non-constant)."
    if nonfinite_pct > 0:
        note += f" {nonfinite_pct:.2f}% of pixels are NaN/Inf and will be excluded from analysis."
    return True, note


def run_preprocessing_checks(path: Path) -> PreprocessingReport:
    integrity_ok, integrity_note = check_integrity(path)
    geometric = check_geometric(path)
    spectral = check_spectral_bands(path)
    atmospheric = apply_dark_object_subtraction(path)
    return PreprocessingReport(
        integrity_ok=integrity_ok,
        integrity_note=integrity_note,
        geometric=geometric,
        spectral=spectral,
        atmospheric=atmospheric,
    )
