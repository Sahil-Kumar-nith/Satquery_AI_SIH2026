"""
Area estimation for a previously-grounded region — Phase 16, follow-up
chain hop 2 ("How much area?" after "Where exactly?").

Closes part of the gap flagged in PHASE15.md §2 ("What this does NOT
do... it does not chain area calculation... automatically") and
LEFT.md §3 ("Multi-step follow-up conversations... the rest of the
chain (area size, then radar cross-check) is not built yet").

HONESTY CONTRACT:
This is a geometric calculation over the PREVIOUS turn's bounding box —
not a new model call, not a segmentation of the actual object, and not a
guaranteed-accurate measurement. It only produces a real area figure
when the source image carries usable georeferencing (a CRS + bounds from
GeoTIFF metadata, in projected/metric units):

- PNG/JPEG benchmark images have no CRS at all -> area is reported as
  unavailable, never guessed.
- GeoTIFFs in a geographic (degree-based) CRS, e.g. plain WGS84
  (EPSG:4326) without a projection, would need proper reprojection to
  convert degrees to metres — this module deliberately does NOT do that
  (no pyproj dependency added for a hackathon follow-up feature) and
  instead reports area as unavailable rather than silently treating
  degrees as if they were metres, which would be badly wrong at most
  latitudes.
- Only GeoTIFFs in a recognisably projected/metric CRS (UTM zones, Web
  Mercator, a handful of common projected EPSG codes) get a real number,
  computed as: pixel-box footprint -> ground distance (using this
  image's own bounds/width/height as its ground-sample distance) ->
  rectangular area in km². This is the box's rectangular footprint, not
  a precise outline of the detected object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AreaResult:
    area_km2: Optional[float]
    area_note: str
    method: str


# Pragmatic prototype check on the CRS string only — not a full units
# lookup via a projections database. Good enough to catch the common
# "GeoTIFF in a UTM/Mercator-family projected CRS" case that matters for
# a hackathon demo; anything else is honestly reported as unavailable
# rather than guessed. State this limitation wherever this is surfaced.
_PROJECTED_CRS_HINTS = ("UTM", "3857", "MERCATOR", ":326", ":327", ":3035", ":5070", ":2163")
_GEOGRAPHIC_CRS_HINTS = ("4326", "WGS 84", "WGS84", "LONGLAT", "LONG/LAT", "GEOGCS")


def _looks_projected(crs: Optional[str]) -> Optional[bool]:
    """Returns True (projected/metric), False (geographic/degrees), or
    None (couldn't tell from the string — treated as unsupported)."""
    if not crs:
        return None
    c = crs.upper()
    if any(h in c for h in _GEOGRAPHIC_CRS_HINTS):
        return False
    if any(h in c for h in _PROJECTED_CRS_HINTS):
        return True
    return None


def estimate_box_area_km2(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    crs: Optional[str],
    bounds: Optional[tuple],
) -> AreaResult:
    """box is (x_min, y_min, x_max, y_max) in PIXEL coordinates on the
    same image the box was originally computed from — image_width/
    image_height/crs/bounds must all come from that same image's own
    upload metadata (api/upload.py's UploadResponse), never a different
    image, or the ground-sample distance will be silently wrong."""
    x_min, y_min, x_max, y_max = box

    if not crs or not bounds or image_width <= 0 or image_height <= 0:
        return AreaResult(
            area_km2=None,
            area_note=(
                "Area unavailable — the source image has no georeferencing metadata "
                "(no CRS/bounds), so pixel dimensions cannot be converted to a ground "
                "distance. This is expected for PNG/JPEG benchmark images."
            ),
            method="unavailable_no_georeferencing",
        )

    projected = _looks_projected(crs)
    if projected is not True:
        reason = (
            f"source CRS ({crs}) appears to be a geographic (degree-based) coordinate "
            "system, not a projected metric one"
            if projected is False
            else f"source CRS ({crs}) could not be confidently identified as projected/metric"
        )
        return AreaResult(
            area_km2=None,
            area_note=(
                f"Area unavailable — {reason}. Converting degrees to km\u00b2 without proper "
                "reprojection would silently misestimate area (the error varies with "
                "latitude), so this is intentionally left unreported rather than guessed. "
                "Reproject the source GeoTIFF to a projected CRS (e.g. the appropriate UTM "
                "zone) to get a real area figure here."
            ),
            method="unavailable_geographic_crs",
        )

    minx, miny, maxx, maxy = bounds
    px_width_m = abs(maxx - minx) / image_width
    px_height_m = abs(maxy - miny) / image_height

    box_width_px = abs(x_max - x_min)
    box_height_px = abs(y_max - y_min)

    box_width_m = box_width_px * px_width_m
    box_height_m = box_height_px * px_height_m
    area_km2 = (box_width_m * box_height_m) / 1_000_000.0

    return AreaResult(
        area_km2=round(area_km2, 4),
        area_note=(
            f"Approximate area from the previously grounded bounding box: "
            f"{box_width_m:.0f} m \u00d7 {box_height_m:.0f} m, using this image's own GeoTIFF "
            f"ground-sample distance ({px_width_m:.2f} m/px \u00d7 {px_height_m:.2f} m/px, derived "
            "from its bounds/width/height — not a hardcoded sensor resolution). This is the "
            "box's rectangular footprint, not a precise segmentation of the actual "
            "object/region — treat it as a rough order-of-magnitude estimate, not a "
            "validated measurement."
        ),
        method="bbox_pixel_to_ground_geotiff_gsd",
    )
