"""
Confidence/evidence map rendering — turns a task's underlying evidence
array into a small base64 PNG heatmap attached to the API response,
plus a "source" attribution block naming which uploaded image(s) and
tool produced it.

Phase 13 (P0/P1 — closes a gap flagged during review: query responses
returned scalar confidence + a bounding box/change-map summary, but no
attached spatial confidence visualization, and no explicit per-result
"source" record separate from the free-text note).

HONESTY CONTRACT:
- change_detection / change_vqa / optical_sar already compute a real
  per-pixel array (diff z-score magnitude; agreement/optical-only/
  sar-only masks). Their confidence maps ARE that real array, just
  colorized — not a fabricated visualization.
- grounding's map is a synthetic Gaussian falloff centered on the
  returned box, scaled by the detector's score. It looks like a
  heatmap but is NOT a learned saliency map — the note says so.
- vqa has no spatial output at all (single free-text answer over the
  whole image). Its "confidence map" is a flat color tint of the
  answer-level confidence bucket (green/amber/red) — explicitly labeled
  as an answer-level indicator, NOT a pixel-level map, so it is never
  mistaken for spatial evidence that doesn't exist.

No matplotlib dependency — colormap is a small hand-rolled
blue -> green -> yellow -> red ramp using only numpy + Pillow (both
already in requirements.txt).
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from PIL import Image

# Simple 4-stop diverging ramp: low evidence -> blue, mid -> green/yellow,
# high -> red. Values in [0, 1].
_STOPS = [
    (0.00, (30, 60, 150)),    # low confidence / no evidence — blue
    (0.35, (40, 160, 90)),    # green
    (0.65, (230, 190, 40)),   # amber
    (1.00, (210, 40, 40)),    # red — high-magnitude evidence / change
]


def _colorize(values01: np.ndarray) -> np.ndarray:
    """values01: float array in [0,1]. Returns uint8 HxWx3."""
    v = np.clip(values01, 0.0, 1.0)
    out = np.zeros(v.shape + (3,), dtype="float32")
    for i in range(len(_STOPS) - 1):
        lo_t, lo_c = _STOPS[i]
        hi_t, hi_c = _STOPS[i + 1]
        band = (v >= lo_t) & (v <= hi_t)
        if not band.any():
            continue
        span = (hi_t - lo_t) or 1e-6
        frac = ((v[band] - lo_t) / span)[..., None]
        lo_arr = np.array(lo_c, dtype="float32")
        hi_arr = np.array(hi_c, dtype="float32")
        out[band] = lo_arr + frac * (hi_arr - lo_arr)
    return out.astype("uint8")


def _to_base64_png(rgb_array: np.ndarray, max_dim: int = 512) -> str:
    img = Image.fromarray(rgb_array, mode="RGB")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class ConfidenceMap:
    png_base64: str
    kind: Literal["pixel_evidence", "synthetic_heatmap", "answer_level_tint"]
    note: str


def from_change_diff(diff: np.ndarray, threshold_mask: np.ndarray) -> ConfidenceMap:
    """Real per-pixel evidence: normalized diff magnitude, with the
    thresholded 'changed' mask pushed toward red so it's visually obvious
    which pixels actually crossed the change threshold vs. background noise."""
    d = diff.astype("float32")
    rng = float(d.max() - d.min()) or 1.0
    norm = (d - d.min()) / rng
    norm = np.where(threshold_mask, np.clip(norm * 0.5 + 0.5, 0.5, 1.0), norm * 0.5)
    rgb = _colorize(norm)
    return ConfidenceMap(
        png_base64=_to_base64_png(rgb),
        kind="pixel_evidence",
        note=(
            "Real per-pixel evidence: normalized before/after intensity-difference "
            "magnitude. Pixels that crossed the statistical change threshold are "
            "pushed toward red; this is the same array the change_map percentage "
            "was computed from, not a separate/approximated visualization."
        ),
    )


def from_optical_sar_masks(agreement: np.ndarray, optical_only: np.ndarray, sar_only: np.ndarray) -> ConfidenceMap:
    """Real per-pixel evidence: 1.0 where both modalities agree (highest
    confidence), 0.6 where only one modality flags the target, 0.0 elsewhere."""
    score = np.zeros(agreement.shape, dtype="float32")
    score[optical_only | sar_only] = 0.55
    score[agreement] = 1.0
    rgb = _colorize(score)
    return ConfidenceMap(
        png_base64=_to_base64_png(rgb),
        kind="pixel_evidence",
        note=(
            "Real per-pixel evidence: red = optical AND SAR agree on the target "
            "(highest confidence), amber = only one modality flagged it, blue = "
            "neither. Same masks used to compute agreement_pct/optical_only_pct/"
            "sar_only_pct — this is not a separate/approximated visualization."
        ),
    )


def from_grounding_box(
    width: int, height: int, box: Optional[tuple], score: float
) -> ConfidenceMap:
    """Synthetic Gaussian falloff centered on the detected box, scaled by
    the detector's own match score. NOT a learned saliency/attention map —
    the returned note says so explicitly."""
    yy, xx = np.mgrid[0:height, 0:width]
    if box is None:
        score_field = np.zeros((height, width), dtype="float32")
    else:
        x_min, y_min, x_max, y_max = box
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        sx = max((x_max - x_min) / 2, 5.0)
        sy = max((y_max - y_min) / 2, 5.0)
        gauss = np.exp(-(((xx - cx) ** 2) / (2 * sx ** 2) + ((yy - cy) ** 2) / (2 * sy ** 2)))
        score_field = np.clip(gauss * min(max(score, 0.0), 1.0) * 1.5, 0, 1).astype("float32")
    rgb = _colorize(score_field)
    return ConfidenceMap(
        png_base64=_to_base64_png(rgb),
        kind="synthetic_heatmap",
        note=(
            "Synthetic visualization only: a Gaussian falloff centered on the "
            "detector's returned box, scaled by its match score. This is NOT a "
            "learned attention/saliency map from the detection model itself — it "
            "exists to make the box's location and confidence visually legible, "
            "not to claim pixel-level model attribution."
        ),
    )


def _load_gray(path) -> np.ndarray:
    """Mirrors the grayscale loader used in models/change.py and
    models/optical_sar.py exactly (same percentile stretch), so the
    confidence map reflects the SAME pixels those heuristics reasoned
    over — not a separately-approximated recomputation."""
    import rasterio  # deferred: heavy dependency, only needed when actually reading a raster
    from preprocessing.raster_to_rgb import to_uint8

    with rasterio.open(path) as src:
        band = src.read(1).astype("float32")
        if src.count >= 3:
            g = src.read(2).astype("float32")
            b = src.read(3).astype("float32")
            band = (band + g + b) / 3.0
    return to_uint8(band)


def _resize_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a2 = np.array(Image.fromarray(a.astype("uint8")).resize((w, h)))
    b2 = np.array(Image.fromarray(b.astype("uint8")).resize((w, h)))
    return a2, b2


def render_change_confidence_map(before_path, after_path, diff_z_threshold: float = 1.5) -> ConfidenceMap:
    """Path-based convenience wrapper: recomputes the same diff/threshold
    logic as models/change.py::compute_change_map so the confidence map
    stays a real (not fabricated) evidence array without requiring
    change.py's internal function signature to change."""
    before = _load_gray(before_path)
    after = _load_gray(after_path)
    before, after = _resize_to_match(before, after)
    diff = np.abs(after.astype("float32") - before.astype("float32"))
    mean, std = float(diff.mean()), float(diff.std() or 1.0)
    mask = diff > (mean + diff_z_threshold * std)
    return from_change_diff(diff, mask)


def render_optical_sar_confidence_map(optical_path, sar_path, target: str) -> ConfidenceMap:
    """Path-based convenience wrapper: recomputes the same percentile-
    threshold agreement masks as models/optical_sar.py::run_optical_sar."""
    optical = _load_gray(optical_path).astype("float32")
    sar = _load_gray(sar_path).astype("float32")
    optical, sar = _resize_to_match(optical, sar)

    if target == "water":
        opt_thresh, sar_thresh = np.percentile(optical, 25), np.percentile(sar, 25)
        opt_mask, sar_mask = optical <= opt_thresh, sar <= sar_thresh
    else:
        opt_thresh, sar_thresh = np.percentile(optical, 85), np.percentile(sar, 85)
        opt_mask, sar_mask = optical >= opt_thresh, sar >= sar_thresh

    agreement = opt_mask & sar_mask
    optical_only = opt_mask & ~sar_mask
    sar_only = sar_mask & ~opt_mask
    return from_optical_sar_masks(agreement, optical_only, sar_only)


def from_answer_confidence(width: int, height: int, confidence: float, level: str) -> ConfidenceMap:
    """VQA/captioning have no spatial output. Flat tint communicates the
    answer-level confidence bucket only — explicitly NOT a spatial map."""
    field = np.full((height, width), float(np.clip(confidence, 0.0, 1.0)), dtype="float32")
    rgb = _colorize(field)
    return ConfidenceMap(
        png_base64=_to_base64_png(rgb),
        kind="answer_level_tint",
        note=(
            f"This task has no spatial output — the flat {level.upper()} tint "
            "represents the single answer-level confidence score for the whole "
            "image, NOT a pixel-level confidence map. Do not present this as "
            "spatial evidence."
        ),
    )
